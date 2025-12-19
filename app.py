import os
import sys
import tempfile
import gc
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import gradio as gr

# Ensure we can import from local src
sys.path.append(os.getcwd())

#############################################
# MemoryOptimizedInstantMesh
#############################################
class MemoryOptimizedInstantMesh:
    def __init__(self, config_path):
        print(f"Loading config from {config_path}")
        self.config = OmegaConf.load(config_path)
        self.config_name = os.path.basename(config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        self.is_flexicubes = True if self.config_name.startswith('instant-mesh') else False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.pipeline = None

    def load_pipeline(self):
        if self.pipeline is not None:
            return
        print("Loading Diffusion Pipeline...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
            trust_remote_code=True  # allow the repo to define its own pipeline class
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline = self.pipeline.to(self.device)



    def unload_pipeline(self):
        if self.pipeline is not None:
            self.pipeline = self.pipeline.to('cpu')
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()

    def load_model(self):
        if self.model is not None:
            return
        print("Loading Reconstruction Model...")
        from src.utils.train_util import instantiate_from_config
        model_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="instant_mesh_base.ckpt",
            repo_type="model"
        )
        self.model = instantiate_from_config(self.model_config)
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items()
                      if k.startswith('lrm_generator.') and 'source_camera' not in k}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        if self.is_flexicubes:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model = self.model.to('cpu')
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

    def remove_background(self, image, session):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return Image.fromarray(rembg.remove(image, session=session))

    def resize_foreground(self, image, scale=0.85):
        image_arr = np.array(image)
        if image_arr.shape[2] == 4:
            mask = image_arr[:, :, 3] > 0
            coords = np.where(mask)
            if len(coords[0]) == 0:
                return image
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            size = max(y_max - y_min, x_max - x_min)
            target_size = int(min(image.size) * scale)
            if size > target_size:
                scale_factor = target_size / size
                new_h = int((y_max - y_min) * scale_factor)
                new_w = int((x_max - x_min) * scale_factor)
                cropped = image.crop((x_min, y_min, x_max, y_max))
                resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
                new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
                paste_x = (image.size[0] - new_w) // 2
                paste_y = (image.size[1] - new_h) // 2
                new_image.paste(resized, (paste_x, paste_y))
                return new_image
        return image

    def generate_mvs(self, input_image, sample_steps, sample_seed):
        self.load_pipeline()
        seed_everything(sample_seed)
        try:
            generator = torch.Generator(device=self.device)
            z123_image = self.pipeline(
                input_image,
                num_inference_steps=sample_steps,
                generator=generator,
            ).images[0]
            show_image = np.asarray(z123_image, dtype=np.uint8)
            show_image = torch.from_numpy(show_image)
            show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
            show_image = Image.fromarray(show_image.numpy())
            return z123_image, show_image
        finally:
            self.unload_pipeline()

    def get_render_cameras(self, batch_size=1, M=120, radius=2.5, elevation=10.0):
        from src.utils.camera_util import FOV_to_intrinsics, get_circular_camera_poses
        c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
        if self.is_flexicubes:
            cameras = torch.linalg.inv(c2ws)
            cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            extrinsics = c2ws.flatten(-2)
            intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
            cameras = torch.cat([extrinsics, intrinsics], dim=-1)
            cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
        return cameras.to(self.device)

    def make3d(self, images):
        self.load_model()
        try:
            images = np.asarray(images, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
            images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
            from src.utils.camera_util import get_zero123plus_input_cameras
            input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)
            render_cameras = self.get_render_cameras(batch_size=1, radius=4.5, elevation=20.0).to(self.device)
            images = images.unsqueeze(0).to(self.device)
            images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
            with torch.no_grad():
                planes = self.model.forward_planes(images, input_cameras)
                frames = self.render_frames(render_cameras, planes)
            video_path = self.save_video(frames[0])
            mesh_path = self.generate_mesh(planes)
            return video_path, mesh_path
        finally:
            del images, input_cameras, render_cameras, planes, frames
            torch.cuda.empty_cache()
            gc.collect()
            self.unload_model()

    def render_frames(self, render_cameras, planes):
        chunk_size = 20 if self.is_flexicubes else 1
        render_size = 384
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if self.is_flexicubes:
                frame = self.model.forward_geometry(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = self.model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
            if i + chunk_size < render_cameras.shape[1]:
                torch.cuda.empty_cache()
        return torch.cat(frames, dim=1)

    def generate_mesh(self, planes):
        from src.utils.mesh_util import save_obj
        mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        with torch.no_grad():
            mesh_out = self.model.extract_mesh(planes, use_texture_map=False, **self.infer_config)
            vertices, faces, vertex_colors = mesh_out
            vertices = vertices[:, [1, 2, 0]]
            vertices[:, -1] *= -1
            faces = faces[:, [2, 1, 0]]
            save_obj(vertices, faces, vertex_colors, mesh_path)
        return mesh_path

    def save_video(self, frames, fps=30):
        video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        frames_np = []
        for i in range(frames.shape[0]):
            frame = (frames[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames_np.append(frame)
        imageio.mimwrite(video_path, np.stack(frames_np), fps=fps, format='FFMPEG')
        del frames_np
        return video_path

#############################################
# Gradio Demo Creation
#############################################
def create_demo(instant_mesh):
    _HEADER_ = '''# InstantMesh: 2D Image to 3D Generation'''
    
    def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")

    def process_input(input_image, do_remove_background):
        if do_remove_background:
            rembg_session = rembg.new_session()
            input_image = instant_mesh.remove_background(input_image, rembg_session)
        input_image = instant_mesh.resize_foreground(input_image, 0.85)
        return input_image

    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label="Input Image", image_mode="RGBA", width=256, height=256, type="pil")
                    processed_image = gr.Image(label="Processed Image", image_mode="RGBA", width=256, height=256, type="pil", interactive=False)
                with gr.Row():
                    with gr.Group():
                        do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                        sample_seed = gr.Number(value=42, label="Seed Value", precision=0)
                        sample_steps = gr.Slider(label="Sample Steps", minimum=30, maximum=75, value=75, step=5)
                with gr.Row():
                    submit = gr.Button("Generate", variant="primary")
            with gr.Column():
                with gr.Row():
                    mv_show_images = gr.Image(label="Generated Multi-views", type="pil", width=379, interactive=False)
                    output_video = gr.Video(label="Video", format="mp4", width=379, autoplay=True, interactive=False)
                with gr.Row():
                    output_model_obj = gr.Model3D(label="Output Model (OBJ)", interactive=False)

        mv_images = gr.State()
        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=process_input, inputs=[input_image, do_remove_background], outputs=[processed_image]
        ).success(
            fn=lambda x, y, z: instant_mesh.generate_mvs(x, y, z), inputs=[processed_image, sample_steps, sample_seed], outputs=[mv_images, mv_show_images]
        ).success(
            fn=lambda x: instant_mesh.make3d(x), inputs=[mv_images], outputs=[output_video, output_model_obj]
        )
    return demo

if __name__ == "__main__":
    # Ensure config path is relative to the container structure
    config_path = 'configs/instant-mesh-base.yaml'
    instant_mesh = MemoryOptimizedInstantMesh(config_path=config_path)
    demo = create_demo(instant_mesh)
    demo.queue(max_size=10)
    # Listen on all interfaces
    demo.launch(server_name="0.0.0.0", server_port=43548)