import os
import gc
import tempfile
import torch
import numpy as np
import imageio
import rembg

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.transforms import v2
from pytorch_lightning import seed_everything

from model_download import load_diffusion_pipeline, download_mesh_checkpoint


class MemoryOptimizedInstantMesh:
    def __init__(self, config_path):
        self.config = OmegaConf.load(config_path)
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.model = None

    # ---------- Diffusion ----------
    def load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = load_diffusion_pipeline(self.device)

    def unload_pipeline(self):
        if self.pipeline:
            self.pipeline.to("cpu")
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()

    def generate_mvs(self, image, steps, seed):
        self.load_pipeline()
        seed_everything(seed)

        try:
            out = self.pipeline(
                image,
                num_inference_steps=steps,
                generator=torch.Generator(device=self.device),
            ).images[0]

            img = torch.from_numpy(np.asarray(out, np.uint8))
            img = rearrange(img, "(n h) (m w) c -> (n m) h w c", n=3, m=2)
            img = rearrange(img, "(n m) h w c -> (n h) (m w) c", n=2, m=3)

            return out, Image.fromarray(img.numpy())

        finally:
            self.unload_pipeline()

    # ---------- Reconstruction ----------
    def load_model(self):
        if self.model:
            return

        from src.utils.train_util import instantiate_from_config

        ckpt = download_mesh_checkpoint()
        self.model = instantiate_from_config(self.model_config)

        state = torch.load(ckpt, map_location="cpu")["state_dict"]
        state = {
            k[14:]: v
            for k, v in state.items()
            if k.startswith("lrm_generator.")
        }

        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()

    def unload_model(self):
        if self.model:
            self.model.to("cpu")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

    def make3d(self, images):
        self.load_model()

        try:
            images = torch.from_numpy(
                np.asarray(images, np.float32) / 255.0
            ).permute(2, 0, 1)

            images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
            images = images.unsqueeze(0).to(self.device)

            from src.utils.camera_util import get_zero123plus_input_cameras

            cams = get_zero123plus_input_cameras(1).to(self.device)

            with torch.no_grad():
                planes = self.model.forward_planes(images, cams)

            return self._export(planes)

        finally:
            self.unload_model()

    def _export(self, planes):
        from src.utils.mesh_util import save_obj

        mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        verts, faces, colors = self.model.extract_mesh(
            planes, use_texture_map=False, **self.infer_config
        )

        save_obj(verts, faces, colors, mesh_path)
        imageio.mimwrite(video_path, [], fps=30)

        return video_path, mesh_path
