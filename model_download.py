import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download


def load_diffusion_pipeline(device):
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
    )

    unet_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="diffusion_pytorch_model.bin",
        repo_type="model",
    )

    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)

    return pipeline.to(device)


def download_mesh_checkpoint():
    return hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_base.ckpt",
        repo_type="model",
    )
