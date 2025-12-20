# Use CUDA 12.4 development image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
# 7.5 is the architecture for T4 GPUs (g4dn.xlarge)
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV FORCE_CUDA="1"
ENV HF_HUB_DISABLE_SYMLINKS=1

# Install system tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# --- FIX: Remove Windows-only and incompatible packages ---
# 1. audioop-lts: Incompatible with Python 3.11
# 2. pywin32, pywinpty, pyreadline3: Windows-only packages that fail on Linux
RUN sed -i -e '/audioop-lts/d' \
           -e '/pywin32/d' \
           -e '/pywinpty/d' \
           -e '/pyreadline3/d' \
           requirements.txt

# Upgrade pip, uninstall conflicting packages, and install requirements
# We uninstall torch/transformers first to avoid conflicts with the versions in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y huggingface-hub diffusers transformers accelerate torch torchvision && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Install nvdiffrast explicitly (requires build tools)
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

COPY . .

EXPOSE 43548

CMD ["python", "app.py"]