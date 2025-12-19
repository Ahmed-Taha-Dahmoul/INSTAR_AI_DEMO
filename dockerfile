# Use CUDA 12.4 development image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
# 7.5 is the specific architecture for NVIDIA T4 (g4dn instances)
ENV TORCH_CUDA_ARCH_LIST="7.5" 
ENV FORCE_CUDA="1"
ENV HF_HUB_DISABLE_SYMLINKS=1

# Install system tools and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip, uninstall conflicting packages, and install requirements
# We uninstall these first to avoid conflicts with the base image's pre-installed versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y huggingface-hub diffusers transformers accelerate && \
    pip install --no-cache-dir -r requirements.txt

# Install nvdiffrast (requires the build tools installed above)
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# Copy the rest of the application (including src/, configs/, and app.py)
COPY . .

# Expose the specific port used in app.py
EXPOSE 43548

# Run the merged python script
CMD ["python", "app.py"]