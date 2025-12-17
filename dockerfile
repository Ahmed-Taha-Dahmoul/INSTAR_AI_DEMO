# Use an official PyTorch base image with CUDA and Devel tools
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# CRITICAL FIXES FOR COMPILATION:
# 1. Force the architecture for Tesla T4 (7.5) so it doesn't try to auto-detect during build
ENV TORCH_CUDA_ARCH_LIST="7.5"
# 2. Force PyTorch to recognize that we want to build with CUDA support
ENV FORCE_CUDA="1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# 1. Install standard python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Install nvdiffrast specifically with --no-build-isolation
# Now it knows to target Arch 7.5 thanks to the ENV var above
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# Copy the rest of the application code
COPY . .

# Expose the Gradio port
EXPOSE 43548

# Run the application
CMD ["python", "app.py"]