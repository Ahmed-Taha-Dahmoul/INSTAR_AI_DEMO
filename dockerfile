# Use an official PyTorch base image with CUDA and Devel tools
# We need 'devel' to compile nvdiffrast
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# git: to clone nvdiffrast
# libgl1/libglib: for opencv and image processing
# build-essential: for compiling C++ extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We install nvdiffrast separately to ensure build deps are ready
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This copies src/, configs/, and app.py
COPY . .

# Expose the Gradio port
EXPOSE 43548

# Run the application
CMD ["python", "app.py"]