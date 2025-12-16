# =========================
# Base image with CUDA + cuDNN
# =========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# =========================
# System dependencies
# =========================
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python setup
# =========================
RUN python3 -m pip install --upgrade pip

# =========================
# Set workdir
# =========================
WORKDIR /app

# =========================
# Copy and install Python dependencies
# =========================
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# =========================
# Copy application code
# =========================
COPY . .

# =========================
# Environment variables
# =========================
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DIFFUSERS_CACHE=/app/.cache/huggingface

# =========================
# Expose Gradio port
# =========================
EXPOSE 43548

# =========================
# Run the app
# =========================
CMD ["python3", "main.py"]
