FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# ---- Install PyTorch with CUDA 11.8 (supports sm_61) ----
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- Install transformers stack ----
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

# Important for CUDA compatibility on older GPUs
ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV CUDA_VISIBLE_DEVICES=0

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]