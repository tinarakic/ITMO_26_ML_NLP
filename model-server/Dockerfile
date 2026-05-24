FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV CUDA_VISIBLE_DEVICES=0

# IMPORTANT: avoids random tokenizer parallelism crashes in slim images
ENV TOKENIZERS_PARALLELISM=false

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
