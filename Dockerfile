# syntax=docker/dockerfile:1.6

ARG TORCH_VER=2.10.0
ARG CUDA_VER=12.8
ARG CUDNN_VER=9
ARG CUPY_VER=13.0.0

############################
# Stage GPU (default runtime)
############################
FROM pytorch/pytorch:${TORCH_VER}-cuda${CUDA_VER}-cudnn${CUDNN_VER}-runtime AS final

ARG CUPY_VER=13.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglvnd0 \
    gcc g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    PIP_NO_CACHE_DIR=1

# torch/torchvision/torchaudio come from the CUDA base image.
COPY requirements.txt ./
RUN python -m pip install --break-system-packages -U pip setuptools wheel \
 && python -m pip install --break-system-packages cupy-cuda12x==${CUPY_VER} \
 && python -m pip install --break-system-packages -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
