# syntax=docker/dockerfile:1.6

ARG TORCH_VER=2.3.1
ARG CUPY_VER=13.0.0

############################
# Stage GPU (default runtime)
############################
FROM pytorch/pytorch:${TORCH_VER}-cuda12.1-cudnn8-runtime AS final

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
RUN pip install -U pip setuptools wheel \
 && pip install cupy-cuda12x==${CUPY_VER} \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
