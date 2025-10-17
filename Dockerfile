# syntax=docker/dockerfile:1.6
ARG PYTHON_VERSION=3.10
ARG TS_VERSION=2.11.0

############################
# Stage GPU (CUDA 12.1)
############################
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS gpu

# Paquetes del sistema necesarios para VTK / PyMeshLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglvnd0 \
    gcc g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala deps Python comunes
COPY requirements-common.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir cupy-cuda12x==13.0.0 \
 && pip install --no-cache-dir -r requirements-common.txt
# Nota: torch/torchvision/torchaudio ya vienen en la imagen base GPU

# Copia tu aplicación
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000
CMD ["uvicorn","som3d_api:app","--host","0.0.0.0","--port","8000","--workers","1"]

############################
# Stage CPU (sin CUDA)
############################
FROM python:3.10-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglvnd0 \
    gcc g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU desde el índice oficial
ARG TORCH_VER=2.3.1
ARG TV_VER=0.18.1
ARG TA_VER=2.3.1

RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==${TORCH_VER} torchvision==${TV_VER} torchaudio==${TA_VER}

# Resto de dependencias
COPY requirements-common.txt ./
RUN pip install --no-cache-dir -r requirements-common.txt

# Copia tu aplicación
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000
CMD ["uvicorn","som3d_api:app","--host","0.0.0.0","--port","8000","--workers","1"]
