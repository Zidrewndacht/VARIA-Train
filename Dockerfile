# Dockerfile:
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Use Python 3.12 (native to Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git wget curl \
    build-essential cmake ninja-build pkg-config \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup venv with Python 3.12
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install RF-DETR
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "rfdetr[train,loggers]"