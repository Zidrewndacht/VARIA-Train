## Dockerfile

```docker
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
```

## config.yaml

```yaml
# config.yaml

# Core Training Parameters
epochs: 100
batch_size: 2              # Per-GPU batch size
devices: 2                 # Number of GPUs to use
lr: 0.0001
seed: 42

# Batch Size Math
target_effective_batch_size: 16

# Model Settings
use_ema: true
checkpoint_interval: 10
skip_best_epochs: 3

# Early Stopping
early_stopping: true
early_stopping_patience: 10
early_stopping_min_delta: 0.005
early_stopping_use_ema: true

# # Logging & Evaluation - 
# # These are currently missing from the training scripts!
# tensorboard: false
# eval_interval: 4
# log_per_class_metrics: true
```

## docker-compose.yml

```yaml
# docker-compose.yaml
services:
  # ======================================================================
  # DATA SYNC SERVICE
  # This is used to create a Docker volume with source training data
  # to avoid constantly reading data slowly from the WSL2 host.
  # ======================================================================
  data-sync:
    image: alpine:latest
    volumes:
##########################################################################
      # Mount host directory (Read-Write so we can burn the flag)
      # FALLBACK: If Conductor doesn't set VARIA_DATASET_DIR, use your manual path.
      # (Using forward slashes here prevents YAML parsing errors with backslashes)
      - "${VARIA_DATASET_DIR:-D:/!staging/@Mestrado/PCBA-AOI/output_sliced}:/src"
##########################################################################     
      # Mount the Docker volume for the destination
      - rfdetr_dataset:/dest
    command: >
      sh -c "
      if [ -f /src/.new_data_flag ]; then
        echo 'Flag found. Wiping and syncing data...';
        rm -rf /dest/* /dest/.[!.]* /dest/..?* 2>/dev/null || true;
        cp -a /src/. /dest/;
        rm -f /src/.new_data_flag;
        rm -f /dest/.new_data_flag;
        echo 'Sync complete. Flag burned.';
      else
        echo 'No flag found. Skipping sync.';
        exit 0;
      fi
      "

  # ======================================================================
  # RF-DETR TRAINING / EVALUATION SERVICE
  # ======================================================================
  rfdetr:
    depends_on:
      data-sync:
        condition: service_completed_successfully
    build: .
    container_name: rfdetr_trainer
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      # The training container just mounts the populated volume
      - rfdetr_dataset:/workspace/data:ro
      
      # FALLBACKS: Map to local relative paths for manual execution
      - "${VARIA_OUTPUT_DIR:-./output}:/workspace/output"
      - "${VARIA_METRICS_DIR:-./metrics}:/workspace/metrics"
      - "${VARIA_CACHE_DIR:-./roboflow_cache}:/root/.roboflow" 
      - "${VARIA_CONFIG_FILE:-./config.yaml}:/workspace/code/config.yaml:ro"
      - "${VARIA_TRAIN_SCRIPT:-./rfdetr_train.py}:/workspace/code/rfdetr_train.py:ro"
      - "${VARIA_EVAL_SCRIPT:-./rfdetr_eval.py}:/workspace/code/rfdetr_eval.py:ro"
      
    working_dir: /workspace/code
    
    # FALLBACK: Defaults to training. Conductor overrides this for evaluation.
    command: ${VARIA_COMMAND:-python rfdetr_train.py}
    
    stdin_open: true 
    tty: true       
    
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - NCCL_DEBUG=INFO
      - NCCL_P2P_DISABLE=1
      - NCCL_IB_DISABLE=1
      - NCCL_SOCKET_IFNAME=eth0
      - OMP_NUM_THREADS=1
      - PYTHONFAULTHANDLER=1
    shm_size: '16gb' 

volumes:
  rfdetr_dataset:
    name: rfdetr_dataset
```

## rfdetr_eval.py

```python
"""rfdetr_eval.py
Evaluate a trained RF-DETR checkpoint on the frozen test set.
"""
import argparse
import json
from pathlib import Path
from rfdetr import RFDETRLarge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth or .ckpt")
    parser.add_argument("--dataset-dir", required=True, help="Path to sliced COCO dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, help="Path to save metrics JSON")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    # RFDETRLarge automatically infers num_classes and resolution from the checkpoint
    model = RFDETRLarge(pretrain_weights=args.checkpoint)
    
    print(f"Evaluating on {args.split} split in {args.dataset_dir}")
    metrics = model.evaluate(
        dataset_dir=args.dataset_dir,
        split=args.split,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
```

## rfdetr_train.py

```python
"""rfdetr_train.py
Train RF-DETR-Large on pre-sliced COCO dataset.

Using Docker:
    docker compose build
    docker compose up
"""
import yaml
from rfdetr import RFDETRLarge
from pathlib import Path

# Fixed container paths (Implementation detail of the Docker environment, must match the docker-compose YAML)
DATASET_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/output")

# RF-DETR saves full training states (optimizer, epoch, etc.) in .ckpt files.
# 'last.ckpt' is the standard file used for resuming interrupted training.
CHECKPOINT_PATH = OUTPUT_DIR / "last.ckpt"

if CHECKPOINT_PATH.exists():
    print(f"Resuming training from {CHECKPOINT_PATH}")
    resume_arg = str(CHECKPOINT_PATH)
else:
    print("No checkpoint found. Starting from scratch.")
    resume_arg = None

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ==============================================================================
# RF-DETR ALBUMENTATIONS CONFIGURATION
# ==============================================================================
#updated according to current documentation: https://albumentations.ai/docs/api-reference/
VARIA_AUG_CONFIG = {
    "Blur": {"p": 0.03},
    "MotionBlur": {
        "p": 0.1
    },
    "RandomRotate90":{"p": 1.0},
    "ShotNoise":{
        "scale_range": (0.03, 0.09),
        "p":0.2, 
    }, 
    "GaussNoise":{
        "std_range": (0.03, 0.09),
        "p":0.2, 
    }, 
    "SaltAndPepper":{
        # "amount_range":(0.001,0.005), #wrong version? "UserWarning: Argument(s) 'amount_range' are not valid for transform SaltAndPepper"
        "amount": (0.001, 0.006),  # v1.x syntax (instead of amount_range)
        "p":0.1,
    },
    "ImageCompression": {   # relevant as capture is done with MJPEG
        "quality_range": (50, 85),
        "p": 0.3
    },
    "RandomBrightnessContrast": {
        "brightness_limit": (-0.2, 0.2),
        "contrast_limit": (-0.2, 0.2),
        "p": 0.25
    },
    "RandomGamma": {
        "gamma_limit": (80, 120),
        "p": 0.25
    },
    "RGBShift": {
        "r_shift_limit": 20,
        "g_shift_limit": 20,
        "b_shift_limit": 20,
        "p": 0.25                 
    },
    "HueSaturationValue": {
        "hue_shift_limit": 10, 
        "sat_shift_limit": 30,
        "val_shift_limit": 20,
        "p": 0.25
    },
}

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    
    # Auto-calculate gradient accumulation
    grad_accum = max(1, cfg["target_effective_batch_size"] // (cfg["batch_size"] * cfg["devices"]))
    actual_effective = cfg["batch_size"] * cfg["devices"] * grad_accum
    
    print(f"Effective batch size: {cfg['batch_size']} (per-GPU) x {cfg['devices']} (GPUs) x {grad_accum} (accum) = {actual_effective}")
    
    model = RFDETRLarge()
    
    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=grad_accum,
        lr=cfg["lr"],
        seed=cfg["seed"],

        # Disabled multi_scale because data does not require this (already pre-sliced multi-scale data) 
        # AND the trainer defaults to a bogus 864x864 blurry upscale of all my exact 704x704 source data otherwise?!
        resolution=704,
        multi_scale=False,  

        # EMA & Checkpoints
        use_ema=cfg["use_ema"],
        checkpoint_interval=cfg["checkpoint_interval"],
        skip_best_epochs=cfg["skip_best_epochs"],
        
        # Early Stopping
        early_stopping=cfg["early_stopping"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
        early_stopping_use_ema=cfg["early_stopping_use_ema"],

        # Hardware config:
        device="cuda",
        strategy="ddp",
        devices=cfg["devices"],

        # Injeção do dicionário de augmentações do Albumentations
        aug_config=VARIA_AUG_CONFIG,

        resume=resume_arg
    )
```

## train.bat

```bat
:: train.bat
@echo off
:: 1.  Start Docker Desktop if it isn’t running yet
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>NUL | find /I "Docker Desktop.exe" >NUL
if errorlevel 1 (
    echo Starting Docker Desktop ...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
)

:: 2.  Wait until the Linux engine answers
:wait_engine
echo Waiting for Docker engine ...
docker version >NUL 2>&1
if errorlevel 1 (
    timeout /t 3 >NUL
    goto :wait_engine
)

docker compose up

pause
pause
```

