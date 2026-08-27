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
        strategy="ddp" if cfg["devices"] > 1 else "auto",
        devices=cfg["devices"],

        # Injeção do dicionário de augmentações do Albumentations
        aug_config=VARIA_AUG_CONFIG,

        resume=resume_arg
    )