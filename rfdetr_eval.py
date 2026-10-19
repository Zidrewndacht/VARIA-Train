#!/usr/bin/env python3
"""
rfdetr_eval.py

Evaluate a trained RF-DETR checkpoint on a frozen COCO test set using the
official model.evaluate() API (added in rfdetr >= 1.8.0).
"""

import argparse
import json
from pathlib import Path

from rfdetr import RFDETRLarge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an RF-DETR checkpoint on a test split."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint (.ckpt or .pth)",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Root directory of the COCO‑formatted dataset",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to save the metrics JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for evaluation (default: 2)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=704,
        help="Input resolution (default: 704)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    # RFDETRLarge automatically infers num_classes from the dataset
    model = RFDETRLarge(pretrain_weights=args.checkpoint)

    print(f"Evaluating on {args.split} split in {args.dataset_dir}")
    # evaluate() accepts the same kwargs as train()
    metrics = model.evaluate(
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        resolution=args.resolution,

        # ------------------------------------------------------------------
        # Restore pre-1.10 defaults (all three flipped in 1.10.0)
        # ------------------------------------------------------------------
        # 1. Per-class AP keys (test/AP/<class>)
        log_per_class_metrics=True,

        # 2. Unconditional test loss (test/loss).
        compute_val_loss=True,

        # 3. Two-forward evaluation: EMA namespace + base-model namespace.
        #    Emits both test/AP/<class> and test/ema/AP/<class>, matching
        #    the 1.9 output exactly.
        eval_base_model=True,
    )

    # Save metrics
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved to {out_path}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()