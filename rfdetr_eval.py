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