"""Export fine-tuned ArcFace checkpoint to ONNX format."""

import sys
from pathlib import Path

import torch

# Add fine_tuning to path to import finetune module
sys.path.insert(0, str(Path(__file__).parent / "fine_tuning"))

from finetune import ArcFaceResNet50, export_to_onnx


def main():
    # Paths
    checkpoint_path = Path("fine_tuning/checkpoints/best.pth")
    output_onnx = Path("fine_tuning/checkpoints/arcface_finetuned.onnx")

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load checkpoint
    print(f"[Load] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 105)
    print(f"[Model] Number of classes: {num_classes}")

    # Create model (ArcFaceResNet50 backbone - outputs 512-dim embeddings)
    model = ArcFaceResNet50()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Export to ONNX
    print(f"[Export] Exporting to {output_onnx}...")
    try:
        export_to_onnx(model, str(output_onnx), device)
        print(f"\n[SUCCESS] Model exported to {output_onnx}")
        print(f"\nNext steps:")
        print(f"  1. To use in inference: copy {output_onnx} to models/")
        print(f"  2. To evaluate: python fine_tuning/evaluate.py --original models/w600k_r50.onnx --finetuned {output_onnx}")
        return True
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
