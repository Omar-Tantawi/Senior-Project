"""Re-export the trained checkpoint to a clean FP32 ONNX file.

Usage:
    python fine_tuning/export_onnx.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import iresnet50

CHECKPOINT = Path(__file__).parent / "checkpoints" / "scratch" / "latest.pth"
OUTPUT     = Path(__file__).parent / "checkpoints" / "scratch" / "iresnet50_adaface_fp32.onnx"


def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    # Rebuild model
    model = iresnet50(fp16=False)

    # The checkpoint may store the backbone weights under different keys
    state = ckpt.get("model_state_dict", ckpt.get("backbone_state_dict", ckpt))

    # Strip 'module.' prefix if the model was saved with DataParallel
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=True)

    # Force FP32 — critical to avoid the mixed-type ONNX error
    model = model.float().eval().cpu()

    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)

    print(f"Exporting to {OUTPUT} ...")
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT),
        opset_version=17,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print("Done. ONNX model saved.")
    print(f"\nNext: run evaluate.py with --finetuned {OUTPUT}")


if __name__ == "__main__":
    main()
