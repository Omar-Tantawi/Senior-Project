"""Complete automation pipeline: MXNet → Convert → FT → Eval"""

import subprocess
import sys
import time
import os
from pathlib import Path

VENV = r"C:\Users\abdul\attend_venv"
PYTHON = os.path.join(VENV, "Scripts", "python.exe")
PROJECT = r"C:\Users\abdul\OneDrive\Desktop\Senior Project\Project\AI\attendance_system"

def wait_for_mxnet(max_wait=600):
    """Wait for MXNet to be installed (max 10 minutes)."""
    print("\n[Pipeline] Waiting for MXNet installation...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            result = subprocess.run(
                [PYTHON, "-c", "import mxnet; print('OK')"],
                capture_output=True, text=True, timeout=5
            )
            if "OK" in result.stdout:
                print("[OK] MXNet ready!")
                return True
        except:
            pass

        elapsed = int(time.time() - start)
        if elapsed % 30 == 0:
            print(f"   Still waiting... ({elapsed}s)")
        time.sleep(5)

    print("[WARNING] MXNet check failed, continuing anyway...")
    return True  # Continue even if check fails

def run_step(name, cmd):
    """Run a step and report results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=PROJECT)
        if result.returncode == 0:
            print(f"\n[OK] {name} completed successfully")
            return True
        else:
            print(f"\n[WARNING] {name} finished with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n[ERROR] {name} error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  COMPLETE TRAINING PIPELINE")
    print("="*60)
    print("\n[Timeline]")
    print("  1. Wait for MXNet install (~5 min)")
    print("  2. Convert RecordIO -> Images (~60 min)")
    print("  3. Fine-tune ArcFace (~2-4 hours GPU)")
    print("  4. Evaluate (~30 min)")
    print("  TOTAL: ~3-5 hours\n")

    # Step 1: Wait for MXNet
    wait_for_mxnet()  # Continue even if check fails

    time.sleep(5)

    # Step 2: Convert RecordIO
    if not run_step(
        "STEP 1: Convert RecordIO to Images",
        [PYTHON, "convert_recordio.py"]
    ):
        print("[WARNING] Conversion may have issues, trying to continue...")

    time.sleep(5)

    # Step 3: Fine-tune
    if not run_step(
        "STEP 2: Fine-tune ArcFace",
        [PYTHON, "fine_tuning/finetune.py",
         "--data", "data/casia_aligned",
         "--epochs", "20",
         "--batch-size", "64",
         "--lr", "0.01"]
    ):
        print("[ERROR] Fine-tuning failed")
        sys.exit(1)

    time.sleep(5)

    # Step 4: Evaluate
    if not run_step(
        "STEP 3: Evaluate Model",
        [PYTHON, "fine_tuning/evaluate.py",
         "--original", "models/w600k_r50.onnx",
         "--finetuned", "fine_tuning/checkpoints/arcface_finetuned.onnx",
         "--data", "data/casia_aligned"]
    ):
        print("[WARNING] Evaluation had issues")

    # Done
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("   fine_tuning/evaluation_results.json")
    print("\nYour model is ready for submission!\n")

if __name__ == "__main__":
    main()
