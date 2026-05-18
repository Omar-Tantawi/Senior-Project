"""Auto Fine-tuning Script - Runs when PyTorch is ready.

This script:
1. Waits for PyTorch to be fully installed
2. Extracts CASIA-WebFace dataset
3. Prepares (detects + aligns) all faces
4. Fine-tunes ArcFace for 20 epochs
5. Runs evaluation

Just run this script and go to sleep!
"""

import subprocess
import sys
import time
import os
from pathlib import Path

VENV_PATH = r"C:\Users\abdul\attend_venv"
PYTHON = os.path.join(VENV_PATH, "Scripts", "python.exe")
PROJECT_PATH = r"C:\Users\abdul\OneDrive\Desktop\Senior Project\Project\AI\attendance_system"
DATA_ZIP = r"C:\Users\abdul\OneDrive\Desktop\CASIA-WebFace.zip"

def check_pytorch():
    """Check if PyTorch is installed and GPU is available."""
    try:
        result = subprocess.run(
            [PYTHON, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        if "True" in result.stdout:
            return True, "GPU"
        elif "False" in result.stdout:
            return True, "CPU"
        return False, None
    except:
        return False, None

def wait_for_pytorch(max_wait_minutes=180):
    """Wait for PyTorch to be ready (up to 3 hours)."""
    print("\n" + "="*60)
    print("  AUTO FINE-TUNING SCRIPT")
    print("="*60)
    print("\n[Setup] Waiting for PyTorch installation to complete...")
    print(f"[Setup] Will check every 30 seconds (max wait: {max_wait_minutes} min)")

    start_time = time.time()
    check_count = 0

    while True:
        ready, device = check_pytorch()

        if ready:
            print(f"\n✅ PyTorch is ready! Using: {device}")
            return True

        check_count += 1
        elapsed = time.time() - start_time
        elapsed_min = int(elapsed / 60)

        if check_count % 2 == 0:  # Print every 60 seconds
            print(f"[Waiting] {elapsed_min} minutes elapsed... still downloading PyTorch")

        if elapsed > max_wait_minutes * 60:
            print(f"\n❌ Timeout! PyTorch not ready after {max_wait_minutes} minutes.")
            print("Please install manually:")
            print(f"  {PYTHON} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return False

        time.sleep(30)

def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n[Running] {description}...")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_PATH, timeout=None)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    # Step 0: Wait for PyTorch (max 2.5 hours)
    # But starts immediately if PyTorch finishes before 2.5 hours
    if not wait_for_pytorch(max_wait_minutes=150):  # 150 minutes = 2.5 hours
        sys.exit(1)

    time.sleep(5)

    # Step 1: Extract CASIA-WebFace
    print("\n" + "="*60)
    print("  STEP 1: Extract Dataset")
    print("="*60)

    if Path(DATA_ZIP).exists():
        extract_dir = os.path.join(PROJECT_PATH, "data", "casia_raw")
        print(f"[Extract] CASIA-WebFace.zip found ({Path(DATA_ZIP).stat().st_size / 1e9:.1f}GB)")
        print(f"[Extract] Extracting to: {extract_dir}")

        try:
            import zipfile
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("✅ Extraction completed")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            sys.exit(1)
    else:
        print(f"❌ CASIA-WebFace.zip not found at {DATA_ZIP}")
        sys.exit(1)

    # Step 2: Prepare dataset (detect + align)
    print("\n" + "="*60)
    print("  STEP 2: Prepare Dataset (Detect + Align Faces)")
    print("="*60)
    print("  This may take 3-6 hours depending on GPU/CPU...")

    extract_dir = os.path.join(PROJECT_PATH, "data", "casia_raw")
    aligned_dir = os.path.join(PROJECT_PATH, "data", "casia_aligned")

    cmd = [
        PYTHON,
        "fine_tuning/prepare_dataset.py",
        "--input", extract_dir,
        "--output", aligned_dir,
        "--max-identities", "2000",
    ]

    if not run_command(cmd, "Face detection and alignment"):
        sys.exit(1)

    # Step 3: Fine-tune
    print("\n" + "="*60)
    print("  STEP 3: Fine-tune ArcFace (20 epochs)")
    print("="*60)
    print("  This may take 2-4 hours on GPU, 12-20 hours on CPU...")

    cmd = [
        PYTHON,
        "fine_tuning/finetune.py",
        "--data", aligned_dir,
        "--epochs", "20",
        "--batch-size", "64",
        "--lr", "0.01",
    ]

    if not run_command(cmd, "Fine-tuning"):
        sys.exit(1)

    # Step 4: Evaluate
    print("\n" + "="*60)
    print("  STEP 4: Evaluate (Compare Before vs After)")
    print("="*60)

    cmd = [
        PYTHON,
        "fine_tuning/evaluate.py",
        "--original", "models/w600k_r50.onnx",
        "--finetuned", "fine_tuning/checkpoints/arcface_finetuned.onnx",
        "--data", aligned_dir,
        "--output", "fine_tuning/evaluation_results.json",
    ]

    if not run_command(cmd, "Evaluation"):
        sys.exit(1)

    # Done!
    print("\n" + "="*60)
    print("  ✅ ALL DONE!")
    print("="*60)
    print("\n📊 Results:")
    print(f"  - Fine-tuned model: fine_tuning/checkpoints/arcface_finetuned.onnx")
    print(f"  - Evaluation results: fine_tuning/evaluation_results.json")
    print(f"  - Checkpoints: fine_tuning/checkpoints/")
    print("\n📁 Check fine_tuning/evaluation_results.json for accuracy numbers")
    print("   (before vs after comparison)")
    print("\nGood morning! Your model is ready for your university report! 🎉\n")

if __name__ == "__main__":
    main()
