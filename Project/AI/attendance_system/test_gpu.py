"""Test GPU support for RTX 5060 (sm_120)."""

import torch

print("="*50)
print("  GPU Compatibility Test")
print("="*50)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Try a tiny calculation on the GPU
    try:
        x = torch.ones(1).cuda()
        print("\n✅ SUCCESS! GPU is ready for training!")
        print("You can now run fine-tuning with GPU acceleration.")
    except Exception as e:
        print(f"\n❌ Still having trouble: {e}")
else:
    print("\n⚠️  CUDA not available. Training will use CPU.")

print("="*50)
