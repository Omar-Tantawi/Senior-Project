"""Extract images from CASIA-WebFace RecordIO format using MXNet ImageIter.

Uses the exact approach from the Kaggle comments.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    import mxnet as mx
    from mxnet import recordio
except ImportError:
    print("[ERROR] MXNet not installed. Install with:")
    print("  pip install mxnet")
    sys.exit(1)

def extract_images_from_rec(rec_path, idx_path, property_path, output_dir, max_identities=None):
    """Extract images from RecordIO using MXNet ImageIter."""

    print(f"\n[Extract] Reading from: {rec_path}")
    print(f"[Extract] Output directory: {output_dir}")

    # Read number of classes from property file
    try:
        with open(property_path, "r") as f:
            num_classes, h, w = [int(i) for i in f.read().split(",")]
        print(f"[Extract] Dataset info: {num_classes} classes, {h}x{w} image size")
    except Exception as e:
        print(f"[ERROR] Could not read property file: {e}")
        return False

    if max_identities:
        num_classes = min(num_classes, max_identities)
        print(f"[Extract] Limiting to {num_classes} classes")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use ImageIter to read RecordIO
    try:
        print("[Extract] Creating ImageIter...")
        data_iter = mx.image.ImageIter(
            batch_size=1,
            data_shape=(3, h, w),
            path_imgrec=rec_path,
            path_imgidx=idx_path,
        )
    except Exception as e:
        print(f"[ERROR] Could not create ImageIter: {e}")
        return False

    # Extract images
    saved_count = 0
    identity_count = 0

    try:
        data_iter.reset()
        batch_num = 0

        while True:
            try:
                batch = data_iter.next()
            except StopIteration:
                break

            # Get image data and label
            data = batch.data[0]  # Image tensor
            label = batch.label[0]  # Label tensor

            if data is None or label is None:
                continue

            # Convert to numpy
            img_array = data.asnumpy()
            label_array = label.asnumpy()

            for i in range(img_array.shape[0]):
                try:
                    img = img_array[i]
                    label_id = int(label_array[i])

                    # Check if we're limiting classes
                    if max_identities and label_id >= max_identities:
                        continue

                    # Convert from CHW to HWC if needed
                    if img.shape[0] == 3:  # CHW format
                        img = img.transpose((1, 2, 0))

                    # Convert to BGR for OpenCV
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Create identity folder
                    identity_dir = os.path.join(output_dir, f"{label_id:06d}")
                    os.makedirs(identity_dir, exist_ok=True)

                    if label_id not in [int(d.name) for d in Path(output_dir).iterdir() if d.is_dir()][:-1]:
                        identity_count += 1

                    # Save image
                    img_name = f"{saved_count % 1000:04d}.jpg"
                    img_path = os.path.join(identity_dir, img_name)

                    success = cv2.imwrite(img_path, img)
                    if success:
                        saved_count += 1
                        if saved_count % 10000 == 0:
                            print(f"  [Progress] Saved {saved_count} images from {identity_count} identities...")

                except Exception as e:
                    if batch_num == 0:
                        print(f"[Warning] Error processing image: {e}")
                    continue

            batch_num += 1

    except Exception as e:
        print(f"[ERROR] Error during extraction: {e}")
        return False

    print(f"\n[Extract] Complete!")
    print(f"  Images saved: {saved_count}")
    print(f"  Identities: {identity_count}")

    return saved_count > 0

def main():
    print("\n" + "="*60)
    print("  Extract CASIA-WebFace RecordIO to Images")
    print("="*60)

    base_dir = r"C:\Users\abdul\OneDrive\Desktop\Senior Project\Project\AI\attendance_system"
    rec_path = os.path.join(base_dir, "data", "casia_raw", "casia-webface", "train.rec")
    idx_path = os.path.join(base_dir, "data", "casia_raw", "casia-webface", "train.idx")
    prop_path = os.path.join(base_dir, "data", "casia_raw", "casia-webface", "property")
    output_dir = os.path.join(base_dir, "data", "casia_aligned")

    # Check files exist
    for path in [rec_path, idx_path, prop_path]:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)

    # Extract with max 2000 identities (~100K images)
    success = extract_images_from_rec(rec_path, idx_path, prop_path, output_dir, max_identities=2000)

    if success:
        print("\n[OK] Extraction successful!")
        print(f"\nNext steps:")
        print(f"  1. Fine-tune: python fine_tuning/finetune.py --data {output_dir}")
        print(f"  2. Evaluate: python fine_tuning/evaluate.py --original models/w600k_r50.onnx --finetuned fine_tuning/checkpoints/arcface_finetuned.onnx --data {output_dir}")
    else:
        print("\n[ERROR] Extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
