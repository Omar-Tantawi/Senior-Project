"""Convert CASIA-WebFace RecordIO format to image files.

Extracts images from MXNet RecordIO binary format to individual JPG files.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import mxnet as mx
from mxnet import recordio

def convert_recordio_to_images(rec_file, idx_file, output_dir, max_identities=None):
    """Convert RecordIO format to image files."""

    print(f"\n[Convert] Reading RecordIO: {rec_file}")
    print(f"[Convert] Index file: {idx_file}")
    print(f"[Convert] Output directory: {output_dir}")

    # Load index
    idx = recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')

    if idx is None:
        print("[ERROR] Could not open RecordIO files")
        return False

    # Get keys
    keys = idx.keys
    print(f"[Convert] Total records: {len(keys)}")

    os.makedirs(output_dir, exist_ok=True)

    # Parse header to understand data format
    # CASIA-WebFace format: [label, image_bytes]
    saved_count = 0
    identity_count = 0
    failed_count = 0

    identities_seen = set()

    for i, key in enumerate(keys):
        if i % 1000 == 0:
            print(f"[Progress] Processing record {i}/{len(keys)}")

        try:
            val = idx.read_idx(key)
            if val is None:
                failed_count += 1
                continue

            # Parse header: first 4 bytes = label (int32), rest = image
            header, s = recordio.unpack(val)

            # header should contain label
            # s should contain image bytes
            label = header.label

            if isinstance(label, mx.nd.NDArray):
                label = int(label.asscalar())
            else:
                label = int(label)

            if max_identities and label >= max_identities:
                continue

            # Extract image from record
            img_bytes = s

            # Decode image
            img_array = mx.image.imdecode(img_bytes)
            if img_array is None:
                failed_count += 1
                continue

            # Convert to numpy and BGR
            img_np = img_array.asnumpy()
            if len(img_np.shape) == 2:
                # Grayscale to BGR
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 3:
                # RGB to BGR
                img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Create identity folder
            identity_dir = os.path.join(output_dir, f"{label:06d}")
            os.makedirs(identity_dir, exist_ok=True)

            if label not in identities_seen:
                identities_seen.add(label)
                identity_count += 1

            # Save image
            img_name = f"{saved_count % 1000:04d}.jpg"
            img_path = os.path.join(identity_dir, img_name)

            success = cv2.imwrite(img_path, img_np)
            if success:
                saved_count += 1
            else:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            if i < 5:
                print(f"[Warning] Record {i}: {e}")

    print(f"\n[Convert] Complete!")
    print(f"  Identities: {identity_count}")
    print(f"  Images saved: {saved_count}")
    print(f"  Failed: {failed_count}")

    return saved_count > 0

def main():
    print("\n" + "="*60)
    print("  Convert CASIA-WebFace RecordIO to Images")
    print("="*60)

    # Paths
    base_dir = r"C:\Users\abdul\OneDrive\Desktop\Senior Project\Project\AI\attendance_system"
    rec_file = os.path.join(base_dir, "data", "casia_raw", "casia-webface", "train.rec")
    idx_file = os.path.join(base_dir, "data", "casia_raw", "casia-webface", "train.idx")
    output_dir = os.path.join(base_dir, "data", "casia_aligned")

    if not os.path.exists(rec_file):
        print(f"[ERROR] RecordIO file not found: {rec_file}")
        sys.exit(1)

    if not os.path.exists(idx_file):
        print(f"[ERROR] Index file not found: {idx_file}")
        sys.exit(1)

    # Convert with max 2000 identities (~100K images)
    success = convert_recordio_to_images(rec_file, idx_file, output_dir, max_identities=2000)

    if success:
        print("\n✅ Conversion successful!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
