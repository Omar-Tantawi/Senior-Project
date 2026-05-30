"""
Curates the merged dataset to ~10k images with a proper train/val/test split.

Strategy:
- Pool all 19,869 images (train + val)
- Stratified sample 10,000 images proportionally by dominant class per image
- Split 70 / 15 / 15  ->  7,000 train | 1,500 val | 1,500 test
- Copies images + labels into C:/ML/curated/{train,val,test}

Seed is fixed for reproducibility.
"""
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED        = 42
TOTAL       = 10_000
TRAIN_N     = 7_000
VAL_N       = 1_500
TEST_N      = 1_500

SRC_ROOT    = Path("C:/ML/merged")
DST_ROOT    = Path("C:/ML/curated")

SPLITS      = ["train", "val"]
CLASS_NAMES = ["hand-raising", "desk-work", "talk", "stand", "discuss"]

random.seed(SEED)

# ── 1. Pool all images and read their dominant class ──────────────────────────
print("Scanning all images...")
by_class = defaultdict(list)   # class_id -> [(img_path, lbl_path)]
no_label  = []

for split in SPLITS:
    img_dir = SRC_ROOT / "images" / split
    lbl_dir = SRC_ROOT / "labels" / split
    for img in img_dir.glob("*.jpg"):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            no_label.append((img, lbl))
            continue
        # dominant class = most frequent class in this image
        classes = [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()]
        if not classes:
            no_label.append((img, lbl))
            continue
        dominant = max(set(classes), key=classes.count)
        by_class[dominant].append((img, lbl))

total_available = sum(len(v) for v in by_class.items())
print(f"Total images with labels: {sum(len(v) for v in by_class.values())}")
print(f"Images without labels (skipped): {len(no_label)}")
print("\nClass distribution in full pool:")
for cls_id, name in enumerate(CLASS_NAMES):
    print(f"  {name:15s}: {len(by_class[cls_id]):5d} images")

# ── 2. Stratified sample 10k ─────────────────────────────────────────────────
pool_total = sum(len(v) for v in by_class.values())
selected = []
for cls_id, items in by_class.items():
    n = round(len(items) / pool_total * TOTAL)
    n = min(n, len(items))
    selected.extend(random.sample(items, n))

# Trim or top-up to exactly TOTAL
random.shuffle(selected)
selected = selected[:TOTAL]
print(f"\nSelected {len(selected)} images (target {TOTAL})")

# ── 3. Split ──────────────────────────────────────────────────────────────────
train_set = selected[:TRAIN_N]
val_set   = selected[TRAIN_N:TRAIN_N + VAL_N]
test_set  = selected[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]

print(f"Split -> train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

# ── 4. Copy files ─────────────────────────────────────────────────────────────
def copy_split(pairs, split_name):
    img_out = DST_ROOT / "images" / split_name
    lbl_out = DST_ROOT / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)
    print(f"  Copied {len(pairs)} images+labels -> {split_name}/")

print("\nCopying files...")
copy_split(train_set, "train")
copy_split(val_set,   "val")
copy_split(test_set,  "test")

# ── 5. Write dataset.yaml ─────────────────────────────────────────────────────
yaml = f"""train: C:/ML/curated/images/train
val:   C:/ML/curated/images/val
test:  C:/ML/curated/images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
(DST_ROOT / "dataset.yaml").write_text(yaml)
print(f"\nWrote dataset.yaml")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Curated dataset ready at C:/ML/curated/")
print(f"  Train : {len(train_set):,} images")
print(f"  Val   : {len(val_set):,} images")
print(f"  Test  : {len(test_set):,} images")
print(f"  Total : {len(selected):,} images")
print("\nClass breakdown in curated set:")
for split_name, pairs in [("train", train_set), ("val", val_set), ("test", test_set)]:
    counts = defaultdict(int)
    for _, lbl in pairs:
        for line in lbl.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"  {split_name}: " + " | ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES))))
