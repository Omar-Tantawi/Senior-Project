"""
Remaps 6-class labels to 5-class by merging read(1) + write(2) -> desk-work(1).
Renumbers talk(3)->2, stand(4)->3, discuss(5)->4.

Old: 0=hand-raising  1=read  2=write  3=talk  4=stand  5=discuss
New: 0=hand-raising  1=desk-work      2=talk  3=stand  4=discuss

Runs in-place on C:/ML/merged/labels/{train,val}/
Safe to re-run — skips files that are already 5-class.
"""
from pathlib import Path

REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
SPLITS = ["train", "val"]
LABELS_ROOT = Path("C:/ML/merged/labels")

def remap_file(path: Path) -> int:
    """Returns number of lines changed."""
    lines = path.read_text().splitlines()
    new_lines, changes = [], 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(parts[0])
        new_cls = REMAP[old_cls]
        new_lines.append(" ".join([str(new_cls)] + parts[1:]))
        if new_cls != old_cls:
            changes += 1
    path.write_text("\n".join(new_lines) + "\n")
    return changes

total_files = total_changes = 0
for split in SPLITS:
    label_dir = LABELS_ROOT / split
    files = list(label_dir.glob("*.txt"))
    split_changes = 0
    for f in files:
        split_changes += remap_file(f)
    print(f"{split}: {len(files)} files, {split_changes} label lines remapped")
    total_files += len(files)
    total_changes += split_changes

print(f"\nDone. {total_files} files, {total_changes} lines remapped total.")
