# Student Behavior Detector — Training Log

## Dataset

**Source:** SCB-03 merged dataset (6-class student behavior)  
**Train:** 14,972 images  
**Val:** 4,897 images  
**Classes:**

| ID | Class | Val Instances | Notes |
|----|-------|--------------|-------|
| 0 | hand-raising | 2,907 | Visually distinctive |
| 1 | read | 6,040 | Visually ambiguous (seated, head down) |
| 2 | write | 2,824 | Visually ambiguous (seated, head down) |
| 3 | talk | 1,249 | Moderate difficulty |
| 4 | stand | 3,190 | Easiest — unique body pose |
| 5 | discuss | 1,630 | Visually ambiguous (similar to talk/read) |

---

## Hardware & Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB GDDR6 |
| TDP | 80 W |
| Driver | 595.97 (CUDA 13.2) |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.56 |
| OS | Windows 11 |

### Setup fixes applied before training
The scripts were originally written for an RTX 5060 desktop machine and a different user's file paths. The following corrections were made:
- `merged/dataset.yaml` — hardcoded paths updated from `C:/Users/Osama Dalati/...` to `C:/Users/LENOVO/Desktop/Class 10/...`
- `config.py` — `BASE_DIR` corrected from `Path(__file__).parent.parent` to `Path(__file__).parent`; `MERGED_DATA_DIR` and `WEIGHTS_DIR` paths updated accordingly
- `train_behavior.py` — import changed from `from attention.config import ...` to local `from config import ...` (no package structure present)
- `workers` default changed from `8` to `0` (required on Windows to avoid DataLoader multiprocessing spawn errors)
- `batch` default changed from `32` to `16` (RTX 4060 Laptop has lower throughput than the RTX 5060 desktop the script was tuned for)

---

## Shared Training Settings (all runs)

| Parameter | Value |
|-----------|-------|
| `device` | `cuda` |
| `workers` | `0` (Windows) |
| `half` | `True` (fp16) |
| `epochs` | `100` (with early stopping) |
| `degrees` | `10` |
| `translate` | `0.1` |
| `scale` | `0.3` |
| `flipud` | `0.0` |
| `fliplr` | `0.5` |
| `mosaic` | `1.0` |
| `auto_augment` | `randaugment` |
| `optimizer` | `auto` (Ultralytics picks MuSGD) |

---

## Run 1 — `behavior_v1`

**Goal:** Baseline fine-tune of the nano model on this hardware.

### Config

| Parameter | Value |
|-----------|-------|
| `base_weights` | `yolo11n.pt` (pretrained, fresh) |
| `imgsz` | `640` |
| `batch` | `16` |
| `cls` | `1.5` |
| `patience` | `20` |

### Training summary

| | |
|---|---|
| Speed | ~2.4 it/s, ~936 batches/epoch |
| Per-epoch time | ~8 min (6.5 min train + 1.5 min val) |
| Stopped at epoch | 32 |
| Best epoch | **12** |
| GPU mem | 2.36 GB |

### Results (best.pt = epoch 12)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **all** | 0.234 | 0.377 | **0.177** | 0.113 |
| hand-raising | 0.336 | 0.431 | 0.277 | 0.158 |
| read | 0.094 | 0.219 | 0.044 | 0.027 |
| write | 0.184 | 0.263 | 0.090 | 0.057 |
| talk | 0.276 | 0.515 | 0.231 | 0.163 |
| stand | 0.429 | 0.628 | 0.386 | 0.258 |
| discuss | 0.088 | 0.205 | 0.033 | 0.017 |

### Observations
- `stand` and `talk` converged well; `read`, `write`, `discuss` are near-useless
- Model peaked at epoch 12 and degraded afterwards — fast overfit on easy classes
- `read`/`write`/`discuss` are visually near-identical from surveillance angle; nano model lacks capacity to separate them

---

## Run 2 — `behavior_v2`

**Goal:** Address class imbalance more aggressively. Warm-start from v1 best.pt, increase `cls` loss, add label smoothing to reduce overconfidence on ambiguous classes.

### Config changes from Run 1

| Parameter | v1 | v2 | Reason |
|-----------|----|----|--------|
| `base_weights` | `yolo11n.pt` | `behavior_v1/best.pt` | Warm start saves convergence epochs |
| `cls` | `1.5` | `2.5` | Harder push on misclassified classes |
| `lr0` | `0.01` | `0.005` (intended) | Lower LR for fine-tuning on fine-tuned weights |
| `label_smoothing` | `0.0` | `0.1` | Reduce overconfidence on read/write/discuss |
| `patience` | `20` | `40` | More room to improve from warm start |
| `name` | `behavior_v1` | `behavior_v2` | |

> **Note:** `lr0=0.005` was silently ignored — Ultralytics `optimizer=auto` overrides `lr0` and selected MuSGD with `lr=0.01`. The intended lower learning rate did not take effect.

### Training summary

| | |
|---|---|
| Stopped at epoch | 41 |
| Best epoch | **1** |
| GPU mem | 2.36 GB |

### Results (best.pt = epoch 1)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **all** | 0.221 | 0.379 | **0.171** | 0.113 |
| hand-raising | 0.233 | 0.568 | 0.254 | 0.153 |
| read | 0.104 | 0.224 | 0.049 | 0.030 |
| write | 0.241 | 0.213 | 0.113 | 0.076 |
| talk | 0.255 | 0.496 | 0.220 | 0.158 |
| stand | 0.410 | 0.625 | 0.360 | 0.244 |
| discuss | 0.078 | 0.159 | 0.032 | 0.016 |

### Observations
- Warm start from v1's best.pt immediately started at 0.171 mAP — the model could not improve from there at all
- `cls=2.5` and `label_smoothing=0.1` slightly improved `write` (0.113 vs 0.090) but hurt `stand`, `talk`, and `hand-raising`
- Overall mAP regressed vs v1 (0.171 vs 0.177) — **v1's best.pt remains superior**
- The model was already at its ceiling; warm-starting with aggressive loss did not break through it

---

## Run 3 — `behavior_v3`

**Goal:** Test whether the ceiling is a model capacity problem. Switch from nano (2.6M params) to small (9.4M params, ~3.6× more capacity), fresh pretrained start, moderate `cls` increase.

### Config changes from Run 1

| Parameter | v1 | v3 | Reason |
|-----------|----|----|--------|
| `base_weights` | `yolo11n.pt` | `yolo11s.pt` | 3.6× more parameters |
| `cls` | `1.5` | `2.0` | Moderate push, less aggressive than v2's 2.5 |
| `patience` | `20` | `30` | Slightly more room given fresh start |
| `name` | `behavior_v1` | `behavior_v3` | |

### Training summary

| | |
|---|---|
| Speed | ~2.1–2.5 it/s |
| Per-epoch time | ~9 min |
| Stopped at epoch | 42 |
| Best epoch | **12** |
| GPU mem | 4.0 GB |

### Results (best.pt = epoch 12)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **all** | 0.220 | 0.396 | **0.178** | 0.120 |
| hand-raising | 0.293 | 0.550 | 0.300 | 0.187 |
| read | 0.107 | 0.240 | 0.053 | 0.033 |
| write | 0.206 | 0.291 | 0.113 | 0.079 |
| talk | 0.229 | 0.506 | 0.209 | 0.152 |
| stand | 0.411 | 0.624 | 0.369 | 0.254 |
| discuss | 0.076 | 0.166 | 0.027 | 0.014 |

### Observations
- **Best overall model** (0.178 mAP50, marginal +0.001 over v1)
- `hand-raising` improved significantly: 0.277 → 0.300 (+8%)
- `write` matched v2's improvement: 0.113
- `talk` slightly regressed vs v1 (0.209 vs 0.231)
- `discuss` worsened (0.027 vs 0.033)
- Model still peaked at exactly epoch 12 — same convergence pattern as yolo11n
- Switching model size produced negligible overall gain; capacity is not the bottleneck
- The hard classes (`read`, `write`, `discuss`) remain near-unusable — these are a data/class-definition problem, not a model problem

---

## Run 4 — `behavior_v4` *(in progress)*

**Goal:** Test whether higher resolution breaks the ceiling on fine-grained posture classes. `read`/`write`/`discuss` differences may be sub-pixel at 640px from surveillance distance; 832px gives the model ~1.7× more spatial detail per person.

### Config changes from Run 3

| Parameter | v3 | v4 | Reason |
|-----------|----|----|--------|
| `imgsz` | `640` | `832` | ~1.7× more spatial detail per person |
| `batch` | `16` | `8` | Memory scales as imgsz² — batch halved to stay within 8 GB VRAM |
| `name` | `behavior_v3` | `behavior_v4` | |

All other settings identical to v3.

### Expected training profile

| | |
|---|---|
| Batches/epoch | ~1,872 (double v3 due to halved batch) |
| Per-epoch time | ~14 min |
| GPU mem | ~3.4 GB |
| Estimated stop | ~42 epochs (~10 hours) |

### Results

*Training in progress — results will be added when complete.*

---

## Summary Comparison

| Run | Model | imgsz | batch | cls | Best epoch | **mAP@50** | mAP@50-95 |
|-----|-------|-------|-------|-----|-----------|------------|-----------|
| v1 | yolo11n | 640 | 16 | 1.5 | 12/32 | 0.177 | 0.113 |
| v2 | yolo11n (warm) | 640 | 16 | 2.5 | 1/41 | 0.171 | 0.113 |
| v3 | yolo11s | 640 | 16 | 2.0 | 12/42 | **0.178** | **0.120** |
| v4 | yolo11s | 832 | 8 | 2.0 | TBD | TBD | TBD |

### Hard class mAP@50 across runs

| Class | v1 | v2 | v3 | v4 |
|-------|----|----|----|----|
| read | 0.044 | 0.049 | 0.053 | TBD |
| write | 0.090 | 0.113 | 0.113 | TBD |
| discuss | 0.033 | 0.032 | 0.027 | TBD |

---

## Key Findings

1. **All runs peaked at epoch 12** regardless of model size or loss settings — a consistent convergence pattern suggesting a dataset-level ceiling, not a training hyperparameter issue.
2. **Model size (nano → small) had negligible impact** (+0.001 mAP50 overall). Capacity is not the bottleneck.
3. **Aggressive cls loss (2.5) backfired** — it shifted the model away from easy classes without recovering hard ones.
4. **`read`, `write`, `discuss` are fundamentally hard** from surveillance-angle footage — seated students with heads down look nearly identical across all three classes. No training strategy has pushed any of these above mAP50=0.12.
5. **Best weights so far:** `weights/behavior_v3/weights/best.pt`

## Potential Next Steps (if v4 also hits the ceiling)

- **Merge ambiguous classes:** Combine `read` + `write` → `desk-work` (5-class model). The model is already functionally treating them as one.
- **Targeted data augmentation:** Synthetically generate more varied `read`/`write`/`discuss` examples with different camera angles.
- **Two-stage detection:** Use a coarse detector (stand/sit/raise-hand) followed by a fine-grained classifier on cropped bounding boxes at higher resolution.
- **Larger model (yolo11m):** Last resort — if v4 shows a trend, medium model may help. Expect ~2× v3 training time.
