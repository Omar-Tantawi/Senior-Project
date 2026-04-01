# Face Recognition Attendance System
### AI Engineering Guide — Full Pipeline

---

## Project Structure

```
attendance_system/
├── config.yaml          ← ALL parameters live here (thresholds, paths, GPU)
├── requirements.txt     ← Python dependencies
├── setup.py             ← Downloads pretrained model weights (run ONCE)
│
├── face_analyzer.py     ← RetinaFace detection + ArcFace embedding wrapper
├── liveness.py          ← MiniFASNet anti-spoofing (rejects photos/screens)
├── embedding_db.py      ← FAISS vector database (store + search embeddings)
├── tracker.py           ← SORT multi-object tracker + temporal voting
│
├── enroll.py            ← STEP 1: Build the student database from photos
├── inference.py         ← STEP 2: Live camera → attendance CSV
├── calibrate.py         ← OPTIONAL: Tune similarity threshold on your data
│
├── data/
│   ├── students/        ← Your enrollment photos go here
│   │   ├── S001_Alice_Johnson/
│   │   │   ├── photo1.jpg
│   │   │   └── photo2.jpg
│   │   └── S002_Bob_Smith/
│   │       └── front.jpg
│   ├── embeddings/      ← Auto-generated FAISS index (after enroll.py)
│   │   ├── index.faiss
│   │   └── metadata.json
│   └── test_photos/     ← Optional: photos for threshold calibration
│
├── models/
│   └── anti_spoof/      ← MiniFASNet weights (downloaded by setup.py)
│
├── logs/                ← Runtime logs
└── output/              ← attendance.csv, annotated frames
```

---

## Quickstart (5 steps)

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Download pretrained model weights
```bash
python setup.py
```
This downloads:
- **RetinaFace-R50** + **ArcFace-R100** (via InsightFace's `buffalo_l` pack, ~500MB)
- **MiniFASNetV2** + **MiniFASNetV1SE** anti-spoofing weights (~5MB each)

No training needed. These models already perform at 99.8%+ accuracy on LFW.

### Step 2 — Organize enrollment photos
```
data/students/
    S001_Alice_Johnson/
        front.jpg          ← looking directly at camera
        left.jpg           ← slight left turn
        right.jpg          ← slight right turn
        smiling.jpg        ← different expression
        glasses.jpg        ← if they wear glasses sometimes
```

**Naming convention:** `<StudentID>_<Name>` — the folder name IS the student identity.

**Best practices:**
- 5–10 photos per student gives best accuracy
- Vary: angle (±30°), expression, lighting, glasses/no glasses
- Photos should be at least 200×200 px, face clearly visible
- Consistent background helps but is not required

### Step 3 — Enroll students
```bash
python enroll.py --student_dir data/students/

# Enroll a single new student without re-doing everyone:
python enroll.py --student_id S007

# List all enrolled students:
python enroll.py --list
```

### Step 4 — (Optional) Calibrate threshold
```bash
python calibrate.py --test_dir data/test_photos/
```
Uses photos NOT used during enrollment to find the optimal similarity threshold.
Updates `config.yaml` recommendation. Skip this step if you're just testing.

### Step 5 — Run live attendance
```bash
# Webcam
python inference.py

# IP camera or RTSP stream
python inference.py --video rtsp://192.168.1.10:554/stream

# Auto-exit after 45 minutes (full class period)
python inference.py --duration 2700 --output output/attendance_today.csv

# Headless (no display, server deployment)
python inference.py --headless --output /data/attendance/$(date +%Y%m%d).csv
```

**Keyboard controls** (while window is open):
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit and save attendance |
| `S` | Save attendance snapshot now |
| `R` | Reset session (new class period) |

---

## How the pipeline works (for each frame)

```
Camera frame
    │
    ▼
RetinaFace ──── detects all face bounding boxes + 5 landmarks
    │             (handles 20+ faces simultaneously)
    │
    ▼
MiniFASNet ──── liveness check for each face
    │             (rejects printed photos, phone screens)
    │             SPOOF → skip this face
    │
    ▼
ArcFace ──────── extracts 512-dim embedding for each real face
    │             (L2-normalized vector)
    │
    ▼
SORT Tracker ── assigns persistent track ID to each face across frames
    │             (avoids re-running recognition on same face every frame)
    │
    ▼
FAISS Search ── cosine similarity search in embedding database
    │             returns (student_id, name, similarity_score)
    │             similarity < threshold → "unknown"
    │
    ▼
Temporal Vote ─ student confirmed "PRESENT" after N consistent recognitions
    │             (default: 5 frames; prevents single-frame false positives)
    │
    ▼
attendance.csv ─ student_id, name, PRESENT/ABSENT, timestamp
```

---

## Pretrained models used

| Model | Task | Trained on | Accuracy | Size |
|-------|------|------------|----------|------|
| RetinaFace-R50 | Face detection + landmarks | WIDER FACE | 96.5% mAP (easy) | ~100MB |
| ArcFace-R100 | Face embedding (512-d) | WebFace600K / MS1MV3 | 99.8% LFW | ~250MB |
| MiniFASNetV2 | Liveness / anti-spoof | SiW + NUAA | ~98% ACER | 2.7MB |

All three are **fully pretrained** — you do NOT need to train from scratch.

---

## Fine-tuning guide (when to do it)

You only need to fine-tune if:
- Recognition accuracy is below 90% on your test set after calibration
- Your students wear uniforms/hijabs that cover more of the face than typical
- Lighting conditions are very unusual (very dim, infrared cameras)

**How to fine-tune ArcFace on your own data:**
```python
# Pseudo-code — full script in fine_tune.py (coming soon)
from insightface.recognition.arcface_torch.configs import get_config
# 1. Load pretrained backbone weights
# 2. Replace final classification head with your N_students classes
# 3. Train with ArcFace loss for 10–20 epochs on your enrollment photos
# 4. Use augmented data (flip, brightness, slight rotation)
# 5. Validate on test_photos, tune threshold with calibrate.py
```

---

## Output — attendance.csv

```csv
student_id,name,status,session_date,timestamp
S001,Alice Johnson,PRESENT,2024-09-15,2024-09-15 09:03:12
S002,Bob Smith,ABSENT,2024-09-15,2024-09-15 09:03:12
S003,Carol White,PRESENT,2024-09-15,2024-09-15 09:03:12
```

---

## Configuration reference (config.yaml)

Key parameters to adjust for your setup:

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `detection.det_size` | [640,640] | Larger if students are far from camera |
| `detection.min_face_size` | 40px | Lower if camera is far away |
| `recognition.similarity_threshold` | 0.45 | Run calibrate.py to find optimal value |
| `voting.min_frames_to_confirm` | 5 | Higher = more security, slower confirmation |
| `liveness.threshold` | 0.7 | Higher = stricter anti-spoof |

---

## References

- **ArcFace** — Deng et al., CVPR 2019: https://arxiv.org/abs/1801.07698
- **RetinaFace** — Deng et al., CVPR 2020: https://arxiv.org/abs/1905.00641
- **MiniFASNet** — Yu et al., CVPR 2020: https://arxiv.org/abs/2001.07663
- **SORT tracker** — Bewley et al., ICIP 2016: https://arxiv.org/abs/1602.00763
- **InsightFace** — https://github.com/deepinsight/insightface
