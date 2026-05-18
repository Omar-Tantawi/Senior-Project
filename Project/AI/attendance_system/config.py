"""
config.py — Central configuration for the Student Attendance Face Recognition System
All hyperparameters, paths, and settings live here.
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"           # Downloaded dataset goes here
PROCESSED_DIR   = DATA_DIR / "processed"     # After alignment & preprocessing
AUGMENTED_DIR   = DATA_DIR / "augmented"     # After augmentation
STUDENT_DB_DIR  = DATA_DIR / "student_db"    # Per-student enrollment photos
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"
LABELS_PATH     = DATA_DIR / "labels.pkl"

# ─── Model Paths ──────────────────────────────────────────────────────────────
MODEL_DIR       = ROOT / "models"
BACKBONE_PATH   = MODEL_DIR / "backbone_arcface.pth"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
ONNX_PATH       = MODEL_DIR / "face_recognition.onnx"

# ─── Logs & Outputs ───────────────────────────────────────────────────────────
LOG_DIR         = ROOT / "logs"
OUTPUT_DIR      = ROOT / "outputs"
ATTENDANCE_CSV  = OUTPUT_DIR / "attendance.csv"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"

# ─── Dataset: VGGFace2 is used for backbone pre-training ─────────────────────
# Fine-tuning is done on your student enrollment images
DATASET_NAME    = "vggface2"           # or "lfw" for lighter setup
NUM_CLASSES_PRETRAIN = 8631            # VGGFace2 identity count
IMAGE_SIZE      = 112                  # ArcFace standard input size

# ─── Camera / Deployment Setup ────────────────────────────────────────────────
# Wall-mounted camera — expect angled faces, variable lighting, crowds
CAMERA_INDEX    = 0                    # 0 = default webcam / RTSP stream URL
FRAME_SKIP      = 2                    # Process every Nth frame for speed
MAX_FACES       = 30                   # Max students in one frame
CONFIDENCE_THRESHOLD = 0.85            # Face detection min confidence
RECOGNITION_THRESHOLD = 0.60           # Cosine similarity threshold (tune this!)
UNKNOWN_LABEL   = "Unknown"

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BACKBONE        = "iresnet50"          # iresnet18 / iresnet50 / iresnet100
LOSS_FUNCTION   = "arcface"            # arcface | cosface | adaface
EMBEDDING_DIM   = 512
ARC_MARGIN      = 0.5                  # Angular margin (m) for ArcFace
ARC_SCALE       = 64                   # Scale factor (s) for ArcFace

BATCH_SIZE      = 64
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
LR_MILESTONES   = [10, 18, 24]        # Epoch steps for LR decay
LR_GAMMA        = 0.1
WEIGHT_DECAY    = 5e-4
MOMENTUM        = 0.9
WARMUP_EPOCHS   = 3

# ─── Fine-Tuning (on student enrollment data) ─────────────────────────────────
FINETUNE_EPOCHS = 15
FINETUNE_LR     = 1e-4
FINETUNE_BATCH  = 16
MIN_IMAGES_PER_STUDENT = 5             # Minimum enrollment images needed

# ─── Preprocessing ────────────────────────────────────────────────────────────
# RetinaFace is used for detection: best for angled & occluded faces
DETECTOR        = "retinaface"         # retinaface | mtcnn
LANDMARK_POINTS = 5                    # 5-point landmark for alignment
FACE_MARGIN     = 0.3                  # Extra margin around detected face box

# Normalization (ArcFace / InsightFace standard)
PIXEL_MEAN      = [0.5, 0.5, 0.5]
PIXEL_STD       = [0.5, 0.5, 0.5]

# ─── Augmentation (critical for your challenging conditions) ──────────────────
# These are specifically designed for:
#   - Angled/wall-mounted camera
#   - Low / uneven lighting
#   - Multiple faces in frame
#   - Partial occlusion (masks, glasses, hair)
AUGMENTATION = {
    "brightness_range": (-0.4, 0.4),   # Handles poor lighting
    "contrast_range":   (0.6, 1.4),
    "rotation_degrees": 30,             # Handles head tilt
    "horizontal_flip":  True,
    "random_crop_scale":(0.7, 1.0),
    "random_erasing_p": 0.3,            # Simulates occlusion (masks/glasses)
    "gaussian_blur_p":  0.2,
    "motion_blur_p":    0.15,           # Simulates camera motion
    "jpeg_quality":     (60, 100),      # Compression robustness
    "clahe_p":          0.4,            # CLAHE for lighting normalization
}

# ─── GPU ──────────────────────────────────────────────────────────────────────
DEVICE          = "cuda"               # Automatically falls back to cpu
NUM_WORKERS     = 4
PIN_MEMORY      = True
MIXED_PRECISION = True                 # fp16 training for speed

# ─── Attendance Logic ─────────────────────────────────────────────────────────
ATTENDANCE_COOLDOWN_SEC = 300          # Don't re-mark same student within 5 min
MIN_CONSECUTIVE_FRAMES  = 3            # Must be seen in N consecutive frames
