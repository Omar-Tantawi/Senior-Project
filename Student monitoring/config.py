from pathlib import Path

BASE_DIR = Path("C:/ML")
SCB_ROOT = BASE_DIR / "SCB-03 dataset" / "SCB-Dataset"
MERGED_DATA_DIR = BASE_DIR / "merged"
WEIGHTS_DIR = BASE_DIR / "weights"

# Pre-trained base weights (downloaded by Ultralytics on first use)
BEHAVIOR_BASE_WEIGHTS = "yolo11n.pt"
POSE_WEIGHTS = "yolo11n-pose.pt"
BEHAVIOR_TRAINED_WEIGHTS = WEIGHTS_DIR / "behavior_best.pt"

# Unified 6-class taxonomy enforced by merge_dataset.py
BEHAVIOR_CLASSES: dict[int, str] = {
    0: "hand-raising",
    1: "read",
    2: "write",
    3: "talk",
    4: "stand",
    5: "discuss",
}

# SCB subset → (folder_name, short_prefix, class_map)
SUBSET_CONFIGS: list[dict] = [
    {
        "root": SCB_ROOT / "SCB5-Handrise-Read-write-2024-9-17",
        "prefix": "hrw",
        "class_map": {0: 0, 1: 1, 2: 2},
    },
    {
        "root": SCB_ROOT / "SCB5-Talk-2024-9-17",
        "prefix": "tlk",
        "class_map": {0: 3},
    },
    {
        "root": SCB_ROOT / "SCB5-Stand-2024-9-17",
        "prefix": "std",
        "class_map": {0: 4},
    },
    {
        "root": SCB_ROOT / "SCB5-Discuss-2024-9-17",
        "prefix": "dsc",
        "class_map": {0: 5},
    },
]

# Engagement scoring hyperparameters
WINDOW_SECONDS: float = 60.0
FPS_SAMPLE: int = 6           # effective FPS after frame-skip (every 5th at 30fps)
FRAME_SKIP: int = 5           # process 1 out of every N frames
GAZE_WEIGHT: float = 0.4
POSTURE_WEIGHT: float = 0.4
DISTRACTION_THRESHOLD: float = 0.45
DISTRACTION_CONSECUTIVE: int = 2

# COCO keypoint indices (17-keypoint model)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Core keypoints required for occlusion guard
CORE_KEYPOINTS: list[int] = [KP_NOSE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]
KEYPOINT_VIS_THRESHOLD: float = 0.5
OCCLUSION_MIN_VISIBLE: int = 2      # need ≥ 2 of 5 core KPs visible
OCCLUSION_BAD_FRAME_RATIO: float = 0.60  # if >60% of window frames are occluded, skip E

# Behavioral delta values (pre-scaled contribution to E)
DELTA_HAND_RAISE: float = +0.2
DELTA_DISTRACTION: float = -0.2    # reserved for future phone-detection class
HAND_RAISE_WINDOW_RATIO: float = 0.10  # >10% of frames with hand-raise triggers delta

# Gaze stability calibration
MAX_EXPECTED_GAZE_VAR: float = 0.25  # variance at fully random gaze → GS=0

# Posture scoring thresholds (radians)
SPINE_ANGLE_MAX: float = 0.698    # 40 degrees — score drops to 0 at this angle
HEAD_DROP_NEUTRAL: float = 0.1    # ratio below which head position is neutral
HEAD_DROP_MAX: float = 0.3        # ratio at which head score drops to 0

# IoU threshold for associating behavior↔pose detections
ASSOCIATION_IOU_THRESHOLD: float = 0.40

# Track eviction: remove track if not seen for this many seconds
TRACK_EVICTION_SECONDS: float = 10.0

# Minimum frames in window before computing E (avoids unreliable early estimates)
MIN_WINDOW_FRAMES: int = 10

# API
API_HOST: str = "0.0.0.0"
API_PORT: int = 8001

# Tracker config — Ultralytics ships botsort.yaml bundled
TRACKER_CONFIG: str = "botsort.yaml"
BEHAVIOR_CONF_THRESHOLD: float = 0.35
POSE_CONF_THRESHOLD: float = 0.30
