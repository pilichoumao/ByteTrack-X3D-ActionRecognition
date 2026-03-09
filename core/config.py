import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)

BYTE_TRACK_DIR = os.path.join(ROOT, "ByteTrack")
MMACTION_DIR = os.path.join(ROOT, "mmaction2")
MMPOSE_DIR = os.path.join(ROOT, "mmpose")
VIDEO_DIR = os.path.join(ROOT, "video")
CORE_DIR = CURRENT_DIR
TMP_DIR = os.path.join(ROOT, "tmp")
OUTPUT_DIR = os.path.join(ROOT, "output")

BT_EXP_FILE = os.path.join(
    BYTE_TRACK_DIR,
    "exps",
    "example",
    "mot",
    "yolox_x_mix_det.py"
)

BT_CKPT = os.path.join(
    BYTE_TRACK_DIR,
    "pretrained",
    "bytetrack_x_mot17.pth.tar"
)

# =========================
# RGB action models (existing branch)
# =========================
ACTION_MODELS = {
    "x3d_s": {
        "config": os.path.join(
            MMACTION_DIR,
            "configs",
            "recognition",
            "x3d",
            "x3d_s_13x6x1_facebook-kinetics400-rgb.py"
        ),
        "ckpt": os.path.join(
            MMACTION_DIR,
            "checkpoints",
            "x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth"
        ),
        "label_map": os.path.join(
            MMACTION_DIR,
            "tools",
            "data",
            "kinetics",
            "label_map_k400.txt"
        ),
        "num_samples": 16,
        "clip_len": 16,
        "stride": 8,
    },
    "tsm_r50_1x1x8": {
        "config": os.path.join(
            MMACTION_DIR,
            "configs",
            "recognition",
            "tsm",
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py"
        ),
        "ckpt": os.path.join(
            MMACTION_DIR,
            "checkpoints",
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth"
        ),
        "label_map": os.path.join(
            MMACTION_DIR,
            "tools",
            "data",
            "kinetics",
            "label_map_k400.txt"
        ),
        "num_samples": 8,
        "clip_len": 16,
        "stride": 8,
    },
}

DEFAULT_ACTION_MODEL = "x3d_s"

# Backward-compatible aliases used by current RGB pipeline code.
X3D_CONFIG = ACTION_MODELS["x3d_s"]["config"]
X3D_CKPT = ACTION_MODELS["x3d_s"]["ckpt"]
X3D_LABEL_MAP = ACTION_MODELS["x3d_s"]["label_map"]

# =========================
# Unified action mode switch
# =========================
ACTION_MODE = "rgb_x3d"
AVAILABLE_ACTION_MODES = [
    "rgb_tsm",
    "rgb_x3d",
    "skeleton_ctrgcn",
    "skeleton_stgcn",
]

RGB_MODE_TO_MODEL = {
    "rgb_tsm": "tsm_r50_1x1x8",
    "rgb_x3d": "x3d_s",
}

# =========================
# Pose (MMPose) settings
# =========================
POSE_CONFIG = os.path.join(
    MMPOSE_DIR,
    "configs",
    "body_2d_keypoint",
    "topdown_heatmap",
    "coco",
    "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
)

# TODO: place the actual pose checkpoint here after download.
POSE_CKPT = os.path.join(
    MMPOSE_DIR,
    "checkpoints",
    "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
)

POSE_DEVICE = "cuda:0"
POSE_CONF_THRESH = 0.2
NUM_KEYPOINTS = 17
NUM_PERSON = 1

# =========================
# Skeleton branch settings
# =========================
SKELETON_CLIP_LEN = 48
SKELETON_STRIDE = 16
SKELETON_USE_SCORE = True
SKELETON_MIN_FRAMES = 16
SKELETON_INFER_EVERY_N_FRAMES = 2
SKELETON_SMOOTH_WINDOW = 5
SKELETON_MIN_UPDATE_SCORE = 0.35

CTRGCN_CONFIG = os.path.join(
    MMACTION_DIR,
    "projects",
    "ctrgcn",
    "configs",
    "ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
)

# TODO: place the actual CTR-GCN checkpoint here after download.
CTRGCN_CKPT = os.path.join(
    MMACTION_DIR,
    "checkpoints",
    "ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth"
)

STGCN_CONFIG = os.path.join(
    MMACTION_DIR,
    "configs",
    "skeleton",
    "stgcn",
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
)

# TODO: place the actual ST-GCN checkpoint here after download.
STGCN_CKPT = os.path.join(
    MMACTION_DIR,
    "checkpoints",
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
)

SKELETON_LABEL_MAP = os.path.join(
    MMACTION_DIR,
    "tools",
    "data",
    "skeleton",
    "label_map_ntu60.txt"
)

SKELETON_MODEL_CONFIGS = {
    "ctrgcn": {
        "config": CTRGCN_CONFIG,
        "ckpt": CTRGCN_CKPT,
        "label_map": SKELETON_LABEL_MAP,
    },
    "stgcn": {
        "config": STGCN_CONFIG,
        "ckpt": STGCN_CKPT,
        "label_map": SKELETON_LABEL_MAP,
    },
}

SKELETON_MODE_TO_RECOGNIZER = {
    "skeleton_ctrgcn": "ctrgcn",
    "skeleton_stgcn": "stgcn",
}
