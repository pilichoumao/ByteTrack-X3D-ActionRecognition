import os

ROOT = r"E:\ActionRecognition"

BYTE_TRACK_DIR = os.path.join(ROOT, "ByteTrack")
MMACTION_DIR = os.path.join(ROOT, "mmaction2")
VIDEO_DIR = os.path.join(ROOT, "video")
CORE_DIR = os.path.join(ROOT, "core")


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

# Action model registry.
# Keep defaults aligned with current behavior (X3D-S).
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

# Backward-compatible aliases used by current pipeline code.
X3D_CONFIG = ACTION_MODELS["x3d_s"]["config"]
X3D_CKPT = ACTION_MODELS["x3d_s"]["ckpt"]
X3D_LABEL_MAP = ACTION_MODELS["x3d_s"]["label_map"]

TMP_DIR = os.path.join(ROOT, "tmp")

# INPUT_VIDEO = os.path.join(VIDEO_DIR, "demo.mp4")

OUTPUT_DIR = os.path.join(ROOT, "output")

# OUTPUT_VIDEO = os.path.join(
#     OUTPUT_DIR,
#     "output_track_x3d.mp4"
# )
