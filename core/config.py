import os

ROOT = r"E:\ActionRecognition"

BYTE_TRACK_DIR = os.path.join(ROOT, "ByteTrack")
MMACTION_DIR = os.path.join(ROOT, "mmaction2")
VIDEO_DIR = os.path.join(ROOT, "video")
CORE_DIR = os.path.join(ROOT, "core")

INPUT_VIDEO = os.path.join(VIDEO_DIR, "demo.mp4")

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

CLIP_LEN = 16
CLIP_STRIDE = 8
EXPAND_RATIO = 1.2

# X3D_CONFIG = os.path.join(
#     MMACTION_DIR,
#     "configs",
#     "recognition",
#     "x3d",
#     "x3d_m_16x5x1_facebook-kinetics400-rgb.py"
# )
#
# X3D_CKPT = os.path.join(
#     MMACTION_DIR,
#     "checkpoints",
#     "x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth"
# )

X3D_CONFIG = os.path.join(
    MMACTION_DIR,
    "configs",
    "recognition",
    "x3d",
    "x3d_s_13x6x1_facebook-kinetics400-rgb.py"
)

X3D_CKPT = os.path.join(
    MMACTION_DIR,
    "checkpoints",
    "x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth"
)

X3D_LABEL_MAP = os.path.join(
    MMACTION_DIR,
    "tools",
    "data",
    "kinetics",
    "label_map_k400.txt"
)

TMP_DIR = os.path.join(ROOT, "tmp")

OUTPUT_DIR = os.path.join(ROOT, "output")

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "output_track_x3d.mp4"
)