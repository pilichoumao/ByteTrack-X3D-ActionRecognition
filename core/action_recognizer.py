import os
import sys
import cv2
import torch

from config import X3D_CONFIG, X3D_CKPT, X3D_LABEL_MAP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MMACTION_DIR = os.path.join(ROOT_DIR, "mmaction2")

if MMACTION_DIR not in sys.path:
    sys.path.insert(0, MMACTION_DIR)

from mmaction.apis import init_recognizer, inference_recognizer


class ActionRecognizer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("X3D device =", self.device)

        self.model = init_recognizer(
            X3D_CONFIG,
            X3D_CKPT,
            device=self.device
        )

        self.labels = []
        with open(X3D_LABEL_MAP, "r", encoding="utf-8") as f:
            for line in f:
                self.labels.append(line.strip())

    def infer_clip(self, clip, temp_video_path):
        if not clip:
            return {
                "label": "unknown",
                "score": 0.0
            }

        h, w = clip[0].shape[:2]

        writer = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            5,
            (w, h)
        )

        for img in clip:
            writer.write(img)

        writer.release()

        result = inference_recognizer(self.model, temp_video_path)

        pred_label = int(result.pred_label.item())
        pred_score = float(result.pred_score[pred_label].item())
        pred_action = self.labels[pred_label]

        # 推理完成后删除临时文件，避免堆积
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception as e:
            print("WARNING: temp file delete failed:", temp_video_path, e)

        return {
            "label": pred_action,
            "score": pred_score
        }