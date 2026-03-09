import os
import sys
from typing import Dict, List

import torch

from config import SKELETON_MODEL_CONFIGS, SKELETON_LABEL_MAP, MMACTION_DIR


class SkeletonActionRecognizer:
    """Skeleton action recognizer wrapper for CTR-GCN / ST-GCN."""

    def __init__(self, recognizer_type="ctrgcn", device=None):
        if recognizer_type not in SKELETON_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown recognizer_type '{recognizer_type}'. "
                f"Available: {sorted(SKELETON_MODEL_CONFIGS.keys())}"
            )

        self.recognizer_type = recognizer_type
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        model_cfg = SKELETON_MODEL_CONFIGS[recognizer_type]
        self.model_config_path = model_cfg["config"]
        self.model_ckpt_path = model_cfg["ckpt"]
        self.label_map_path = model_cfg.get("label_map", SKELETON_LABEL_MAP)

        self._init_mmaction_runtime()
        self.model = self._build_model()
        self.labels = self._load_labels(self.label_map_path)

        print("Skeleton device =", self.device)
        print("Skeleton recognizer =", self.recognizer_type)

    def _init_mmaction_runtime(self):
        if MMACTION_DIR not in sys.path:
            sys.path.insert(0, MMACTION_DIR)

        # CTR-GCN config in mmaction2/projects/ctrgcn uses
        # custom_imports = dict(imports='models', ...),
        # which requires PYTHONPATH to include this project root.
        if self.recognizer_type == "ctrgcn":
            ctrgcn_project_dir = os.path.join(MMACTION_DIR, "projects", "ctrgcn")
            if ctrgcn_project_dir not in sys.path:
                sys.path.insert(0, ctrgcn_project_dir)

        try:
            from mmaction.apis import init_recognizer, inference_recognizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Cannot import mmaction2 APIs. Please ensure mmaction2 dependencies "
                "(mmengine/mmcv) are installed in your current environment."
            ) from exc

        self.init_recognizer = init_recognizer
        self.inference_recognizer = inference_recognizer

    def _build_model(self):
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Skeleton model config not found: {self.model_config_path}\n"
                "Please check core/config.py skeleton model paths."
            )
        if not os.path.exists(self.model_ckpt_path):
            raise FileNotFoundError(
                f"Skeleton model checkpoint not found: {self.model_ckpt_path}\n"
                "Please download checkpoint and update core/config.py."
            )
        return self.init_recognizer(
            self.model_config_path,
            self.model_ckpt_path,
            device=self.device,
        )

    @staticmethod
    def _load_labels(label_map_path: str) -> List[str]:
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"SKELETON_LABEL_MAP not found: {label_map_path}\n"
                "Please check core/config.py -> SKELETON_LABEL_MAP."
            )
        labels = []
        with open(label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def _infer_with_mmaction(self, data: Dict):
        """Single place that calls mmaction skeleton inference.

        If your local mmaction2 version requires a custom test pipeline, replace
        this method and keep the public infer_clip() signature unchanged.
        """
        with torch.no_grad():
            return self.inference_recognizer(self.model, data)

    def infer_clip(self, skeleton_clip: Dict) -> Dict:
        if not skeleton_clip:
            return {"label": "unknown", "score": 0.0, "scores": []}

        result = self._infer_with_mmaction(skeleton_clip)

        pred_label = int(result.pred_label.item())
        pred_scores = result.pred_score.detach().cpu().tolist()
        pred_score = float(pred_scores[pred_label])

        if 0 <= pred_label < len(self.labels):
            pred_action = self.labels[pred_label]
        else:
            pred_action = str(pred_label)

        return {
            "label": pred_action,
            "score": pred_score,
            "scores": [float(x) for x in pred_scores],
        }
