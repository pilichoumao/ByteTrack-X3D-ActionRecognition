import os
import sys
from typing import Dict, List

import numpy as np

from config import POSE_CONFIG, POSE_CKPT, POSE_CONF_THRESH, POSE_DEVICE, MMPOSE_DIR


class PoseEstimator:
    """Track-level top-down pose wrapper.

    Input: frame + one track bbox
    Output:
        {
          "keypoints": ndarray [V, 2],
          "keypoint_scores": ndarray [V],
          "bbox": [x1, y1, x2, y2],
          "track_id": int,
          "valid": bool
        }
    """

    def __init__(
        self,
        pose_config: str = POSE_CONFIG,
        pose_ckpt: str = POSE_CKPT,
        device: str = POSE_DEVICE,
        conf_thresh: float = POSE_CONF_THRESH,
    ):
        self.pose_config = pose_config
        self.pose_ckpt = pose_ckpt
        self.device = device
        self.conf_thresh = conf_thresh

        self._init_mmpose_runtime()
        self.model = self._build_model()

    def _init_mmpose_runtime(self) -> None:
        if MMPOSE_DIR not in sys.path:
            sys.path.insert(0, MMPOSE_DIR)

        try:
            from mmpose.apis import init_model, inference_topdown  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Cannot import mmpose. Please install mmpose and its dependencies "
                "(mmengine/mmcv/mmdet), then retry."
            ) from exc

        self.init_model = init_model
        self.inference_topdown = inference_topdown

    def _build_model(self):
        if not os.path.exists(self.pose_config):
            raise FileNotFoundError(
                f"POSE_CONFIG not found: {self.pose_config}\n"
                "Please check core/config.py -> POSE_CONFIG."
            )
        if not os.path.exists(self.pose_ckpt):
            raise FileNotFoundError(
                f"POSE_CKPT not found: {self.pose_ckpt}\n"
                "Please download checkpoint and update core/config.py -> POSE_CKPT."
            )
        return self.init_model(self.pose_config, self.pose_ckpt, device=self.device)

    @staticmethod
    def _to_float_bbox(bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [x1, y1, x2, y2]

    @staticmethod
    def _empty_result(track_id: int, bbox: List[float]) -> Dict:
        return {
            "keypoints": np.zeros((0, 2), dtype=np.float32),
            "keypoint_scores": np.zeros((0,), dtype=np.float32),
            "bbox": bbox,
            "track_id": int(track_id),
            "valid": False,
        }

    def infer_track(self, frame, bbox: List[float], track_id: int) -> Dict:
        bbox = self._to_float_bbox(bbox)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return self._empty_result(track_id, bbox)

        # Keep one-person top-down inference to match "single track -> single person".
        person_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        try:
            pose_results = self.inference_topdown(
                self.model,
                frame,
                person_bbox,
                bbox_format="xyxy",
            )
        except TypeError:
            # Backward-compatible fallback for older mmpose signatures.
            pose_results = self.inference_topdown(self.model, frame, person_bbox)

        if pose_results is None or len(pose_results) == 0:
            return self._empty_result(track_id, bbox)

        data_sample = pose_results[0]
        pred_instances = getattr(data_sample, "pred_instances", None)
        if pred_instances is None:
            return self._empty_result(track_id, bbox)

        keypoints = getattr(pred_instances, "keypoints", None)
        keypoint_scores = getattr(pred_instances, "keypoint_scores", None)
        if keypoints is None:
            return self._empty_result(track_id, bbox)

        keypoints = np.asarray(keypoints)
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        keypoints = keypoints.astype(np.float32)

        if keypoint_scores is None:
            keypoint_scores = np.ones((keypoints.shape[0],), dtype=np.float32)
        else:
            keypoint_scores = np.asarray(keypoint_scores)
            if keypoint_scores.ndim == 2:
                keypoint_scores = keypoint_scores[0]
            keypoint_scores = keypoint_scores.astype(np.float32)

        valid = bool(keypoints.shape[0] > 0 and float(np.mean(keypoint_scores)) >= self.conf_thresh)
        return {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "bbox": bbox,
            "track_id": int(track_id),
            "valid": valid,
        }
