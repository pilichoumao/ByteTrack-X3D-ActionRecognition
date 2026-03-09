from typing import Dict, List, Optional, Tuple

import numpy as np


class SkeletonFormatter:
    """Convert buffered skeleton sequence into MMACTION2-style inference dict.

    Shape convention:
    - T: temporal length
    - V: number of joints
    - M: number of persons (currently fixed to 1)
    - C: coordinate dims, fixed to 2 for 2D keypoints
    """

    def __init__(self, clip_len=48, num_keypoints=17, num_person=1, use_score=True):
        self.clip_len = int(clip_len)
        self.num_keypoints = int(num_keypoints)
        self.num_person = int(num_person)
        self.use_score = bool(use_score)

    def _normalize_length(self, items: List[dict]) -> List[dict]:
        if not items:
            return []
        items = sorted(items, key=lambda x: int(x["frame_id"]))
        if len(items) >= self.clip_len:
            return items[-self.clip_len:]

        pad_count = self.clip_len - len(items)
        pad_item = items[0]
        padded = [pad_item.copy() for _ in range(pad_count)] + items
        return padded

    def _fit_keypoints(self, kpts: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kpts = np.asarray(kpts, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        out_kpts = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        out_scores = np.zeros((self.num_keypoints,), dtype=np.float32)

        n = min(self.num_keypoints, kpts.shape[0])
        if n > 0:
            out_kpts[:n, :2] = kpts[:n, :2]
            out_scores[:n] = scores[:n]
        return out_kpts, out_scores

    def format_clip(self, items: List[dict], image_shape: Optional[Tuple[int, int]] = None) -> Dict:
        seq = self._normalize_length(items)
        if len(seq) == 0:
            raise ValueError("Empty skeleton clip cannot be formatted.")

        t = len(seq)
        m = self.num_person  # currently 1 for single-track inference
        v = self.num_keypoints

        keypoint = np.zeros((m, t, v, 2), dtype=np.float32)
        keypoint_score = np.zeros((m, t, v), dtype=np.float32)

        for ti, item in enumerate(seq):
            kpt, score = self._fit_keypoints(item["keypoints"], item["keypoint_scores"])
            keypoint[0, ti] = kpt
            if self.use_score:
                keypoint_score[0, ti] = score
            else:
                keypoint_score[0, ti] = 1.0

        h, w = (0, 0)
        if image_shape is not None:
            h, w = int(image_shape[0]), int(image_shape[1])

        return {
            "frame_dir": "online_track",
            "total_frames": t,
            "label": -1,
            "start_index": 0,
            "modality": "Pose",
            "img_shape": (h, w),
            "original_shape": (h, w),
            "keypoint": keypoint,              # [M, T, V, 2]
            "keypoint_score": keypoint_score,  # [M, T, V]
        }
