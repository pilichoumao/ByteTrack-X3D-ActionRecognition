from collections import deque
from typing import Dict, List

import numpy as np


class SkeletonBufferManager:
    """Per-track skeleton temporal buffer.

    Stored item:
        {
          "frame_id": int,
          "keypoints": np.ndarray [V, 2],
          "keypoint_scores": np.ndarray [V]
        }
    """

    def __init__(self, clip_len=48, stride=16, min_frames=16):
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.min_frames = int(min_frames)

        self.buffers: Dict[int, deque] = {}
        self.last_infer_frame: Dict[int, int] = {}
        self.last_seen_frame: Dict[int, int] = {}

    def update(self, track_id, frame_id, keypoints, keypoint_scores):
        if keypoints is None or keypoint_scores is None:
            return

        keypoints = np.asarray(keypoints, dtype=np.float32)
        keypoint_scores = np.asarray(keypoint_scores, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[0] == 0:
            return

        tid = int(track_id)
        if tid not in self.buffers:
            self.buffers[tid] = deque(maxlen=self.clip_len)

        self.buffers[tid].append(
            {
                "frame_id": int(frame_id),
                "keypoints": keypoints.copy(),
                "keypoint_scores": keypoint_scores.copy(),
            }
        )
        self.last_seen_frame[tid] = int(frame_id)

    def is_ready(self, track_id) -> bool:
        tid = int(track_id)
        if tid not in self.buffers or len(self.buffers[tid]) == 0:
            return False
        if len(self.buffers[tid]) < self.min_frames:
            return False

        cur_frame = int(self.buffers[tid][-1]["frame_id"])
        last_frame = int(self.last_infer_frame.get(tid, -10**9))
        return (cur_frame - last_frame) >= self.stride

    def get_clip(self, track_id) -> List[dict]:
        tid = int(track_id)
        if tid not in self.buffers:
            return []
        return [item.copy() for item in list(self.buffers[tid])]

    def mark_inferred(self, track_id, frame_id):
        self.last_infer_frame[int(track_id)] = int(frame_id)

    def remove_track(self, track_id):
        tid = int(track_id)
        self.buffers.pop(tid, None)
        self.last_infer_frame.pop(tid, None)
        self.last_seen_frame.pop(tid, None)

    def cleanup_expired_tracks(self, current_frame_id, max_missing_frames=60):
        expired_ids = []
        for tid, last_seen in list(self.last_seen_frame.items()):
            if int(current_frame_id) - int(last_seen) > int(max_missing_frames):
                expired_ids.append(tid)

        for tid in expired_ids:
            self.remove_track(tid)
        return expired_ids
