from collections import deque
import cv2


class ClipBufferManager:
    def __init__(self, clip_len=16, stride=8, expand_ratio=1.2, crop_size=224):
        self.clip_len = clip_len
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.crop_size = crop_size

        # track_id -> deque([{"frame_id": int, "crop": np.ndarray}, ...])
        self.buffers = {}

        # track_id -> 上一次做动作识别时，对应的最新 frame_id
        self.last_infer_frame = {}

        # track_id -> 最近一次被 update() 的 frame_id
        self.last_seen_frame = {}

    def update(self, track_id, frame, bbox, frame_id):
        crop = self.crop_person(frame, bbox)
        if crop is None:
            return

        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.clip_len)

        self.buffers[track_id].append({
            "frame_id": frame_id,
            "crop": crop
        })

        self.last_seen_frame[track_id] = frame_id

    def get_ready_track_ids(self):
        ready_ids = []

        for track_id, buf in self.buffers.items():
            if len(buf) < self.clip_len:
                continue

            last_frame = self.last_infer_frame.get(track_id, -9999)
            cur_frame = buf[-1]["frame_id"]

            if cur_frame - last_frame >= self.stride:
                ready_ids.append(track_id)

        return ready_ids

    def get_clip(self, track_id):
        if track_id not in self.buffers:
            return []

        buf = self.buffers[track_id]
        if len(buf) == 0:
            return []

        self.last_infer_frame[track_id] = buf[-1]["frame_id"]

        # 返回副本，避免外部误改内部缓存结构
        return [item["crop"].copy() for item in buf]

    def remove_track(self, track_id):
        self.buffers.pop(track_id, None)
        self.last_infer_frame.pop(track_id, None)
        self.last_seen_frame.pop(track_id, None)

    def cleanup_expired_tracks(self, current_frame_id, max_missing_frames=60):
        """
        清理长时间未出现的 track，避免无效 ID 持续堆积。

        参数:
            current_frame_id: 当前处理到的视频帧号
            max_missing_frames: 若某个 track 超过这么多帧未出现，则删除

        返回:
            expired_ids: 本次被清理掉的 track_id 列表
        """
        expired_ids = []

        for track_id, last_seen in list(self.last_seen_frame.items()):
            if current_frame_id - last_seen > max_missing_frames:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self.remove_track(track_id)

        return expired_ids

    def crop_person(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * self.expand_ratio
        bh = (y2 - y1) * self.expand_ratio

        nx1 = max(0, int(cx - bw / 2.0))
        ny1 = max(0, int(cy - bh / 2.0))
        nx2 = min(w - 1, int(cx + bw / 2.0))
        ny2 = min(h - 1, int(cy + bh / 2.0))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        return crop