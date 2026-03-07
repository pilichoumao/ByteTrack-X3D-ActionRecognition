from collections import deque
import cv2


class ClipBufferManager:
    def __init__(self, clip_len=16, stride=8, expand_ratio=1.2, crop_size=224):
        self.clip_len = clip_len
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.crop_size = crop_size

        self.buffers = {}
        self.last_infer_frame = {}

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
        buf = self.buffers[track_id]
        self.last_infer_frame[track_id] = buf[-1]["frame_id"]
        return [item["crop"] for item in buf]

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