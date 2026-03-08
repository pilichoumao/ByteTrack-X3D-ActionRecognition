import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from types import SimpleNamespace

from config import BT_EXP_FILE, BT_CKPT

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
BYTE_TRACK_DIR = os.path.join(ROOT_DIR, "ByteTrack")

if BYTE_TRACK_DIR not in sys.path:
    sys.path.insert(0, BYTE_TRACK_DIR)

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess, fuse_model
from yolox.tracker.byte_tracker import BYTETracker


class ByteTrackAdapter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = (self.device == "cuda")
        print("ByteTrack device =", self.device)

        if self.device == "cuda":
            cudnn.benchmark = True

        self.exp = get_exp(BT_EXP_FILE, None)

        self.model = self.exp.get_model()
        ckpt = torch.load(BT_CKPT, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        # fuse conv + bn
        self.model = fuse_model(self.model)

        # FP16
        if self.fp16:
            self.model.half()

        args = SimpleNamespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            aspect_ratio_thresh=1.6,
            min_box_area=10,
            mot20=False
        )
        self.tracker = BYTETracker(args, frame_rate=30)

    def update(self, frame):
        img, ratio = preproc(frame, self.exp.test_size, (0, 0, 0), (1, 1, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        if self.fp16:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(
                outputs,
                self.exp.num_classes,
                self.exp.test_conf,
                self.exp.nmsthre
            )

        output_results = outputs[0]
        if output_results is None:
            return []

        online_targets = self.tracker.update(
            output_results,
            [frame.shape[0], frame.shape[1]],
            self.exp.test_size
        )

        results = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = int(t.track_id)
            score = float(t.score)

            x1 = float(tlwh[0])
            y1 = float(tlwh[1])
            x2 = float(tlwh[0] + tlwh[2])
            y2 = float(tlwh[1] + tlwh[3])

            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "score": score
            })

        return results


if __name__ == "__main__":
    cap = cv2.VideoCapture(r"E:\ActionRecognition\video\demo.mp4")
    ret, frame = cap.read()
    cap.release()

    tracker = ByteTrackAdapter()
    results = tracker.update(frame)
    print("results =", results)