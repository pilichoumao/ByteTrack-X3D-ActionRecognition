from collections import Counter, deque

from action_recognizer import ActionRecognizer
from clip_buffer import ClipBufferManager
from config import (
    ACTION_MODELS,
    NUM_KEYPOINTS,
    NUM_PERSON,
    RGB_MODE_TO_MODEL,
    SKELETON_INFER_EVERY_N_FRAMES,
    SKELETON_MIN_FRAMES,
    SKELETON_MODE_TO_RECOGNIZER,
    SKELETON_STRIDE,
    SKELETON_USE_SCORE,
)
from pose_estimator import PoseEstimator
from skeleton_action_recognizer import SkeletonActionRecognizer
from skeleton_buffer import SkeletonBufferManager
from skeleton_formatter import SkeletonFormatter


class RGBActionRunner:
    def __init__(self, mode, action_model_override=""):
        self.mode = mode
        self.model_name = action_model_override or RGB_MODE_TO_MODEL.get(mode)
        if self.model_name not in ACTION_MODELS:
            raise ValueError(
                f"Invalid RGB action model: {self.model_name}. "
                f"Available: {sorted(ACTION_MODELS.keys())}"
            )

        self.model_profile = ACTION_MODELS[self.model_name]
        self.clip_manager = ClipBufferManager(
            clip_len=self.model_profile["clip_len"],
            stride=self.model_profile["stride"],
            expand_ratio=1.2,
            crop_size=224,
            num_samples=self.model_profile["num_samples"],
        )
        self.action_model = ActionRecognizer(model_name=self.model_name)
        self.action_results = {}
        self.action_history = {}
        self.total_action_calls = 0
        self.total_pose_calls = 0
        self.total_pose_valid = 0

    def process_frame(self, frame, track_results, frame_id, sync_cuda, dry_run=False):
        _ = frame_id
        _ = dry_run
        frame_action_time = 0.0
        pose_results_for_vis = {}

        for obj in track_results:
            self.clip_manager.update(obj["track_id"], frame, obj["bbox"], frame_id)

        ready_ids = self.clip_manager.get_ready_track_ids()
        infer_records = []
        for track_id in ready_ids:
            clip = self.clip_manager.get_clip(track_id)
            if not clip:
                continue

            sync_cuda()
            import time
            t2 = time.perf_counter()
            pred = self.action_model.infer_clip(clip)
            sync_cuda()
            t3 = time.perf_counter()

            infer_time = t3 - t2
            frame_action_time += infer_time
            self.total_action_calls += 1

            if track_id not in self.action_history:
                self.action_history[track_id] = deque(maxlen=5)
            self.action_history[track_id].append(pred)

            labels = [x["label"] for x in self.action_history[track_id]]
            best_label = Counter(labels).most_common(1)[0][0]
            same_label_scores = [
                x["score"] for x in self.action_history[track_id] if x["label"] == best_label
            ]
            best_score = sum(same_label_scores) / len(same_label_scores)
            self.action_results[track_id] = {"label": best_label, "score": best_score}

            infer_records.append((track_id, pred, self.action_results[track_id], infer_time))

        return frame_action_time, pose_results_for_vis, infer_records

    def cleanup_expired_tracks(self, frame_id, max_missing_frames):
        expired_ids = self.clip_manager.cleanup_expired_tracks(frame_id, max_missing_frames)
        for tid in expired_ids:
            self.action_history.pop(tid, None)
            self.action_results.pop(tid, None)
        return expired_ids

    @property
    def cache_size(self):
        return len(self.clip_manager.buffers)


class SkeletonActionRunner:
    def __init__(
        self,
        mode,
        clip_len,
        smooth_window,
        min_update_score,
        pose_conf_thresh,
        dry_run=False,
    ):
        self.mode = mode
        self.recognizer_type = SKELETON_MODE_TO_RECOGNIZER[mode]
        self.dry_run = bool(dry_run)
        self.smooth_window = int(smooth_window)
        self.min_update_score = float(min_update_score)

        self.pose_estimator = PoseEstimator(conf_thresh=pose_conf_thresh)
        self.skeleton_buffer = SkeletonBufferManager(
            clip_len=clip_len,
            stride=SKELETON_STRIDE,
            min_frames=SKELETON_MIN_FRAMES,
        )
        self.skeleton_formatter = SkeletonFormatter(
            clip_len=clip_len,
            num_keypoints=NUM_KEYPOINTS,
            num_person=NUM_PERSON,
            use_score=SKELETON_USE_SCORE,
        )
        self.skeleton_action_model = None
        if not self.dry_run:
            self.skeleton_action_model = SkeletonActionRecognizer(recognizer_type=self.recognizer_type)

        self.action_results = {}
        self.action_history = {}
        self.total_action_calls = 0
        self.total_pose_calls = 0
        self.total_pose_valid = 0

    def process_frame(self, frame, track_results, frame_id, sync_cuda, dry_run=False):
        _ = dry_run
        frame_action_time = 0.0
        pose_results_for_vis = {}
        infer_records = []

        for obj in track_results:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            try:
                self.total_pose_calls += 1
                pose_out = self.pose_estimator.infer_track(frame, bbox, track_id)
            except Exception as exc:
                infer_records.append(("pose_error", track_id, str(exc), 0.0))
                continue

            if not pose_out["valid"]:
                continue

            self.total_pose_valid += 1
            self.skeleton_buffer.update(
                track_id=track_id,
                frame_id=frame_id,
                keypoints=pose_out["keypoints"],
                keypoint_scores=pose_out["keypoint_scores"],
            )
            pose_results_for_vis[track_id] = {
                "keypoints": pose_out["keypoints"],
                "keypoint_scores": pose_out["keypoint_scores"],
            }

        if frame_id % SKELETON_INFER_EVERY_N_FRAMES != 0:
            return frame_action_time, pose_results_for_vis, infer_records

        import time

        for track_id in list(self.skeleton_buffer.buffers.keys()):
            if not self.skeleton_buffer.is_ready(track_id):
                continue

            try:
                clip_items = self.skeleton_buffer.get_clip(track_id)
                clip_data = self.skeleton_formatter.format_clip(
                    clip_items,
                    image_shape=frame.shape[:2],
                )
                latest_frame = clip_items[-1]["frame_id"]
            except Exception as exc:
                infer_records.append(("format_error", track_id, str(exc), 0.0))
                continue

            if self.dry_run:
                self.skeleton_buffer.mark_inferred(track_id, latest_frame)
                self.action_results[track_id] = {"label": "pose_ready", "score": 1.0}
                infer_records.append(("dry_run", track_id, "pose_ready", 0.0))
                continue

            sync_cuda()
            t2 = time.perf_counter()
            try:
                pred = self.skeleton_action_model.infer_clip(clip_data)
            except Exception as exc:
                infer_records.append(("infer_error", track_id, str(exc), 0.0))
                continue
            sync_cuda()
            t3 = time.perf_counter()

            infer_time = t3 - t2
            frame_action_time += infer_time
            self.total_action_calls += 1
            self.skeleton_buffer.mark_inferred(track_id, latest_frame)

            if track_id not in self.action_history:
                self.action_history[track_id] = deque(maxlen=self.smooth_window)
            self.action_history[track_id].append(pred)

            if float(pred["score"]) >= self.min_update_score:
                labels = [x["label"] for x in self.action_history[track_id]]
                best_label = Counter(labels).most_common(1)[0][0]
                same_label_scores = [
                    x["score"] for x in self.action_history[track_id] if x["label"] == best_label
                ]
                best_score = sum(same_label_scores) / len(same_label_scores)
                self.action_results[track_id] = {"label": best_label, "score": best_score}

            infer_records.append(("ok", track_id, pred, infer_time))

        return frame_action_time, pose_results_for_vis, infer_records

    def cleanup_expired_tracks(self, frame_id, max_missing_frames):
        expired_ids = self.skeleton_buffer.cleanup_expired_tracks(frame_id, max_missing_frames)
        for tid in expired_ids:
            self.action_history.pop(tid, None)
            self.action_results.pop(tid, None)
        return expired_ids

    @property
    def cache_size(self):
        return len(self.skeleton_buffer.buffers)
