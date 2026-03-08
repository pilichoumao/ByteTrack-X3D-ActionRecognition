import os
import cv2
import time
import statistics
from collections import deque, Counter

from config import INPUT_VIDEO, OUTPUT_VIDEO, TMP_DIR, OUTPUT_DIR
from tracker_adapter import ByteTrackAdapter
from clip_buffer import ClipBufferManager
from action_recognizer import ActionRecognizer
from visualizer import draw_tracks

try:
    import torch
except ImportError:
    torch = None


def sync_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def print_stats(name, data):
    if not data:
        print(f"{name}: no data")
        return
    print(
        f"{name}: "
        f"mean={statistics.mean(data) * 1000:.2f} ms, "
        f"median={statistics.median(data) * 1000:.2f} ms, "
        f"min={min(data) * 1000:.2f} ms, "
        f"max={max(data) * 1000:.2f} ms"
    )


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("video:", INPUT_VIDEO)
    print("size:", width, height)
    print("fps:", fps)
    print("tmp dir:", TMP_DIR)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker = ByteTrackAdapter()
    clip_manager = ClipBufferManager(
        clip_len=16,
        stride=8,
        expand_ratio=1.2,
        crop_size=224
    )
    action_model = ActionRecognizer()

    action_results = {}
    action_history = {}

    frame_id = 0

    # 超过这么多帧未出现，就清理该 track 的缓存
    max_missing_frames = 60

    # ===== timing stats =====
    track_time_list = []          # 每帧 YOLO+ByteTrack 耗时
    action_time_list = []         # 每次动作识别调用耗时
    action_time_per_frame = []    # 每帧动作识别总耗时
    total_frame_time_list = []    # 每帧总耗时

    total_action_calls = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        # ===== 1) YOLO + ByteTrack timing =====
        sync_cuda()
        t0 = time.perf_counter()
        track_results = tracker.update(frame)
        sync_cuda()
        t1 = time.perf_counter()

        tracker_time = t1 - t0
        track_time_list.append(tracker_time)

        # ===== clip update =====
        active_track_ids = set()
        for obj in track_results:
            track_id = obj["track_id"]
            bbox = obj["bbox"]

            active_track_ids.add(track_id)
            clip_manager.update(track_id, frame, bbox, frame_id)

        ready_ids = clip_manager.get_ready_track_ids()

        # 本帧所有动作识别累计耗时
        frame_action_time = 0.0

        # ===== 2) Action recognition timing =====
        for track_id in ready_ids:
            clip = clip_manager.get_clip(track_id)

            if not clip:
                continue

            temp_video_path = os.path.join(TMP_DIR, f"track_{track_id}.mp4")

            sync_cuda()
            t2 = time.perf_counter()
            pred = action_model.infer_clip(
                clip,
                temp_video_path=temp_video_path
            )
            sync_cuda()
            t3 = time.perf_counter()

            infer_time = t3 - t2
            action_time_list.append(infer_time)
            frame_action_time += infer_time
            total_action_calls += 1

            if track_id not in action_history:
                action_history[track_id] = deque(maxlen=5)

            action_history[track_id].append(pred)

            labels = [x["label"] for x in action_history[track_id]]
            best_label = Counter(labels).most_common(1)[0][0]
            same_label_scores = [
                x["score"] for x in action_history[track_id]
                if x["label"] == best_label
            ]
            best_score = sum(same_label_scores) / len(same_label_scores)

            action_results[track_id] = {
                "label": best_label,
                "score": best_score
            }

            print(
                "frame", frame_id,
                "track", track_id,
                "raw_action =", pred["label"],
                "show_action =", action_results[track_id]["label"],
                "score =", round(action_results[track_id]["score"], 4),
                "infer_time_ms =", round(infer_time * 1000, 2)
            )

        # ===== 3) cleanup expired tracks =====
        expired_ids = clip_manager.cleanup_expired_tracks(
            current_frame_id=frame_id,
            max_missing_frames=max_missing_frames
        )

        for tid in expired_ids:
            action_history.pop(tid, None)
            action_results.pop(tid, None)

        if expired_ids:
            print(f"frame {frame_id} | cleaned expired track ids: {expired_ids}")

        vis_frame = draw_tracks(frame, track_results, action_results)
        writer.write(vis_frame)

        frame_end = time.perf_counter()
        total_frame_time = frame_end - frame_start

        action_time_per_frame.append(frame_action_time)
        total_frame_time_list.append(total_frame_time)

        if frame_id % 30 == 0:
            print(
                f"frame {frame_id} | "
                f"tracks={len(track_results)} | "
                f"track_time={tracker_time * 1000:.2f} ms | "
                f"action_time_frame={frame_action_time * 1000:.2f} ms | "
                f"total_frame_time={total_frame_time * 1000:.2f} ms | "
                f"cache_tracks={len(clip_manager.buffers)}"
            )

        frame_id += 1

    cap.release()
    writer.release()

    print("finished")
    print("output:", OUTPUT_VIDEO)

    print("\n===== Timing Summary =====")
    print_stats("YOLO+ByteTrack per frame", track_time_list)
    print_stats("Action recognition per call", action_time_list)
    print_stats("Action recognition per frame", action_time_per_frame)
    print_stats("Total pipeline per frame", total_frame_time_list)

    if total_frame_time_list:
        avg_total = statistics.mean(total_frame_time_list)
        print(f"Approx pipeline FPS = {1.0 / avg_total:.2f}")

    print(f"Total frames = {frame_id}")
    print(f"Total action calls = {total_action_calls}")
    print(f"Remaining cached tracks = {len(clip_manager.buffers)}")


if __name__ == "__main__":
    main()