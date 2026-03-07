import os
import cv2
from collections import deque, Counter

from config import INPUT_VIDEO, OUTPUT_VIDEO, TMP_DIR, OUTPUT_DIR
from tracker_adapter import ByteTrackAdapter
from clip_buffer import ClipBufferManager
from action_recognizer import ActionRecognizer
from visualizer import draw_tracks


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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        track_results = tracker.update(frame)

        for obj in track_results:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            clip_manager.update(track_id, frame, bbox, frame_id)

        ready_ids = clip_manager.get_ready_track_ids()

        for track_id in ready_ids:
            clip = clip_manager.get_clip(track_id)

            temp_video_path = os.path.join(TMP_DIR, f"track_{track_id}.mp4")

            pred = action_model.infer_clip(
                clip,
                temp_video_path=temp_video_path
            )

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
                "score =", round(action_results[track_id]["score"], 4)
            )

        vis_frame = draw_tracks(frame, track_results, action_results)
        writer.write(vis_frame)

        if frame_id % 30 == 0:
            print("frame", frame_id, "tracks:", len(track_results))

        frame_id += 1

    cap.release()
    writer.release()

    print("finished")
    print("output:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()