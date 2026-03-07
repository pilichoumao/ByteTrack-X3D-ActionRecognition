import cv2

def draw_tracks(frame, track_results, action_results=None):
    vis = frame.copy()

    if action_results is None:
        action_results = {}

    for obj in track_results:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        track_id = obj["track_id"]
        score = obj["score"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if track_id in action_results:
            action_label = action_results[track_id]["label"]
            action_score = action_results[track_id]["score"]
            text = f"ID {track_id} | {action_label} | {action_score:.2f}"
        else:
            text = f"ID {track_id} | track {score:.2f}"

        text_y = y1 - 10
        if text_y < 20:
            text_y = y1 + 25

        cv2.putText(
            vis,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis