import cv2


COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_skeleton(vis, keypoints, keypoint_scores=None, score_thr=0.2):
    if keypoints is None:
        return

    kpts = keypoints
    scores = keypoint_scores
    num_kpts = len(kpts)

    def is_valid(i):
        if i < 0 or i >= num_kpts:
            return False
        if scores is None:
            return True
        return float(scores[i]) >= float(score_thr)

    for i in range(num_kpts):
        if not is_valid(i):
            continue
        x, y = int(kpts[i][0]), int(kpts[i][1])
        cv2.circle(vis, (x, y), 3, (0, 200, 255), -1)

    for i, j in COCO17_SKELETON:
        if not (is_valid(i) and is_valid(j)):
            continue
        x1, y1 = int(kpts[i][0]), int(kpts[i][1])
        x2, y2 = int(kpts[j][0]), int(kpts[j][1])
        cv2.line(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)


def draw_tracks(
    frame,
    track_results,
    action_results=None,
    pose_results=None,
    skeleton_score_thr=0.2,
):
    vis = frame.copy()

    if action_results is None:
        action_results = {}
    if pose_results is None:
        pose_results = {}

    for obj in track_results:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        track_id = obj["track_id"]
        score = obj["score"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if track_id in action_results:
            action_label = action_results[track_id].get("label", "unknown")
            action_score = float(action_results[track_id].get("score", 0.0))
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
            2,
        )

        if track_id in pose_results:
            pose_item = pose_results[track_id]
            _draw_skeleton(
                vis,
                pose_item.get("keypoints"),
                pose_item.get("keypoint_scores"),
                score_thr=skeleton_score_thr,
            )

    return vis
