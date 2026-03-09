import argparse
import os
import statistics
import time

import cv2

from config import (
    ACTION_MODE,
    AVAILABLE_ACTION_MODES,
    OUTPUT_DIR,
    RGB_MODE_TO_MODEL,
    SKELETON_CLIP_LEN,
    SKELETON_MIN_UPDATE_SCORE,
    SKELETON_SMOOTH_WINDOW,
    TMP_DIR,
    POSE_CONF_THRESH,
)
from pipeline_runners import RGBActionRunner, SkeletonActionRunner
from tracker_adapter import ByteTrackAdapter
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


def is_rgb_mode(mode_name: str) -> bool:
    return mode_name.startswith("rgb_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument(
        "--mode",
        type=str,
        default=ACTION_MODE,
        choices=AVAILABLE_ACTION_MODES,
        help="pipeline mode: rgb_tsm/rgb_x3d/skeleton_ctrgcn/skeleton_stgcn",
    )
    parser.add_argument(
        "--action-model",
        type=str,
        default="",
        help="RGB mode only. Optional override model key, e.g. x3d_s / tsm_r50_1x1x8",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="optional custom output suffix; when empty, auto includes mode/model",
    )
    parser.add_argument(
        "--dry-run-skeleton",
        action="store_true",
        help="skeleton mode only: run pose+buffer without skeleton classifier",
    )
    parser.add_argument("--max-frames", type=int, default=-1, help="debug: stop early")
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="draw skeleton keypoints/links on output video in skeleton mode",
    )
    parser.add_argument(
        "--skeleton-score-thr",
        type=float,
        default=0.2,
        help="keypoint score threshold for skeleton drawing",
    )
    parser.add_argument(
        "--pose-conf-thr",
        type=float,
        default=-1.0,
        help="pose valid threshold override; -1 uses config default",
    )
    parser.add_argument(
        "--skeleton-smooth-window",
        type=int,
        default=-1,
        help="skeleton smoothing window override; -1 uses config default",
    )
    parser.add_argument(
        "--skeleton-min-update-score",
        type=float,
        default=-1.0,
        help="skeleton minimum raw score to update shown label; -1 uses config default",
    )
    return parser.parse_args()


def build_output_path(input_video, output_dir, mode, effective_model, suffix):
    video_name = os.path.basename(input_video)
    name, _ = os.path.splitext(video_name)
    mode_tag = mode
    if effective_model:
        mode_tag = f"{mode}_{effective_model}"

    if suffix:
        out_name = f"{name}{suffix}.mp4"
    else:
        out_name = f"{name}_{mode_tag}.mp4"
    return os.path.join(output_dir, out_name)


def main():
    args = parse_args()
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pose_conf_thr = args.pose_conf_thr if args.pose_conf_thr >= 0 else POSE_CONF_THRESH
    skeleton_smooth_window = (
        args.skeleton_smooth_window if args.skeleton_smooth_window > 0 else SKELETON_SMOOTH_WINDOW
    )
    skeleton_min_update_score = (
        args.skeleton_min_update_score
        if args.skeleton_min_update_score >= 0
        else SKELETON_MIN_UPDATE_SCORE
    )

    runner = None
    effective_model = ""
    if is_rgb_mode(args.mode):
        runner = RGBActionRunner(mode=args.mode, action_model_override=args.action_model)
        effective_model = runner.model_name
    else:
        runner = SkeletonActionRunner(
            mode=args.mode,
            clip_len=SKELETON_CLIP_LEN,
            smooth_window=skeleton_smooth_window,
            min_update_score=skeleton_min_update_score,
            pose_conf_thresh=pose_conf_thr,
            dry_run=args.dry_run_skeleton,
        )
        effective_model = runner.recognizer_type

    output_video = build_output_path(
        input_video=args.video,
        output_dir=OUTPUT_DIR,
        mode=args.mode,
        effective_model=effective_model,
        suffix=args.suffix,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print("video:", args.video)
    print("size:", width, height)
    print("fps:", fps)
    print("mode:", args.mode)
    print("effective model:", effective_model)
    if not is_rgb_mode(args.mode):
        print("dry-run-skeleton:", bool(args.dry_run_skeleton))
        print("show-skeleton:", bool(args.show_skeleton))
        print("skeleton-score-thr:", args.skeleton_score_thr)
        print("pose-conf-thr:", pose_conf_thr)
        print("skeleton-smooth-window:", skeleton_smooth_window)
        print("skeleton-min-update-score:", skeleton_min_update_score)

    tracker = ByteTrackAdapter()
    max_missing_frames = 60
    frame_id = 0

    track_time_list = []
    action_time_list = []
    action_time_per_frame = []
    total_frame_time_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_id >= args.max_frames:
            print(f"reach max_frames={args.max_frames}, stop early")
            break

        frame_start = time.perf_counter()

        sync_cuda()
        t0 = time.perf_counter()
        track_results = tracker.update(frame)
        sync_cuda()
        t1 = time.perf_counter()
        tracker_time = t1 - t0
        track_time_list.append(tracker_time)

        frame_action_time, pose_results_for_vis, infer_records = runner.process_frame(
            frame=frame,
            track_results=track_results,
            frame_id=frame_id,
            sync_cuda=sync_cuda,
            dry_run=args.dry_run_skeleton,
        )

        for rec in infer_records:
            if is_rgb_mode(args.mode):
                track_id, pred, shown, infer_time = rec
                action_time_list.append(infer_time)
                print(
                    "frame", frame_id,
                    "track", track_id,
                    "raw_action =", pred["label"],
                    "show_action =", shown["label"],
                    "score =", round(shown["score"], 4),
                    "infer_time_ms =", round(infer_time * 1000, 2),
                )
            else:
                tag = rec[0]
                if tag == "ok":
                    _, track_id, pred, infer_time = rec
                    action_time_list.append(infer_time)
                    print(
                        "frame", frame_id,
                        "track", track_id,
                        "raw_skeleton_action =", pred["label"],
                        "raw_score =", round(pred["score"], 4),
                        "show_action =", runner.action_results.get(track_id, {}).get("label", "unknown"),
                        "show_score =", round(runner.action_results.get(track_id, {}).get("score", 0.0), 4),
                        "infer_time_ms =", round(infer_time * 1000, 2),
                    )
                elif tag == "dry_run":
                    _, track_id, msg, _ = rec
                    print("frame", frame_id, "track", track_id, "skeleton_dry_run =", msg)
                else:
                    _, track_id, err, _ = rec
                    print(f"frame {frame_id} | track {track_id} {tag}: {err}")

        expired_ids = runner.cleanup_expired_tracks(
            frame_id=frame_id,
            max_missing_frames=max_missing_frames,
        )
        if expired_ids:
            print(f"frame {frame_id} | cleaned expired track ids: {expired_ids}")

        vis_pose = pose_results_for_vis if (args.show_skeleton and not is_rgb_mode(args.mode)) else None
        vis_frame = draw_tracks(
            frame,
            track_results,
            runner.action_results,
            pose_results=vis_pose,
            skeleton_score_thr=args.skeleton_score_thr,
        )
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
                f"cache_tracks={runner.cache_size}"
            )

        frame_id += 1

    cap.release()
    writer.release()

    print("finished")
    print("output:", output_video)
    print("\n===== Timing Summary =====")
    print_stats("YOLO+ByteTrack per frame", track_time_list)
    print_stats("Action recognition per call", action_time_list)
    print_stats("Action recognition per frame", action_time_per_frame)
    print_stats("Total pipeline per frame", total_frame_time_list)

    if total_frame_time_list:
        avg_total = statistics.mean(total_frame_time_list)
        print(f"Approx pipeline FPS = {1.0 / avg_total:.2f}")

    print(f"Total frames = {frame_id}")
    print(f"Total action calls = {runner.total_action_calls}")
    print(f"Total pose calls = {runner.total_pose_calls}")
    print(f"Total valid poses = {runner.total_pose_valid}")
    if runner.total_pose_calls > 0:
        print(f"Pose valid ratio = {100.0 * runner.total_pose_valid / runner.total_pose_calls:.2f}%")
    print(f"Remaining cached tracks = {runner.cache_size}")


if __name__ == "__main__":
    main()
