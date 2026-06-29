"""
VoxMM preprocessing: extract mouth ROI clips and WAV slices per on-screen segment.

Two face detection modes:
  --use-bbox-only: Use VoxMM bbox directly as face_boxes for FAN (fast, no RetinaFace)
  default:         Use RetinaFace to detect faces, select best match via IoU with VoxMM bbox
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess VoxMM videos into mouth ROI clips.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--facetrack-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--avec-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-bbox-only", action="store_true",
                        help="Skip RetinaFace; use VoxMM bbox directly as face input to FAN")
    parser.add_argument("--detector-threshold", type=float, default=0.3)
    parser.add_argument("--detector-model", default="mobilenet0.25",
                        choices=["mobilenet0.25", "resnet50"])
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    video_dir = Path(args.video_dir)
    wav_dir = Path(args.wav_dir)
    meta_dir = Path(args.metadata_dir)
    ft_dir = Path(args.facetrack_dir)
    output_dir = Path(args.output_dir)
    roi_dir = output_dir / "video"
    wav_out_dir = output_dir / "wav"

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.force:
            import shutil
            shutil.rmtree(output_dir)
        elif not args.skip_existing:
            raise FileExistsError(f"Output dir not empty: {output_dir}. Use --force or --skip-existing.")

    roi_dir.mkdir(parents=True, exist_ok=True)
    wav_out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve AVEC path before chdir
    avec_repo_path = Path(args.avec_repo).resolve()
    sys.path.insert(0, str(avec_repo_path))
    os.chdir(str(avec_repo_path))

    import torch
    import torchvision
    import torchvision.transforms.functional as TF
    import skimage.transform
    import av

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Face detection mode: {'bbox-only (no RetinaFace)' if args.use_bbox_only else f'RetinaFace ({args.detector_model})'}", flush=True)

    detector = None
    if not args.use_bbox_only:
        from ibug.face_detection import RetinaFacePredictor
        detector = RetinaFacePredictor(
            device=device, threshold=args.detector_threshold,
            model=RetinaFacePredictor.get_model(args.detector_model)
        )

    from ibug.face_alignment import FANPredictor
    fan = FANPredictor(device=device, model=None)

    mean_face = np.load(avec_repo_path / "media" / "20words_mean_face.npy")
    STABLE_PTS = [33, 36, 39, 42, 45]
    START_IDX, STOP_IDX = 48, 68
    CROP_H, CROP_W = 96, 96
    STD_SIZE = (256, 256)
    WINDOW_MARGIN = 12

    split_ids = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    if args.limit:
        split_ids = split_ids[:args.limit]

    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["utterance_id", "audio_path", "video_path", "reference", "language"]

    existing_ids: set[str] = set()
    if args.skip_existing and manifest_path.exists():
        with manifest_path.open() as f:
            for row in csv.DictReader(f):
                existing_ids.add(row["utterance_id"])

    write_header = not (args.skip_existing and manifest_path.exists())
    manifest_handle = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_written = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.time()

    for vid_idx, vid_id in enumerate(split_ids):
        video_path = video_dir / f"{vid_id}.mp4"
        wav_path = wav_dir / f"{vid_id}.wav"
        meta_path = meta_dir / f"{vid_id}.json"
        ft_path = ft_dir / f"{vid_id}.json"

        if not video_path.exists():
            print(f"[{vid_idx+1}/{len(split_ids)}] SKIP {vid_id}: no video", flush=True)
            continue
        if not wav_path.exists():
            print(f"[{vid_idx+1}/{len(split_ids)}] SKIP {vid_id}: no wav", flush=True)
            continue

        meta = json.loads(meta_path.read_text())
        ft = json.loads(ft_path.read_text())
        ft_map = {t["track_index"]: t for t in ft["face_tracks"]}
        ft_frame_w, ft_frame_h = ft["frame_size"]

        segments = [s for s in meta["segments"] if s.get("on-screen") and s.get("face_track")]
        print(f"[{vid_idx+1}/{len(split_ids)}] {vid_id}: {len(segments)} segments", flush=True)

        vid_written = 0
        vid_failed = 0

        for seg in segments:
            utterance_id = f"{vid_id}_{seg['segment_index']:05d}"

            if utterance_id in existing_ids:
                total_skipped += 1
                continue

            seg_start = seg["start"]
            seg_end = seg["end"]
            text = seg["text"].strip()
            if not text:
                continue

            ft_entry = seg["face_track"][0]
            track_idx = ft_entry["index"]
            ft_ts_start, ft_ts_end = ft_entry["timestamp"]

            if track_idx not in ft_map:
                vid_failed += 1
                continue

            track = ft_map[track_idx]
            track_bboxes = [
                bbox for ts, bbox in zip(track["time_stamp"], track["bbox"])
                if ft_ts_start <= ts <= ft_ts_end
            ]
            if not track_bboxes:
                track_bboxes = [track["bbox"][0]]

            avg_bbox_norm = np.mean(track_bboxes, axis=0)  # [x1,y1,x2,y2] normalized

            roi_path = roi_dir / f"{utterance_id}_mouth.mp4"
            wav_out_path = wav_out_dir / f"{utterance_id}.wav"

            try:
                t_extract = time.time()
                frames = _extract_frames(str(video_path), seg_start, seg_end, av)
                t_extract = time.time() - t_extract

                if len(frames) == 0:
                    vid_failed += 1
                    continue

                t_fan = time.time()
                mouth_roi = _process_frames(
                    frames, avg_bbox_norm, ft_frame_w, ft_frame_h,
                    detector, fan, mean_face,
                    STABLE_PTS, START_IDX, STOP_IDX,
                    CROP_H, CROP_W, STD_SIZE, WINDOW_MARGIN,
                    args.iou_threshold, args.use_bbox_only,
                    device, torch, skimage
                )
                t_fan = time.time() - t_fan

                if mouth_roi is None or len(mouth_roi) == 0:
                    vid_failed += 1
                    continue

                t_post = time.time()
                mouth_roi = _apply_preprocessing(mouth_roi, TF, torch)
                _write_video(str(roi_path), mouth_roi, fps=25, av=av)
                _slice_wav(str(wav_path), str(wav_out_path), seg_start, seg_end)
                t_post = time.time() - t_post
                print(f"    {utterance_id}: frames={len(frames)} extract={t_extract:.2f}s fan={t_fan:.2f}s post={t_post:.2f}s", flush=True)

                writer.writerow({
                    "utterance_id": utterance_id,
                    "audio_path": str(wav_out_path),
                    "video_path": str(roi_path),
                    "reference": text.upper(),
                    "language": "en",
                })
                manifest_handle.flush()
                vid_written += 1
                total_written += 1

            except Exception as exc:
                print(f"  ERROR {utterance_id}: {exc}", flush=True)
                vid_failed += 1
                total_failed += 1
                continue

        total_failed += vid_failed
        elapsed = time.time() - t0
        print(f"  Done {vid_id}: written={vid_written} failed={vid_failed} | total={total_written} elapsed={elapsed:.0f}s", flush=True)

    manifest_handle.close()
    print(f"\nFinished. Written={total_written} Skipped={total_skipped} Failed={total_failed}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def _extract_frames(video_path: str, start: float, end: float, av) -> list:
    frames = []
    with av.open(video_path) as container:
        container.seek(int(max(0, start - 0.5) * 1e6))
        for frame in container.decode(video=0):
            t = frame.time
            if t < start:
                continue
            if t > end + 0.1:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _bbox_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def _bbox_to_face_boxes(bbox_norm, frame_h, frame_w) -> np.ndarray:
    """Convert normalized VoxMM bbox to ibug face_boxes format (1, 15)."""
    x1 = bbox_norm[0] * frame_w
    y1 = bbox_norm[1] * frame_h
    x2 = bbox_norm[2] * frame_w
    y2 = bbox_norm[3] * frame_h
    # format: [x1, y1, x2, y2, score, kp1x, kp1y, ..., kp5x, kp5y]
    row = [x1, y1, x2, y2, 1.0] + [0.0] * 10
    return np.array([row], dtype=np.float32)


def _process_frames(
    frames, avg_bbox_norm, ft_frame_w, ft_frame_h,
    detector, fan, mean_face,
    stable_pts, start_idx, stop_idx,
    crop_h, crop_w, std_size, window_margin,
    iou_threshold, use_bbox_only,
    device, torch, skimage
):
    h, w = frames[0].shape[:2]
    # Pixel bbox in the downloaded video's resolution
    ref_bbox_pixel = [
        avg_bbox_norm[0] * w, avg_bbox_norm[1] * h,
        avg_bbox_norm[2] * w, avg_bbox_norm[3] * h,
    ]

    landmarks_list = []
    for frame in frames:
        if use_bbox_only:
            face_boxes = _bbox_to_face_boxes(avg_bbox_norm, h, w)
        else:
            detected = detector(frame, rgb=True)
            if len(detected) == 0:
                landmarks_list.append(None)
                continue
            # Select face with best IoU to VoxMM bbox
            best_iou, best_idx = -1, 0
            for i, face in enumerate(detected):
                iou = _bbox_iou(list(face[:4]), ref_bbox_pixel)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou < iou_threshold:
                # fallback to first detection
                face_boxes = detected[:1]
            else:
                face_boxes = detected[best_idx:best_idx+1]

        lms, _ = fan(frame, face_boxes, rgb=True)
        if len(lms) > 0:
            landmarks_list.append(lms[0])
        else:
            landmarks_list.append(None)

    landmarks_list = _interpolate_landmarks(landmarks_list)
    if landmarks_list is None:
        return None

    return _crop_mouth_patch(
        frames, landmarks_list, mean_face,
        stable_pts, start_idx, stop_idx,
        crop_h, crop_w, std_size, window_margin, skimage
    )


def _interpolate_landmarks(landmarks):
    valid = [i for i, l in enumerate(landmarks) if l is not None]
    if not valid:
        return None
    landmarks[:valid[0]] = [landmarks[valid[0]]] * valid[0]
    landmarks[valid[-1]+1:] = [landmarks[valid[-1]]] * (len(landmarks) - valid[-1] - 1)
    for i in range(1, len(valid)):
        a, b = valid[i-1], valid[i]
        if b - a > 1:
            for j in range(1, b - a):
                alpha = j / (b - a)
                landmarks[a + j] = landmarks[a] * (1 - alpha) + landmarks[b] * alpha
    return landmarks


def _estimate_similarity_matrix(src, dst):
    """Estimate similarity transform matrix (cv2-compatible 2x3) from src->dst point pairs."""
    import cv2
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    tform = cv2.estimateAffinePartial2D(src, dst)[0]  # 2x3 matrix
    return tform


def _apply_affine(frame, M, out_size):
    """Apply 2x3 affine matrix using cv2. out_size=(h,w)."""
    import cv2
    return cv2.warpAffine(frame, M, (out_size[1], out_size[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _transform_landmarks(lm, M):
    """Apply 2x3 affine matrix to Nx2 landmark array."""
    lm_h = np.hstack([lm, np.ones((len(lm), 1))])  # Nx3
    return (M @ lm_h.T).T  # Nx2


def _crop_mouth_patch(frames, landmarks, mean_face, stable_pts, start_idx, stop_idx,
                       crop_h, crop_w, std_size, window_margin, skimage):
    import cv2
    num_frames = len(frames)
    margin = min(num_frames, window_margin)
    q_frame = collections.deque()
    q_lm = collections.deque()
    sequence = []
    M = None

    for i, (frame, lm) in enumerate(zip(frames, landmarks)):
        q_frame.append(frame)
        q_lm.append(lm)

        if len(q_frame) == margin:
            smoothed = np.mean(q_lm, axis=0)
            cur_lm = q_lm.popleft()
            cur_frame = q_frame.popleft()
            M = _estimate_similarity_matrix(smoothed[stable_pts], mean_face[stable_pts])
            if M is None:
                continue
            warped = _apply_affine(cur_frame, M, std_size)
            trans_lm = _transform_landmarks(cur_lm, M)
            patch = _cut_patch(warped, trans_lm[start_idx:stop_idx], crop_h // 2, crop_w // 2)
            if patch is not None:
                sequence.append(patch)

        if i == num_frames - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                cur_lm = q_lm.popleft()
                if M is None:
                    break
                warped = _apply_affine(cur_frame, M, std_size)
                trans_lm = _transform_landmarks(cur_lm, M)
                patch = _cut_patch(warped, trans_lm[start_idx:stop_idx], crop_h // 2, crop_w // 2)
                if patch is not None:
                    sequence.append(patch)

    return np.array(sequence) if sequence else None


def _cut_patch(img, landmarks, height, width):
    cx, cy = np.mean(landmarks, axis=0)
    cy = float(np.clip(cy, height, img.shape[0] - height))
    cx = float(np.clip(cx, width, img.shape[1] - width))
    y1 = int(round(cy) - round(height))
    y2 = int(round(cy) + round(height))
    x1 = int(round(cx) - round(width))
    x2 = int(round(cx) + round(width))
    if y1 < 0 or y2 > img.shape[0] or x1 < 0 or x2 > img.shape[1]:
        return None
    return np.copy(img[y1:y2, x1:x2])


def _apply_preprocessing(frames_np, TF, torch):
    """Grayscale + normalize(-1,1) + CenterCrop(88,88). Input: (T,H,W,C) uint8."""
    result = []
    for frame in frames_np:
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # C,H,W
        t = TF.rgb_to_grayscale(t)       # 1,H,W
        t = (t - 0.5) / 0.5             # normalize to [-1,1]
        t = TF.center_crop(t, [88, 88]) # 1,88,88
        result.append(t.squeeze(0).numpy())
    return np.array(result)  # T,H,W


def _write_video(path: str, frames: np.ndarray, fps: int, av) -> None:
    """Write grayscale frames (T,H,W) float [-1,1] as MP4."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(path, mode='w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        for frame_gray in frames:
            frame_uint8 = ((frame_gray + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            rgb = np.stack([frame_uint8] * 3, axis=-1)
            frame_av = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame_av):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _slice_wav(src: str, dst: str, start: float, end: float) -> None:
    with wave.open(src, 'rb') as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        total = wf.getnframes()
        s = min(int(start * sr), total)
        e = min(int(end * sr), total)
        wf.setpos(s)
        data = wf.readframes(e - s)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(dst, 'wb') as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(data)


if __name__ == "__main__":
    main()
