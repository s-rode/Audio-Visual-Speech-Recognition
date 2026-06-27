from __future__ import annotations

import argparse
import csv
import importlib
import os
from pathlib import Path
import shutil
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AVEC-style mouth ROI preprocessing for a manifest. This script is "
            "intended to run inside the AVEC Python environment."
        )
    )
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--repo-path", default="external/AVEC")
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--id-column", default="utterance_id")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess_manifest(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        repo_path=Path(args.repo_path),
        video_column=args.video_column,
        audio_column=args.audio_column,
        id_column=args.id_column,
        reference_column=args.reference_column,
        language_column=args.language_column,
        force=args.force,
        skip_existing=args.skip_existing,
    )


def preprocess_manifest(
    manifest_path: Path,
    output_dir: Path,
    repo_path: Path,
    video_column: str,
    audio_column: str,
    id_column: str,
    reference_column: str,
    language_column: str,
    force: bool,
    skip_existing: bool,
) -> None:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    output_dir = output_dir.resolve()
    roi_dir = output_dir / "video"
    repo_path = repo_path.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        if force:
            shutil.rmtree(output_dir)
        elif not skip_existing:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force or --skip-existing.")

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    modules = _load_avec_modules(repo_path)
    cropper = _build_cropper(modules["nnet"], repo_path)

    output_manifest = output_dir / "manifest.csv"
    fieldnames = _fieldnames(rows, [id_column, audio_column, video_column, reference_column, language_column, "video_preprocessing"])
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            utterance_id = row.get(id_column) or f"utt-{index:05d}"
            source_video = _resolve_required(row, video_column, manifest_dir, utterance_id)
            source_audio = _resolve_required(row, audio_column, manifest_dir, utterance_id)
            roi_video = roi_dir / f"{utterance_id}_mouth.mp4"
            if not roi_video.exists() or not skip_existing:
                _preprocess_video(
                    source_video=source_video,
                    output_video=roi_video,
                    cropper=cropper,
                    torch=modules["torch"],
                    torchvision=modules["torchvision"],
                )

            out_row = dict(row)
            out_row[id_column] = utterance_id
            out_row[audio_column] = os.path.relpath(source_audio, output_dir)
            out_row[video_column] = os.path.relpath(roi_video, output_dir)
            out_row["video_preprocessing"] = "avec_mouth_roi"
            writer.writerow(out_row)

    print(f"Wrote AVEC mouth ROI videos: {roi_dir}")
    print(f"Wrote preprocessed manifest: {output_manifest}")


def _load_avec_modules(repo_path: Path) -> dict[str, object]:
    if not repo_path.exists():
        raise FileNotFoundError(f"AVEC repo not found: {repo_path}")
    sys.path.insert(0, str(repo_path))
    previous_cwd = Path.cwd()
    os.chdir(repo_path)
    try:
        return {
            "nnet": importlib.import_module("nnet"),
            "torch": importlib.import_module("torch"),
            "torchvision": importlib.import_module("torchvision"),
            "np": importlib.import_module("numpy"),
            "av": importlib.import_module("av"),
        }
    finally:
        os.chdir(previous_cwd)


def _build_cropper(nnet, repo_path: Path):
    mean_face = repo_path / "media" / "20words_mean_face.npy"
    cropper = nnet.LipDetectCrop(mean_face_landmarks_path=str(mean_face))
    if not hasattr(cropper, "face_detector") or not hasattr(cropper, "landmark_detector"):
        raise RuntimeError(
            "AVEC LipDetectCrop could not initialize face/landmark detectors. "
            "Install ibug.face_detection and ibug.face_alignment in the AVEC environment."
        )
    return cropper


def _preprocess_video(source_video: Path, output_video: Path, cropper, torch, torchvision) -> None:
    video, fps = _read_video(source_video, torch)
    landmarks = cropper.detect_landmarks(video)
    landmarks = cropper.landmarks_interpolate(landmarks)
    if not landmarks:
        roi = torchvision.transforms.functional.resize(
            video.permute(3, 0, 1, 2),
            size=(cropper.crop_height, cropper.crop_width),
        ).permute(1, 2, 3, 0)
        roi = roi.numpy()
    else:
        roi = cropper.crop_patch(video.numpy(), landmarks)
    if roi is None or len(roi) == 0:
        raise RuntimeError(f"Failed to produce mouth ROI video for {source_video}")
    _write_video(output_video, roi, fps=fps)


def _read_video(path: Path, torch):
    np = importlib.import_module("numpy")
    av = importlib.import_module("av")
    frames = []
    fps = 25.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return torch.from_numpy(np.stack(frames, axis=0)), fps


def _write_video(path: Path, frames, fps: float) -> None:
    av = importlib.import_module("av")
    np = importlib.import_module("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=max(1, round(fps)))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(frame_array, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    names = list(rows[0].keys())
    for name in preferred:
        if name not in names:
            names.append(name)
    return names


def _resolve_required(row: dict[str, str], column: str, manifest_dir: Path, utterance_id: str) -> Path:
    value = row.get(column)
    if not value:
        raise ValueError(f"Row {utterance_id} has no {column!r} value.")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


if __name__ == "__main__":
    main()
