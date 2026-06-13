from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop raw manifest videos into AV-HuBERT/MMS-LLaMA mouth ROI clips "
            "using the preprocessing scripts vendored in external/MMS-LLaMA."
        )
    )
    parser.add_argument("manifest", help="Input manifest.csv with a video_path column.")
    parser.add_argument("output_dir", help="Directory for staged raw videos, landmarks, ROI videos, and manifest.csv.")
    parser.add_argument("--mms-repo", default="external/MMS-LLaMA", help="Path to the MMS-LLaMA checkout.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable.")
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable for MMS-LLaMA/AV-HuBERT preprocessing scripts. Defaults to <mms-repo>/.venv/bin/python.",
    )
    parser.add_argument("--cnn-detector", required=True, help="Path to dlib mmod_human_face_detector.dat.")
    parser.add_argument("--face-predictor", required=True, help="Path to dlib shape_predictor_68_face_landmarks.dat.")
    parser.add_argument("--mean-face", required=True, help="Path to 20words_mean_face.npy.")
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--id-column", default="utterance_id")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--copy", action="store_true", help="Copy videos into the staging directory instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing ROI videos when present.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        preprocess_manifest(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            mms_repo=Path(args.mms_repo),
            ffmpeg=args.ffmpeg,
            python_executable=Path(args.python_executable) if args.python_executable else None,
            cnn_detector=Path(args.cnn_detector),
            face_predictor=Path(args.face_predictor),
            mean_face=Path(args.mean_face),
            video_column=args.video_column,
            audio_column=args.audio_column,
            id_column=args.id_column,
            reference_column=args.reference_column,
            language_column=args.language_column,
            copy=args.copy,
            force=args.force,
            skip_existing=args.skip_existing,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


def preprocess_manifest(
    manifest_path: Path,
    output_dir: Path,
    mms_repo: Path,
    ffmpeg: str,
    python_executable: Path | None,
    cnn_detector: Path,
    face_predictor: Path,
    mean_face: Path,
    video_column: str,
    audio_column: str,
    id_column: str,
    reference_column: str,
    language_column: str,
    copy: bool,
    force: bool,
    skip_existing: bool,
) -> None:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    mms_repo = mms_repo.resolve()
    output_dir = output_dir.resolve()
    script_python = python_executable or (mms_repo / ".venv" / "bin" / "python")
    if not script_python.exists():
        raise FileNotFoundError(f"MMS-LLaMA Python executable not found: {script_python}")

    prep_dir = mms_repo / "avhubert" / "preparation"
    detect_script = prep_dir / "detect_landmark.py"
    align_script = prep_dir / "align_mouth.py"
    for required in [manifest_path, detect_script, align_script, cnn_detector, face_predictor, mean_face]:
        if not required.exists():
            raise FileNotFoundError(required)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force and not skip_existing:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force or --skip-existing.")
        if force:
            shutil.rmtree(output_dir)

    raw_dir = output_dir / "raw_video"
    landmark_dir = output_dir / "landmark"
    roi_dir = output_dir / "video"
    raw_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    staged_rows = []
    file_list = output_dir / "file.list"
    with file_list.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            utterance_id = row.get(id_column) or f"utt-{index:05d}"
            video_value = row.get(video_column)
            audio_value = row.get(audio_column)
            if not video_value:
                raise ValueError(f"Row {utterance_id} has no {video_column!r} value.")
            if not audio_value:
                raise ValueError(f"Row {utterance_id} has no {audio_column!r} value.")
            source_video = _resolve_path(video_value, manifest_dir)
            source_audio = _resolve_path(audio_value, manifest_dir)
            staged_video = raw_dir / f"{utterance_id}.mp4"
            if not staged_video.exists():
                _stage_file(source_video, staged_video, copy=copy)
            handle.write(utterance_id + "\n")
            staged_rows.append((row, utterance_id, source_audio))

    if not skip_existing or not all((roi_dir / f"{utterance_id}.mp4").exists() for _, utterance_id, _ in staged_rows):
        _run(
            [
                str(script_python),
                str(detect_script),
                "--root",
                str(raw_dir),
                "--landmark",
                str(landmark_dir),
                "--manifest",
                str(file_list),
                "--cnn_detector",
                str(cnn_detector.resolve()),
                "--face_predictor",
                str(face_predictor.resolve()),
                "--ffmpeg",
                _which_or_value(ffmpeg),
                "--rank",
                "0",
                "--nshard",
                "1",
            ],
            cwd=mms_repo,
        )
        _run(
            [
                str(script_python),
                str(align_script),
                "--video-direc",
                str(raw_dir),
                "--landmark-direc",
                str(landmark_dir),
                "--filename-path",
                str(file_list),
                "--save-direc",
                str(roi_dir),
                "--mean-face",
                str(mean_face.resolve()),
                "--ffmpeg",
                _which_or_value(ffmpeg),
                "--rank",
                "0",
                "--nshard",
                "1",
            ],
            cwd=mms_repo,
        )

    output_manifest = output_dir / "manifest.csv"
    fieldnames = _fieldnames(rows, [id_column, audio_column, video_column, reference_column, language_column])
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, utterance_id, source_audio in staged_rows:
            roi_video = roi_dir / f"{utterance_id}.mp4"
            if not roi_video.exists():
                raise RuntimeError(f"Mouth ROI video was not produced: {roi_video}")
            out_row = dict(row)
            out_row[id_column] = utterance_id
            out_row[audio_column] = os.path.relpath(source_audio, output_dir)
            out_row[video_column] = os.path.relpath(roi_video, output_dir)
            writer.writerow(out_row)

    print(f"Wrote AV-HuBERT mouth ROI videos: {roi_dir}")
    print(f"Wrote preprocessed manifest: {output_manifest}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    names = list(rows[0].keys())
    for name in preferred:
        if name not in names:
            names.append(name)
    return names


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def _stage_file(source: Path, target: Path, copy: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _which_or_value(command: str) -> str:
    resolved = shutil.which(command)
    return resolved or command


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


if __name__ == "__main__":
    main()
