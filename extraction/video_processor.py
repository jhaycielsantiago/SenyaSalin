"""
Batch video-to-landmark processor.

Walks data/raw/<LABEL>/ directories, extracts landmarks from each video,
and writes landmark JSON files to data/processed/<LABEL>/.

Usage:
    python -m extraction.video_processor --raw-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from .landmark_extractor import GestureLandmarkRecord, LandmarkExtractor

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def process_video(
    video_path: Path,
    label: str,
    signer_id: str,
    dataset_version: str,
    extractor: LandmarkExtractor,
    max_frames: Optional[int] = None,
) -> GestureLandmarkRecord:
    """
    Extract landmarks from a single video file.

    Args:
        video_path: Path to the source video.
        label: Gesture class label (e.g. "HELP").
        signer_id: Signer identifier for dataset tracking.
        dataset_version: Dataset version string.
        extractor: A shared LandmarkExtractor instance.
        max_frames: If set, stops after this many frames (useful for long clips).

    Returns:
        GestureLandmarkRecord with all extracted frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    session_id = f"{label}_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    record = GestureLandmarkRecord(
        label=label,
        signer_id=signer_id,
        session_id=session_id,
        dataset_version=dataset_version,
        fps=fps,
        source="video_file",
    )

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_index >= max_frames:
                break

            timestamp_ms = (frame_index / fps) * 1000.0
            lm_frame = extractor.extract_frame(frame, frame_index, timestamp_ms)
            record.frames.append(lm_frame)
            frame_index += 1
    finally:
        cap.release()

    return record


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    signer_id: str = "unknown",
    dataset_version: str = "0.1",
    max_frames: Optional[int] = None,
) -> dict[str, int]:
    """
    Process all videos in raw_dir/<LABEL>/ subdirectories.

    Returns a summary dict: {label: num_processed}.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    summary: dict[str, int] = {}

    with LandmarkExtractor() as extractor:
        label_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

        for label_dir in tqdm(label_dirs, desc="Processing labels"):
            label = label_dir.name
            videos = [
                v for v in label_dir.iterdir()
                if v.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not videos:
                print(f"  [SKIP] {label} — no supported video files found")
                continue

            label_output = output_dir / label
            label_output.mkdir(parents=True, exist_ok=True)

            count = 0
            for video_path in tqdm(videos, desc=f"  {label}", leave=False):
                try:
                    record = process_video(
                        video_path=video_path,
                        label=label,
                        signer_id=signer_id,
                        dataset_version=dataset_version,
                        extractor=extractor,
                        max_frames=max_frames,
                    )
                    out_file = label_output / f"{video_path.stem}.json"
                    record.save(out_file)
                    count += 1
                except Exception as e:
                    print(f"  [ERROR] {video_path.name}: {e}", file=sys.stderr)

            summary[label] = count
            print(f"  {label}: {count} videos processed")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from raw gesture videos."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Input raw video directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output landmark JSON directory")
    parser.add_argument("--signer-id", default="unknown", help="Signer identifier")
    parser.add_argument("--dataset-version", default="0.1", help="Dataset version tag")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    args = parser.parse_args()

    summary = process_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        signer_id=args.signer_id,
        dataset_version=args.dataset_version,
        max_frames=args.max_frames,
    )

    print("\nExtraction complete:")
    for label, count in summary.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
