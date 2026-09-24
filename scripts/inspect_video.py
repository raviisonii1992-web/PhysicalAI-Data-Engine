import argparse
from pathlib import Path

import cv2


def inspect_video(video_path: str) -> None:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = frame_count / fps if fps > 0 else 0.0

    print("=" * 50)
    print("PhysicalAI - Video Inspection")
    print("=" * 50)
    print(f"File         : {path.name}")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps:.2f}")
    print(f"Frame Count  : {frame_count}")
    print(f"Duration     : {duration_seconds:.2f} seconds")
    print("=" * 50)

    capture.release()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect video metadata."
    )

    parser.add_argument(
        "video",
        help="Path to the input video file",
    )

    args = parser.parse_args()

    inspect_video(args.video)


if __name__ == "__main__":
    main()