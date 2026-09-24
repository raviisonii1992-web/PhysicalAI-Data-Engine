import argparse
from pathlib import Path

import cv2


def read_frames(video_path: str, max_frames: int = 10) -> None:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_index = 0

    while frame_index < max_frames:
        success, frame = capture.read()

        if not success:
            print("No more frames available.")
            break

        timestamp_seconds = frame_index / fps if fps > 0 else 0.0

        height, width = frame.shape[:2]

        print(
            f"Frame: {frame_index:04d} | "
            f"Time: {timestamp_seconds:.3f}s | "
            f"Shape: {width}x{height}"
        )

        frame_index += 1

    capture.release()


def main():
    parser = argparse.ArgumentParser(
        description="Read video frames and print timestamps."
    )

    parser.add_argument(
        "video",
        help="Path to input video",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of frames to read",
    )

    args = parser.parse_args()

    read_frames(
        video_path=args.video,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()