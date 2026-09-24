import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_every: int = 30,
) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % sample_every == 0:
            timestamp_seconds = frame_index / fps if fps > 0 else 0.0

            output_filename = (
                f"frame_{frame_index:06d}_"
                f"{timestamp_seconds:08.2f}s.jpg"
            )

            output_path = output_dir / output_filename

            cv2.imwrite(str(output_path), frame)

            print(
                f"Saved: {output_filename}"
            )

            saved_count += 1

        frame_index += 1

    capture.release()

    print()
    print("=" * 50)
    print(f"Frames processed : {frame_index}")
    print(f"Frames saved     : {saved_count}")
    print(f"Sample interval  : every {sample_every} frames")
    print(f"Output directory : {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Extract sampled frames from a video."
    )

    parser.add_argument(
        "video",
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        default="data/frames",
        help="Directory where extracted frames are saved",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="Save one frame every N frames",
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()