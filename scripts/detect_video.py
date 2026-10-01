from pathlib import Path

import cv2

from physical_ai.ingestion.video_reader import VideoReader
from physical_ai.perception.detector import ObjectDetector
from physical_ai.perception.visualization import draw_detections


VIDEO_PATH = "data/raw/complex_traffic.mp4"

MODEL_PATH = "models/detection/yolo11n.pt"

OUTPUT_PATH = "outputs/videos/complex_traffic_detected.mp4"


def main():

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=0,
        confidence_threshold=0.25,
    )

    with VideoReader(VIDEO_PATH) as video:
        codec = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            codec,
            video.fps,
            (video.width, video.height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        print()
        print("=" * 60)
        print("PhysicalAI - Video Detection")
        print("=" * 60)

        print(f"Input video  : {VIDEO_PATH}")

        print(f"Output video : {OUTPUT_PATH}")

        print(f"Resolution   : {video.width}x{video.height}")

        print(f"FPS          : {video.fps:.2f}")

        print(f"Frames       : {video.frame_count}")

        print()

        try:
            for frame_index, timestamp, frame in video.frames():
                detections = detector.detect(frame)

                annotated_frame = draw_detections(
                    frame,
                    detections,
                )

                writer.write(annotated_frame)

                if frame_index % 30 == 0:
                    print(
                        f"Frame {frame_index:04d} | "
                        f"Time {timestamp:6.2f}s | "
                        f"Detections {len(detections)}"
                    )

        finally:
            writer.release()

    print()
    print("=" * 60)
    print("Video processing completed")
    print(f"Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
