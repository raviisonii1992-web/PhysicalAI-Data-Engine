import time

from physical_ai.ingestion.video_reader import VideoReader
from physical_ai.perception.detector import ObjectDetector


VIDEO_PATH = "data/raw/test_drive.mp4"
MODEL_PATH = "models/detection/yolo11n.pt"

MAX_FRAMES = 100


def main():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=0,
        confidence_threshold=0.25,
    )

    processed_frames = 0

    start_time = time.perf_counter()

    with VideoReader(VIDEO_PATH) as video:
        for frame_index, timestamp, frame in video.frames():
            detector.detect(frame)

            processed_frames += 1

            if processed_frames >= MAX_FRAMES:
                break

    elapsed_time = time.perf_counter() - start_time

    fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

    average_ms = elapsed_time / processed_frames * 1000

    print()
    print("=" * 60)
    print("PhysicalAI - Detector Benchmark")
    print("=" * 60)

    print(f"Frames processed : {processed_frames}")

    print(f"Total time       : {elapsed_time:.2f}s")

    print(f"Average latency  : {average_ms:.2f} ms/frame")

    print(f"Pipeline FPS     : {fps:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
