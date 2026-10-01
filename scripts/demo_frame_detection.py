from physical_ai.ingestion.video_reader import VideoReader
from physical_ai.perception.detector import ObjectDetector


VIDEO_PATH = "data/raw/test_drive.mp4"

MODEL_PATH = "models/detection/yolo11n.pt"


detector = ObjectDetector(
    model_path=MODEL_PATH,
    device=0,
    confidence_threshold=0.25,
)


with VideoReader(VIDEO_PATH) as video:
    for frame_index, timestamp, frame in video.frames():
        detections = detector.detect(frame)

        print()
        print(f"Frame: {frame_index}")

        print(f"Timestamp: {timestamp:.3f}s")

        print(f"Detections: {len(detections)}")

        for detection in detections:
            print(f"  {detection.class_name:<12} {detection.confidence:.3f}")

        if frame_index == 4:
            break
