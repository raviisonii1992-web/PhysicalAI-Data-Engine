from physical_ai.perception.detector import ObjectDetector


MODEL_PATH = "models/detection/yolo11n.pt"

IMAGE_PATH = "data/frames/test_drive/frame_001020_00034.03s.jpg"


detector = ObjectDetector(
    model_path=MODEL_PATH,
    device=0,
    confidence_threshold=0.25,
)

detections = detector.detect(IMAGE_PATH)

print()
print("=" * 60)
print("PhysicalAI - Detector Demo")
print("=" * 60)

print(f"Number of detections: {len(detections)}")

print()

for index, detection in enumerate(
    detections,
    start=1,
):
    print(f"Detection #{index}")

    print(f"  Class      : {detection.class_name}")

    print(f"  Confidence : {detection.confidence:.3f}")

    print(
        f"  Box        : "
        f"({detection.bbox.x1:.1f}, "
        f"{detection.bbox.y1:.1f}) "
        f"→ "
        f"({detection.bbox.x2:.1f}, "
        f"{detection.bbox.y2:.1f})"
    )

    print(f"  Width      : {detection.bbox.width:.1f}px")

    print(f"  Height     : {detection.bbox.height:.1f}px")

    print("-" * 60)
