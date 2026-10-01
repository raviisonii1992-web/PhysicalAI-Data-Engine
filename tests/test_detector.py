from physical_ai.perception.detector import ObjectDetector


MODEL_PATH = "models/detection/yolo11n.pt"

IMAGE_PATH = "data/frames/test_drive/frame_001020_00034.03s.jpg"


def test_detector_returns_detections():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=0,
        confidence_threshold=0.25,
    )

    detections = detector.detect(IMAGE_PATH)

    assert isinstance(detections, list)
    assert len(detections) > 0


def test_detection_fields():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=0,
        confidence_threshold=0.25,
    )

    detections = detector.detect(IMAGE_PATH)

    detection = detections[0]

    assert detection.class_id >= 0
    assert detection.class_name != ""
    assert 0 <= detection.confidence <= 1
    assert detection.bbox.width > 0
    assert detection.bbox.height > 0
