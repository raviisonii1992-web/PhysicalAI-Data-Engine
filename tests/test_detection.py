from physical_ai.perception.detection import (
    BoundingBox,
    Detection,
)


def test_bounding_box_dimensions():

    bbox = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=220,
    )

    assert bbox.width == 100
    assert bbox.height == 200


def test_detection_model():

    bbox = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=220,
    )

    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
    )

    assert detection.class_id == 2
    assert detection.class_name == "car"
    assert detection.confidence == 0.9
    assert detection.bbox.width == 100
