from physical_ai.perception.detection import BoundingBox, Detection


bbox = BoundingBox(
    x1=133.5,
    y1=50.3,
    x2=243.5,
    y2=108.7,
)

detection = Detection(
    class_id=2,
    class_name="car",
    confidence=0.89,
    bbox=bbox,
)

print(detection)

print(f"Class: {detection.class_name}")
print(f"Confidence: {detection.confidence}")
print(f"Width: {detection.bbox.width}")
print(f"Height: {detection.bbox.height}")
