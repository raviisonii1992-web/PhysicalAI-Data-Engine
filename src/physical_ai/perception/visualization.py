import cv2

from physical_ai.perception.detection import Detection


def draw_detections(
    frame,
    detections: list[Detection],
):
    annotated_frame = frame.copy()

    for detection in detections:
        bbox = detection.bbox

        x1 = int(bbox.x1)
        y1 = int(bbox.y1)
        x2 = int(bbox.x2)
        y2 = int(bbox.y2)

        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated_frame
