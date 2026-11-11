from physical_ai.perception.association import build_iou_cost_matrix
from physical_ai.perception.detection import BoundingBox


def main():
    predicted_tracks = [
        BoundingBox(
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        ),
        BoundingBox(
            x1=300,
            y1=100,
            x2=400,
            y2=200,
        ),
    ]

    detections = [
        BoundingBox(
            x1=110,
            y1=105,
            x2=205,
            y2=205,
        ),
        BoundingBox(
            x1=295,
            y1=95,
            x2=405,
            y2=205,
        ),
    ]

    cost_matrix = build_iou_cost_matrix(
        track_boxes=predicted_tracks,
        detection_boxes=detections,
    )

    print("Cost matrix:")
    print(cost_matrix)


if __name__ == "__main__":
    main()
