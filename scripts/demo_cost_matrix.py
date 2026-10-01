from physical_ai.perception.detection import BoundingBox
from physical_ai.perception.tracking_utils import calculate_iou


def main():

    tracks = [
        BoundingBox(100, 100, 200, 200),
        BoundingBox(300, 100, 400, 200),
    ]

    detections = [
        BoundingBox(110, 105, 205, 205),
        BoundingBox(295, 95, 405, 205),
    ]

    print("Cost matrix")
    print()

    for track_index, track_box in enumerate(tracks):
        row = []

        for detection_index, detection_box in enumerate(detections):
            iou = calculate_iou(
                track_box,
                detection_box,
            )

            cost = 1.0 - iou

            row.append(cost)

        print(f"Track {track_index}: {row}")


if __name__ == "__main__":
    main()
