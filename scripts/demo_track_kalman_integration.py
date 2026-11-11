from physical_ai.perception.detection import BoundingBox
from physical_ai.perception.tracking import TrackedObject


def main():
    track = TrackedObject(
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.91,
        bbox=BoundingBox(
            x1=100,
            y1=200,
            x2=220,
            y2=320,
        ),
    )

    track.activate()

    print("Initial bbox:")
    print(track.bbox)
    print()

    print("Initial Kalman state:")
    print(track.motion_filter.state)
    print()

    predicted_bbox = track.predict()

    print("Predicted bbox:")
    print(predicted_bbox)
    print()

    detection_bbox = BoundingBox(
        x1=112,
        y1=197,
        x2=232,
        y2=317,
    )

    print("YOLO detection:")
    print(detection_bbox)
    print()

    track.update(
        bbox=detection_bbox,
        confidence=0.86,
    )

    print("Filtered bbox:")
    print(track.bbox)
    print()

    print("Updated Kalman state:")
    print(track.motion_filter.state)
    print()

    print("History:")
    print(track.history)


if __name__ == "__main__":
    main()
