from physical_ai.perception.detection import (
    BoundingBox,
    Detection,
)
from physical_ai.perception.tracker import ObjectTracker


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.90,
) -> Detection:
    return Detection(
        class_id=2,
        class_name="car",
        confidence=confidence,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )


def print_tracks(
    frame_number: int,
    tracks,
) -> None:
    print()
    print(f"Frame {frame_number}")
    print("-" * 40)

    for track in tracks:
        print(
            f"Track {track.track_id} | "
            f"state={track.state.value} | "
            f"bbox={track.bbox} | "
            f"age={track.age} | "
            f"hits={track.hits} | "
            f"time_since_update={track.time_since_update}"
        )


def main():
    tracker = ObjectTracker(
        max_association_cost=0.60,
        max_lost_frames=2,
    )

    frame_1 = [
        make_detection(100, 100, 200, 200),
        make_detection(300, 100, 400, 200),
    ]

    tracks = tracker.update(frame_1)
    print_tracks(1, tracks)

    frame_2 = [
        make_detection(110, 102, 210, 202),
        make_detection(290, 103, 390, 203),
    ]

    tracks = tracker.update(frame_2)
    print_tracks(2, tracks)

    frame_3 = [
        make_detection(120, 104, 220, 204),
    ]

    tracks = tracker.update(frame_3)
    print_tracks(3, tracks)


if __name__ == "__main__":
    main()
