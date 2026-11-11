from physical_ai.perception.association import (
    build_iou_cost_matrix,
    solve_assignment,
)
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
    print()

    result = solve_assignment(
        cost_matrix=cost_matrix,
        max_cost=0.60,
    )

    print("Matches:")
    for match in result.matches:
        print(
            f"Track {match.track_index} "
            f"-> Detection {match.detection_index} "
            f"cost={match.cost:.3f}"
        )

    print()
    print("Unmatched tracks:", result.unmatched_tracks)
    print("Unmatched detections:", result.unmatched_detections)


if __name__ == "__main__":
    main()
