import numpy as np
import lap


def main():

    cost_matrix = np.array(
        [
            [0.20, 0.90],
            [0.85, 0.25],
            [0.95, 0.80],
        ]
    )

    total_cost, track_to_detection, detection_to_track = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=0.60,
    )

    print("Cost matrix:")
    print(cost_matrix)

    print()
    print(f"Total cost: {total_cost:.3f}")

    print()
    print("Track → Detection")

    for track_index, detection_index in enumerate(track_to_detection):
        print(f"Track {track_index} → Detection {detection_index}")

    print()
    print("Detection → Track")

    for detection_index, track_index in enumerate(detection_to_track):
        print(f"Detection {detection_index} → Track {track_index}")


if __name__ == "__main__":
    main()
