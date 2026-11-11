from dataclasses import dataclass

import lap
import numpy as np

from physical_ai.perception.detection import BoundingBox
from physical_ai.perception.tracking_utils import calculate_iou


@dataclass
class Match:
    track_index: int
    detection_index: int
    cost: float


@dataclass
class AssociationResult:
    matches: list[Match]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def build_iou_cost_matrix(
    track_boxes: list[BoundingBox],
    detection_boxes: list[BoundingBox],
) -> np.ndarray:
    cost_matrix = np.zeros(
        (
            len(track_boxes),
            len(detection_boxes),
        ),
        dtype=float,
    )

    for track_index, track_box in enumerate(track_boxes):
        for detection_index, detection_box in enumerate(detection_boxes):
            iou = calculate_iou(
                track_box,
                detection_box,
            )

            cost_matrix[
                track_index,
                detection_index,
            ] = 1.0 - iou

    return cost_matrix


def solve_assignment(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> AssociationResult:
    if cost_matrix.size == 0:
        num_tracks, num_detections = cost_matrix.shape

        return AssociationResult(
            matches=[],
            unmatched_tracks=list(range(num_tracks)),
            unmatched_detections=list(range(num_detections)),
        )

    _, track_to_detection, detection_to_track = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=max_cost,
    )

    matches: list[Match] = []
    unmatched_tracks: list[int] = []

    for track_index, detection_index in enumerate(track_to_detection):
        if detection_index == -1:
            unmatched_tracks.append(track_index)
            continue

        matches.append(
            Match(
                track_index=track_index,
                detection_index=int(detection_index),
                cost=float(
                    cost_matrix[
                        track_index,
                        detection_index,
                    ]
                ),
            )
        )

    unmatched_detections: list[int] = []

    for detection_index, track_index in enumerate(detection_to_track):
        if track_index == -1:
            unmatched_detections.append(detection_index)

    return AssociationResult(
        matches=matches,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
    )
