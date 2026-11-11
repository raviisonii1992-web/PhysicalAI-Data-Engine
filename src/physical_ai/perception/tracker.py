from physical_ai.perception.association import (
    build_iou_cost_matrix,
    solve_assignment,
)
from physical_ai.perception.detection import Detection
from physical_ai.perception.tracking import (
    TrackedObject,
    TrackState,
)


class ObjectTracker:
    def __init__(
        self,
        max_association_cost: float = 0.60,
        max_lost_frames: int = 30,
    ) -> None:
        self.max_association_cost = max_association_cost
        self.max_lost_frames = max_lost_frames
        self.tracks: list[TrackedObject] = []
        self.next_track_id = 1

    def _create_track(
        self,
        detection: Detection,
    ) -> TrackedObject:
        track = TrackedObject(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
        )

        self.next_track_id += 1
        return track

    def _association_candidates(
        self,
    ) -> list[TrackedObject]:
        return [
            track
            for track in self.tracks
            if track.state != TrackState.REMOVED
        ]

    def update(
        self,
        detections: list[Detection],
    ) -> list[TrackedObject]:
        candidate_tracks = self._association_candidates()

        for track in candidate_tracks:
            track.increment_age()

        predicted_boxes = [
            track.predict()
            for track in candidate_tracks
        ]

        detection_boxes = [
            detection.bbox
            for detection in detections
        ]

        cost_matrix = build_iou_cost_matrix(
            track_boxes=predicted_boxes,
            detection_boxes=detection_boxes,
        )

        association = solve_assignment(
            cost_matrix=cost_matrix,
            max_cost=self.max_association_cost,
        )

        for match in association.matches:
            track = candidate_tracks[match.track_index]
            detection = detections[match.detection_index]

            track.update(
                bbox=detection.bbox,
                confidence=detection.confidence,
            )

        for track_index in association.unmatched_tracks:
            candidate_tracks[track_index].mark_lost()

        for track in candidate_tracks:
            if (
                track.state == TrackState.LOST
                and track.time_since_update > self.max_lost_frames
            ):
                track.mark_removed()

        for detection_index in association.unmatched_detections:
            detection = detections[detection_index]
            self.tracks.append(
                self._create_track(detection)
            )

        return [
            track
            for track in self.tracks
            if track.state != TrackState.REMOVED
        ]
