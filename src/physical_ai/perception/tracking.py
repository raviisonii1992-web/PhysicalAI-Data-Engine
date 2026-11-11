from dataclasses import dataclass, field
from enum import Enum

from physical_ai.perception.detection import BoundingBox
from physical_ai.perception.kalman_filter import KalmanFilter2D


class TrackState(Enum):
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    state: TrackState = TrackState.NEW
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    motion_filter: KalmanFilter2D = field(init=False)

    def __post_init__(self) -> None:
        center_x, center_y = self.center

        self.motion_filter = KalmanFilter2D(
            x=center_x,
            y=center_y,
        )

        self.history.append(self.center)

    @property
    def center(self) -> tuple[float, float]:
        center_x = (self.bbox.x1 + self.bbox.x2) / 2.0
        center_y = (self.bbox.y1 + self.bbox.y2) / 2.0
        return center_x, center_y

    def activate(self) -> None:
        self.state = TrackState.ACTIVE

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def increment_age(self) -> None:
        self.age += 1
        self.time_since_update += 1

    def _bbox_from_center(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> BoundingBox:
        half_width = width / 2.0
        half_height = height / 2.0

        return BoundingBox(
            x1=center_x - half_width,
            y1=center_y - half_height,
            x2=center_x + half_width,
            y2=center_y + half_height,
        )

    def predict(self) -> BoundingBox:
        predicted_state = self.motion_filter.predict()

        predicted_x = float(predicted_state[0])
        predicted_y = float(predicted_state[1])

        return self._bbox_from_center(
            center_x=predicted_x,
            center_y=predicted_y,
            width=self.bbox.width,
            height=self.bbox.height,
        )

    def update(
        self,
        bbox: BoundingBox,
        confidence: float,
    ) -> None:
        measured_x = (bbox.x1 + bbox.x2) / 2.0
        measured_y = (bbox.y1 + bbox.y2) / 2.0

        updated_state = self.motion_filter.update(
            measured_x=measured_x,
            measured_y=measured_y,
        )

        filtered_x = float(updated_state[0])
        filtered_y = float(updated_state[1])

        self.bbox = self._bbox_from_center(
            center_x=filtered_x,
            center_y=filtered_y,
            width=bbox.width,
            height=bbox.height,
        )

        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.ACTIVE
        self.history.append(self.center)
