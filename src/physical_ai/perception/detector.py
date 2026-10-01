from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO

from physical_ai.perception.detection import (
    BoundingBox,
    Detection,
)


ImageInput = Union[str, np.ndarray]


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        device: int | str = 0,
        confidence_threshold: float = 0.25,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.device = device
        self.confidence_threshold = confidence_threshold

        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image: ImageInput,
    ) -> list[Detection]:

        results = self.model(
            image,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )

        result = results[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())

            class_name = result.names[class_id]

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
            )

            detections.append(detection)

        return detections
