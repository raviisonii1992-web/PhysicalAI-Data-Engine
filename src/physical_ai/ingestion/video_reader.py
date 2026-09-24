from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        self.frame_count = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0

        return self.frame_count / self.fps

    def metadata(self) -> dict:
        return {
            "file": self.video_path.name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }

    def frames(self):
        frame_index = 0

        while True:
            success, frame = self.capture.read()

            if not success:
                break

            timestamp = (
                frame_index / self.fps
                if self.fps > 0
                else 0.0
            )

            yield frame_index, timestamp, frame

            frame_index += 1

    def release(self):
        self.capture.release()
    

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()