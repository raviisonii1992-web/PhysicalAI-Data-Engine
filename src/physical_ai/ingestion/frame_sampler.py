from pathlib import Path

import cv2

from physical_ai.ingestion.video_reader import VideoReader


class FrameSampler:
    def __init__(self, sample_every: int = 30):
        if sample_every <= 0:
            raise ValueError("sample_every must be greater than 0")

        self.sample_every = sample_every

    def sample(self, video: VideoReader):
        for frame_index, timestamp, frame in video.frames():

            if frame_index % self.sample_every == 0:
                yield frame_index, timestamp, frame

    def save(
        self,
        video: VideoReader,
        output_dir: str,
    ) -> int:

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for frame_index, timestamp, frame in self.sample(video):

            filename = (
                f"frame_{frame_index:06d}_"
                f"{timestamp:08.2f}s.jpg"
            )

            file_path = output_path / filename

            success = cv2.imwrite(
                str(file_path),
                frame,
            )

            if not success:
                raise RuntimeError(
                    f"Failed to save frame: {file_path}"
                )

            saved_count += 1

        return saved_count