from physical_ai.ingestion.frame_sampler import FrameSampler
from physical_ai.ingestion.video_reader import VideoReader


VIDEO_PATH = "data/raw/test_drive.mp4"


def test_frame_sampler_count():
    with VideoReader(VIDEO_PATH) as video:
        sampler = FrameSampler(sample_every=30)

        sampled_frames = list(sampler.sample(video))

        assert len(sampled_frames) == 93


def test_frame_sampler_first_frame():
    with VideoReader(VIDEO_PATH) as video:
        sampler = FrameSampler(sample_every=30)

        frame_index, timestamp, frame = next(sampler.sample(video))

        assert frame_index == 0
        assert timestamp == 0.0
        assert frame is not None