from physical_ai.ingestion.frame_sampler import FrameSampler
from physical_ai.ingestion.video_reader import VideoReader


VIDEO_PATH = "data/raw/test_drive.mp4"
OUTPUT_DIR = "data/frames/sampler_test"


with VideoReader(VIDEO_PATH) as video:

    sampler = FrameSampler(
        sample_every=30
    )

    saved_count = sampler.save(
        video,
        OUTPUT_DIR,
    )

    print(f"Frames saved: {saved_count}")