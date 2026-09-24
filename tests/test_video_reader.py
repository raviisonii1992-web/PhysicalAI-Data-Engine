from physical_ai.ingestion.video_reader import VideoReader


VIDEO_PATH = "data/raw/test_drive.mp4"


def test_video_metadata():
    with VideoReader(VIDEO_PATH) as video:
        metadata = video.metadata()

        assert metadata["file"] == "test_drive.mp4"
        assert metadata["fps"] > 0
        assert metadata["frame_count"] > 0
        assert metadata["width"] > 0
        assert metadata["height"] > 0
        assert metadata["duration"] > 0


def test_video_first_frame():
    with VideoReader(VIDEO_PATH) as video:
        frame_index, timestamp, frame = next(video.frames())

        assert frame_index == 0
        assert timestamp == 0.0
        assert frame is not None
        assert frame.shape[0] == video.height
        assert frame.shape[1] == video.width