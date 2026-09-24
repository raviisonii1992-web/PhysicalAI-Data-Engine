from physical_ai.ingestion.video_reader import VideoReader


with VideoReader("data/raw/test_drive.mp4") as video:

    print(video.metadata())

    for frame_index, timestamp, frame in video.frames():

        print(
            f"Frame {frame_index} | "
            f"Time {timestamp:.3f}s | "
            f"Shape {frame.shape}"
        )

        if frame_index == 4:
            break