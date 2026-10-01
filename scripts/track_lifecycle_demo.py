from collections import defaultdict

from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"
VIDEO_PATH = "data/raw/test_drive.mp4"


def main():
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        device=0,
        tracker="botsort.yaml",
        conf=0.25,
        persist=True,
        stream=True,
        verbose=False,
    )

    last_seen = {}
    seen_frames = defaultdict(list)

    for frame_index, result in enumerate(results):
        current_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()

            for track_id in track_ids:
                current_ids.add(track_id)

                if track_id in last_seen:
                    gap = frame_index - last_seen[track_id]

                    if gap > 1:
                        print(
                            f"ID {track_id} reappeared after missing {gap - 1} frame(s)"
                        )

                last_seen[track_id] = frame_index
                seen_frames[track_id].append(frame_index)

        if frame_index >= 120:
            break

    print()
    print("Track summary")

    for track_id, frames in sorted(seen_frames.items()):
        print(
            f"ID {track_id:<3} "
            f"first={frames[0]:<4} "
            f"last={frames[-1]:<4} "
            f"observed={len(frames)}"
        )


if __name__ == "__main__":
    main()
