from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"
VIDEO_PATH = "data/raw/test_drive.mp4"

TRACK_ID_TO_WATCH = 1


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

    previous_center = None

    for frame_index, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue

        track_ids = result.boxes.id.int().cpu().tolist()

        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, box in zip(
            track_ids,
            boxes,
        ):
            if track_id != TRACK_ID_TO_WATCH:
                continue

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            print(
                f"Frame {frame_index:<3} center=({center_x:.1f}, {center_y:.1f})",
                end="",
            )

            if previous_center is not None:
                previous_x, previous_y = previous_center

                velocity_x = center_x - previous_x
                velocity_y = center_y - previous_y

                predicted_x = center_x + velocity_x
                predicted_y = center_y + velocity_y

                print(
                    f"  velocity=({velocity_x:.1f}, {velocity_y:.1f})"
                    f"  predicted_next=({predicted_x:.1f}, {predicted_y:.1f})"
                )

            else:
                print("  velocity=(unknown)")

            previous_center = (
                center_x,
                center_y,
            )

        if frame_index >= 10:
            break


if __name__ == "__main__":
    main()
