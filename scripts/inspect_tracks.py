from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"
VIDEO_PATH = "data/raw/India_Traffic.mp4"


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

    for frame_index, result in enumerate(results):
        print()
        print("=" * 50)
        print(f"Frame {frame_index}")
        print("=" * 50)

        if result.boxes is None:
            print("No detections")
            continue

        if result.boxes.id is None:
            print("Detections exist, but no track IDs")
            continue

        track_ids = result.boxes.id.int().cpu().tolist()

        class_ids = result.boxes.cls.int().cpu().tolist()

        confidences = result.boxes.conf.cpu().tolist()

        boxes = result.boxes.xyxy.cpu().tolist()

        for (
            track_id,
            class_id,
            confidence,
            box,
        ) in zip(
            track_ids,
            class_ids,
            confidences,
            boxes,
        ):
            class_name = result.names[class_id]

            x1, y1, x2, y2 = box

            print(
                f"ID={track_id:<3} "
                f"Class={class_name:<12} "
                f"Conf={confidence:.3f} "
                f"Box=({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})",
                f"Width={x2 - x1:.1f}",
                f"Height={y2 - y1:.1f}",
            )

        if frame_index >= 5:
            break


if __name__ == "__main__":
    main()
