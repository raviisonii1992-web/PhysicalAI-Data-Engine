from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"
VIDEO_PATH = "data/raw/indian_traffic.mp4"


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
        print(f"Frame: {frame_index}")

        if result.boxes is None:
            print("No detections")
            continue

        if result.boxes.id is None:
            print("No track IDs assigned")
            continue

        track_ids = result.boxes.id.int().cpu().tolist()

        class_ids = result.boxes.cls.int().cpu().tolist()

        confidences = result.boxes.conf.cpu().tolist()

        for track_id, class_id, confidence in zip(
            track_ids,
            class_ids,
            confidences,
        ):
            class_name = result.names[class_id]

            print(f"  ID {track_id:<4} {class_name:<12} {confidence:.3f}")

        if frame_index >= 20:
            break


if __name__ == "__main__":
    main()
