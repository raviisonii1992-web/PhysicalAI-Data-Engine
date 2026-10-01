from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"
VIDEO_PATH = "data/raw/India_Traffic.mp4"

OUTPUT_PATH = "outputs/videos/India_Traffic_tracked.mp4"


def main():

    model = YOLO(MODEL_PATH)

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            results = model.track(
                frame,
                device=0,
                tracker="botsort.yaml",
                conf=0.25,
                persist=True,
                verbose=False,
            )

            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
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
                    x1, y1, x2, y2 = map(int, box)

                    class_name = result.names[class_id]

                    label = f"{class_name} #{track_id} {confidence:.2f}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(y1 - 10, 20),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)

            if frame_index % 30 == 0:
                print(f"Processed frame {frame_index}")

            frame_index += 1

    finally:
        capture.release()
        writer.release()

    print()
    print(f"Saved tracked video to:")
    print(output_path)


if __name__ == "__main__":
    main()
