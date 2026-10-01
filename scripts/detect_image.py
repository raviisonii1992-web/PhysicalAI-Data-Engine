from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "models/detection/yolo11n.pt"

IMAGE_PATH = "data/frames/test_drive/frame_001020_00034.03s.jpg"

OUTPUT_PATH = "outputs/images/detected_frame.jpg"


def main():

    model = YOLO(MODEL_PATH)

    results = model(
        IMAGE_PATH,
        device=0,
    )

    result = results[0]

    image = result.orig_img.copy()

    print()
    print("=" * 60)
    print("PhysicalAI - Object Detection Results")
    print("=" * 60)

    print(f"Number of detections: {len(result.boxes)}")

    for detection_index, box in enumerate(
        result.boxes,
        start=1,
    ):
        class_id = int(box.cls[0].item())

        class_name = result.names[class_id]

        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        label = f"{detection_index} {class_name} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        print(f"Detection #{detection_index}: {class_name} {confidence:.3f}")

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(f"Could not save image to {output_path}")

    print()
    print(f"Saved annotated image to:")
    print(output_path)


if __name__ == "__main__":
    main()
