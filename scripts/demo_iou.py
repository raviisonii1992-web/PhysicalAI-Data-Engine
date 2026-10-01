from physical_ai.perception.detection import BoundingBox
from physical_ai.perception.tracking_utils import calculate_iou


def main():

    box_a = BoundingBox(
        x1=100,
        y1=100,
        x2=200,
        y2=200,
    )

    box_b = BoundingBox(
        x1=150,
        y1=150,
        x2=250,
        y2=250,
    )

    iou = calculate_iou(
        box_a,
        box_b,
    )

    print(f"IoU = {iou:.3f}")


if __name__ == "__main__":
    main()
