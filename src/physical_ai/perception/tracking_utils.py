from physical_ai.perception.detection import BoundingBox


def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:

    intersection_x1 = max(
        box_a.x1,
        box_b.x1,
    )

    intersection_y1 = max(
        box_a.y1,
        box_b.y1,
    )

    intersection_x2 = min(
        box_a.x2,
        box_b.x2,
    )

    intersection_y2 = min(
        box_a.y2,
        box_b.y2,
    )

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = intersection_width * intersection_height

    area_a = box_a.width * box_a.height

    area_b = box_b.width * box_b.height

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area
