import cv2
import numpy as np


COLORS = [
    (0, 255, 0),   # empty -> green
    (0, 0, 255)    # occupied -> red
]


def draw_boxes(image: np.ndarray, detections, class_names):
    """
    Draw bounding boxes + labels on image
    """

    img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls = det["class_id"]
        score = det["score"]

        color = COLORS[cls % len(COLORS)]
        label = f"{class_names[cls]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), color, -1)

        cv2.putText(
            img,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return img

