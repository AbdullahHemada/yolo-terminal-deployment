from ultralytics import YOLO
import numpy as np


class YOLOModel:
    """
    Loads YOLO model once and runs inference.
    Returns raw detections in structured format.
    """

    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, image: np.ndarray):
        """
        image: numpy BGR or RGB
        returns:
            list of dicts:
            [
                {
                    'box': [x1, y1, x2, y2],
                    'score': float,
                    'class_id': int
                }
            ]
        """

        results = self.model(image, conf=self.conf)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": conf,
                "class_id": cls
            })

        return detections

