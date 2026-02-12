import cv2
from utils.detector import YOLOModel
from utils.visualization import draw_boxes


MODEL_PATH = "model/best.pt"
IMAGE_PATH = "assets/demo.png"
OUTPUT_PATH = "output.png"


def load_labels(path):
    with open(path) as f:
        return [l.strip() for l in f.readlines()]


def main():
    class_names = load_labels("model/labels.txt")

    model = YOLOModel(MODEL_PATH)

    image = cv2.imread(IMAGE_PATH)

    detections = model.predict(image)

    result = draw_boxes(image, detections, class_names)

    cv2.imwrite(OUTPUT_PATH, result)

    print("Saved result to output.png")


if __name__ == "__main__":
    main()

