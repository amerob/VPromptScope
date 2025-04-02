
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml, confidence=0.4, class_score=0.25):

        with open(data_yaml, mode="r") as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml["names"]  # List of class names
        self.nc = data_yaml["nc"]        # Number of classes
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.labels)}

        self.confidence = confidence
        self.class_score = class_score

        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image, classes_to_detect=None):
        if classes_to_detect is None:
            classes_to_detect = self.labels

        classes_to_detect_ids = [
            self.class_name_to_id[name] for name in classes_to_detect if name in self.class_name_to_id
        ]

        row, col, _ = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]

            if confidence > self.confidence:
                scores = row[5:]
                class_id = np.argmax(scores)
                class_score = scores[class_id]

                if class_score > self.class_score and class_id in classes_to_detect_ids:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    confidences.append(confidence)
                    classes.append(class_id)

        if boxes:
            nms_result = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, 0.45)
            indices = nms_result[0] if isinstance(nms_result, tuple) and len(nms_result) > 0 else nms_result

            for ind in indices:
                x, y, w, h = boxes[ind]
                bb_conf = int(confidences[ind] * 100)
                class_id = classes[ind]
                class_name = self.labels[class_id]
                color = self.generate_colors(class_id)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(image, (x, y - 30), (x + w, y), color, -1)
                cv2.putText(image, f"{class_name} {bb_conf}%", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        return image

    def generate_colors(self, ID):

        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])