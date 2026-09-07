from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
import cv2


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]


class ObjectDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if self.model_path.exists():
            # Use the locally available model.
            self.model = YOLO(str(self.model_path))

        else:
            # If the model is not present (e.g. Streamlit Cloud),
            # let Ultralytics download the YOLO11n model automatically.
            self.model = YOLO("yolo11n.pt")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.model.names[class_id]

            detection = Detection(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                bounding_box=(x1, y1, x2, y2)
            )

            detections.append(detection)

        return detections
