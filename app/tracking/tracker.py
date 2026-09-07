from dataclasses import dataclass
import numpy as np
import supervision as sv


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]


class ObjectTracker:

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(self, detections):

        if not detections:
            return []

        xyxy = np.array(
            [d.bounding_box for d in detections],
            dtype=np.float32,
        )

        confidence = np.array(
            [d.confidence for d in detections],
            dtype=np.float32,
        )

        class_id = np.array(
            [d.class_id for d in detections],
            dtype=np.int32,
        )

        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self.tracker.update_with_detections(
            supervision_detections
        )

        if tracked.tracker_id is None:
            return []

        tracks = []

        for i in range(len(tracked)):

            track_id = tracked.tracker_id[i]

            if track_id is None:
                continue

            track_class_id = int(tracked.class_id[i])

            track_confidence = float(
                tracked.confidence[i]
            )

            class_name = "unknown"

            for detection in detections:
                if detection.class_id == track_class_id:
                    class_name = detection.class_name
                    break

            track_box = tuple(
                map(float, tracked.xyxy[i])
            )

            tracks.append(
                Track(
                    track_id=int(track_id),
                    class_id=track_class_id,
                    class_name=class_name,
                    confidence=track_confidence,
                    bounding_box=track_box,
                )
            )

        return tracks