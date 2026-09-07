from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Point = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]


@dataclass
class Position:
    frame_number: int
    timestamp: float
    center: Point
    foot_point: Point
    bounding_box: BoundingBox


@dataclass
class TrackHistory:
    track_id: int
    class_name: str
    positions: List[Position] = field(default_factory=list)

    @property
    def latest_position(self) -> Position | None:
        if not self.positions:
            return None

        return self.positions[-1]

    @property
    def first_position(self) -> Position | None:
        if not self.positions:
            return None

        return self.positions[0]


class TrackHistoryManager:

    def __init__(self):
        self.histories: Dict[int, TrackHistory] = {}

    def update(
        self,
        tracks,
        frame_number: int,
        fps: float,
    ) -> None:

        if fps <= 0:
            raise ValueError("FPS must be greater than zero.")

        timestamp = frame_number / fps

        for track in tracks:

            x1, y1, x2, y2 = track.bounding_box

            center = (
                (x1 + x2) / 2,
                (y1 + y2) / 2,
            )

            foot_point = (
                (x1 + x2) / 2,
                y2,
            )

            position = Position(
                frame_number=frame_number,
                timestamp=timestamp,
                center=center,
                foot_point=foot_point,
                bounding_box=track.bounding_box,
            )

            if track.track_id not in self.histories:

                self.histories[track.track_id] = TrackHistory(
                    track_id=track.track_id,
                    class_name=track.class_name,
                )

            self.histories[
                track.track_id
            ].positions.append(position)

    def get_history(
        self,
        track_id: int,
    ) -> TrackHistory | None:

        return self.histories.get(track_id)

    def get_all_histories(
        self,
    ) -> Dict[int, TrackHistory]:

        return self.histories

    def clear(self) -> None:
        self.histories.clear()