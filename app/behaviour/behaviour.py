from dataclasses import dataclass
from typing import List, Tuple
import math


Point = Tuple[float, float]


# ============================================================
# EVENTS
# ============================================================

@dataclass
class LoiteringEvent:
    track_id: int
    zone_name: str
    start_time: float
    end_time: float
    duration: float
    movement_speed: float


@dataclass
class IntrusionEvent:
    track_id: int
    zone_name: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class AccessEvent:
    track_id: int
    zone_name: str
    access_type: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class AbandonedObjectEvent:
    track_id: int
    class_name: str
    start_time: float
    end_time: float
    duration: float
    position: Point


@dataclass
class WrongDirectionEvent:
    track_id: int
    start_time: float
    end_time: float
    duration: float
    direction: str


# ============================================================
# ZONES
# ============================================================

@dataclass
class Zone:
    name: str
    polygon: List[Point]


@dataclass
class ZoneVisit:
    zone_name: str
    track_id: int
    start_time: float
    end_time: float
    duration: float


# ============================================================
# MOVEMENT
# ============================================================

class MovementAnalyzer:

    @staticmethod
    def distance(
        point_a: Point,
        point_b: Point,
    ) -> float:

        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]

        return math.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def total_distance(
        positions,
    ) -> float:

        if len(positions) < 2:
            return 0.0

        total = 0.0

        for i in range(1, len(positions)):

            total += MovementAnalyzer.distance(
                positions[i - 1].center,
                positions[i].center,
            )

        return total

    @staticmethod
    def displacement(
        positions,
    ) -> float:

        if len(positions) < 2:
            return 0.0

        return MovementAnalyzer.distance(
            positions[0].center,
            positions[-1].center,
        )

    @staticmethod
    def duration(
        positions,
    ) -> float:

        if len(positions) < 2:
            return 0.0

        return (
            positions[-1].timestamp
            - positions[0].timestamp
        )

    @staticmethod
    def average_speed(
        positions,
    ) -> float:

        duration = MovementAnalyzer.duration(
            positions
        )

        if duration <= 0:
            return 0.0

        return (
            MovementAnalyzer.total_distance(
                positions
            )
            / duration
        )

    @staticmethod
    def window_positions(
        positions,
        window_seconds: float,
    ):

        if not positions:
            return []

        latest_time = positions[-1].timestamp

        return [
            position
            for position in positions
            if latest_time - position.timestamp
            <= window_seconds
        ]

    @staticmethod
    def window_distance(
        positions,
        window_seconds: float,
    ) -> float:

        window = MovementAnalyzer.window_positions(
            positions,
            window_seconds,
        )

        if len(window) < 2:
            return 0.0

        return MovementAnalyzer.total_distance(
            window
        )

    @staticmethod
    def window_speed(
        positions,
        window_seconds: float,
    ) -> float:

        window = MovementAnalyzer.window_positions(
            positions,
            window_seconds,
        )

        if len(window) < 2:
            return 0.0

        return MovementAnalyzer.average_speed(
            window
        )


# ============================================================
# STATIONARY
# ============================================================

class StationaryDetector:

    def __init__(
        self,
        movement_threshold: float = 10.0,
        window_seconds: float = 3.0,
        minimum_duration: float = 3.0,
    ):

        self.movement_threshold = movement_threshold
        self.window_seconds = window_seconds
        self.minimum_duration = minimum_duration

    def is_stationary(
        self,
        positions,
    ) -> bool:

        if len(positions) < 2:
            return False

        duration = MovementAnalyzer.duration(
            positions
        )

        if duration < self.minimum_duration:
            return False

        speed = MovementAnalyzer.window_speed(
            positions,
            self.window_seconds,
        )

        return speed <= self.movement_threshold


# ============================================================
# ZONE ANALYSIS
# ============================================================

class ZoneAnalyzer:

    @staticmethod
    def point_inside_polygon(
        point: Point,
        polygon: List[Point],
    ) -> bool:

        if len(polygon) < 3:
            return False

        x, y = point

        inside = False
        j = len(polygon) - 1

        for i in range(len(polygon)):

            xi, yi = polygon[i]
            xj, yj = polygon[j]

            intersects = (
                (yi > y) != (yj > y)
                and
                x < (
                    (xj - xi)
                    * (y - yi)
                    / (yj - yi)
                ) + xi
            )

            if intersects:
                inside = not inside

            j = i

        return inside

    @staticmethod
    def position_inside_zone(
        position,
        zone: Zone,
    ) -> bool:

        return ZoneAnalyzer.point_inside_polygon(
            position.foot_point,
            zone.polygon,
        )

    @staticmethod
    def history_inside_zone(
        positions,
        zone: Zone,
    ):

        return [
            position
            for position in positions
            if ZoneAnalyzer.position_inside_zone(
                position,
                zone,
            )
        ]

    @staticmethod
    def calculate_zone_visits(
        track_id: int,
        positions,
        zone: Zone,
        exit_grace_period: float = 0.5,
    ) -> List[ZoneVisit]:

        if not positions:
            return []

        visits = []

        inside = False
        start_time = None
        outside_start_time = None

        for position in positions:

            current_inside = (
                ZoneAnalyzer.position_inside_zone(
                    position,
                    zone,
                )
            )

            if current_inside:

                if not inside:

                    inside = True
                    start_time = position.timestamp

                outside_start_time = None

            else:

                if inside:

                    if outside_start_time is None:
                        outside_start_time = position.timestamp

                    outside_duration = (
                        position.timestamp
                        - outside_start_time
                    )

                    if (
                        outside_duration
                        >= exit_grace_period
                    ):

                        end_time = outside_start_time

                        visits.append(
                            ZoneVisit(
                                zone_name=zone.name,
                                track_id=track_id,
                                start_time=start_time,
                                end_time=end_time,
                                duration=(
                                    end_time
                                    - start_time
                                ),
                            )
                        )

                        inside = False
                        start_time = None
                        outside_start_time = None

        if inside and start_time is not None:

            end_time = positions[-1].timestamp

            visits.append(
                ZoneVisit(
                    zone_name=zone.name,
                    track_id=track_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=(
                        end_time
                        - start_time
                    ),
                )
            )

        return visits


# ============================================================
# LOITERING
# ============================================================

class LoiteringDetector:

    def __init__(
        self,
        minimum_duration: float = 3.0,
        movement_window: float = 2.0,
        maximum_speed: float = 30.0,
    ):

        self.minimum_duration = minimum_duration
        self.movement_window = movement_window
        self.maximum_speed = maximum_speed

    def detect(
        self,
        visit: ZoneVisit,
        positions,
    ) -> LoiteringEvent | None:

        if visit.duration < self.minimum_duration:
            return None

        visit_positions = [
            position
            for position in positions
            if (
                visit.start_time
                <= position.timestamp
                <= visit.end_time
            )
        ]

        if len(visit_positions) < 2:
            return None

        for position in visit_positions:

            window_end = (
                position.timestamp
                + self.movement_window
            )

            window_positions = [
                p
                for p in visit_positions
                if (
                    position.timestamp
                    <= p.timestamp
                    <= window_end
                )
            ]

            if len(window_positions) < 2:
                continue

            movement_speed = (
                MovementAnalyzer.average_speed(
                    window_positions
                )
            )

            if movement_speed <= self.maximum_speed:

                return LoiteringEvent(
                    track_id=visit.track_id,
                    zone_name=visit.zone_name,
                    start_time=window_positions[0].timestamp,
                    end_time=window_positions[-1].timestamp,
                    duration=visit.duration,
                    movement_speed=movement_speed,
                )

        return None


# ============================================================
# INTRUSION
# ============================================================

class IntrusionDetector:

    def __init__(
        self,
        minimum_duration: float = 0.5,
    ):

        self.minimum_duration = minimum_duration

    def detect(
        self,
        track_id: int,
        positions,
        zone: Zone,
    ) -> List[IntrusionEvent]:

        visits = ZoneAnalyzer.calculate_zone_visits(
            track_id,
            positions,
            zone,
        )

        events = []

        for visit in visits:

            if visit.duration < self.minimum_duration:
                continue

            events.append(
                IntrusionEvent(
                    track_id=track_id,
                    zone_name=visit.zone_name,
                    start_time=visit.start_time,
                    end_time=visit.end_time,
                    duration=visit.duration,
                )
            )

        return events


# ============================================================
# ACCESS CONTROL
# ============================================================

class AccessController:

    def __init__(
        self,
        minimum_duration: float = 0.5,
    ):

        self.minimum_duration = minimum_duration

    def analyze(
        self,
        track_id: int,
        positions,
        zone: Zone,
        authorized: bool = False,
    ) -> List[AccessEvent]:

        if authorized:
            return []

        visits = ZoneAnalyzer.calculate_zone_visits(
            track_id,
            positions,
            zone,
        )

        events = []

        for visit in visits:

            if visit.duration < self.minimum_duration:
                continue

            events.append(
                AccessEvent(
                    track_id=track_id,
                    zone_name=visit.zone_name,
                    access_type="UNAUTHORIZED",
                    start_time=visit.start_time,
                    end_time=visit.end_time,
                    duration=visit.duration,
                )
            )

        return events


# ============================================================
# ABANDONED OBJECT
# ============================================================

class AbandonedObjectDetector:

    def __init__(
        self,
        stationary_duration: float = 5.0,
        movement_threshold: float = 30.0,
    ):

        self.stationary_duration = stationary_duration
        self.movement_threshold = movement_threshold

    def detect(
        self,
        track_id: int,
        class_name: str,
        positions,
    ) -> List[AbandonedObjectEvent]:

        if not positions:
            return []

        events = []

        stationary_start = None
        reference_position = None

        for position in positions:

            current_point = position.foot_point

            if reference_position is None:

                reference_position = current_point
                stationary_start = position.timestamp

                continue

            movement = MovementAnalyzer.distance(
                reference_position,
                current_point,
            )

            if movement <= self.movement_threshold:

                duration = (
                    position.timestamp
                    - stationary_start
                )

                if duration >= self.stationary_duration:

                    events.append(
                        AbandonedObjectEvent(
                            track_id=track_id,
                            class_name=class_name,
                            start_time=stationary_start,
                            end_time=position.timestamp,
                            duration=duration,
                            position=current_point,
                        )
                    )

                    stationary_start = position.timestamp
                    reference_position = current_point

            else:

                reference_position = current_point
                stationary_start = position.timestamp

        return events


# ============================================================
# WRONG DIRECTION
# ============================================================

class WrongDirectionDetector:

    def __init__(
        self,
        expected_direction: str = "RIGHT",
        minimum_distance: float = 50.0,
        minimum_duration: float = 1.0,
    ):

        self.expected_direction = expected_direction.upper()
        self.minimum_distance = minimum_distance
        self.minimum_duration = minimum_duration

    def detect(
        self,
        track_id: int,
        positions,
    ) -> WrongDirectionEvent | None:

        if len(positions) < 2:
            return None

        first = positions[0]
        last = positions[-1]

        dx = last.center[0] - first.center[0]
        dy = last.center[1] - first.center[1]

        duration = (
            last.timestamp
            - first.timestamp
        )

        if duration < self.minimum_duration:
            return None

        distance = math.sqrt(
            dx ** 2 + dy ** 2
        )

        if distance < self.minimum_distance:
            return None

        actual_direction = self._get_direction(
            dx,
            dy,
        )

        if actual_direction != self.expected_direction:

            return WrongDirectionEvent(
                track_id=track_id,
                start_time=first.timestamp,
                end_time=last.timestamp,
                duration=duration,
                direction=actual_direction,
            )

        return None

    @staticmethod
    def _get_direction(
        dx: float,
        dy: float,
    ) -> str:

        if abs(dx) >= abs(dy):

            if dx > 0:
                return "RIGHT"

            return "LEFT"

        if dy > 0:
            return "DOWN"

        return "UP"