import cv2
import numpy as np
from pathlib import Path

from config.settings import BASE_DIR

from app.detection.detector import ObjectDetector
from app.tracking.tracker import ObjectTracker

from app.behaviour.history import TrackHistoryManager

from app.behaviour.behaviour import (
    MovementAnalyzer,
    StationaryDetector,
    Zone,
    ZoneAnalyzer,
    LoiteringDetector,
    IntrusionDetector,
    AccessController,
    AbandonedObjectDetector,
)


PROJECT_ROOT = BASE_DIR

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "sample_2.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo11n.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "zone_visualization.mp4"
)

CONFIDENCE_THRESHOLD = 0.25
MAX_FRAMES = 300


ZONE = Zone(
    name="Restricted Area",
    polygon=[
        (51, 822),
        (286, 1045),
        (1872, 945),
        (1060, 737),
        (1754, 642),
        (1640, 881),
    ],
)


def create_pipeline():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    tracker = ObjectTracker()

    history_manager = TrackHistoryManager()

    return detector, tracker, history_manager


def collect_histories():

    detector, tracker, history_manager = create_pipeline()

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    frame_number = 0

    while frame_number < MAX_FRAMES:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)

        tracks = tracker.update(detections)

        history_manager.update(
            tracks=tracks,
            frame_number=frame_number,
            fps=fps,
        )

        frame_number += 1

    cap.release()

    return history_manager.get_all_histories(), fps


def test_movement():

    print("\n========== MOVEMENT ANALYSIS ==========\n")

    histories, _ = collect_histories()

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        positions = history.positions

        total_distance = (
            MovementAnalyzer.total_distance(
                positions
            )
        )

        displacement = (
            MovementAnalyzer.displacement(
                positions
            )
        )

        duration = (
            MovementAnalyzer.duration(
                positions
            )
        )

        average_speed = (
            MovementAnalyzer.average_speed(
                positions
            )
        )

        window_speed = (
            MovementAnalyzer.window_speed(
                positions,
                window_seconds=3.0,
            )
        )

        print(
            f"Track ID: {track_id}"
        )

        print(
            f"Positions: {len(positions)}"
        )

        print(
            f"Duration: {duration:.2f}s"
        )

        print(
            f"Total distance: "
            f"{total_distance:.2f} pixels"
        )

        print(
            f"Displacement: "
            f"{displacement:.2f} pixels"
        )

        print(
            f"Average speed: "
            f"{average_speed:.2f} pixels/s"
        )

        print(
            f"3-second window speed: "
            f"{window_speed:.2f} pixels/s"
        )

        print("-" * 50)


def test_stationary():

    print("\n========== STATIONARY ANALYSIS ==========\n")

    histories, _ = collect_histories()

    stationary_detector = StationaryDetector(
        movement_threshold=10.0,
        window_seconds=3.0,
        minimum_duration=3.0,
    )

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        stationary = (
            stationary_detector.is_stationary(
                history.positions
            )
        )

        print(
            f"Track ID: {track_id} | "
            f"Positions: {len(history.positions)} | "
            f"Stationary: {stationary}"
        )


def test_zones():

    print(
        "\n========== FOOT POINT ZONE ANALYSIS ==========\n"
    )

    histories, _ = collect_histories()

    people_in_zone = []

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        positions_inside = (
            ZoneAnalyzer.history_inside_zone(
                positions=history.positions,
                zone=ZONE,
            )
        )

        if positions_inside:

            people_in_zone.append(track_id)

            first_position = positions_inside[0]
            last_position = positions_inside[-1]

            print(
                f"Track ID: {track_id}"
            )

            print(
                f"Positions inside zone: "
                f"{len(positions_inside)}"
            )

            print(
                f"First zone timestamp: "
                f"{first_position.timestamp:.2f}s"
            )

            print(
                f"Last zone timestamp: "
                f"{last_position.timestamp:.2f}s"
            )

            print(
                f"First foot point: "
                f"{first_position.foot_point}"
            )

            print(
                f"Last foot point: "
                f"{last_position.foot_point}"
            )

            print("-" * 50)

    print(
        f"\nPeople who entered the zone: "
        f"{len(people_in_zone)}"
    )

    print(
        f"Track IDs: {people_in_zone}"
    )


def test_zone_duration():

    print(
        "\n========== ZONE VISIT ANALYSIS ==========\n"
    )

    histories, _ = collect_histories()

    total_visits = 0

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        visits = (
            ZoneAnalyzer.calculate_zone_visits(
                track_id=track_id,
                positions=history.positions,
                zone=ZONE,
            )
        )

        if not visits:
            continue

        print(
            f"Track ID: {track_id}"
        )

        print(
            f"Visits: {len(visits)}"
        )

        print("-" * 50)

        for visit_number, visit in enumerate(
            visits,
            start=1,
        ):

            total_visits += 1

            print(
                f"Visit {visit_number}"
            )

            print(
                f"  Zone: {visit.zone_name}"
            )

            print(
                f"  Entered: "
                f"{visit.start_time:.2f}s"
            )

            print(
                f"  Exited: "
                f"{visit.end_time:.2f}s"
            )

            print(
                f"  Duration: "
                f"{visit.duration:.2f}s"
            )

            print()

    print(
        f"Total zone visits: {total_visits}"
    )


def test_loitering():

    print(
        "\n========== LOITERING ANALYSIS ==========\n"
    )

    minimum_duration = 3.0
    movement_window = 2.0
    maximum_speed = 30.0

    print(
        f"Minimum zone duration: "
        f"{minimum_duration:.2f}s"
    )

    print(
        f"Movement window: "
        f"{movement_window:.2f}s"
    )

    print(
        f"Maximum movement speed: "
        f"{maximum_speed:.2f} pixels/s"
    )

    print()

    histories, _ = collect_histories()

    detector = LoiteringDetector(
        minimum_duration=minimum_duration,
        movement_window=movement_window,
        maximum_speed=maximum_speed,
    )

    total_visits = 0
    total_events = 0

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        visits = ZoneAnalyzer.calculate_zone_visits(
            track_id=track_id,
            positions=history.positions,
            zone=ZONE,
        )

        for visit_number, visit in enumerate(
            visits,
            start=1,
        ):

            total_visits += 1

            print(
                f"Track ID: {track_id} | "
                f"Visit: {visit_number}"
            )

            print(
                f"  Zone duration: "
                f"{visit.duration:.2f}s"
            )

            event = detector.detect(
                visit=visit,
                positions=history.positions,
            )

            if event is None:

                print(
                    "  Loitering: False"
                )

            else:

                total_events += 1

                print(
                    "  Loitering: True"
                )

                print(
                    f"  Low-movement window: "
                    f"{event.start_time:.2f}s → "
                    f"{event.end_time:.2f}s"
                )

                print(
                    f"  Window movement speed: "
                    f"{event.movement_speed:.2f} pixels/s"
                )

            print("-" * 50)

    print(
        f"\nTotal zone visits: {total_visits}"
    )

    print(
        f"Total loitering events: {total_events}"
    )


def test_intrusion():

    print(
        "\n========== INTRUSION ANALYSIS ==========\n"
    )

    minimum_duration = 0.5

    print(
        f"Minimum intrusion duration: "
        f"{minimum_duration:.2f}s\n"
    )

    histories, _ = collect_histories()

    detector = IntrusionDetector(
        minimum_duration=minimum_duration
    )

    total_events = 0

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        events = detector.detect(
            track_id=track_id,
            positions=history.positions,
            zone=ZONE,
        )

        for event_number, event in enumerate(
            events,
            start=1,
        ):

            total_events += 1

            print(
                f"Track ID: {track_id} | "
                f"Intrusion: {event_number}"
            )

            print(
                f"  Zone: {event.zone_name}"
            )

            print(
                f"  Entered: "
                f"{event.start_time:.2f}s"
            )

            print(
                f"  Exited: "
                f"{event.end_time:.2f}s"
            )

            print(
                f"  Duration: "
                f"{event.duration:.2f}s"
            )

            print("-" * 50)

    print(
        f"\nTotal intrusion events: {total_events}"
    )


def test_access_control():

    print(
        "\n========== ACCESS CONTROL ANALYSIS ==========\n"
    )

    minimum_duration = 0.5

    print(
        f"Minimum unauthorized access duration: "
        f"{minimum_duration:.2f}s\n"
    )

    histories, _ = collect_histories()

    controller = AccessController(
        minimum_duration=minimum_duration
    )

    total_events = 0

    for track_id, history in histories.items():

        if history.class_name != "person":
            continue

        events = controller.analyze(
            track_id=track_id,
            positions=history.positions,
            zone=ZONE,
            authorized=False,
        )

        for event_number, event in enumerate(
            events,
            start=1,
        ):

            total_events += 1

            print(
                f"Track ID: {track_id} | "
                f"Unauthorized Access: {event_number}"
            )

            print(
                f"  Zone: {event.zone_name}"
            )

            print(
                f"  Access: {event.access_type}"
            )

            print(
                f"  Entered: "
                f"{event.start_time:.2f}s"
            )

            print(
                f"  Exited: "
                f"{event.end_time:.2f}s"
            )

            print(
                f"  Duration: "
                f"{event.duration:.2f}s"
            )

            print("-" * 50)

    print(
        f"\nTotal unauthorized access events: "
        f"{total_events}"
    )


def test_abandoned_objects():

    print(
        "\n========== ABANDONED OBJECT ANALYSIS ==========\n"
    )

    stationary_duration = 5.0
    movement_threshold = 30.0

    print(
        f"Stationary duration: "
        f"{stationary_duration:.2f}s"
    )

    print(
        f"Movement threshold: "
        f"{movement_threshold:.2f} pixels\n"
    )

    histories, _ = collect_histories()

    detector = AbandonedObjectDetector(
        stationary_duration=stationary_duration,
        movement_threshold=movement_threshold,
    )

    target_classes = {
        "suitcase",
        "backpack",
        "handbag",
        "bag",
    }

    total_events = 0

    for track_id, history in histories.items():

        if history.class_name not in target_classes:
            continue

        events = detector.detect(
            track_id=track_id,
            class_name=history.class_name,
            positions=history.positions,
        )

        for event_number, event in enumerate(
            events,
            start=1,
        ):

            total_events += 1

            print(
                f"Track ID: {track_id} | "
                f"Event: {event_number}"
            )

            print(
                f"  Object: {event.class_name}"
            )

            print(
                f"  Started: "
                f"{event.start_time:.2f}s"
            )

            print(
                f"  Detected until: "
                f"{event.end_time:.2f}s"
            )

            print(
                f"  Stationary duration: "
                f"{event.duration:.2f}s"
            )

            print(
                f"  Position: "
                f"({event.position[0]:.1f}, "
                f"{event.position[1]:.1f})"
            )

            print("-" * 50)

    print(
        f"\nTotal abandoned-object events: "
        f"{total_events}"
    )


def test_zone_visualization():

    print(
        "\n========== ZONE VISUALIZATION ==========\n"
    )

    detector, tracker, history_manager = create_pipeline()

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    polygon = ZONE.polygon

    frame_number = 0

    while frame_number < MAX_FRAMES:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)

        tracks = tracker.update(detections)

        history_manager.update(
            tracks=tracks,
            frame_number=frame_number,
            fps=fps,
        )

        points = [
            (int(x), int(y))
            for x, y in polygon
        ]

        cv2.polylines(
            frame,
            [np.array(points)],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3,
        )

        for track in tracks:

            if track.class_name != "person":
                continue

            x1, y1, x2, y2 = map(
                int,
                track.bounding_box
            )

            foot_x = int(
                (x1 + x2) / 2
            )

            foot_y = int(y2)

            inside = (
                ZoneAnalyzer.point_inside_polygon(
                    (foot_x, foot_y),
                    polygon,
                )
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID: {track.track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                frame,
                (foot_x, foot_y),
                6,
                (0, 0, 255),
                -1,
            )

            status = (
                "INSIDE"
                if inside
                else "OUTSIDE"
            )

            cv2.putText(
                frame,
                status,
                (x1, min(y2 + 25, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        frame_number += 1

    cap.release()
    writer.release()

    print(
        f"Frames processed: {frame_number}"
    )

    print(
        f"Output saved to: {OUTPUT_PATH}"
    )


def select_zone():

    points = []

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            points.append((x, y))

            print(
                f"Point {len(points)}: "
                f"({x}, {y})"
            )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Could not read the first frame."
        )

    window_name = "Select Zone"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print(
        "\n========== ZONE SELECTION ==========\n"
    )

    print("Left click: add polygon point")
    print("Press ENTER: finish")
    print("Press R: reset points")
    print("Press ESC: exit\n")

    while True:

        display = frame.copy()

        for i, point in enumerate(points):

            cv2.circle(
                display,
                point,
                6,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                display,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        if len(points) >= 2:

            for i in range(1, len(points)):

                cv2.line(
                    display,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2,
                )

        if len(points) >= 3:

            cv2.line(
                display,
                points[-1],
                points[0],
                (255, 0, 0),
                2,
            )

        cv2.imshow(
            window_name,
            display,
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 13:

            if len(points) < 3:

                print(
                    "\nYou need at least 3 points."
                )

                continue

            break

        elif key == ord("r"):

            points.clear()

            print(
                "\nPoints reset."
            )

        elif key == 27:

            cv2.destroyAllWindows()

            print(
                "\nSelection cancelled."
            )

            return

    cv2.destroyAllWindows()

    print(
        "\n========== SELECTED ZONE ==========\n"
    )

    print("polygon=[")

    for point in points:

        print(
            f"    ({point[0]}, {point[1]}),"
        )

    print("]")

    print(
        "\nCopy these coordinates into your Zone object."
    )


def main():

    print("\n")
    print("=" * 60)
    print("BEHAVIOR TESTS")
    print("=" * 60)

    test_movement()
    test_stationary()
    test_zones()
    test_zone_duration()
    test_loitering()
    test_intrusion()
    test_access_control()
    test_abandoned_objects()


if __name__ == "__main__":
    main()