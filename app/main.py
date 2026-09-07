from pathlib import Path
import cv2

from app.detection.detector import ObjectDetector
from app.tracking.tracker import ObjectTracker
from app.behaviour.history import TrackHistoryManager
from app.behaviour.behaviour import (
    Zone,
    ZoneAnalyzer,
    LoiteringDetector,
    IntrusionDetector,
    AccessController,
    AbandonedObjectDetector,
    WrongDirectionDetector,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"

VIDEO_PATH = (
    BASE_DIR
    / "data"
    / "samples"
    / "sample.mp4"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "outputs"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "security_analysis.mp4"
)

SNAPSHOT_DIR = (
    OUTPUT_DIR
    / "snapshots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SNAPSHOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.25

MAX_FRAMES = None

# Process every 2nd frame.
#
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
FRAME_SKIP = 2

PERSON_CLASS = "person"

ABANDONED_OBJECT_CLASSES = {
    "suitcase",
    "backpack",
    "handbag",
    "bag",
}


# ============================================================
# RESTRICTED ZONE
# ============================================================

# Current prototype restricted zone.
#
# The Streamlit UI will eventually provide the selected zones.
# For now, this is the active security zone used by the engine.

RESTRICTED_ZONE = Zone(
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


# ============================================================
# SECURITY SYSTEM
# ============================================================

class AirportSecuritySystem:

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
    ):

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        self.tracker = ObjectTracker()

        # ----------------------------------------------------
        # Track history
        # ----------------------------------------------------

        self.history = TrackHistoryManager()

        # ----------------------------------------------------
        # Loitering
        # ----------------------------------------------------

        self.loitering_detector = LoiteringDetector(
            minimum_duration=3.0,
            movement_window=2.0,
            maximum_speed=30.0,
        )

        # ----------------------------------------------------
        # Intrusion
        # ----------------------------------------------------

        self.intrusion_detector = IntrusionDetector(
            minimum_duration=0.5,
        )

        # ----------------------------------------------------
        # Access control
        # ----------------------------------------------------

        self.access_controller = AccessController(
            minimum_duration=0.5,
        )

        # ----------------------------------------------------
        # Abandoned object
        # ----------------------------------------------------

        self.abandoned_detector = (
            AbandonedObjectDetector(
                stationary_duration=5.0,
                movement_threshold=30.0,
            )
        )

        # ----------------------------------------------------
        # Wrong direction
        # ----------------------------------------------------

        self.direction_detector = (
            WrongDirectionDetector(
                expected_direction="RIGHT",
                minimum_distance=50.0,
                minimum_duration=1.0,
            )
        )

        # ----------------------------------------------------
        # Passenger snapshot tracking
        # ----------------------------------------------------

        self.snapshot_track_ids = set()

        self.passenger_snapshots = []


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame,
        frame_number: int,
        fps: float,
    ):

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        detections = self.detector.detect(
            frame
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        tracks = self.tracker.update(
            detections
        )

        # ----------------------------------------------------
        # Track history
        # ----------------------------------------------------

        self.history.update(
            tracks=tracks,
            frame_number=frame_number,
            fps=fps,
        )

        return detections, tracks


    # ========================================================
    # SAVE PASSENGER SNAPSHOT
    # ========================================================

    def save_passenger_snapshot(
        self,
        frame,
        track,
        frame_number: int,
    ):

        track_id = track.track_id

        # ----------------------------------------------------
        # Only save one snapshot for each track
        # ----------------------------------------------------

        if track_id in self.snapshot_track_ids:

            return None

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            track.bounding_box,
        )

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # Add some padding around passenger
        # ----------------------------------------------------

        padding = 30

        x1 = max(
            0,
            x1 - padding,
        )

        y1 = max(
            0,
            y1 - padding,
        )

        x2 = min(
            width,
            x2 + padding,
        )

        y2 = min(
            height,
            y2 + padding,
        )

        # ----------------------------------------------------
        # Validate crop
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            return None

        passenger_crop = frame[
            y1:y2,
            x1:x2,
        ].copy()

        # ----------------------------------------------------
        # Add evidence information
        # ----------------------------------------------------

        cv2.rectangle(
            passenger_crop,
            (0, 0),
            (
                passenger_crop.shape[1] - 1,
                passenger_crop.shape[0] - 1,
            ),
            (0, 0, 255),
            4,
        )

        cv2.putText(
            passenger_crop,
            "SECURITY ZONE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            passenger_crop,
            f"TRACK ID: {track_id}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Snapshot path
        # ----------------------------------------------------

        snapshot_path = (
            SNAPSHOT_DIR
            / f"passenger_track_{track_id}.jpg"
        )

        success = cv2.imwrite(
            str(snapshot_path),
            passenger_crop,
        )

        if not success:

            return None

        # ----------------------------------------------------
        # Remember this track
        # ----------------------------------------------------

        self.snapshot_track_ids.add(
            track_id
        )

        self.passenger_snapshots.append(
            str(snapshot_path)
        )

        return snapshot_path


    # ========================================================
    # ANALYZE BEHAVIOR
    # ========================================================

    def analyze_behavior(self):

        all_events = []

        histories = (
            self.history.get_all_histories()
        )

        for track_id, track_history in histories.items():

            positions = (
                track_history.positions
            )

            if not positions:

                continue

            class_name = (
                track_history.class_name
            )

            # ------------------------------------------------
            # PERSON BEHAVIOR
            # ------------------------------------------------

            if class_name == PERSON_CLASS:

                # --------------------------------------------
                # Zone visits
                # --------------------------------------------

                visits = (
                    ZoneAnalyzer.calculate_zone_visits(
                        track_id=track_id,
                        positions=positions,
                        zone=RESTRICTED_ZONE,
                    )
                )

                # --------------------------------------------
                # Loitering
                # --------------------------------------------

                for visit in visits:

                    event = (
                        self.loitering_detector.detect(
                            visit=visit,
                            positions=positions,
                        )
                    )

                    if event is not None:

                        all_events.append(
                            event
                        )

                # --------------------------------------------
                # Intrusion
                # --------------------------------------------

                intrusion_events = (
                    self.intrusion_detector.detect(
                        track_id=track_id,
                        positions=positions,
                        zone=RESTRICTED_ZONE,
                    )
                )

                all_events.extend(
                    intrusion_events
                )

                # --------------------------------------------
                # Unauthorized access
                # --------------------------------------------

                access_events = (
                    self.access_controller.analyze(
                        track_id=track_id,
                        positions=positions,
                        zone=RESTRICTED_ZONE,
                        authorized=False,
                    )
                )

                all_events.extend(
                    access_events
                )

                # --------------------------------------------
                # Wrong direction
                # --------------------------------------------

                direction_event = (
                    self.direction_detector.detect(
                        track_id=track_id,
                        positions=positions,
                    )
                )

                if direction_event is not None:

                    all_events.append(
                        direction_event
                    )

            # ------------------------------------------------
            # ABANDONED OBJECT
            # ------------------------------------------------

            if (
                class_name.lower()
                in ABANDONED_OBJECT_CLASSES
            ):

                abandoned_events = (
                    self.abandoned_detector.detect(
                        track_id=track_id,
                        class_name=class_name,
                        positions=positions,
                    )
                )

                all_events.extend(
                    abandoned_events
                )

        return all_events


    # ========================================================
    # DRAW FRAME
    # ========================================================

    @staticmethod
    def draw_frame(
        frame,
        detections,
        tracks,
        zone: Zone,
    ):

        # ----------------------------------------------------
        # Draw restricted zone
        # ----------------------------------------------------

        points = [
            (int(x), int(y))
            for x, y in zone.polygon
        ]

        for i in range(
            len(points)
        ):

            start = points[i]

            end = points[
                (i + 1) % len(points)
            ]

            cv2.line(
                frame,
                start,
                end,
                (0, 255, 255),
                3,
            )

        # ----------------------------------------------------
        # Zone label
        # ----------------------------------------------------

        if points:

            label_x = points[0][0]
            label_y = points[0][1]

            cv2.rectangle(
                frame,
                (
                    label_x,
                    max(
                        0,
                        label_y - 40,
                    ),
                ),
                (
                    label_x + 230,
                    label_y,
                ),
                (255, 255, 255),
                -1,
            )

            cv2.putText(
                frame,
                zone.name,
                (
                    label_x + 8,
                    max(
                        28,
                        label_y - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Draw tracks
        # ----------------------------------------------------

        for track in tracks:

            x1, y1, x2, y2 = map(
                int,
                track.bounding_box,
            )

            foot_x = int(
                (x1 + x2) / 2
            )

            foot_y = int(y2)

            # ----------------------------------------------
            # Determine whether object is inside zone
            # ----------------------------------------------

            inside = (
                ZoneAnalyzer.point_inside_polygon(
                    (foot_x, foot_y),
                    zone.polygon,
                )
            )

            # ----------------------------------------------
            # Passenger inside security zone
            # ----------------------------------------------

            if (
                inside
                and track.class_name == PERSON_CLASS
            ):

                box_color = (
                    0,
                    0,
                    255,
                )

                label = (
                    f"PASSENGER "
                    f"ID:{track.track_id} "
                    f"| SECURITY ZONE"
                )

                thickness = 4

            # ----------------------------------------------
            # Other tracked objects inside zone
            # ----------------------------------------------

            elif inside:

                box_color = (
                    0,
                    165,
                    255,
                )

                label = (
                    f"{track.class_name} "
                    f"ID:{track.track_id} "
                    f"| INSIDE"
                )

                thickness = 3

            # ----------------------------------------------
            # Object outside zone
            # ----------------------------------------------

            else:

                box_color = (
                    0,
                    255,
                    0,
                )

                label = (
                    f"{track.class_name} "
                    f"ID:{track.track_id}"
                )

                thickness = 2

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness,
            )

            # ----------------------------------------------
            # Foot point
            # ----------------------------------------------

            cv2.circle(
                frame,
                (foot_x, foot_y),
                5,
                (255, 0, 0),
                -1,
            )

            # ----------------------------------------------
            # Label background
            # ----------------------------------------------

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )[0]

            text_width = text_size[0]
            text_height = text_size[1]

            text_x = x1

            text_y = max(
                y1 - 8,
                text_height + 8,
            )

            cv2.rectangle(
                frame,
                (
                    text_x,
                    text_y - text_height - 8,
                ),
                (
                    text_x + text_width + 10,
                    text_y + 4,
                ),
                box_color,
                -1,
            )

            # ----------------------------------------------
            # Track label
            # ----------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    text_x + 5,
                    text_y - 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame


    # ========================================================
    # RUN VIDEO
    # ========================================================

    def run(
        self,
        video_path: Path,
        output_path: Path | None = None,
        max_frames: int | None = None,
        frame_skip: int = 1,
    ):

        # ----------------------------------------------------
        # Validate frame skip
        # ----------------------------------------------------

        if frame_skip < 1:

            raise ValueError(
                "frame_skip must be >= 1"
            )

        # ----------------------------------------------------
        # Validate video
        # ----------------------------------------------------

        video_path = Path(
            video_path
        )

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        # ----------------------------------------------------
        # Video information
        # ----------------------------------------------------

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 30.0

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        # ----------------------------------------------------
        # Video writer
        # ----------------------------------------------------

        writer = None

        if output_path is not None:

            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            output_fps = (
                fps / frame_skip
            )

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_fps,
                (width, height),
            )

            if not writer.isOpened():

                writer.release()

                writer = None

                raise RuntimeError(
                    f"Could not create output video: "
                    f"{output_path}"
                )

        # ----------------------------------------------------
        # Clear previous snapshots
        # ----------------------------------------------------

        self.snapshot_track_ids.clear()

        self.passenger_snapshots.clear()

        for old_snapshot in SNAPSHOT_DIR.glob(
            "passenger_track_*.jpg"
        ):

            try:

                old_snapshot.unlink()

            except OSError:

                pass

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        frame_number = 0

        processed_frames = 0

        total_detections = 0

        unique_track_ids = set()

        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        try:

            while True:

                success, frame = (
                    capture.read()
                )

                if not success:

                    break

                frame_number += 1

                # --------------------------------------------
                # Maximum frame limit
                # --------------------------------------------

                if (
                    max_frames is not None
                    and frame_number > max_frames
                ):

                    break

                # --------------------------------------------
                # Frame skipping
                # --------------------------------------------

                if (
                    frame_number
                    % frame_skip
                    != 0
                ):

                    continue

                processed_frames += 1

                # --------------------------------------------
                # Detection + tracking
                # --------------------------------------------

                detections, tracks = (
                    self.process_frame(
                        frame=frame,
                        frame_number=frame_number,
                        fps=fps,
                    )
                )

                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                total_detections += (
                    len(detections)
                )

                for track in tracks:

                    unique_track_ids.add(
                        track.track_id
                    )

                # --------------------------------------------
                # Passenger security-zone detection
                # --------------------------------------------

                for track in tracks:

                    if (
                        track.class_name
                        != PERSON_CLASS
                    ):

                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        track.bounding_box,
                    )

                    foot_x = int(
                        (x1 + x2) / 2
                    )

                    foot_y = int(y2)

                    inside = (
                        ZoneAnalyzer.point_inside_polygon(
                            (foot_x, foot_y),
                            RESTRICTED_ZONE.polygon,
                        )
                    )

                    if inside:

                        self.save_passenger_snapshot(
                            frame=frame,
                            track=track,
                            frame_number=frame_number,
                        )

                # --------------------------------------------
                # Draw output frame
                # --------------------------------------------

                if writer is not None:

                    output_frame = (
                        self.draw_frame(
                            frame=frame,
                            detections=detections,
                            tracks=tracks,
                            zone=RESTRICTED_ZONE,
                        )
                    )

                    writer.write(
                        output_frame
                    )

        finally:

            capture.release()

            if writer is not None:

                writer.release()

        # ====================================================
        # BEHAVIOR ANALYSIS
        # ====================================================

        events = (
            self.analyze_behavior()
        )

        # ====================================================
        # RESULTS
        # ====================================================

        return {
            "total_detections":
                total_detections,

            "total_tracks":
                len(unique_track_ids),

            "total_events":
                len(events),

            "total_frames":
                processed_frames,

            "events":
                events,

            "output_video":
                (
                    str(output_path)
                    if output_path is not None
                    else None
                ),

            "passenger_snapshots":
                self.passenger_snapshots.copy(),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    system = AirportSecuritySystem(
        model_path=MODEL_PATH,
        confidence_threshold=(
            CONFIDENCE_THRESHOLD
        ),
    )

    results = system.run(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        max_frames=MAX_FRAMES,
        frame_skip=FRAME_SKIP,
    )

    print()
    print("=" * 60)
    print("AIRPORT SECURITY SYSTEM")
    print("=" * 60)

    print(
        f"Detections: "
        f"{results['total_detections']}"
    )

    print(
        f"Unique tracked objects: "
        f"{results['total_tracks']}"
    )

    print(
        f"Frames processed: "
        f"{results['total_frames']}"
    )

    print(
        f"Detected events: "
        f"{results['total_events']}"
    )

    print(
        f"Passenger snapshots: "
        f"{len(results['passenger_snapshots'])}"
    )

    for snapshot in results[
        "passenger_snapshots"
    ]:

        print(
            f"Snapshot: {snapshot}"
        )

    for event in results["events"]:

        print()
        print(event)

    print()
    print(
        f"Output video: "
        f"{results['output_video']}"
    )


if __name__ == "__main__":
    main()