import cv2
from pathlib import Path

from config.settings import BASE_DIR
from app.detection.detector import ObjectDetector
from app.tracking.tracker import ObjectTracker
from app.behaviour.history import TrackHistoryManager


PROJECT_ROOT = BASE_DIR

VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "sample_2.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.25
MAX_FRAMES = 100


def create_pipeline():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    tracker = ObjectTracker()

    history_manager = TrackHistoryManager()

    return detector, tracker, history_manager


def read_video():

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    return cap, fps


def test_detection():

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    frame_number = 0

    print("\n========== DETECTION TEST ==========\n")

    while frame_number < 10:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)

        print(
            f"Frame: {frame_number}"
        )

        print(
            f"Detections: {len(detections)}"
        )

        for detection in detections:

            print(
                f"Class: {detection.class_name} | "
                f"Confidence: {detection.confidence:.2f} | "
                f"Box: {detection.bounding_box}"
            )

        frame_number += 1

    cap.release()

    print(
        f"\nFrames processed: {frame_number}"
    )


def test_tracking():

    detector, tracker, _ = create_pipeline()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    frame_number = 0
    total_tracks = 0
    unique_track_ids = set()

    print("\n========== TRACKING TEST ==========\n")

    while frame_number < MAX_FRAMES:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)

        tracks = tracker.update(detections)

        total_tracks += len(tracks)

        print(
            f"Frame {frame_number:03d} | "
            f"Detections: {len(detections)} | "
            f"Tracks: {len(tracks)}"
        )

        for track in tracks:

            unique_track_ids.add(track.track_id)

            print(
                f"    ID: {track.track_id} | "
                f"Class: {track.class_name} | "
                f"Confidence: {track.confidence:.2f} | "
                f"BBox: {track.bounding_box}"
            )

        frame_number += 1

    cap.release()

    print("\n========== RESULTS ==========\n")

    print(f"Frames processed: {frame_number}")
    print(f"Total tracks: {total_tracks}")
    print(f"Unique IDs: {len(unique_track_ids)}")
    print(f"IDs: {sorted(unique_track_ids)}")

    if total_tracks == 0:
        print("\nFAILED: No tracks produced.")
    else:
        print(
            "\nSUCCESS: Tracking produced tracked objects."
        )


def test_history():

    detector, tracker, history_manager = create_pipeline()

    cap, fps = read_video()

    frame_number = 0

    print("\n========== HISTORY TEST ==========\n")
    print(f"Video FPS: {fps}")

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

        print(
            f"Frame {frame_number:03d} | "
            f"Tracks: {len(tracks)}"
        )

        frame_number += 1

    cap.release()

    print("\n========== HISTORY RESULTS ==========\n")

    histories = history_manager.get_all_histories()

    print(
        f"Frames processed: {frame_number}"
    )

    print(
        f"Unique tracked objects: {len(histories)}"
    )

    for track_id, history in histories.items():

        print(
            f"\nTrack ID: {track_id}"
        )

        print(
            f"Class: {history.class_name}"
        )

        print(
            f"Positions recorded: "
            f"{len(history.positions)}"
        )

        if history.positions:

            first = history.positions[0]
            last = history.positions[-1]

            print(
                f"First position: {first.center}"
            )

            print(
                f"Last position: {last.center}"
            )

            print(
                f"First timestamp: "
                f"{first.timestamp:.2f}s"
            )

            print(
                f"Last timestamp: "
                f"{last.timestamp:.2f}s"
            )


def main():

    print("\n")
    print("=" * 60)
    print("CORE TESTS")
    print("=" * 60)

    test_detection()
    test_tracking()
    test_history()


if __name__ == "__main__":
    main()