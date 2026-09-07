from pathlib import Path
import tempfile

import cv2
import streamlit as st

from app.main import AirportSecuritySystem


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airport Security System",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "outputs"
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
# SESSION STATE
# ============================================================

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "uploaded_video_name" not in st.session_state:
    st.session_state.uploaded_video_name = None

if "first_frame" not in st.session_state:
    st.session_state.first_frame = None

if "results" not in st.session_state:
    st.session_state.results = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_video(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_file.write(
        uploaded_file.read()
    )

    temporary_file.close()

    return Path(
        temporary_file.name
    )


def extract_first_frame(video_path):

    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():
        return None

    success, frame = video.read()

    video.release()

    if not success:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return frame


def create_grid_zones(
    frame,
    rows=3,
    columns=3,
):

    height, width = (
        frame.shape[:2]
    )

    zones = []

    zone_number = 1

    for row in range(rows):

        for column in range(columns):

            x1 = int(
                column
                * width
                / columns
            )

            x2 = int(
                (column + 1)
                * width
                / columns
            )

            y1 = int(
                row
                * height
                / rows
            )

            y2 = int(
                (row + 1)
                * height
                / rows
            )

            polygon = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]

            zones.append(
                {
                    "id": zone_number,
                    "row": row,
                    "column": column,
                    "polygon": polygon,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

            zone_number += 1

    return zones


def draw_grid(
    frame,
    zones,
):

    image = frame.copy()

    for zone in zones:

        x1 = zone["x1"]
        y1 = zone["y1"]
        x2 = zone["x2"]
        y2 = zone["y2"]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

        label = (
            f"ZONE {zone['id']}"
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2,
        )[0]

        padding_x = 12
        padding_y = 10

        label_width = (
            text_size[0]
            + padding_x * 2
        )

        label_height = (
            text_size[1]
            + padding_y * 2
        )

        label_x = x1 + 10
        label_y = y1 + 10

        cv2.rectangle(
            image,
            (
                label_x,
                label_y,
            ),
            (
                label_x + label_width,
                label_y + label_height,
            ),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            label,
            (
                label_x + padding_x,
                label_y
                + text_size[1]
                + padding_y // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def build_zone_configuration(
    frame,
):

    zones = create_grid_zones(
        frame
    )

    zone_configuration = []

    for zone in zones:

        selected_type = (
            st.session_state.get(
                f"zone_type_{zone['id']}",
                "none",
            )
        )

        if selected_type != "none":

            zone_configuration.append(
                {
                    "name":
                        f"Zone {zone['id']}",

                    "type":
                        selected_type,

                    "polygon":
                        zone["polygon"],
                }
            )

    return zone_configuration


def get_event_value(
    event,
    key,
    default=None,
):

    if isinstance(
        event,
        dict,
    ):

        return event.get(
            key,
            default,
        )

    return getattr(
        event,
        key,
        default,
    )


def display_video(
    video_path,
):

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        return False

    try:

        video_bytes = (
            video_path.read_bytes()
        )

        if not video_bytes:
            return False

        st.video(
            video_bytes,
            format="video/mp4",
        )

        return True

    except Exception as error:

        st.error(
            f"Could not display processed video: {error}"
        )

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "Security Settings"
    )

    st.caption(
        "Configure the detection and behaviour thresholds."
    )

    # ========================================================
    # DETECTION
    # ========================================================

    st.subheader(
        "Detection"
    )

    confidence_threshold = st.number_input(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        format="%.2f",
    )

    # ========================================================
    # LOITERING
    # ========================================================

    st.divider()

    st.subheader(
        "Loitering"
    )

    loitering_min_duration = st.number_input(
        "Minimum duration (seconds)",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )

    loitering_window = st.number_input(
        "Movement window (seconds)",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )

    loitering_max_speed = st.number_input(
        "Maximum movement speed (pixels/sec)",
        min_value=0.0,
        value=25.0,
        step=1.0,
    )

    # ========================================================
    # INTRUSION
    # ========================================================

    st.divider()

    st.subheader(
        "Zone / Intrusion"
    )

    intrusion_min_duration = st.number_input(
        "Minimum zone duration (seconds)",
        min_value=0.0,
        value=0.5,
        step=0.1,
    )

    exit_grace_period = st.number_input(
        "Exit grace period (seconds)",
        min_value=0.0,
        value=0.5,
        step=0.1,
    )

    # ========================================================
    # ABANDONED OBJECT
    # ========================================================

    st.divider()

    st.subheader(
        "Abandoned Object"
    )

    abandoned_stationary_duration = st.number_input(
        "Stationary duration (seconds)",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )

    abandoned_max_movement = st.number_input(
        "Maximum movement (pixels)",
        min_value=0.0,
        value=30.0,
        step=1.0,
    )

    # ========================================================
    # WRONG DIRECTION
    # ========================================================

    st.divider()

    st.subheader(
        "Wrong Direction"
    )

    expected_direction = st.selectbox(
        "Expected direction",
        [
            "left_to_right",
            "right_to_left",
            "top_to_bottom",
            "bottom_to_top",
        ],
    )

    wrong_direction_min_distance = st.number_input(
        "Minimum movement distance (pixels)",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    wrong_direction_min_duration = st.number_input(
        "Minimum movement duration (seconds)",
        min_value=0.0,
        value=1.0,
        step=0.5,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Airport Security Monitoring System"
)

st.write(
    "Upload surveillance footage, configure security "
    "thresholds, assign security types to zones, "
    "and run the analysis."
)


# ============================================================
# STEP 1 — UPLOAD VIDEO
# ============================================================

st.header(
    "1. Upload Surveillance Video"
)

uploaded_video = st.file_uploader(
    "Choose a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_video is not None:

    if (
        st.session_state.video_path is None
        or
        st.session_state.uploaded_video_name
        != uploaded_video.name
    ):

        video_path = (
            save_uploaded_video(
                uploaded_video
            )
        )

        st.session_state.video_path = (
            video_path
        )

        st.session_state.uploaded_video_name = (
            uploaded_video.name
        )

        st.session_state.first_frame = (
            extract_first_frame(
                video_path
            )
        )

        st.session_state.results = None


# ============================================================
# VIDEO PREVIEW
# ============================================================

if (
    st.session_state.first_frame
    is not None
):

    st.subheader(
        "Video Preview"
    )

    st.caption(
        "This is the first frame of your uploaded video."
    )

    st.image(
        st.session_state.first_frame,
        width="stretch",
    )


# ============================================================
# STEP 2 — CONFIGURE ZONES
# ============================================================

if (
    st.session_state.first_frame
    is not None
):

    st.header(
        "2. Configure Security Zones"
    )

    st.info(
        """
        ### How zones work

        The video is automatically divided into **9 zones**.

        You do **not** need to draw anything.

        Simply look at the image and decide what each area
        represents.

        For example:

        **Zone 1 → Restricted**

        **Zone 2 → Passenger**

        **Zone 3 → Staff**

        **Zone 4 → Boarding**

        **Zone 5 → Baggage**

        If an area does not need monitoring, select
        **None**.
        """
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid_zones = create_grid_zones(
        st.session_state.first_frame
    )

    grid_image = draw_grid(
        st.session_state.first_frame,
        grid_zones,
    )

    st.subheader(
        "Security Zone Map"
    )

    st.caption(
        "Use this numbered map to identify each area."
    )

    st.image(
        grid_image,
        width="stretch",
    )

    # --------------------------------------------------------
    # Zone assignment
    # --------------------------------------------------------

    st.subheader(
        "Assign Zone Types"
    )

    zone_types = [
        "none",
        "restricted",
        "passenger",
        "staff",
        "boarding",
        "baggage",
        "custom",
    ]

    zone_columns = st.columns(
        3
    )

    for index, zone in enumerate(
        grid_zones
    ):

        column = (
            zone_columns[
                index % 3
            ]
        )

        with column:

            st.selectbox(
                f"Zone {zone['id']}",
                zone_types,
                key=(
                    f"zone_type_"
                    f"{zone['id']}"
                ),
            )


# ============================================================
# BUILD ZONE CONFIGURATION
# ============================================================

zone_configuration = []

if (
    st.session_state.first_frame
    is not None
):

    zone_configuration = (
        build_zone_configuration(
            st.session_state.first_frame
        )
    )


# ============================================================
# SHOW SELECTED ZONES
# ============================================================

if zone_configuration:

    st.subheader(
        "Selected Security Zones"
    )

    for zone in zone_configuration:

        st.write(
            f"**{zone['name']}** → "
            f"`{zone['type']}`"
        )

else:

    if (
        st.session_state.first_frame
        is not None
    ):

        st.warning(
            "No security zones are currently enabled. "
            "Select a zone type above."
        )


# ============================================================
# STEP 3 — POLICY SUMMARY
# ============================================================

st.header(
    "3. Security Policy Summary"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Active Zones",
        len(zone_configuration),
    )

with col2:

    st.metric(
        "Confidence",
        f"{confidence_threshold:.2f}",
    )

with col3:

    st.metric(
        "Loitering",
        f"{loitering_min_duration:.1f}s",
    )

with col4:

    st.metric(
        "Intrusion",
        f"{intrusion_min_duration:.1f}s",
    )


# ============================================================
# STEP 4 — RUN ANALYSIS
# ============================================================

st.header(
    "4. Run Security Analysis"
)

if (
    st.session_state.video_path
    is None
):

    st.warning(
        "Please upload a video first."
    )

else:

    st.success(
        f"Ready to analyze "
        f"'{st.session_state.uploaded_video_name}'."
    )


run_analysis = st.button(
    "Run Security Analysis",
    type="primary",
    use_container_width=True,
)


if run_analysis:

    if (
        st.session_state.video_path
        is None
    ):

        st.error(
            "Please upload a video first."
        )

    else:

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        try:

            # ------------------------------------------------
            # Configuration
            # ------------------------------------------------

            configuration = {

                "confidence_threshold":
                    confidence_threshold,

                "loitering": {

                    "min_duration":
                        loitering_min_duration,

                    "movement_window":
                        loitering_window,

                    "max_speed":
                        loitering_max_speed,
                },

                "intrusion": {

                    "min_duration":
                        intrusion_min_duration,

                    "exit_grace_period":
                        exit_grace_period,
                },

                "abandoned_object": {

                    "stationary_duration":
                        abandoned_stationary_duration,

                    "max_movement":
                        abandoned_max_movement,
                },

                "wrong_direction": {

                    "expected_direction":
                        expected_direction,

                    "min_distance":
                        wrong_direction_min_distance,

                    "min_duration":
                        wrong_direction_min_duration,
                },

                "zones":
                    zone_configuration,
            }

            # ------------------------------------------------
            # Initialize system
            # ------------------------------------------------

            status_text.write(
                "Initializing security system..."
            )

            system = AirportSecuritySystem(
                model_path=(
                    ROOT_DIR
                    / "models"
                    / "yolo11n.pt"
                ),
                confidence_threshold=(
                    confidence_threshold
                ),
            )

            # ------------------------------------------------
            # Output path
            # ------------------------------------------------

            output_video = (
                OUTPUT_DIR
                / "security_analysis.mp4"
            )

            # ------------------------------------------------
            # Run engine
            # ------------------------------------------------

            status_text.write(
                "Processing video..."
            )

            results = system.run(
                video_path=(
                    st.session_state.video_path
                ),
                output_path=output_video,
                frame_skip=2,
            )

            st.session_state.results = (
                results
            )

            progress_bar.progress(
                1.0
            )

            status_text.success(
                "Analysis completed successfully."
            )

        except Exception as error:

            progress_bar.empty()

            status_text.empty()

            st.error(
                f"Analysis failed: {error}"
            )

            st.exception(
                error
            )


# ============================================================
# STEP 5 — SECURITY DASHBOARD
# ============================================================

if (
    st.session_state.results
    is not None
):

    results = (
        st.session_state.results
    )

    st.header(
        "5. Security Dashboard"
    )

    # --------------------------------------------------------
    # Compatibility
    # --------------------------------------------------------

    if isinstance(
        results,
        list,
    ):

        events = results

        results = {

            "total_detections":
                0,

            "total_tracks":
                0,

            "total_events":
                len(events),

            "total_frames":
                0,

            "events":
                events,

            "output_video":
                str(
                    OUTPUT_DIR
                    / "security_analysis.mp4"
                ),

            "passenger_snapshots":
                [],
        }

    else:

        events = results.get(
            "events",
            [],
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_detections = results.get(
        "total_detections",
        0,
    )

    total_tracks = results.get(
        "total_tracks",
        0,
    )

    total_events = results.get(
        "total_events",
        len(events),
    )

    total_frames = results.get(
        "total_frames",
        0,
    )

    passenger_snapshots = (
        results.get(
            "passenger_snapshots",
            [],
        )
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Detections",
            total_detections,
        )

    with col2:

        st.metric(
            "Tracked Objects",
            total_tracks,
        )

    with col3:

        st.metric(
            "Security Events",
            total_events,
        )

    with col4:

        st.metric(
            "Frames Processed",
            total_frames,
        )

    # ========================================================
    # PASSENGERS INSIDE SECURITY ZONE
    # ========================================================

    st.subheader(
        "Passengers Detected in Security Zone"
    )

    if passenger_snapshots:

        st.success(
            f"{len(passenger_snapshots)} "
            f"passenger(s) detected inside "
            f"the security zone."
        )

        snapshot_columns = st.columns(
            3
        )

        for index, snapshot_path in enumerate(
            passenger_snapshots
        ):

            snapshot_path = Path(
                snapshot_path
            )

            if not snapshot_path.exists():
                continue

            with snapshot_columns[
                index % 3
            ]:

                st.image(
                    snapshot_path,
                    caption=(
                        f"Passenger "
                        f"{index + 1}"
                    ),
                    width="stretch",
                )

    else:

        st.info(
            "No passenger snapshots were generated."
        )

    # ========================================================
    # EVENT SUMMARY
    # ========================================================

    st.subheader(
        "Event Summary"
    )

    event_counts = {}

    for event in events:

        event_type = (
            get_event_value(
                event,
                "event_type",
                None,
            )
        )

        if not event_type:

            event_type = (
                event.__class__.__name__
            )

        event_counts[event_type] = (
            event_counts.get(
                event_type,
                0,
            )
            + 1
        )

    if event_counts:

        event_columns = st.columns(
            len(event_counts)
        )

        for column, (
            event_type,
            count,
        ) in zip(
            event_columns,
            event_counts.items(),
        ):

            with column:

                st.metric(
                    event_type,
                    count,
                )

    else:

        st.success(
            "No security events detected."
        )

    # ========================================================
    # SECURITY EVENTS
    # ========================================================

    st.subheader(
        "Security Events"
    )

    if events:

        event_rows = []

        for event in events:

            event_type = (
                get_event_value(
                    event,
                    "event_type",
                    None,
                )
            )

            if not event_type:

                event_type = (
                    event.__class__.__name__
                )

            event_rows.append(
                {
                    "Event":
                        event_type,

                    "Track ID":
                        get_event_value(
                            event,
                            "track_id",
                            "",
                        ),

                    "Zone":
                        get_event_value(
                            event,
                            "zone_name",
                            "",
                        ),

                    "Timestamp":
                        get_event_value(
                            event,
                            "timestamp",
                            "",
                        ),

                    "Duration":
                        get_event_value(
                            event,
                            "duration",
                            "",
                        ),

                    "Confidence":
                        get_event_value(
                            event,
                            "confidence",
                            "",
                        ),
                }
            )

        st.dataframe(
            event_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No security events detected."
        )

    # ========================================================
    # EVENT DETAILS
    # ========================================================

    if events:

        st.subheader(
            "Event Details"
        )

        for index, event in enumerate(
            events,
            start=1,
        ):

            event_type = (
                get_event_value(
                    event,
                    "event_type",
                    None,
                )
            )

            if not event_type:

                event_type = (
                    event.__class__.__name__
                )

            track_id = (
                get_event_value(
                    event,
                    "track_id",
                    "N/A",
                )
            )

            with st.expander(
                f"{index}. {event_type} "
                f"— Track {track_id}"
            ):

                timestamp = (
                    get_event_value(
                        event,
                        "timestamp",
                        "N/A",
                    )
                )

                zone_name = (
                    get_event_value(
                        event,
                        "zone_name",
                        None,
                    )
                )

                duration = (
                    get_event_value(
                        event,
                        "duration",
                        None,
                    )
                )

                confidence = (
                    get_event_value(
                        event,
                        "confidence",
                        None,
                    )
                )

                snapshot_path = (
                    get_event_value(
                        event,
                        "snapshot_path",
                        None,
                    )
                )

                st.write(
                    f"**Timestamp:** "
                    f"{timestamp}"
                )

                st.write(
                    f"**Track ID:** "
                    f"{track_id}"
                )

                if zone_name:

                    st.write(
                        f"**Zone:** "
                        f"{zone_name}"
                    )

                if duration is not None:

                    st.write(
                        f"**Duration:** "
                        f"{duration} seconds"
                    )

                if confidence is not None:

                    st.write(
                        f"**Confidence:** "
                        f"{confidence}"
                    )

                if snapshot_path:

                    snapshot_path = Path(
                        snapshot_path
                    )

                    if snapshot_path.exists():

                        st.image(
                            snapshot_path,
                            caption=(
                                "Evidence snapshot"
                            ),
                            width="stretch",
                        )

    # ========================================================
    # PROCESSED VIDEO
    # ========================================================

    st.subheader(
        "Processed Tracking Video"
    )

    st.caption(
        "This video contains the YOLO detections, "
        "tracking IDs, security-zone boundary, "
        "and highlighted security detections."
    )

    output_video = results.get(
        "output_video",
        None,
    )

    if output_video:

        output_video = Path(
            output_video
        )

    else:

        output_video = (
            OUTPUT_DIR
            / "security_analysis.mp4"
        )

    # --------------------------------------------------------
    # Check processed video
    # --------------------------------------------------------

    if output_video.exists():

        st.success(
            "Processed video is ready."
        )

        # ----------------------------------------------------
        # Video player
        # ----------------------------------------------------

        display_video(
            output_video
        )

        # ----------------------------------------------------
        # DOWNLOAD BUTTON
        # ----------------------------------------------------

        st.download_button(
            label="Download Processed Video",
            data=output_video.read_bytes(),
            file_name="security_analysis.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        st.caption(
            f"File: {output_video.name} "
            f"({output_video.stat().st_size / (1024 * 1024):.2f} MB)"
        )

    else:

        st.error(
            "The analysis completed, but the "
            "processed video could not be found."
        )

    # ========================================================
    # ALL EVIDENCE SNAPSHOTS
    # ========================================================

    st.subheader(
        "Evidence Snapshots"
    )

    all_snapshot_paths = []

    # --------------------------------------------------------
    # Passenger snapshots
    # --------------------------------------------------------

    for snapshot_path in (
        passenger_snapshots
    ):

        snapshot_path = Path(
            snapshot_path
        )

        if (
            snapshot_path.exists()
            and snapshot_path
            not in all_snapshot_paths
        ):

            all_snapshot_paths.append(
                snapshot_path
            )

    # --------------------------------------------------------
    # Event snapshots
    # --------------------------------------------------------

    for event in events:

        snapshot_path = (
            get_event_value(
                event,
                "snapshot_path",
                None,
            )
        )

        if snapshot_path:

            snapshot_path = Path(
                snapshot_path
            )

            if (
                snapshot_path.exists()
                and snapshot_path
                not in all_snapshot_paths
            ):

                all_snapshot_paths.append(
                    snapshot_path
                )

    if all_snapshot_paths:

        snapshot_columns = st.columns(
            3
        )

        for index, snapshot_path in enumerate(
            all_snapshot_paths
        ):

            with snapshot_columns[
                index % 3
            ]:

                st.image(
                    snapshot_path,
                    caption=snapshot_path.name,
                    width="stretch",
                )

    else:

        st.info(
            "No evidence snapshots were generated."
        )