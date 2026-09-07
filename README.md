# Airport Security Monitoring System

An AI-powered computer vision system for analyzing airport surveillance footage, tracking people and objects, detecting potentially suspicious behavior, and generating visual security evidence.

---

## Overview

The Airport Security Monitoring System is a modular computer vision application designed to analyze surveillance video and identify security-related events.

The system combines:

- YOLO11 object detection
- ByteTrack multi-object tracking
- Track history management
- Spatial and zone analysis
- Temporal behavior analysis
- Security event detection
- Evidence snapshot generation
- Annotated video generation
- Streamlit dashboard

The project follows a modular pipeline rather than relying on one large neural network to detect every type of behavior.

```text
Surveillance Video
        │
        ▼
Object Detection
        │
        ▼
Object Tracking
        │
        ▼
Track History
        │
        ▼
Spatial / Zone Analysis
        │
        ▼
Temporal Behavior Analysis
        │
        ▼
Security Event Detection
        │
        ├───────────────┐
        ▼               ▼
Evidence           Annotated Video
        │               │
        └───────┬───────┘
                ▼
        Streamlit Dashboard
```

---

# Project Goals

The main objective of this project is to demonstrate how machine learning and traditional software engineering can be combined to build a practical computer vision security system.

The project focuses on:

- Object detection
- Multi-object tracking
- Computer vision
- Spatial reasoning
- Temporal reasoning
- Behavioral analysis
- Video processing
- Event detection
- Evidence generation
- Python software architecture
- Modular programming
- Automated testing
- Streamlit application development
- Machine learning deployment

---

# Key Features

## 1. Object Detection

The system uses YOLO11 through the Ultralytics framework to detect objects in surveillance footage.

Objects relevant to the security system include:

- Person
- Suitcase
- Backpack
- Handbag
- Bag

The YOLO model can also detect other object classes supported by its pretrained weights.

---

## 2. Multi-Object Tracking

The system uses ByteTrack to maintain object identities across frames.

Instead of treating every detection as a completely new object, the tracker attempts to maintain the same identity over time.

Example:

```text
Frame 1 → Person → Track ID 7
Frame 2 → Person → Track ID 7
Frame 3 → Person → Track ID 7
Frame 4 → Person → Track ID 7
```

This allows the behavior engine to reason about what an object has been doing over time.

---

## 3. Track History

The system stores historical information about tracked objects.

For example:

```text
Track ID: 7

Frame 1 → (400, 500)
Frame 2 → (410, 505)
Frame 3 → (425, 515)
Frame 4 → (440, 530)
```

This information can be used for:

- Movement analysis
- Direction analysis
- Stationary detection
- Loitering detection
- Zone visits
- Temporal behavior analysis

---

## 4. Security Zone Monitoring

The system supports spatial security analysis using defined regions.

Possible zone types include:

- Restricted
- Passenger
- Staff
- Boarding
- Baggage
- Custom
- None

The Streamlit dashboard provides a 3×3 zone configuration interface that allows the user to define the intended monitoring layout.

The core processing pipeline currently uses the configured restricted security region defined in the application logic.

---

## 5. Intrusion Detection

The system can detect people entering restricted areas.

Conceptually:

```text
Person detected
      ↓
Person tracked
      ↓
Person enters restricted zone
      ↓
Intrusion condition satisfied
      ↓
Intrusion Event
```

An intrusion event can contain information such as:

- Track ID
- Event type
- Zone
- Timestamp
- Duration
- Position

---

## 6. Loitering Detection

The system monitors how long tracked objects remain within monitored areas.

Conceptually:

```text
Person enters area
      ↓
Timer starts
      ↓
Person remains in area
      ↓
Duration threshold exceeded
      ↓
Loitering Event
```

The minimum loitering duration can be configured from the Streamlit dashboard.

---

## 7. Abandoned Object Detection

The system can monitor objects such as:

- Suitcases
- Backpacks
- Handbags
- Bags

A simplified abandoned-object workflow is:

```text
Object detected
      ↓
Object tracked
      ↓
Object movement decreases
      ↓
Object becomes stationary
      ↓
Stationary timer starts
      ↓
Threshold exceeded
      ↓
Potential Abandoned Object Event
```

The stationary duration and movement thresholds can be configured.

---

## 8. Wrong-Direction Detection

The system can analyze the direction in which tracked objects are moving.

The dashboard allows the user to configure an expected movement direction.

For example:

```text
Expected Direction → Right
```

If a tracked object moves sufficiently against the expected direction, the system can generate a wrong-direction event.

---

## 9. Access Control Analysis

The behavior engine contains access-control logic that can be used with monitored zones and security rules.

This provides a foundation for future extensions where different object types or people could have different permissions for specific areas.

---

## 10. Passenger Security Snapshots

When a person enters the configured security area, the system can generate an evidence snapshot.

Example:

```text
data/
└── outputs/
    └── snapshots/
        ├── passenger_track_1.jpg
        ├── passenger_track_5.jpg
        └── passenger_track_12.jpg
```

The generated snapshots contain visual annotations such as:

```text
SECURITY ZONE
TRACK ID: 12
```

The system generates a maximum of one passenger snapshot per tracked ID during a processing run.

---

## 11. Annotated Output Video

The system generates a processed video containing visual annotations.

Depending on the detected objects and their locations, the video can contain:

- Bounding boxes
- Object labels
- Tracking IDs
- Security zones
- Security status labels
- Movement-related information

The processed video can be viewed directly inside the Streamlit dashboard.

It can also be downloaded from the dashboard.

---

## 12. Streamlit Dashboard

The Streamlit application provides an interactive interface for the system.

The dashboard supports:

- Video upload
- Video preview
- Detection configuration
- Behavior configuration
- Movement configuration
- Zone configuration
- Video processing
- Detection statistics
- Tracking statistics
- Security event summaries
- Event details
- Passenger snapshots
- Evidence review
- Processed video playback
- Processed video download

---

# System Architecture

The project separates the major responsibilities into different modules.

```text
                         ┌──────────────────────┐
                         │  Surveillance Video  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       YOLO11         │
                         │  Object Detection    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      ByteTrack       │
                         │  Object Tracking     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Track History     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Spatial / Zone       │
                         │ Analysis             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Temporal Behavior    │
                         │ Analysis              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Security Event       │
                         │ Detection             │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │ Evidence        │   │ Annotated       │
                │ Snapshots       │   │ Video           │
                └────────┬────────┘   └────────┬────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Dashboard  │
                         └──────────────────────┘
```

---

# Processing Pipeline

The complete processing pipeline is:

```text
1. Read video frame
        ↓
2. Run YOLO11 detection
        ↓
3. Filter detections
        ↓
4. Run ByteTrack
        ↓
5. Assign tracking IDs
        ↓
6. Update track history
        ↓
7. Analyze object positions
        ↓
8. Analyze security zones
        ↓
9. Analyze movement
        ↓
10. Analyze temporal behavior
        ↓
11. Generate security events
        ↓
12. Generate evidence snapshots
        ↓
13. Draw annotations
        ↓
14. Write processed frame
        ↓
15. Continue until video ends
```

---

# Project Structure

```text
airport_security_system/
│
├── streamlit_app.py
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── detection/
│   │   └── detector.py
│   │
│   ├── tracking/
│   │   └── tracker.py
│   │
│   └── behaviour/
│       ├── __init__.py
│       ├── history.py
│       └── behaviour.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── db/
│   ├── __init__.py
│   └── database.py
│
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_behaviour.py
│
├── data/
│   ├── samples/
│   └── outputs/
│       └── snapshots/
│
├── models/
│   └── yolo11n.pt
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Directory Responsibilities

## `streamlit_app.py`

The main user interface.

Responsibilities include:

- Uploading videos
- Displaying the uploaded video
- Configuring processing parameters
- Configuring zones
- Running the security system
- Displaying results
- Displaying events
- Displaying evidence
- Playing the processed video
- Providing the processed-video download functionality

---

## `app/main.py`

Contains the main security-system orchestration logic.

Responsibilities include:

- Loading the detection model
- Initializing tracking
- Initializing behavior analysis
- Processing video frames
- Updating track history
- Checking security zones
- Generating snapshots
- Drawing annotations
- Generating the final processed video
- Returning processing results

---

## `app/detection/detector.py`

Contains the object detection implementation.

The module provides the `ObjectDetector` class and detection data structures.

It is responsible for:

- Loading YOLO
- Running inference
- Filtering detections
- Returning detection results

---

## `app/tracking/tracker.py`

Contains the object tracking implementation.

The module wraps ByteTrack and provides tracking information.

It is responsible for:

- Receiving detections
- Assigning tracking IDs
- Maintaining object identities

---

## `app/behaviour/history.py`

Maintains historical information about tracked objects.

It is used by the behavior engine to reason about object movement over time.

---

## `app/behaviour/behaviour.py`

Contains the behavior-analysis and event-detection logic.

The module includes components for:

- Zones
- Zone analysis
- Movement analysis
- Stationary detection
- Loitering detection
- Intrusion detection
- Access control
- Abandoned object detection
- Wrong-direction detection
- Security events

---

## `config/settings.py`

Contains configuration-related settings used by the project.

---

## `db/database.py`

Contains the project's database-related foundation for future persistence and event storage.

---

## `tests/`

Contains automated tests for core functionality and behavior logic.

---

## `data/samples/`

Used for local sample surveillance videos.

---

## `data/outputs/`

Used for generated processing results.

Generated files can include:

- Processed videos
- Evidence snapshots

---

## `models/`

Contains the YOLO model used by the application.

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO11 | Object detection |
| Ultralytics | YOLO implementation |
| ByteTrack | Multi-object tracking |
| Supervision | Computer vision utilities and tracking integration |
| OpenCV | Video processing |
| NumPy | Numerical operations |
| Pandas | Data handling and event presentation |
| Pillow | Image processing |
| Streamlit | Web dashboard |
| Pytest | Automated testing |
| Git | Version control |
| GitHub | Source-code hosting |

---

# Requirements

The project requires Python and the dependencies listed in `requirements.txt`.

Main dependencies include:

```text
streamlit
ultralytics
supervision
opencv-python-headless
numpy
pandas
Pillow
```

For local development, a virtual environment is recommended.

---

# Installation

## 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd airport_security_system
```

---

## 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# Model Setup

The project expects the YOLO model at:

```text
models/yolo11n.pt
```

Make sure the required model file is available before running the application.

The model file is intentionally excluded from Git through `.gitignore` because machine-learning model files can be large.

For deployment, the model should be provided through an appropriate model-storage or deployment mechanism rather than committing large binary files directly to the repository.

---

# Running the Application

Start the Streamlit dashboard with:

```bash
streamlit run streamlit_app.py
```

After starting Streamlit, open the URL displayed in the terminal.

The application will provide the complete user interface for uploading and analyzing surveillance videos.

---

# Running the Core Pipeline

The core processing pipeline can also be executed directly through:

```bash
python app/main.py
```

This mode is useful for local testing of the core security-processing pipeline without using the Streamlit interface.

---

# Using the Dashboard

## Step 1 — Upload a Video

Upload a surveillance video through the Streamlit interface.

The system accepts video files supported by OpenCV.

---

## Step 2 — Preview the Video

After uploading, the dashboard extracts and displays the first frame.

This allows the user to understand the camera view before configuring zones.

---

## Step 3 — Configure Processing Parameters

The dashboard provides configuration options for parameters such as:

- Detection confidence threshold
- Loitering duration
- Movement window
- Maximum movement speed
- Intrusion duration
- Exit grace period
- Abandoned-object stationary duration
- Abandoned-object movement threshold
- Expected movement direction
- Wrong-direction distance
- Wrong-direction duration

These parameters allow the system to be adjusted for different surveillance scenarios.

---

## Step 4 — Configure Zones

The dashboard provides a 3×3 zone layout.

Each grid cell can be assigned a zone type.

Available zone types include:

```text
None
Restricted
Passenger
Staff
Boarding
Baggage
Custom
```

The purpose of this interface is to provide a simple visual way of defining the intended security layout.

---

## Step 5 — Start Processing

Start the analysis from the dashboard.

The system will:

1. Read the video
2. Detect objects
3. Track objects
4. Maintain track history
5. Analyze movement
6. Analyze zones
7. Detect behaviors
8. Generate events
9. Generate evidence
10. Create an annotated video

---

## Step 6 — Review Results

After processing, the dashboard displays:

- Total detections
- Total tracked objects
- Total events
- Number of processed frames
- Detected security events
- Passenger snapshots
- Evidence snapshots
- Processed video

---

## Step 7 — Download the Processed Video

The dashboard provides a download button for the processed video.

The generated video is saved as:

```text
security_analysis.mp4
```

The video can be downloaded directly from the Streamlit interface.

---

# Output

After processing, the system can generate:

```text
data/
└── outputs/
    ├── security_analysis.mp4
    │
    └── snapshots/
        ├── passenger_track_1.jpg
        ├── passenger_track_2.jpg
        └── passenger_track_3.jpg
```

---

# Event Output

Security events are returned by the processing system.

Events can include:

- Loitering
- Intrusion
- Access-related events
- Abandoned object
- Wrong direction

The dashboard displays the detected events in a structured format.

Example:

```text
Event Type: Intrusion
Track ID: 12
Zone: Restricted
Timestamp: 00:01:24
```

The exact information depends on the detected event and processing conditions.

---

# Evidence Generation

Evidence generation is an important part of the system.

Instead of only reporting that an event occurred, the system can generate visual evidence associated with tracked people.

For example:

```text
Security Event
      ↓
Track ID identified
      ↓
Bounding box located
      ↓
Frame cropped
      ↓
Security annotation added
      ↓
Snapshot saved
```

Snapshots are stored under:

```text
data/outputs/snapshots/
```

---

# Configuration

The core system contains configurable parameters such as:

```python
CONFIDENCE_THRESHOLD = 0.25
MAX_FRAMES = None
FRAME_SKIP = 2
PERSON_CLASS = "person"

ABANDONED_OBJECT_CLASSES = {
    "suitcase",
    "backpack",
    "handbag",
    "bag",
}
```

The Streamlit dashboard exposes additional behavior-related settings to the user.

---

# Frame Skipping

The application supports frame skipping to reduce processing cost.

For example:

```text
FRAME_SKIP = 1
```

Processes every frame.

```text
FRAME_SKIP = 2
```

Processes every second frame.

```text
FRAME_SKIP = 3
```

Processes every third frame.

Increasing frame skipping can improve processing speed but may reduce temporal accuracy.

For security-sensitive applications, frame skipping should be selected carefully.

---

# Testing

The project contains automated tests under:

```text
tests/
```

Run the test suite using:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

Testing focuses on verifying core functionality and behavior-analysis logic.

---

# GitHub Setup

Before pushing the project to GitHub, verify that unnecessary generated files are excluded.

The repository should not contain:

- Virtual environments
- Python cache files
- Temporary files
- Generated output videos
- Generated evidence images
- Environment secrets
- Large machine-learning model binaries

The `.gitignore` file is configured to exclude these files.

---

# Git Commands

Initialize Git if required:

```bash
git init
```

Add the project files:

```bash
git add .
```

Create the first commit:

```bash
git commit -m "Initial airport security monitoring system"
```

Connect the local repository to GitHub:

```bash
git remote add origin YOUR_GITHUB_REPOSITORY_URL
```

Push the project:

```bash
git branch -M main
git push -u origin main
```

---

# Deployment

The Streamlit application can be deployed using a platform that supports Streamlit applications.

A typical deployment process is:

```text
GitHub Repository
       ↓
Cloud Deployment Platform
       ↓
Install requirements.txt
       ↓
Start streamlit_app.py
       ↓
Public Web Application
```

The deployment platform must have access to:

- Python
- Required Python packages
- YOLO model
- Sufficient CPU/GPU resources
- Sufficient memory
- Storage for temporary processing files

---

# Deployment Considerations

## Model Files

The YOLO model is not committed to the repository because model binaries can be large.

The deployment environment therefore needs another way to obtain the model.

Possible approaches include:

- Downloading the model during deployment
- Using external model storage
- Using a model registry
- Including the model through deployment-specific storage

---

## CPU vs GPU

YOLO inference can be computationally expensive.

A CPU deployment may work for smaller videos but can be considerably slower.

A GPU-enabled deployment is preferable when processing large amounts of surveillance footage or requiring near-real-time performance.

---

## Temporary Storage

Video processing generates temporary and output files.

A production deployment should define:

- Maximum upload size
- Maximum video duration
- Temporary file lifetime
- Output storage policy
- Evidence retention policy

---

# Performance Considerations

Video processing performance depends on:

- Video resolution
- Number of frames
- YOLO model size
- Detection confidence
- Frame skipping
- CPU/GPU availability
- Number of tracked objects
- Number of behavioral rules

For example:

```text
Higher resolution
      ↓
More pixels
      ↓
More computation
      ↓
Slower processing
```

Similarly:

```text
More frames processed
      ↓
More detection operations
      ↓
Higher processing time
```

Frame skipping can be used when processing speed is more important than analyzing every frame.

---

# Limitations

This project is a computer vision prototype and should not be treated as a complete real-world airport security solution.

Important limitations include:

- Object detection accuracy depends on the trained model.
- Tracking can fail when objects become heavily occluded.
- Poor lighting can reduce detection performance.
- Camera angle affects zone and movement analysis.
- Behavior rules are heuristic and depend on configured thresholds.
- False positives and false negatives are possible.
- Real airport environments are significantly more complex than controlled test footage.
- A production system would require extensive testing using representative surveillance data.
- Current zone configuration functionality is primarily designed as a dashboard interface, while the core processing pipeline currently relies on its configured security-region logic.
- The system does not establish that a person or object is actually dangerous merely because an event is detected.

Security events should therefore be treated as potential events requiring human review rather than automatic conclusions.

---

# Responsible Use

This project is intended for educational, research, prototyping, and portfolio purposes.

Automated computer vision should assist human operators rather than replace appropriate human judgment in high-stakes security environments.

A production security system would require:

- Extensive validation
- Human oversight
- False-positive analysis
- False-negative analysis
- Security testing
- Reliability testing
- Privacy safeguards
- Access controls
- Audit logging
- Appropriate legal and regulatory review

---

# Security and Privacy

Surveillance footage may contain sensitive personal information.

When using this project:

- Use footage that you have permission to process.
- Do not upload private surveillance footage to untrusted services.
- Protect generated evidence.
- Avoid exposing surveillance data publicly.
- Do not commit sensitive footage to GitHub.
- Do not commit credentials or API keys.
- Implement appropriate access controls for production deployments.
- Define appropriate data-retention policies.

Generated videos and snapshots should remain excluded from version control unless there is a specific reason to include them.

---

# Future Improvements

Potential future improvements include:

## Real-Time CCTV Processing

Support live camera streams instead of uploaded video files.

```text
CCTV Camera
     ↓
Video Stream
     ↓
YOLO Detection
     ↓
Tracking
     ↓
Behavior Analysis
     ↓
Real-Time Alerts
```

---

## Advanced Zone Configuration

Allow users to draw arbitrary polygons directly on the video rather than relying primarily on a fixed region or grid.

---

## Database Event Storage

Persist security events in a database.

Possible architecture:

```text
Detection
    ↓
Event
    ↓
Database
    ↓
Dashboard
```

---

## Alerting

Add notifications for important events.

Potential channels include:

- Email
- Web notifications
- SMS
- Internal security dashboards
- Messaging systems

---

## Real-Time Alerts

Instead of analyzing recorded footage only, events could trigger immediate alerts.

Example:

```text
Restricted Zone Intrusion
        ↓
Event Generated
        ↓
Alert Service
        ↓
Security Operator
```

---

## Improved Tracking

Potential future improvements include:

- Stronger multi-object tracking
- Re-identification
- Occlusion handling
- Track recovery
- Cross-camera tracking

---

## Better Behavioral Models

The current architecture primarily relies on explicit rules.

Future versions could incorporate learned behavioral models for:

- Crowd analysis
- Anomalous movement
- Unusual trajectories
- Crowd formation
- Panic-like movement
- Advanced abandoned-object reasoning

---

## Production MLOps

A production version could include:

- Docker
- CI/CD
- Model versioning
- Experiment tracking
- Monitoring
- Logging
- Metrics
- Cloud deployment
- Kubernetes
- Model serving
- Automated testing
- Model performance monitoring

---

# Development Philosophy

The project follows a simple development philosophy:

```text
Build
  ↓
Test
  ↓
Verify
  ↓
Works
  ↓
Move On
```

The system is intentionally modular.

Each major responsibility should have a clear location in the codebase.

The goal is to avoid creating unnecessary files or unnecessary abstractions.

A small feature that naturally belongs inside an existing module should generally be implemented there rather than creating another module without a clear reason.

---

# Design Principles

## Separation of Responsibilities

Each component has a specific responsibility.

```text
Detector
   ↓
Find objects

Tracker
   ↓
Maintain identities

History
   ↓
Remember movement

Behavior Engine
   ↓
Interpret movement

Main System
   ↓
Orchestrate everything

Streamlit
   ↓
Provide user interface
```

---

## Detection Is Not Behavior

A detection only answers:

```text
"What is visible?"
```

For example:

```text
Person
Suitcase
Backpack
```

Behavior analysis asks:

```text
"What is this tracked object doing?"
```

For example:

```text
Is it entering a restricted area?
Is it staying too long?
Is it moving in the wrong direction?
Has it stopped moving?
```

This distinction is fundamental to the architecture.

---

# Why Tracking Is Important

Without tracking:

```text
Frame 1 → Person
Frame 2 → Person
Frame 3 → Person
```

The system cannot reliably know whether these detections belong to the same person.

With tracking:

```text
Frame 1 → Person → ID 7
Frame 2 → Person → ID 7
Frame 3 → Person → ID 7
```

Now the system can reason about the person's behavior over time.

---

# Why Temporal Reasoning Is Important

Many security behaviors cannot be identified from a single frame.

For example, a stationary suitcase may simply be a suitcase being carried.

A temporal system can reason:

```text
Object appears
     ↓
Object remains in approximately same location
     ↓
Movement remains below threshold
     ↓
Stationary duration increases
     ↓
Potential abandoned object
```

This is why the project combines computer vision with traditional programming logic.

---

# Example System Flow

A simplified example:

```text
Person enters camera view
        ↓
YOLO detects person
        ↓
ByteTrack assigns Track ID 15
        ↓
Track history begins
        ↓
Person moves toward restricted zone
        ↓
Position is analyzed
        ↓
Person enters restricted region
        ↓
Intrusion rule evaluates condition
        ↓
Security event generated
        ↓
Evidence snapshot generated
        ↓
Annotated frame written
        ↓
Event displayed in dashboard
```

---

# Portfolio Value

This project demonstrates more than simply training a machine-learning model.

It demonstrates the integration of:

```text
Machine Learning
      +
Computer Vision
      +
Python Engineering
      +
Object Tracking
      +
Algorithmic Logic
      +
Software Architecture
      +
Testing
      +
User Interface
      +
Deployment
```

This makes the project useful as a portfolio demonstration of practical ML engineering skills.



# Skills Demonstrated

By building and deploying this project, the following skills are demonstrated:

### Machine Learning

- Object detection
- Model inference
- Model integration
- Computer vision

### Python

- Classes
- Dataclasses
- Modules
- Packages
- Type hints
- File handling
- Error handling
- Object-oriented programming

### Computer Vision

- Bounding boxes
- Video processing
- Frame processing
- Image cropping
- Spatial reasoning
- Object tracking

### Algorithms

- Tracking
- Distance calculations
- Movement analysis
- Direction analysis
- Time-based rules
- Threshold-based event detection

### Software Engineering

- Modular architecture
- Separation of concerns
- Configuration
- Testing
- Git
- GitHub
- Project organization

### Deployment

- Streamlit
- Dependency management
- Environment configuration
- Cloud deployment preparation

---

# Project Status

Current implementation includes:

-  YOLO11 object detection
-  ByteTrack object tracking
-  Track history
-  Zone analysis
-  Loitering detection
-  Intrusion detection
-  Access-control logic
-  Abandoned-object detection
-  Wrong-direction detection
-  Passenger security snapshots
-  Annotated output video
-  Streamlit dashboard
-  Video upload
-  Video preview
-  Event reporting
-  Evidence display
-  Processed video download
-  Automated tests
-  GitHub-ready project structure

---

# Roadmap

## Phase 1 — Core Computer Vision

-  Object detection
-  Object tracking
-  Track history
-  Video processing

## Phase 2 — Behavior Analysis

-  Zone analysis
-  Loitering detection
-  Intrusion detection
-  Abandoned object detection
-  Wrong-direction detection
-  Access-control logic

## Phase 3 — Evidence

-  Security snapshots
-  Annotated video
-  Event reporting

## Phase 4 — Dashboard

-  Streamlit interface
-  Video upload
-  Configuration
-  Results dashboard
-  Evidence display
-  Video download

## Phase 5 — Deployment

-  GitHub repository
-  Cloud deployment
-  Production configuration
-  Model hosting
-  Resource optimization

## Phase 6 — Production Improvements

-  Real-time CCTV streams
-  Advanced polygon drawing
-  Persistent event database
-  Alert system
-  Authentication
-  Logging
-  Monitoring
-  Docker
-  CI/CD
-  Cloud infrastructure
-  Kubernetes
-  Advanced behavioral models

---

# Troubleshooting

## Streamlit Does Not Start

Make sure the virtual environment is activated and dependencies are installed.

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run streamlit_app.py
```

---

## YOLO Model Not Found

Verify that the model exists at:

```text
models/yolo11n.pt
```

If the model is not present, provide it through the appropriate model-download or model-storage mechanism.

---

## Video Cannot Be Processed

Check:

- The video file is valid.
- OpenCV can read the video.
- The file format is supported.
- The video is not corrupted.
- The system has sufficient disk space.

---

## Processing Is Too Slow

Try:

- Increasing frame skip.
- Using a smaller YOLO model.
- Reducing video resolution.
- Using a GPU.
- Processing shorter videos.

For example:

```text
FRAME_SKIP = 2
```

processes approximately every second frame.

---

## No Security Events Are Detected

Behavior detection depends heavily on:

- Object detection accuracy
- Tracking quality
- Zone configuration
- Threshold configuration
- Camera angle
- Video quality
- Object movement

Try testing with a clear video where the expected behavior is obvious.



# Acknowledgements

This project uses open-source technologies and libraries including:

- Ultralytics YOLO
- Supervision
- OpenCV
- NumPy
- Pandas
- Pillow
- Streamlit
- Pytest

The project is intended as a practical demonstration of computer vision, machine learning engineering, and Python software development.


# Final Architecture

The complete project can be summarized as:

```text
                    AIRPORT SECURITY SYSTEM
                              │
                              ▼
                       Surveillance Video
                              │
                              ▼
                       ┌──────────────┐
                       │    YOLO11    │
                       │   Detection  │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  ByteTrack   │
                       │   Tracking   │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │Track History │
                       └──────┬───────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Spatial Reasoning  │
                    │   + Zone Analysis  │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Temporal Behavior  │
                    │      Analysis       │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  Security Events   │
                    └──────────┬─────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Evidence                  Video
             Snapshots               Annotation
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌────────────────────┐
                    │ Streamlit Dashboard│
                    └────────────────────┘
```