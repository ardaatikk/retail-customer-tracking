# Retail Customer Tracking & Behavior Analytics

[![CI](https://github.com/ardaatikk/retail-customer-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/ardaatikk/retail-customer-tracking/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-blueviolet)](https://docs.ultralytics.com/models/yolo11/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-44%20Passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An end-to-end computer vision pipeline for tracking customers in indoor environments and transforming raw video into interpretable behavioral analytics.

The current implementation provides **single-camera customer tracking**, zone-based behavior analysis, dwell-time measurement, trajectory reconstruction, and occupancy heatmaps. The architecture is designed to be extended toward **multi-camera identity association and global customer analytics**.

---

## Overview

Traditional object detection can answer **where a person is in a frame**. This project extends that idea into a temporal analytics system capable of answering questions such as:

- How many customers were observed?
- How long did each customer remain visible?
- Which areas did they visit?
- How much time was spent in each zone?
- What path did each customer follow?
- Which areas accumulated the most customer presence?

The pipeline combines pretrained person detection, multi-object tracking, customer-state management, spatial zones, and analytics into a reproducible workflow.

---

## Features

- Person detection using **pretrained YOLO11n**
- Multi-object tracking using **ByteTrack**
- Persistent local track IDs
- Customer confirmation to reduce short-lived false detections
- Ground-point estimation from person bounding boxes
- Camera-specific polygonal zones
- Zone transition tracking
- Per-zone dwell-time calculation
- Customer trajectory reconstruction
- Occupancy / dwell heatmap generation
- Per-customer behavior summaries
- Per-zone analytics
- Parquet-based structured outputs
- Reproducible CITIC-MTMC dataset download
- Automated unit testing with **44 tests**
- GitHub Actions continuous integration
- Architecture prepared for future multi-camera tracking

> The detector is **not fine-tuned** in the current version. The pipeline uses pretrained YOLO11n weights and restricts detection to the person class.

---

## System Architecture

```text
                    Video
                      │
                      ▼
             Pretrained YOLO11n
              Person Detection
                      │
                      ▼
                  ByteTrack
                      │
                      ▼
               local_track_id
                      │
                      ▼
            Customer State Manager
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Zone Assignment    Track Storage
             │                 │
             └────────┬────────┘
                      ▼
                 Analytics
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Trajectory    Heatmap     Summaries
```

Detection, tracking, state management, zone assignment, persistence, and analytics are separated into independent modules so that individual components can be extended without redesigning the complete pipeline.

---

## Example Results

The following outputs were generated from the `Showroom` camera of the CITIC-MTMC dataset.

### Customer Trajectories

Customer movement is reconstructed using the bottom-center point of each tracked bounding box as an approximation of the person's position on the ground plane.

![Customer Trajectory](artifacts/figures/capture1_Showroom_trajectory.png)

### Occupancy / Dwell Heatmap

Each valid customer position contributes to the heatmap. Areas with higher accumulated presence therefore represent locations where customers spent more observed time.

![Customer Heatmap](artifacts/figures/capture1_Showroom_heatmap.png)

---

## Example Analytics

For the current `capture1 / Showroom` experiment:

| Metric | Result |
|---|---:|
| Confirmed customers | 2 |
| Confirmed track points | 944 |
| Spatial analytics points | 696 |
| Bottom-truncated points excluded from spatial analytics | 248 |

### Customer Behavior

| Customer | Observed Time | Zone Path |
|---|---:|---|
| Person #1 | 19.44 s | Entrance → Main Area → Demo Area → Main Area → Entrance |
| Person #2 | 19.92 s | Entrance → Main Area → Demo Area → Main Area → Entrance |

### Zone Analytics

| Zone | Customers | Total Dwell Time | Average Dwell Time |
|---|---:|---:|---:|
| Entrance | 2 | 10.04 s | 5.02 s |
| Main Area | 2 | 18.00 s | 9.00 s |
| Demo Area | 2 | 11.32 s | 5.66 s |

The total measured zone dwell time equals the total confirmed customer observation time for this experiment.

---

## Dataset

This project currently uses the **CITIC-MTMC** indoor multi-camera tracking dataset.

The dataset contains multiple synchronized indoor camera views and provides a suitable environment for developing and evaluating tracking and spatial analytics pipelines.

Current development uses:

```text
capture1
└── Showroom
```

The dataset itself is **not included in this repository**.

Dataset resources:

- Repository: `https://github.com/GTEC-UDC/citic-mtmc-dataset`
- DOI: `10.5281/zenodo.17511785`

The project includes a download utility that retrieves the dataset archive from Zenodo.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ardaatikk/retail-customer-tracking.git
cd retail-customer-tracking
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Main dependencies include:

- Ultralytics
- OpenCV
- Pandas
- PyArrow
- Requests
- LAP
- Pytest

YOLO11n weights are not stored in the repository.

---

## Dataset Setup

Download the CITIC-MTMC archive:

```bash
python -m src.data.download
```

After downloading, extract the dataset under:

```text
data/raw/citic_mtmc/dataset/
```

The expected structure includes the capture directories and camera videos.

Validate the local dataset:

```bash
python -m src.data.validate
```

A valid installation should detect:

```text
Captures: 3
Cameras per capture: 5
Videos: 15
Ground-truth files: 3
CITIC-MTMC dataset validation passed.
```

Raw datasets are excluded from version control.

---

## Zone Configuration

Zones are represented as camera-specific polygons and stored as JSON configuration files.

The repository already includes the Showroom zone configuration used by the current pipeline:

```text
src/config/zones/Showroom.json
```

The Showroom configuration defines three behavioral areas:

- **Entrance**
- **Main Area**
- **Demo Area**

The tracking pipeline automatically loads this configuration and assigns each confirmed customer's ground point to the corresponding zone.

For a new camera view, zones can be created interactively using:

```bash
python -m src.zones.configure
```

The configuration utility allows polygonal zones to be drawn directly on a camera reference frame and saves the resulting coordinates for use by the tracking pipeline.

The included Showroom configuration is already complete and does not need to be recreated to run the current example.

This camera-specific configuration approach also allows additional zone files to be introduced as the project expands to multiple camera views.

---

## Running the Pipeline

### 1. Customer Tracking

Run the tracking pipeline:

```bash
python -m src.pipeline.tracking_pipeline
```

The pipeline performs:

```text
Video
  ↓
Person Detection
  ↓
ByteTrack
  ↓
Customer Confirmation
  ↓
Zone Assignment
  ↓
Track Persistence
```

Structured tracking data is written to:

```text
data/processed/tracks.parquet
```

The schema includes fields such as:

```text
camera_id
capture_id
frame_id
timestamp
local_track_id
customer_id
bounding box
ground point
zone
confidence
```

Generated videos and processed datasets are excluded from Git.

### 2. Analytics

After tracking is complete:

```bash
python -m src.pipeline.analytics_pipeline
```

This generates:

```text
artifacts/figures/capture1_Showroom_trajectory.png
artifacts/figures/capture1_Showroom_heatmap.png

data/processed/customer_summary.parquet
data/processed/zone_summary.parquet
```

---

## Spatial Analytics

Bounding-box center coordinates are useful for visualization, but they do not accurately represent a person's position on the floor.

For spatial analytics, this project therefore uses:

```text
ground_x = (x1 + x2) / 2
ground_y = y2
```

which approximates the person's ground contact point.

When a bounding box is clipped exactly at the bottom edge of the frame, the resulting ground point does not reliably represent the person's feet. These samples remain available for tracking and temporal analytics but are excluded from trajectory and heatmap generation.

This separation prevents spatial artifacts without discarding valid customer observation time.

---

## Customer Confirmation

Raw tracker IDs are not automatically treated as customers.

A track must satisfy multiple conditions before being confirmed, including:

- Minimum observation duration
- Minimum displacement
- Minimum number of movement frames
- Minimum average detection confidence

This helps suppress short-lived false detections and static objects incorrectly detected as people.

Once confirmed, the track receives a sequential customer identifier while preserving its original `local_track_id`.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

Current test suite:

```text
44 passed
```

Tests cover:

- Tracking geometry
- Customer-state management
- Customer confirmation
- Zone assignment
- Zone dwell times
- Analytics summaries
- Dataset validation
- Parquet track persistence
- Detector behavior with mocked YOLO inference

Network access and real model inference are not required by the unit tests.

---

## Continuous Integration

GitHub Actions is configured to run the unit test suite automatically on:

- Pushes to `main`
- Pull requests targeting `main`

The CI environment installs the development dependencies and runs:

```bash
pytest -q
```

---

## Project Structure

```text
retail-customer-tracking/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   ├── figures/
│   └── videos/
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── analytics/
│   │   ├── heatmap.py
│   │   ├── summary.py
│   │   └── trajectory.py
│   ├── config/
│   │   ├── reference_frames/
│   │   ├── zones/
│   │   ├── paths.py
│   │   └── settings.py
│   ├── data/
│   │   ├── download.py
│   │   ├── track_writer.py
│   │   └── validate.py
│   ├── detection/
│   │   └── detector.py
│   ├── pipeline/
│   │   ├── analytics_pipeline.py
│   │   └── tracking_pipeline.py
│   ├── tracking/
│   │   ├── customer_state.py
│   │   └── tracker.py
│   ├── visualization/
│   │   └── draw.py
│   └── zones/
│       ├── configure.py
│       └── zone_manager.py
├── tests/
│   └── unit/
├── .gitignore
├── LICENSE
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

---

## Design Decisions

### Pretrained Detection

YOLO11n is used as a pretrained person detector rather than being trained specifically for this dataset.

This keeps the current project focused on tracking, behavioral analytics, and system design. Domain-specific detector fine-tuning can be introduced later if real deployment footage reveals significant detection limitations.

### Local Track IDs

Tracker-generated identifiers are explicitly represented as:

```text
local_track_id
```

rather than global customer identities.

This distinction is intentional and prepares the data model for future multi-camera tracking.

### Camera-Specific Tracking

Tracking state belongs to an individual video stream. Each camera therefore requires an independent tracker instance.

Analytics modules are designed to consume structured tracking records rather than depending directly on a specific camera implementation.

---

## Multi-Camera Roadmap

The current version provides a complete single-camera analytics pipeline.

The next development milestone is multi-camera customer tracking using the remaining CITIC-MTMC camera views.

Planned architecture:

```text
Camera 1 ─→ YOLO ─→ ByteTrack ─→ Local Tracks ─┐
Camera 2 ─→ YOLO ─→ ByteTrack ─→ Local Tracks ─┤
Camera 3 ─→ YOLO ─→ ByteTrack ─→ Local Tracks ─┼─→ Cross-Camera Association
Camera 4 ─→ YOLO ─→ ByteTrack ─→ Local Tracks ─┤            │
Camera 5 ─→ YOLO ─→ ByteTrack ─→ Local Tracks ─┘            ▼
                                                       global_track_id
                                                              │
                                                              ▼
                                                       Global Analytics
```

Planned extensions include:

- Processing all five camera views
- Per-camera tracker instances
- Person Re-Identification embeddings
- Temporal cross-camera constraints
- Camera-topology-aware association
- Global customer identities
- Cross-camera trajectories
- Camera transition analytics
- Global visit-duration estimation
- Multi-camera behavioral summaries

The existing `camera_id` and `local_track_id` fields are intentionally preserved to support this extension without redesigning the tracking data model.

---

## Limitations

The current implementation has several intentional limitations:

- Tracking is currently performed on a single camera.
- Customer IDs are session-level identifiers, not cross-camera identities.
- YOLO11n uses pretrained weights without domain-specific fine-tuning.
- Store time represents the observed duration within the current camera rather than the customer's complete visit duration.
- Ground position is approximated from the bottom-center of the bounding box rather than projected into a calibrated world coordinate system.
- Zone definitions are manually configured per camera.
- Heatmaps represent accumulated observed presence rather than unique-person density.

These limitations define the next stages of development rather than being hidden assumptions of the system.

---

## License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## Author

**Arda Atik**

Artificial Intelligence Engineer