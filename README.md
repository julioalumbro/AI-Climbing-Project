# AI Climbing Coach 

An AI-powered climbing analysis application that combines computer vision, pose estimation, and Google's Gemini AI to analyze recorded bouldering footage, extract movement metrics, and generate personalized, evidence-aware coaching feedback.

## Overview

AI Climbing Coach is my personal project created to explore how artificial intelligence and computer vision can help climbers better understand their movement and technique.

Traditional climbing improvement often relies on personal observation, coaching, or reviewing recorded footage. This project aims to make video review more informative by automatically tracking body movements, measuring specific movement patterns, and generating feedback that climbers can use to evaluate their technique.

The application combines two complementary approaches:

- **MediaPipe Pose Landmarker:** Extracts body landmarks from video frames and provides measurable movement data.
- **Google Gemini:** Interprets climbing footage and generates natural-language observations and coaching feedback.

By combining visual interpretation with numerical measurements, the application aims to provide more transparent and useful climbing analysis.

## Features

### 1. Video Upload and Processing

- Upload recorded climbing videos through a web interface.
- Process videos using a Python-based computer vision pipeline.
- Extract body landmarks frame by frame.
- Generate annotated climbing footage with a skeleton overlay.
- Review original and annotated footage using an integrated video player.

### 2. Movement Tracking and Metrics

Uses MediaPipe Pose Landmarker and OpenCV to extract movement measurements.

Current measurements include:

- Hip position and vertical velocity.
- Left and right elbow angles.
- Arm-bend detection and duration tracking.
- Longest detected deep-bend intervals.
- Total detected deep-bend duration.
- Frame-level tracking validity and landmark visibility.

Measurements are exported to JSON for further analysis.

### 3. Rule-Based Coaching Feedback

Analyzes recorded movement measurements to identify predefined movement patterns.

Feedback includes:

- Movement observations.
- Possible interpretations.
- Coaching suggestions.
- Evidence limitations.
- Relevant video timestamps.

Users can select a movement's timestamp to review the corresponding footage.

### 4. Gemini AI Coaching

Integrates Google's Gemini API to generate natural-language coaching recommendations using extracted movement metrics.

Generated coaching include:

- Overall climbing assessment.
- Identified strengths.
- Potential areas for improvement.
- Limitations of the available measurements.

Coaching is generated on demand through the web interface.

Previously generated coaching responses can be retrieved from the application's cache to avoid unnecessary repeated API calls.

### 5. Hybrid Movement Analysis

Combines Gemini's interpretation of climbing footage with MediaPipe's movement measurements.

The hybrid analysis pipeline:

1. Uses Gemini to identify climbing movements and observations from the uploaded video.
2. Associates observations with relevant video timestamps.
3. Matches identified movement intervals with available MediaPipe measurements.
4. Evaluates the availability of supporting measurement evidence.
5. Produces feedback that communicates both observations and measurement limitations.

The system distinguishes between:

- **Measurement supported:** Relevant measurement evidence is available.
- **Partially supported:** Some relevant measurements are available.
- **Insufficient measurement data:** Available measurements cannot adequately support the evaluation.

These classifications describe measurement coverage, not whether Gemini's visual interpretation is necessarily correct.

### 6.  Web Interface (React)

The React frontend provides:

- Video upload and analysis controls.
- Original and skeleton-overlay video playback.
- Movement metrics dashboard.
- On-demand Gemini AI Coaching.
- Collapsible Hybrid Movement Analysis.
- Timestamp-based movement review.
- Rule-based coaching feedback.
- Backend connection and pipeline status indicators.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Langs | Python, TypeScript |
| Computer vision | OpenCV |
| Pose estimation | MediaPipe Pose Landmarker |
| Generative AI | Google Gemini API |
| Backend | FastAPI |
| Frontend | React, Vite |
| Styling | Tailwind CSS |
| Video processing | FFmpeg |
| Data storage | JSON |
| Development tools | PyCharm, Git, GitHub |

## System Architecture

The application uses a multi-stage analysis pipeline.

```text
                   Climbing Video
                         |
                         v
                 React Frontend
                         |
                         v
                  FastAPI Backend
                         |
             +-----------+-----------+
             |                       |
             v                       v
      MediaPipe Pipeline      Gemini Video Analysis
             |                       |
             v                       v
       Pose Landmarks          Visual Observations
             |                       |
             v                       v
      Movement Metrics         Movement Intervals
             |                       |
             +-----------+-----------+
                         |
                         v
                Movement Matching
                         |
                         v
                Hybrid Feedback
                         |
                         v
                  React Frontend
```

A separate coaching workflow allows users to request additional Gemini-generated feedback based on extracted movement metrics.

## Project Structure

The project was developed incrementally across multiple phases.

```text
AI-Climbing-Coach/
|
|-- phase01/
|   |-- Initial video processing and pose tracking
|
|-- phase02/
|   |-- Joint-angle calculations and movement metrics
|
|-- ...
|
|-- phase06/
|   |
|   |-- backend/
|   |   |-- main.py
|   |   |-- analysis_service.py
|   |   |-- feedback_service.py
|   |   |-- gemini_video_service.py
|   |   |-- movement_matching_service.py
|   |   |-- hybrid_feedback_service.py
|   |   |-- results/
|   |
|   |-- frontend/
|       |-- src/
|       |   |-- App.tsx
|       |   |-- VideoUploader.tsx
|       |
|       |-- package.json
|
|-- README.md
```

Earlier phases focus on computer vision fundamentals and movement measurement. Phase 06 integrates these components into a full-stack application.
The phases aren't ordered by system architecture but rather the steps I took to build this project
*Note: This diagram highlights major components rather than every file in the repository.*


### Prerequisites

have the following tools are installed:

- Python
- Node.js and npm
- Git
- FFmpeg
- A Google Gemini API key for Gemini-powered functionality (did require credits spent)

The application also requires the MediaPipe Pose Landmarker model asset used by the video processing pipeline.

## How to Use

1. Start the FastAPI backend and React frontend.
2. Open the application in your browser.
3. Select a recorded climbing video.
4. Upload and analyze the video.
5. Wait for the processing pipeline to complete.
6. Review the original or skeleton-overlay footage.
7. Examine the extracted movement metrics.
8. Expand Hybrid Movement Analysis to review movement observations and measurement evidence.
9. Select movement timestamps to inspect relevant moments in the video.
10. Click **Generate AI Coaching** to request an additional Gemini-generated coaching assessment.

## Analysis Outputs

The backend stores analysis artifacts in a results directory associated with each uploaded video.

Example:

```text
phase06/backend/results/
|
|-- <upload_id>/
    |-- metrics.json
    |-- gemini_movements.json
    |-- hybrid_feedback.json
    |-- annotated video output
```

### `metrics.json`

Contains extracted frame-level movement measurements and tracking information.

### `gemini_movements.json`

Contains Gemini-generated movement observations and associated time intervals.

### `hybrid_feedback.json`

Contains movement observations matched with available measurement evidence and generated feedback.

### Annotated Video

Provides a visual representation of detected body landmarks and the pose skeleton overlaid on the climbing footage.

## API

The backend exposes endpoints for application health, video analysis, and AI coaching.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Check backend availability |
| Video analysis endpoint | Upload and process climbing footage |
| `POST /api/coaching/{upload_id}` | Generate or retrieve Gemini coaching |

The video analysis response includes movement measurements, generated feedback, annotated video information, and hybrid analysis results.

Relevant hybrid response fields include:

```json
{
  "hybrid_status": "completed",
  "hybrid_error": null,
  "hybrid_feedback": []
}
```

The hybrid feedback array contains movement-specific observations, timestamps, evidence classifications, metric evaluations, and feedback.

## Development Approach


The project was built incrementally to establish a reliable foundation before introducing generative AI.

### Phase 01 - Video Processing and Pose Detection

- Implemented video loading with OpenCV.
- Integrated MediaPipe Pose Landmarker.
- Extracted body landmarks from recorded footage.
- Calculated hip midpoint positions and vertical velocity.
- Added tracking-validity checks.
- Exported frame-level measurements to JSON.

### Phase 02 - Joint-Angle Analysis

- Implemented joint-angle calculations using vector mathematics.
- Calculated elbow angles from shoulder, elbow, and wrist landmarks.
- Added visibility and coordinate validation.
- Expanded frame-level movement measurements.

### Phase 03-05 - Intermediate Development

- Expanded movement tracking and duration analysis.
- Introduced movement classification and rule-based feedback.
- Improved annotated video generation and analysis outputs.

### Phase 06 - Full-Stack AI Integration

- Developed a FastAPI backend.
- Built a React and TypeScript frontend.
- Integrated video upload and browser playback.
- Added on-demand Gemini coaching.
- Integrated Gemini video observations with MediaPipe measurements.
- Implemented evidence-aware hybrid feedback.
- Improved the interface with collapsible analysis sections and timestamp-based navigation.

## Technical Challenges

### Handling Incomplete Pose Detection

Climbing movements often involve occlusion, unusual body positions, and inconsistent landmark visibility.

**Approach:** Implemented landmark visibility checks, coordinate validation, and tracking-validity indicators to reduce the use of unreliable measurements.

### Calculating Joint Angles

Joint-angle calculations require careful handling of coordinate geometry and numerical edge cases.

**Approach:** Used vector dot products and inverse cosine to calculate angles, with safeguards against zero-length vectors and floating-point errors.

### Combining Different AI Outputs

Gemini produces natural-language observations, while MediaPipe provides numerical movement measurements.

**Approach:** Developed a movement-matching layer that associates Gemini's timestamped observations with corresponding measurement intervals.

### Communicating Uncertainty

A model-generated observation does not automatically imply that numerical measurements confirm the observation.

**Approach:** Added measurement-evidence classifications and explicit limitations to distinguish available measurements from unsupported conclusions.

### Video Compatibility

Annotated videos must be encoded in a format supported by modern web browsers.

**Approach:** Integrated FFmpeg-based H.264 conversion for browser-compatible playback.

## Current Limitations

AI Climbing Coach is an experimental analysis tool, it's not a substitute for professional climbing instruction.

Known limitations include:

- **Single-camera perspective:** Two-dimensional pose estimation cannot fully capture depth, rotation, or three-dimensional movement.
- **Occlusion:** Climbing holds, walls, and body positioning may obstruct landmarks.
- **Pose estimation errors:** Unusual positions, lighting, clothing, and camera angles can affect tracking accuracy.
- **Limited biomechanical measurements:** The current system does not directly measure grip force, contact pressure, or complete body mechanics.
- **Threshold sensitivity:** Movement classifications depend on predefined measurement thresholds.
- **Generative AI uncertainty:** Gemini may produce incomplete or incorrect observations and recommendations.
- **Evidence limitations:** Available MediaPipe measurements do not necessarily validate the correctness of a Gemini-generated observation.

The application communicates measurement limitations to help users interpret its feedback appropriately.

## Future Improvements

Potential future development includes:

- Tracking additional joints and movement patterns.
- Improving measurement smoothing and handling missing landmarks.
- Refining movement classification thresholds.
- Adding configurable minimum-duration thresholds for detected movement events.
- Expanding footwork and body-position analysis.
- Improving recognition of climbing-specific techniques.
- Evaluating the accuracy of AI-generated feedback against expert climbing assessments.
- Supporting more detailed climb-to-climb comparisons.
- Exploring additional computer vision models for improved movement understanding.

## Motivation

Bouldering has been an important personal interest of mine, and I wanted to build something that combined my passion for climbing with my interest in artificial intelligence and software engineering.

I was particularly interested in exploring whether AI could do more than simply identify body positions. I wanted to investigate how computer vision measurements and generative AI could work together to provide meaningful, interpretable feedback about climbing technique.

This project has allowed me to gain hands-on experience with computer vision, video processing, AI integration, backend development, frontend development, and the challenges of building an end-to-end AI application.

## Disclaimer

AI Climbing Coach is an experimental personal project intended for learning, research, and exploratory movement analysis.

Its recommendations are not guaranteed to be accurate and should not be treated as professional coaching, medical advice, or safety guidance.

Users should apply their own judgment and follow appropriate climbing safety practices.

## Author
Gabe Alumbro :)
Developed as a personal project combining software engineering, artificial intelligence, and a passion for bouldering.
