# AI Climbing Coach — Development Documentation

## Project Overview

AI Climbing Coach is a computer vision project designed to analyze climbing technique from recorded bouldering videos. The long-term goal is to develop an AI-powered system capable of identifying movement patterns, evaluating body positioning, and providing actionable feedback to help climbers improve their technique and movement efficiency.

The project uses **Python, OpenCV, and MediaPipe Pose Landmarker** to extract body landmarks and calculate movement metrics. Development is organized into phases, beginning with pose estimation and progressing toward whole-body movement analysis and technique feedback.

---

# Phase 01 — Video Processing and Pose Estimation

## Objective

Establish a reliable video-processing pipeline capable of detecting and tracking a climber's body landmarks frame by frame.

This phase provides the foundation for all future movement analysis.

## Implementation

### 1. Video Processing Pipeline
- Implemented video loading and frame-by-frame processing using OpenCV.
- Integrated MediaPipe's Pose Landmarker Tasks API for human pose estimation.
- Configured the landmarker to operate in VIDEO mode using frame timestamps.
- Extracted body landmarks and their normalized coordinates from recorded climbing videos.

### 2. Center of Mass Approximation
- Calculated the midpoint between the left and right hip landmarks as an approximation of the climber's center of mass.
- Tracked the horizontal and vertical hip coordinates across frames.
- Used these measurements to establish a baseline for analyzing body movement.

### 3. Hip Vertical Velocity
- Implemented a calculation for vertical hip velocity using changes in normalized hip position over time.
- Enabled detection of upward and downward hip movement in image coordinates.
- Established a foundation for analyzing movement dynamics during climbing sequences.

### 4. Tracking Validation
- Incorporated landmark visibility checks to identify potentially unreliable pose measurements.
- Implemented tracking-validity conditions to prevent missing or invalid coordinates from affecting calculations.
- Handled frames where body landmarks could not be reliably detected.

### 5. Data Logging
- Developed a JSON-based logging system to store frame-level movement measurements.
- Recorded hip coordinates, vertical velocity, landmark visibility, and tracking validity.
- Structured the output for use in later analysis phases.

## Key Accomplishments

- Established a functional computer vision pipeline for recorded climbing videos.
- Successfully extracted and tracked body landmarks using MediaPipe.
- Implemented initial body-position and movement measurements.
- Created a reusable foundation for expanding into upper- and lower-body analysis.

---

# Phase 02 — Movement Detection and Whole-Body Analysis

## Objective

Expand the initial pose-estimation pipeline to analyze climbing movements, identify candidate foot movement events, and investigate coordination between the upper and lower body.

This phase focuses on transforming raw landmark data into structured movement features that can eventually support technique evaluation.

## Implementation

### 1. Upper-Body and Joint Analysis

Expanded frame-level measurements to include:

- Left and right elbow angles.
- Elbow-position classifications.
- Left and right knee angles.
- Left and right hip angles.
- Arm and leg tracking-validity indicators.

These measurements provide information about joint positioning and changes in body posture throughout a climb.

### 2. Lower-Body Movement Tracking

Implemented foot movement analysis using MediaPipe foot landmarks and hip-relative positioning.

Tracked:

- Horizontal and vertical foot displacement relative to the hips.
- Horizontal and vertical foot velocity.
- Overall foot speed.
- Landmark tracking validity.

Using hip-relative coordinates helps describe foot movement relative to the climber's body rather than solely relative to the camera frame.

### 3. Foot Movement Event Detection

Developed an analysis pipeline to identify periods of elevated foot movement.

The system:

- Calculates foot-speed distributions and percentile-based thresholds.
- Identifies frames containing elevated foot speeds.
- Groups nearby frames into movement events.
- Calculates movement duration, displacement, and path length.
- Measures path efficiency to characterize movement trajectories.
- Identifies candidate movement cycles based on nearby movement events and direction changes.

Movement-cycle detection identifies potential release-return patterns, but does not independently confirm foot contact, release, or the intent of a movement.

### 4. Cross-Video Movement Validation

Tested the movement-detection pipeline on multiple climbing videos.

Manually verified three movement cycles:

- A left-foot release-return sequence.
- A right-foot release-return sequence.
- A right-foot readjustment into a heel hook.

The heel-hook readjustment was initially detected as a candidate movement cycle, demonstrating that direction changes and foot displacement alone cannot reliably distinguish intentional repositioning from release-return movements.

This highlighted the need for additional movement context and manual validation.

### 5. Movement Stability and Tracking Coverage

Implemented movement stability analysis using average foot speeds across three windows:

- Before a movement event.
- Between paired movement events.
- After the second event.

Added tracking coverage calculations to determine how many frames contained valid measurements.

This helped identify unreliable analysis windows, including a heel-hook post-event window with only 10% valid foot-speed coverage.

### 6. Upper- and Lower-Body Integration

Combined foot movement events with upper-body positioning and hip movement measurements.

For each detected movement cycle, the system analyzes:

- Left and right elbow angles before, during, and after the first foot movement.
- Changes in elbow flexion and extension.
- Average hip vertical velocity across the same windows.
- Foot movement stability.
- Tracking coverage and missing measurements.

This enables investigation of how upper-body positioning changes during lower-body movements.

### 7. Movement Coordination Analysis

Compared manually verified release-return movements against the heel-hook readjustment.

Observed that:

- Both verified foot releases coincided with a shift toward upward hip movement.
- The heel-hook readjustment occurred while the hips were already moving upward.
- Elbow flexion occurred during all three movements, but its timing and magnitude differed.
- Missing landmark measurements affected which movement features could be analyzed reliably.

These findings are exploratory observations rather than validated rules for classifying technique.

### 8. Unified Movement Analysis

Developed a structured analysis record that combines:

- Foot movement event information.
- Foot stability measurements.
- Elbow positioning.
- Elbow-angle changes.
- Hip vertical movement.
- Tracking coverage.
- Manually verified movement labels.

Exported these records as JSON files for further processing and comparison.

### 9. Cross-Video Feature Extraction

Created `compare_movements.py` to load and compare movement-analysis records across different climbing videos.

Implemented:

- Automatic discovery of movement-analysis JSON files.
- Extraction of elbow-angle and hip-velocity features.
- Tracking-coverage checks to exclude insufficiently supported measurements.
- Structural validation of movement records.
- Export of a consolidated `movement_features.json` dataset.

This design allows new climbing videos to be incorporated into the comparison pipeline without manually hardcoding their analysis filenames.

## Key Accomplishments

- Expanded pose estimation into upper- and lower-body movement analysis.
- Developed a candidate foot movement detection system.
- Identified limitations of movement classification using foot trajectories alone.
- Integrated elbow positioning, hip movement, and foot stability into unified records.
- Validated movement measurements and tracking coverage.
- Built a reusable cross-video feature extraction pipeline.
- Created a structured dataset for future technique analysis.

## Current Limitations

- Movement detection identifies candidate events rather than definitively recognizing climbing techniques.
- Landmark occlusion and camera perspective can affect measurement accuracy.
- Hip-relative coordinates and normalized velocities do not directly represent physical distances or forces.
- The current dataset contains only three manually verified movement cycles from two videos.
- The two verified release-return cycles overlap within the same climbing sequence and are not independent observations.
- Additional labeled videos are needed before developing reliable movement classifiers or technique recommendations.

---

# Next Steps — Phase 03: Climbing Technique Feedback

Phase 03 will focus on translating the movement features developed in Phases 01 and 02 into understandable, evidence-based climbing feedback.

Planned objectives include:

1. **Movement Summarization:** Generate readable descriptions of detected body movements.
2. **Coordination Analysis:** Investigate relationships between foot movement, elbow positioning, and hip motion.
3. **Technique Evaluation:** Explore methods for identifying potential movement inefficiencies using validated climbing examples.
4. **Feedback Generation:** Develop a system that provides useful, appropriately qualified suggestions for improving climbing technique.

The long-term vision is to integrate these analytical capabilities with a vision-language model (VLM), enabling the system to interpret climbing movements using both quantitative pose measurements and visual context.
