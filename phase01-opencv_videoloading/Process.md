Overall Process of Project
                    ┌───────────────────┐
                    │   React Frontend  │
                    │                   │
                    │ Upload Video      │
                    │ View Analysis     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    FastAPI        │
                    │    Backend        │
                    └─────────┬─────────┘
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          ┌───────────┐ ┌──────────┐ ┌───────────┐
          │   Video   │ │  Pose    │ │ Database  │
          │ Processing │ │ Analysis │ │           │
          └─────┬─────┘ └────┬─────┘ └───────────┘
                │             │
                └──────┬──────┘
                       ▼
               ┌───────────────┐
               │ ML Movement   │
               │ Classifier    │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │ AI Coaching   │
               │     LLM       │
               └───────────────┘

-- use FFmpeg and GStreamer
-- For video analysis we use mediapipe and pose_landmarker_full to connect joints and body parts together. Created a PoseLandmarker obj that draws a skeleton based off of approximate locations of body parts. It uses multiple models to predict landmarks. first model predicts pressence of human bodies within a frame, the second locates the landmarks on the bodies
--



-- FFmpeg for slicing, dicing // resizing 10,000 video clips to a specific res and fps
-- GSreamer ingets low-latency cam feeds and pushes frames into AI model
