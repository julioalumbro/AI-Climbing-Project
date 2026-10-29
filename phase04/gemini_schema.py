from pydantic import BaseModel, Field
from typing import Literal

# how these two classes work together:
# GeminiMovementAnalysis --> look at each frame using frameObservation
# --> Movement-level analysis (summary/interpretation/uncertainties)

# observations and interpretation are separated because interpretations != visible evidence
# prevents the coaching system from presenting uncertain interpretations as established facts


# eventual pipeline: OpenCV --> MediaPipe --> GeminiMovementAnalysis --> Evidence reconciliation --> Coaching Recs
# coaching reccs == supported by evidence
class FrameObservation(BaseModel):
    frame: int = Field(
        description="Original video frame number"
    )

    foot_position: str = Field(
        description=(
            "Visible position of the target foot "
            "relative to the climber's hips and torso"
        )
    )
    foothold_contact: Literal["yes", "no", "unclear"] = Field(
        description=(
            "Whether the target foot visibly contacts"
            "a climbing hold"
        )
    )

    body_position: str = Field(
        description=(
            "Visible position and orientation of "
            "the climber's hips and torso"
        )
    )

    arm_position: str = Field(
        description=(
            "Visible position of the climber's "
            "arms and elbows"
        )
    )

    uncertainties: list[str] = Field(
        description=(
            "Visual details that cannot be "
            "determined confidently from this frame"
        )
    )


class GeminiMovementAnalysis(BaseModel):
    movement_type: str = Field(
        description=(
            "Type of climbing movement being analyzed, "
            "such as left_release_return or right_heel_hook"
        )
    )

    target_foot: Literal["left", "right"] = Field(
        description="Anatomical foot being analyzed"
    )

    frames: list[FrameObservation] = Field(
        description=(
            "Visual observations for each provided frame, "
            "in chronological order"
        )
    )

    movement_summary: str = Field(
        description=(
            "Summary of visible changes in foot, hip, "
            "and arm positioning across the frames"
        )
    )

    movement_interpretation: str = Field(
        description=(
            "Possible climbing technique or movement pattern "
            "suggested by the sequence. Distinguish interpretation "
            "from directly visible evidence"
        )
    )

    uncertainties: list[str] = Field(
        description=(
            "Details that cannot be confidently determined "
            "from the available frames"
        )
    )
