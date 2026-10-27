# file is specifically to distinguish between what data shows and what might indicate a technique issue
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
COORDINATION_FILE = BASE_DIR / "coordination_analysis.json"

# directly maps the original climbing video to the detected movement
# eventually gonna extract short video clips containing detected movements and provide these clips alongside
# our coach questions to the VLM.
SOURCE_VIDEO_MAP = {
    "metrics.json": "climbingTestDeepBentElbow.mov",
    "metrics_tension.json": "climbingTestTension.mov"
}

# creating movement observation

def loadCoordinationAnalyses(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["coordination_analyses"]


def identifyFootMovementObservation(movement):
    stability = movement.get("foot_stability")

    if stability is None:
        return {
            "observation": "unavailable",
            "reason": "No foot stability data"
        }

    speed_change = stability["speed_changes"]["between_to_after"]
    if speed_change is None:
        return {
            "observation": "insufficient_tracking",
            "reason": "Insufficient reliable data to compare foot speed"
        }

    if speed_change > 0:
        observation = "increased_post_event_foot_speed"
    elif speed_change < 0:
        observation = "decreased_post_event_foot_speed"
    else:
        observation = "unchanged_post_event_foot_speed"

    return {
        "observation": observation,
        "speed_change": speed_change
    }

def identifyUpperBodyObservation(movement):
    coordination = movement.get("elbow_hip_coordination")

    if coordination is None:
        return {
            "observation": "unavailable",
            "reason": "No elbow-hip coordination data"
        }
    flexing_arms = coordination.get("flexing_arms", [])
    hip_observation = coordination.get("hip_observation")

    upward_hip_patterns = {
        "downward_to_upward",
        "stationary_to_upward",
        "continued_upward"
    }

    if hip_observation not in upward_hip_patterns:
        observation = "no_confirmed_upward_hip_pattern"

    elif len(flexing_arms) == 2:
        observation = "bilateral_elbow_flexion_with_upward_hips"
    elif len(flexing_arms) == 1:
        observation = "single_observed_elbow_flexion_with_upward_hips"
    else:
        observation = "upward_hips_without_observed_elbow_flexion"

    return {
        "observation": observation,
        "flexing_arms": flexing_arms,
        "hip_observation": hip_observation
    }

# start of producing potential technique issues + coaching suggestions
def generateFootCoachingFeedback(foot_observation):
    observation = foot_observation["observation"]

    if observation == "increased_post_event_foot_speed":
        return {
            "potential_issue": "continued_foot_movement",
            "confidence": "needs_video_review",
            "suggestion": (
                "Review whether the foot continued repositioning "
                "after the movement. Consider practicing deliberate "
                "foot placement and controlled weight transfer."
            )
        }
    elif observation == "decreased_post_event_foot_speed":
        return {
            "potential_issue": None,
            "confidence": "needs_video_review",
            "suggestion": (
                "Foot movement decreased after the event. "
                "Review whether the foot established secure contact "
                "before initiating the next movement."
            )
        }
    elif observation == "insufficient_tracking":
        return {
            "potential_issue": None,
            "confidence": "insufficient_data",
            "suggestion": (
                "Post-event foot tracking was insufficient. "
                "Review the original footage before assessing "
                "foot placement or stability."
            )
        }

def generateUpperBodyCoachingFeedback(upper_body_observation):
    observation = upper_body_observation["observation"]
    if observation == "bilateral_elbow_flexion_with_upward_hips":
        return {
            "potential_issue": "review_arm_reliance",
            "confidence": "needs_video_review",
            "suggestion": (
                "Both elbows flexed during a movement cycle with upward hip motion "
                "Review whether the climber used their legs effectively "
                "or relied heavily on pulling with their arms"
            )
        }
    elif observation == "single_observed_elbow_flexion_with_upward_hips":
        return {
            "potential_issue": None,
            "confidence": "needs_video_review",
            "suggestion": (
                "One elbow showed increased flexion alongside upward hip movement. "
                "Review the climber's body positioning and weight transfer. "
                "Do not assume the other arm was inactive."
            )
        }
    return {
        "potential_issue": None,
        "confidence": "insufficient_data",
        "suggestion": (
            "No specific upper-body coaching suggestion is available "
            "from the current measurements"
        )
    }


# combined feedback fun
def buildTechniqueFeedback(movement):
    foot_observation = identifyFootMovementObservation(movement)
    upper_body_observation = identifyUpperBodyObservation(movement)

    foot_feedback = generateFootCoachingFeedback(foot_observation)
    upper_body_feedback = generateUpperBodyCoachingFeedback(upper_body_observation)

    return {
        "source": movement["source"],
        "foot": movement["foot"],
        "movement_type": movement["movement_type"],
        "frame_ranges": movement["frame_ranges"],
        "foot_analysis": {
            "observation": foot_observation,
            "coaching_feedback": foot_feedback
        },
        "upper_body_analysis": {
            "observation": upper_body_observation,
            "coaching_feedback": upper_body_feedback
        }
    }

# save the feedback inside phase03
def saveTechniqueFeedback(feedback_records, output_path):
    output = {
        "total_movements": len(feedback_records),
        "technique_feedback": feedback_records
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"\nTechnique feedback saved to: {output_path}")

# we need to distinguish sufficiently between different climbing techniques
# make our suggestions depend on verified movement type
def generateMovementSpecificFeedback(movement):
    movement_type = movement.get("movement_type")

    foot_observation = identifyFootMovementObservation(movement)
    observation = foot_observation["observation"]

    if movement_type == "release_return":

        if observation == "increased_post_event_foot_speed":
            suggestion = (
                "During this foot release-return, foot movement "
                "increased after the second event. Review whether "
                "the foot continued repositioning before the next move."
            )

        elif observation == "decreased_post_event_foot_speed":
            suggestion = (
                "During this foot release-return, foot movement "
                "decreased after the second event. Review whether "
                "the foot established secure contact."
            )

        else:
            suggestion = (
                "Foot tracking was insufficient to assess movement "
                "after the release-return. Review the original footage."
            )

    elif movement_type == "heel_hook_readjustment":

        if observation == "insufficient_tracking":
            suggestion = (
                "The heel-hook readjustment could not be reliably "
                "evaluated after the movement. Review whether the "
                "heel remained engaged and supported body positioning."
            )

        else:
            suggestion = (
                "Review whether the heel-hook adjustment maintained "
                "controlled positioning and effective heel engagement."
            )

    else:
        suggestion = (
            "Review the movement for controlled foot positioning "
            "and deliberate weight transfer."
        )

    return {
        "movement_type": movement_type,
        "observation": observation,
        "suggestion": suggestion
    }

# using func composition: one function combines the results of other functions
# movement feedback gives us a movement specific suggestion
# upper_body feedback gives us upper body suggestion
# this func brings those res together into one dictionary
# can be used later on to export or display
def generateCombinedCoachingFeedback(movement):
    movement_feedback = generateMovementSpecificFeedback(movement)

    upper_body_observation = identifyUpperBodyObservation(movement)

    upper_body_feedback = generateUpperBodyCoachingFeedback(upper_body_observation)

    return {
        "source": movement["source"],
        "foot": movement["foot"],
        "movement_type": movement["movement_type"],
        "frame_ranges": movement["frame_ranges"],
        "foot_feedback": movement_feedback,
        "upper_body_feedback": upper_body_feedback
    }

def generateContextualCoachingFocus(movement):
    movement_type = movement.get("movement_type")

    foot_observation = identifyFootMovementObservation(movement)
    upper_observation = identifyUpperBodyObservation(movement)

    foot_pattern = foot_observation["observation"]
    upper_patter = upper_observation["observation"]

    if foot_pattern == "insufficient_tracking":
        return {
            "focus": "verify_movement_in_video",
            "suggestion": (
                "Foot tracking is insufficient for a reliable "
                "post-event comparison. Review the video before "
                "assessing foot placement or movement efficiency."
            )
        }
    if movement_type == "release_return":
        if foot_pattern == "increased_post_event_foot_speed":
            return {
                "focus": "foot_repositioning_and_weight_transfer",
                "suggestion": (
                    "Review whether the foot continued repositioning "
                    "after the release-return. Examine how the climber "
                    "transferred weight through the foot while moving upward."
                )
            }
        elif foot_pattern == "decreased_post_event_foot_speed":
            return {
                "focus": "foot_contact_and_body_position",
                "suggestion": (
                    "Foot movement decreased after the release-return. "
                    "Review whether the climber established secure foot "
                    "contact before continuing upward."
                )
            }
    if movement_type == "heel_hook_readjustment":
        return {
            "focus": "heel_engagement_and_body_tension",
            "suggestion": (
                "Review whether the heel remained engaged during "
                "the readjustment and how body positioning changed."
            )
        }
    return {
        "focus": "general_movement_review",
        "suggestion": (
            "Review foot positioning and upper-body movement "
            "during this climbing sequence."
        )
    }

def generateUpperBodyContext(movement):
    upper_observation = identifyUpperBodyObservation(movement)

    pattern = upper_observation["observation"]

    if pattern == "bilateral_elbow_flexion_with_upward_hips":
        return {
            "context": "bilateral_elbow_flexion",
            "suggestion": (
                "Both elbows showed increased flexion during a movement "
                "cycle with upward hip motion. When reviewing the footage, "
                "consider how the climber used their feet and arms "
                "to support the upward movement."
            )
        }
    elif pattern == "single_observed_elbow_flexion_with_upward_hips":
        return {
            "context": "partial_elbow_evidence",
            "suggestion": (
                "One elbow showed increased flexion alongside upward "
                "hip movement. Review body positioning and weight "
                "transfer without assuming the other elbow was inactive."
            )
        }
    return {
        "context": "limited_upper_body_evidence",
        "suggestion": (
            "The available measurements do not establish a clear "
            "elbow-flexion and upward-hip movement pattern."
        )
    }
# why this structure: each recommendation now contains three important pieces of information
# coaching_focus (primary area to investigate)
# upper_body_context --> describes the elbow-hip evidence (if available)
# recommendations --> contains readable suggestions for the climber
def generateUnifiedCoachingRecommendation(movement):
    foot_context = generateContextualCoachingFocus(movement)
    upper_context = generateUpperBodyContext(movement)

    return {
        "source": movement["source"],
        "video_file": SOURCE_VIDEO_MAP.get(movement["source"]),
        "foot": movement["foot"],
        "movement_type": movement["movement_type"],
        "frame_ranges": movement["frame_ranges"],
        "coaching_focus": foot_context["focus"],
        "upper_body_context": upper_context["context"],
        "recommendations": {
            "foot": foot_context["suggestion"],
            "upper_body": upper_context["suggestion"],
        },
        "requires_video_review": True
    }

def saveUnifiedRecommendations(recommendations, output_path):
    output = {
        "total_movements": len(recommendations),
        "recommendations": recommendations
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

        print(f"Unified recommendations saved to: {output_path}")
if __name__ == "__main__":
    movements = loadCoordinationAnalyses(COORDINATION_FILE)

    feedback_records = [
        buildTechniqueFeedback(movement)
        for movement in movements
    ]

    OUTPUT_FILE = BASE_DIR / "technique_feedback.json"

    saveTechniqueFeedback(feedback_records, OUTPUT_FILE)

    print("\n--- Technique Feedback Summary ---")

    for feedback in feedback_records:
        foot_issue = feedback["foot_analysis"]["coaching_feedback"]["potential_issue"]
        upper_issue = feedback["upper_body_analysis"]["coaching_feedback"]["potential_issue"]

        print(
            f"{feedback['source']} | "
            f"{feedback['foot']} foot | "
            f"Foot: {foot_issue} | "
            f"Upper body: {upper_issue}"
        )

    print("\n--- Movement-Specific Coaching Feedback ---")

    for movement in movements:
        feedback = generateMovementSpecificFeedback(movement)

        print(f"\n{movement['source']} | {movement['foot']} foot")
        print("Movement:", feedback["movement_type"])
        print("Observation:", feedback["observation"])
        print("Suggestion:", feedback["suggestion"])

    print("\n--- Combined Movement-Specific Coaching Feedback ---")

    for movement in movements:
        feedback = generateCombinedCoachingFeedback(movement)

        print(f"\n{feedback['source']} | {feedback['foot']} foot")

        print("Movement:", feedback["movement_type"])
        print("Foot suggestion:", feedback["foot_feedback"]["suggestion"])
        print(
            "Upper-body suggestion:",
            feedback["upper_body_feedback"]["suggestion"]
        )

    print("\n--- Contextual Coaching Focus ---")

    for movement in movements:
        feedback = generateContextualCoachingFocus(movement)

        print(f"\n{movement['source']} | {movement['foot']} foot")
        print("Focus:", feedback["focus"])
        print("Suggestion:", feedback["suggestion"])

    print("\n--- Upper-Body Context ---")

    for movement in movements:
        context = generateUpperBodyContext(movement)

        print(f"\n{movement['source']} | {movement['foot']} foot")
        print("Context:", context["context"])
        print("Suggestion:", context["suggestion"])

    print("\n--- Unified Coaching Recommendations ---")

    for movement in movements:
        recommendation = generateUnifiedCoachingRecommendation(movement)

        print(f"\n{movement['source']} | {movement['foot']} foot")
        print(json.dumps(recommendation, indent=4))

    recommendations = [
        generateUnifiedCoachingRecommendation(movement)
        for movement in movements
    ]

    OUTPUT_FILE = BASE_DIR / "unified_coaching_recommendations.json"

    saveUnifiedRecommendations(recommendations, OUTPUT_FILE)

    print("\n--- Source Video Mapping ---")

    for recommendation in recommendations:
        print(
            f"{recommendation['source']} -> "
            f"{recommendation['video_file']}"
        )