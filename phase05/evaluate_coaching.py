import re
def evaluate_evidence_alignment(evidence_report, coaching_analysis):
    """
    Evaluate whether coaching recommendations reference existing
    evidence and whether that evidence is supported.

    Evidence statuses:
    - supported: Gemini and MediaPipe agree.
    - contradicted: Gemini and MediaPipe disagree.
    - not_independently_verified: Visual observation lacks
      independent verification.
    """

    # Map reconciliation claims to their statuses.
    evidence_statuses = {
        item["claim"]: item["status"]
        for item in evidence_report.get("reconciliations", [])
    }

    # Include visual observations without overwriting reconciliations.
    for item in evidence_report.get("visual_observations", []):
        evidence_statuses.setdefault(
            item["claim"],
            item["status"]
        )

    results = []

    for index, recommendation in enumerate(
        coaching_analysis.get("recommendations", [])
    ):
        referenced_claims = recommendation.get(
            "supporting_evidence", []
        )

        missing_claims = [
            claim for claim in referenced_claims
            if claim not in evidence_statuses
        ]

        contradicted_claims = [
            claim for claim in referenced_claims
            if evidence_statuses.get(claim) == "contradicted"
        ]

        unverified_claims = [
            claim for claim in referenced_claims
            if evidence_statuses.get(claim)
            == "not_independently_verified"
        ]

        insufficient_claims = [
            claim for claim in referenced_claims
            if evidence_statuses.get(claim) == "insufficient_evidence"
        ]

        supported_claims = [
            claim for claim in referenced_claims
            if evidence_statuses.get(claim) == "supported"
        ]

        results.append({
            "recommendation_index": index,
            "referenced_claims": referenced_claims,
            "missing_claims": missing_claims,
            "supported_claims": supported_claims,
            "contradicted_claims": contradicted_claims,
            "unverified_claims": unverified_claims,
            "insufficient_claims": insufficient_claims,
            "evidence_alignment_passed": (
                len(referenced_claims) > 0
                and len(missing_claims) == 0
                and len(contradicted_claims) == 0
                and len(unverified_claims) == 0
                and len(supported_claims) == len(referenced_claims)
            )
        })

    return results


def evaluate_uncertainty_handling(evidence_report, coaching_analysis):
    """
    Check whether recommendations referencing uncertain evidence
    include an uncertainty explanation.

    This checks for the presence of an explanation, not whether
    the explanation accurately addresses the uncertainty.
    """

    alignment_results = evaluate_evidence_alignment(
        evidence_report,
        coaching_analysis
    )

    results = []

    for index, recommendation in enumerate(
        coaching_analysis.get("recommendations", [])
    ):
        alignment = alignment_results[index]

        uncertain_claims = (
            alignment["contradicted_claims"]
            + alignment["unverified_claims"]
            + alignment["insufficient_claims"]
        )


        uncertainty_text = recommendation.get("uncertainty", "")
        has_uncertainty_explanation = (
            isinstance(uncertainty_text, str)
            and bool(uncertainty_text.strip())
        )

        results.append({
            "recommendation_index": index,
            "uncertain_claims": uncertain_claims,
            "has_uncertainty_explanation": has_uncertainty_explanation,
            "uncertainty_check_passed": (
                not uncertain_claims
                or has_uncertainty_explanation
            )
        })

    return results


def extract_joint_movement_ground_truth(evidence_report):
    """
    Extract joint movement directions from MediaPipe measurements.

    These measurements serve as our reference for checking
    coaching descriptions, although they are not independently
    verified ground truth.
    """

    joint_movements = {}

    for evaluation in evidence_report.get("evaluations", []):
        claim = evaluation.get("claim", "")
        movement = evaluation.get("movement")

        if "angle change" not in claim.lower():
            continue

        if movement not in ("flexion", "extension"):
            continue

        joint_name = claim.lower().replace(" angle change", "")

        joint_movements[joint_name] = {
            "movement": movement,
            "first_angle": evaluation.get("first_angle"),
            "last_angle": evaluation.get("last_angle"),
            "angle_change_degrees": evaluation.get(
                "angle_change_degrees"
            )
        }

    return joint_movements

def evaluate_joint_movement_descriptions(
    evidence_report,
    coaching_analysis
):
    """
    Detect explicit contradictions between joint movement
    descriptions and MediaPipe measurements.

    Only evaluates simple flexion/extension descriptions.
    Does not assess suggested adjustments.
    """

    joint_movements = extract_joint_movement_ground_truth(
        evidence_report
    )

    results = []

    joint_aliases = {
        "left elbow": ["left elbow", "left arm"],
        "right elbow": ["right elbow", "right arm"],
        "left knee": ["left knee"],
        "right knee": ["right knee"]
    }

    for index, recommendation in enumerate(
        coaching_analysis.get("recommendations", [])
    ):
        observation = recommendation.get("observation", "").lower()

        contradictions = []
        consistent_movements = []

        for joint_name, measurement in joint_movements.items():
            aliases = joint_aliases.get(joint_name, [joint_name])

            movement_terms = r"(flexion|extension|flexed|extended)"

            matches = []

            for alias in aliases:
                escaped_alias = re.escape(alias)

                # Example: "left elbow flexion"
                direct_pattern = (
                    rf"\b{escaped_alias}\b\s+{movement_terms}\b"
                )

                # Example: "flexion in the left arm"
                reverse_pattern = (
                    rf"\b{movement_terms}\b\s+"
                    rf"(?:in|of|at)\s+(?:the\s+)?"
                    rf"{escaped_alias}\b"
                )

                matches.extend(re.findall(direct_pattern, observation))
                matches.extend(re.findall(reverse_pattern, observation))

            for description in matches:
                described_movement = (
                    "flexion"
                    if description in ("flexion", "flexed")
                    else "extension"
                )

                result = {
                    "joint": joint_name,
                    "measured_movement": measurement["movement"],
                    "described_movement": described_movement
                }

                if described_movement != measurement["movement"]:
                    contradictions.append(result)
                else:
                    consistent_movements.append(result)

        results.append({
            "recommendation_index": index,
            "contradictions": contradictions,
            "consistent_movements": consistent_movements,
            "description_check_passed": (
                len(contradictions) == 0
            ),
            "checked_movement_count": (
                len(contradictions) + len(consistent_movements)
            )
        })

    return results


