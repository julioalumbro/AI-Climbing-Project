"""Phase05 Step 5.3.2: deterministic tests for recommendation evidence checks.

Run from phase05:
    python -m unittest discover -s tests -p "test_recommendation_safety.py" -v

These tests make no Gemini API calls and do not modify production files.
"""

import unittest

from evaluate_coaching import (
    evaluate_evidence_alignment,
    evaluate_uncertainty_handling,
)


# build a compact evidence report fixture for recommendation safety tests.
def evidence(*claim_status_pairs, visual_observations=()):
    """Build a minimal evidence report with reconciled and visual claims."""
    return {
        "reconciliations": [
            {"claim": claim, "status": status}
            for claim, status in claim_status_pairs
        ],
        "visual_observations": [
            {"claim": claim, "status": status}
            for claim, status in visual_observations
        ],
    }


# build a coaching response fixture with configurable supporting claims.
def coaching(*supporting_claims, uncertainty=""):
    """Build one synthetic coaching recommendation."""
    return {
        "recommendations": [
            {
                "observation": "Synthetic observation for testing only.",
                "suggested_adjustment": "Try a controlled comparison.",
                "supporting_evidence": list(supporting_claims),
                "uncertainty": uncertainty,
            }
        ]
    }


# verify that recommendations do not treat missing or contradicted claims as supported.
class RecommendationEvidenceTests(unittest.TestCase):
    """Test whether recommendations cite only supported, existing evidence."""

    # verify the expected behavior for supported evidence passes.
    def test_supported_evidence_passes(self):
        """Accept a recommendation referencing an existing supported claim."""
        result = evaluate_evidence_alignment(
            evidence(("Left elbow movement", "supported")),
            coaching("Left elbow movement"),
        )[0]
        self.assertTrue(result["evidence_alignment_passed"])
        self.assertEqual(result["supported_claims"], ["Left elbow movement"])

    # verify the expected behavior for missing claim is rejected.
    def test_missing_claim_is_rejected(self):
        """Reject a recommendation citing a claim absent from the evidence report."""
        result = evaluate_evidence_alignment(evidence(), coaching("Right knee movement"))[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["missing_claims"], ["Right knee movement"])

    # verify the expected behavior for contradicted claim is rejected.
    def test_contradicted_claim_is_rejected(self):
        """Reject a recommendation that uses disputed hip movement as support."""
        result = evaluate_evidence_alignment(
            evidence(("Hip movement direction", "contradicted")),
            coaching("Hip movement direction"),
        )[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["contradicted_claims"], ["Hip movement direction"])

    # verify the expected behavior for insufficient evidence is rejected.
    def test_insufficient_evidence_is_rejected(self):
        """Reject a recommendation supported only by unavailable right-knee tracking."""
        result = evaluate_evidence_alignment(
            evidence(("Right knee movement", "insufficient_evidence")),
            coaching("Right knee movement"),
        )[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["supported_claims"], [])

    # verify the expected behavior for unverified visual claim is rejected.
    def test_unverified_visual_claim_is_rejected(self):
        """Do not accept an unverified heel-contact observation as confirmed support."""
        result = evaluate_evidence_alignment(
            evidence(visual_observations=(("Right heel contact at frame 697", "not_independently_verified"),)),
            coaching("Right heel contact at frame 697"),
        )[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["unverified_claims"], ["Right heel contact at frame 697"])

    # verify the expected behavior for no supporting evidence is rejected.
    def test_no_supporting_evidence_is_rejected(self):
        """Reject a recommendation with an empty supporting_evidence list."""
        result = evaluate_evidence_alignment(
            evidence(("Left elbow movement", "supported")),
            coaching(),
        )[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["referenced_claims"], [])

    # verify the expected behavior for mixed supported and contradicted is rejected.
    def test_mixed_supported_and_contradicted_is_rejected(self):
        """Reject a mixed citation set even when one of its claims is supported."""
        result = evaluate_evidence_alignment(
            evidence(
                ("Left elbow movement", "supported"),
                ("Hip movement direction", "contradicted"),
            ),
            coaching("Left elbow movement", "Hip movement direction"),
        )[0]
        self.assertFalse(result["evidence_alignment_passed"])
        self.assertEqual(result["supported_claims"], ["Left elbow movement"])
        self.assertEqual(result["contradicted_claims"], ["Hip movement direction"])


# verify that uncertainty explanations are required when supporting evidence is weak.
class RecommendationUncertaintyTests(unittest.TestCase):
    """Test whether uncertain cited claims have an uncertainty explanation."""

    # verify the expected behavior for contradicted claim without explanation fails.
    def test_contradicted_claim_without_explanation_fails(self):
        """Flag a disputed hip claim when its uncertainty explanation is blank."""
        result = evaluate_uncertainty_handling(
            evidence(("Hip movement direction", "contradicted")),
            coaching("Hip movement direction", uncertainty="  "),
        )[0]
        self.assertFalse(result["uncertainty_check_passed"])
        self.assertEqual(result["uncertain_claims"], ["Hip movement direction"])

    # verify the expected behavior for contradicted claim with explanation passes uncertainty.
    def test_contradicted_claim_with_explanation_passes_uncertainty(self):
        """Accept a stated caveat about disputed hip evidence, not its evidence alignment."""
        report = evidence(("Hip movement direction", "contradicted"))
        analysis = coaching("Hip movement direction", uncertainty="Hip direction is disputed.")
        uncertainty = evaluate_uncertainty_handling(report, analysis)[0]
        alignment = evaluate_evidence_alignment(report, analysis)[0]
        self.assertTrue(uncertainty["uncertainty_check_passed"])
        self.assertFalse(alignment["evidence_alignment_passed"])

    # verify the expected behavior for unverified visual without explanation fails.
    def test_unverified_visual_without_explanation_fails(self):
        """Flag a visual heel-contact claim with no uncertainty explanation."""
        result = evaluate_uncertainty_handling(
            evidence(visual_observations=(("Heel contact", "not_independently_verified"),)),
            coaching("Heel contact"),
        )[0]
        self.assertFalse(result["uncertainty_check_passed"])
        self.assertEqual(result["uncertain_claims"], ["Heel contact"])

    # verify the expected behavior for unverified visual with explanation passes uncertainty.
    def test_unverified_visual_with_explanation_passes_uncertainty(self):
        """Accept a caveat about unverified contact while alignment still fails."""
        report = evidence(visual_observations=(("Heel contact", "not_independently_verified"),))
        analysis = coaching("Heel contact", uncertainty="Visual contact is not independently verified.")
        uncertainty = evaluate_uncertainty_handling(report, analysis)[0]
        alignment = evaluate_evidence_alignment(report, analysis)[0]
        self.assertTrue(uncertainty["uncertainty_check_passed"])
        self.assertFalse(alignment["evidence_alignment_passed"])

    # verify the expected behavior for supported claim needs no uncertainty explanation.
    def test_supported_claim_needs_no_uncertainty_explanation(self):
        """Allow a supported claim to pass uncertainty handling without a caveat."""
        result = evaluate_uncertainty_handling(
            evidence(("Left elbow movement", "supported")),
            coaching("Left elbow movement"),
        )[0]
        self.assertTrue(result["uncertainty_check_passed"])
        self.assertEqual(result["uncertain_claims"], [])

    # verify the expected behavior for insufficient evidence requires uncertainty.
    def test_insufficient_evidence_requires_uncertainty(self):
        """
        Ensure insufficient evidence requires an uncertainty explanation
        and cannot be treated as verified supporting evidence.
        """
        report = evidence(("Right knee movement", "insufficient_evidence"))
        analysis = coaching("Right knee movement")

        uncertainty = evaluate_uncertainty_handling(report, analysis)[0]
        alignment = evaluate_evidence_alignment(report, analysis)[0]

        # insufficient evidence without an explanation should fail.
        self.assertFalse(uncertainty["uncertainty_check_passed"])

        # the right knee should be identified as uncertain.
        self.assertEqual(
            uncertainty["uncertain_claims"],
            ["Right knee movement"]
        )

        # insufficient evidence cannot support a recommendation.
        self.assertFalse(alignment["evidence_alignment_passed"])

if __name__ == "__main__":
    unittest.main()
