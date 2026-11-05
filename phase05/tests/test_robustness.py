"""Local robustness tests for Phase04 evidence categorization and Phase05 descriptions.

Run from project root: python -m unittest discover -s phase05/tests -p 'test_robustness.py' -v
No Gemini API calls are made.
"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'phase04'))
sys.path.insert(0, str(PROJECT_ROOT / 'phase05'))

from generate_evidence_coaching import categorize_evidence, build_coaching_prompt
import evaluate_coaching as evaluator


def report(*reconciliations, visual_observations=None):
    return {
        'video_file': 'synthetic_test.mov',
        'movement_type': 'heel_hook_readjustment',
        'reconciliations': list(reconciliations),
        'visual_observations': visual_observations or [],
    }


def claim(name, status):
    return {'claim': name, 'status': status}


class EvidenceCategorizationTests(unittest.TestCase):
    def test_missing_measurement_is_not_supported(self):
        """Ensure unavailable right-knee measurements remain insufficient evidence, not supported claims."""
        evidence = report(claim('Right knee movement', 'insufficient_evidence'))
        categories = categorize_evidence(evidence)
        self.assertEqual(categories['supported'], [])
        self.assertEqual(len(categories['insufficient_evidence']), 1)

    def test_contradiction_is_kept_separate_from_supported(self):
        """Ensure conflicting hip evidence stays contradicted while verified elbow evidence stays supported."""
        evidence = report(
            claim('Hip movement direction', 'contradicted'),
            claim('Left elbow movement', 'supported'),
        )
        categories = categorize_evidence(evidence)
        self.assertEqual([x['claim'] for x in categories['contradicted']], ['Hip movement direction'])
        self.assertEqual([x['claim'] for x in categories['supported']], ['Left elbow movement'])

    def test_inconclusive_is_not_promoted_to_supported(self):
        """Ensure uncertain reconciliation results are not incorrectly promoted to supported evidence."""
        categories = categorize_evidence(report(claim('Hip movement direction', 'inconclusive')))
        self.assertEqual(categories['supported'], [])
        self.assertEqual(len(categories['inconclusive']), 1)

    def test_unknown_status_fails_loudly(self):
        """Reject unexpected reconciliation statuses instead of silently accepting invalid evidence."""
        with self.assertRaises(ValueError):
            categorize_evidence(report(claim('Unexpected claim', 'unrecognized_status')))

    def test_prompt_marks_visual_observations_as_unverified(self):
        """Confirm frame-level observations and their unverified status appear in the coaching prompt."""
        evidence = report(
            claim('Left elbow movement', 'supported'),
            visual_observations=[{
                'frame': 697,
                'claim': 'Target foot foothold contact',
                'observed_value': 'yes',
                'status': 'not_independently_verified',
                'source': 'gemini',
            }],
        )
        prompt = build_coaching_prompt(evidence, categorize_evidence(evidence))
        self.assertIn('697', prompt)
        self.assertIn('not_independently_verified', prompt)
        self.assertIn('FRAME-LEVEL VISUAL CONTEXT:', prompt)


class JointDescriptionTests(unittest.TestCase):
    def evaluate(self, observation, ground_truth):
        coaching = {'recommendations': [{'observation': observation}]}
        with patch.object(evaluator, 'extract_joint_movement_ground_truth', return_value=ground_truth):
            return evaluator.evaluate_joint_movement_descriptions({}, coaching)[0]

    def test_supported_joint_description_passes(self):
        """Accept a left-elbow flexion description when the measured movement is flexion."""
        result = self.evaluate('Left elbow flexion was observed.', {'left elbow': {'movement': 'flexion'}})
        self.assertTrue(result['description_check_passed'])
        self.assertEqual(result['checked_movement_count'], 1)

    def test_reverse_phrase_is_detected(self):
        """Recognize reverse wording such as “flexion in the left arm” as left-elbow flexion."""
        result = self.evaluate('There was flexion in the left arm.', {'left elbow': {'movement': 'flexion'}})
        self.assertTrue(result['description_check_passed'])
        self.assertEqual(result['checked_movement_count'], 1)

    def test_contradictory_description_is_flagged(self):
        """Flag an extension description when the measured left-elbow movement is flexion."""
        result = self.evaluate('Left elbow extension was observed.', {'left elbow': {'movement': 'flexion'}})
        self.assertFalse(result['description_check_passed'])
        self.assertEqual(result['checked_movement_count'], 1)
        self.assertEqual(len(result['contradictions']), 1)

    def test_no_ground_truth_is_not_counted_as_checked(self):
        """Do not count a joint description as checked when no measurement is available."""
        result = self.evaluate('Left elbow flexion was observed.', {})
        self.assertEqual(result['checked_movement_count'], 0)
        self.assertEqual(result['contradictions'], [])
        # Current evaluator reports True when nothing is checked. This test
        # documents that behavior; it is NOT evidence that the claim is verified.

    def test_unrecognized_phrase_is_not_silently_counted(self):
        """Do not count a movement phrase that the current regex cannot recognize."""
        result = self.evaluate('The elbow bent slightly.', {'left elbow': {'movement': 'flexion'}})
        self.assertEqual(result['checked_movement_count'], 0)
        self.assertEqual(result['contradictions'], [])


if __name__ == '__main__':
    unittest.main()
