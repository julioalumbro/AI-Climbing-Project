
Phase05 — Coaching Evaluation and Robustness Testing
Overview
Phase05 evaluates the AI Climbing Coach recommendations produced in Phase04. The goal is to make coaching feedback more evidence-grounded, actionable, and appropriately cautious when pose tracking or visual interpretation is incomplete.

The pipeline combines deterministic local checks with an optional Gemini-based qualitative evaluation. Phase05 also compares coaching prompt versions, preserves evaluation outputs using fingerprints and caches, and tests how the evaluator behaves under missing or conflicting evidence.

Scope: A passing test or high Gemini evaluation score does not establish that a climbing recommendation is biomechanically correct or effective in practice. The current evaluation measures internal consistency, evidence references, and model-assessed coaching quality; human review remains important.

Relationship to Previous Phases
Phase01–02: Process climbing video with OpenCV and MediaPipe; extract body landmarks and movement metrics, including joint angles and hip motion.
Phase03: Identify movement events and select relevant frames.
Phase04: Use Gemini to interpret selected frames, combine visual observations with MediaPipe measurements, reconcile evidence, and generate structured coaching recommendations.
Phase05 (this phase): Evaluate recommendations, refine prompts, compare versions, and test evidence-handling robustness.
Phase06 (planned): Build a video-upload and coaching-visualization dashboard.
Evaluation Approach
1. Local deterministic checks
The local evaluator in evaluate_coaching.py checks the following:

Check	Purpose	Important limitation
evaluate_evidence_alignment()	Verify that each recommendation references existing supported evidence; reject missing, contradicted, insufficient, or visually unverified claims as supporting proof.	Checks references and statuses, not whether the suggested technique will work.
evaluate_uncertainty_handling()	Require a nonempty uncertainty explanation when a recommendation references contradicted, insufficient, or not-independently-verified evidence.	Checks that a caveat exists, not whether it is complete or accurate.
evaluate_joint_movement_descriptions()	Compare recognizable flexion/extension wording in coaching observations against MediaPipe-derived movement direction.	Uses limited regex patterns and treats MediaPipe as a reference measurement, not independently verified ground truth.
The evidence statuses used by the pipeline include:

supported — the reconciliation supports the claim.
contradicted — the available sources disagree under the reconciliation rules.
insufficient_evidence — the required measurement is unavailable or unreliable.
not_independently_verified — a visual observation has not been independently verified.
inconclusive — evidence does not support a definitive conclusion.
A recommendation may pass uncertainty handling while failing evidence alignment. For example, explaining that hip motion is disputed does not make disputed hip motion valid supporting evidence.

2. Optional Gemini qualitative evaluation
run_coaching_evaluation.py can evaluate recommendations on five dimensions, each scored from 1 to 5:

Dimension	What is assessed
Evidence accuracy	Does the recommendation accurately represent the available evidence?
Reasoning quality	Is the interpretation coherent and appropriately qualified?
Actionability	Can the climber test or apply the suggestion?
Climbing relevance	Does the suggestion address meaningful climbing mechanics?
Uncertainty handling	Are tracking gaps, contradictions, and unverified observations acknowledged?
The maximum score is 25/25. These are Gemini-generated judgments, not independent measurements of real-world coaching effectiveness.

3. Explicit API generation and caching
The evaluation workflow supports cached outputs and fingerprints to avoid unnecessary regeneration and help distinguish different versions of evidence, coaching, and evaluation results.

Running local tests does not call Gemini.
Use --generate only when you explicitly intend to request a Gemini evaluation.
Use a separate --cache path for a new experiment so earlier comparison outputs are preserved.
Review fingerprint/cache status before assuming a cached evaluation applies to a changed coaching output.
Climbing Test Cases
Case A — Left release and return
Video: climbingTestDeepBentElbow.mov
Movement: left_release_return
Frames: 332, 358, 373
Observed MediaPipe changes: left knee approximately −53.86°, left elbow −34.53°, right elbow −53.59° (flexion); hip horizontal displacement approximately +0.102 normalized image coordinates.
Reconciliation: hip, left-knee, and left-elbow claims supported; right-elbow claim contradicted; foot contact not independently verified.
Recorded Gemini evaluation: 23/25 for the improved coaching version. An earlier baseline scored 21/25.
Case B — Right heel-hook readjustment
Video: climbingTestTension.mov
Movement: heel_hook_readjustment
Target foot: right
Frames: 664 (before), 668 (transition), 697 (after)
Supported measurement: left-elbow flexion approximately −15.48°.
Contradicted interpretation: Gemini described image-right hip movement; MediaPipe measured approximately +0.0139 normalized horizontal displacement, below the configured 0.05 movement threshold.
Insufficient evidence: right-knee and right-elbow tracking.
Unverified visual context: the selected frames visually suggest a right-foot repositioning sequence, but do not independently establish secure heel contact, loading, or stability.
Coaching iteration
Version	Evidence	Reasoning	Actionability	Climbing relevance	Uncertainty	Total
Original heel-hook coaching	5	5	4	5	5	24/25
Timing-aware heel-hook coaching	5	5	5	5	5	25/25
The timing-aware prompt included frame-level observations while preserving their not_independently_verified status. The resulting recommendation proposed a controlled comparison: experiment with active left-arm pulling versus a more extended arm while preparing to lift and place the heel, then compare control and perceived effort. It did not assert that one strategy was universally superior.

Gemini's evaluation suggested adding tactile checkpoints, such as perceived hamstring engagement or foot-placement precision. This remains an idea for future iteration rather than a validated improvement.

Confidence distinction: The qualitative evaluator reported high confidence in its assessment, while the coaching output reported low confidence in the underlying movement interpretation because important tracking evidence was missing. The evaluator also indicated human review required.

Local Robustness Tests
Tests live under phase05/tests/:

phase05/
├── evaluate_coaching.py
├── run_coaching_evaluation.py
├── tests/
│   ├── test_robustness.py
│   └── test_recommendation_safety.py
└── README.md
test_robustness.py — 10 tests
Tests evidence categorization and joint-description consistency:

Missing joint measurements remain insufficient, not supported.
Contradicted evidence stays separate from supported evidence.
Inconclusive evidence is not promoted to supported.
Unknown reconciliation statuses raise an error.
Unverified visual context retains its status in the coaching prompt.
Supported joint descriptions are recognized.
Alternate wording such as “flexion in the left arm” is recognized.
Flexion/extension contradictions are flagged.
Missing joint reference measurements are not counted as checked.
Unrecognized phrasing is not falsely counted as checked.
test_recommendation_safety.py — 13 tests
Tests recommendation evidence references and uncertainty handling:

Supported evidence passes alignment.
Missing, contradicted, insufficient, and unverified visual evidence fail alignment.
Empty supporting-evidence lists fail alignment.
Mixed supported/contradicted citations fail alignment.
Contradicted and unverified claims require an uncertainty explanation.
Insufficient evidence also requires an uncertainty explanation (added during robustness testing).
Supported claims do not require an uncertainty explanation solely because they are supported.
Providing a caveat can pass uncertainty handling without making unsupported evidence valid.
Run the tests
From the phase05 directory:

# Run all local tests; no Gemini API calls
python -m unittest discover -s tests -p "test_*.py" -v
Or run each file separately:

python -m unittest discover -s tests -p "test_robustness.py" -v
python -m unittest discover -s tests -p "test_recommendation_safety.py" -v
Recorded results: test_robustness.py passed 10/10 tests; after updating the insufficient-evidence caveat rule, test_recommendation_safety.py passed 13/13 tests. These results were obtained in separate local runs; a final combined run is recommended after the latest code changes.

Reproducing the Heel-Hook Coaching Evaluation
From the phase05 directory, with the Phase04 evidence and coaching JSON files present:

# Local checks and cached evaluation only; no new Gemini request
python run_coaching_evaluation.py `
    --evidence ../phase04/evidence_report_tension_verified_v2.json `
    --coaching ../phase04/coaching_outputs/climbingTestTension/heel_hook_readjustment/frames_664_697/coaching_analysis_v2.json `
    --cache comparison_outputs/climbingTestTension/heel_hook_readjustment/frames_664_697/timing_v2/evaluation_cache.json
To explicitly allow a new Gemini evaluation, append --generate:

python run_coaching_evaluation.py `
    --evidence ../phase04/evidence_report_tension_verified_v2.json `
    --coaching ../phase04/coaching_outputs/climbingTestTension/heel_hook_readjustment/frames_664_697/coaching_analysis_v2.json `
    --cache comparison_outputs/climbingTestTension/heel_hook_readjustment/frames_664_697/timing_v2/evaluation_cache.json `
    --generate
The --generate option may incur API usage and should be used intentionally. A valid cached result may be reused depending on the script's fingerprint checks.

Output Preservation and Comparison
Comparison artifacts are organized by video, movement, and selected frame range. For example:

phase05/comparison_outputs/
└── climbingTestTension/
    └── heel_hook_readjustment/
        └── frames_664_697/
            ├── baseline_v1/
            └── timing_v2/
                └── evaluation_cache.json
Keep baseline coaching JSON, metadata, and evaluation cache files in a separate directory before regenerating coaching. Do not assume the baseline files have been archived unless their presence is verified locally. Avoid overwriting source videos, Phase04 evidence, or prior comparison results.

Known Limitations
Tracking and occlusion: Right-side joints may be unavailable, especially during heel-hook repositioning. Missing data is not evidence of no movement.
Reconciliation thresholds: A small measured hip displacement below a threshold does not prove zero motion; it only fails that particular movement criterion.
Visual observation versus verified contact: A frame can suggest foot placement without establishing load transfer, heel-hook security, or stability.
Regex coverage: Joint-description checks recognize a limited set of flexion/extension phrases. Zero recognized descriptions can currently produce description_check_passed=True; always inspect checked_movement_count alongside the pass flag.
Uncertainty checks are shallow: A nonempty explanation passes the local uncertainty-presence check even if the explanation is incomplete or misleading.
Evidence references are not biomechanics validation: A recommendation may cite supported measurements but still be ineffective, unsafe, or contextually inappropriate.
Model-based scoring is not independent validation: Gemini may rate its own ecosystem's outputs favorably; the sample comprises only two climbing movements.
Human review: A knowledgeable climber or coach should review recommendations, particularly when confidence is low or tracking is incomplete.
Phase05 Completion Checklist

Local evidence-alignment and uncertainty checks

Joint movement description checks

Gemini qualitative evaluation rubric

Heel-hook prompt refinement and version comparison

Fingerprint-aware evaluation caching and explicit generation flag

Evidence robustness tests (10 passing)

Recommendation safety tests (13 passing)

Treat insufficient_evidence as requiring an uncertainty explanation
Run the full combined suite after the latest change
Confirm baseline comparison artifacts are preserved
Human-review representative coaching outputs before dashboard presentation
Next: Phase06
Phase06 is planned to provide a user-facing interface for uploading climbing videos and viewing movement events, selected frames, measured metrics, evidence status, coaching recommendations, and uncertainty notices.

The dashboard should visually distinguish supported, contradicted, insufficient, and unverified claims. It should not display Gemini-generated coaching scores as proof of biomechanical correctness, and should surface tracking gaps rather than hiding them.

0/25