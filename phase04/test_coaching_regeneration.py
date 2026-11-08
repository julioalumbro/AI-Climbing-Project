# test coaching regeneration, cache integrity, and safe output persistence.
"""Regression tests for Phase04 coaching cache integrity and recovery.

All coaching output files are created inside temporary directories. Calls to
generate_coaching_analysis are mocked in tests that exercise regeneration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from coaching_schema import CoachingAnalysis
from generate_evidence_coaching import (
    EVIDENCE_PATH,
    load_evidence_report,
    categorize_evidence,
    build_coaching_prompt,
    regenerate_coaching_analysis,
    check_coaching_cache,
    atomic_write_json,
    save_coaching_metadata,
    save_coaching_analysis,
    calculate_coaching_fingerprint,
    calculate_analysis_fingerprint,
    calculate_evidence_fingerprint,
)




# verify mock regeneration behavior and expected failure handling.
def test_mock_regeneration():
    """Verify mocked regeneration archives old coaching, saves the new analysis and matching fingerprints, and produces a valid cache."""

    # load real evidence, but never modify real coaching files.
    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    # use a temporary directory for all test outputs.
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # create an existing cached analysis.
        old_analysis = CoachingAnalysis(
            movement_summary="Previous cached analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="low"
        )

        coaching_path.write_text(
            old_analysis.model_dump_json(indent=2),
            encoding="utf-8"
        )

        # simulate a new gemini response.
        mock_analysis = CoachingAnalysis(
            movement_summary="New mock coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # patch gemini so no api request can occur.
        with patch(
            "generate_evidence_coaching.generate_coaching_analysis",
            return_value=mock_analysis
        ) as mocked_gemini:
            regenerate_coaching_analysis(
                prompt,
                evidence,
                categorized,
                coaching_path,
                metadata_path
            )

        mocked_gemini.assert_called_once_with(prompt)

        # check that the original analysis was archived.
        archive_path = (
            test_dir
            / "archive"
            / "version_1"
            / "coaching_analysis_v2.json"
        )

        assert archive_path.is_file()

        archived = CoachingAnalysis.model_validate_json(
            archive_path.read_text(encoding="utf-8")
        )

        assert archived.movement_summary == "Previous cached analysis"

        # check that the active analysis was replaced.
        current = CoachingAnalysis.model_validate_json(
            coaching_path.read_text(encoding="utf-8")
        )

        assert current.movement_summary == "New mock coaching analysis"

        # check that the metadata file was created.
        assert metadata_path.is_file()

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        # verify the new analysis fingerprint is stored.
        assert "analysis_fingerprint" in metadata

        expected_fingerprint = calculate_analysis_fingerprint(current)

        assert metadata["analysis_fingerprint"] == expected_fingerprint

        # verify metadata does not describe the old analysis.
        old_fingerprint = calculate_analysis_fingerprint(old_analysis)

        assert metadata["analysis_fingerprint"] != old_fingerprint

        # check that the new metadata matches current configuration.
        status = check_coaching_cache(
            evidence,
            prompt,
            coaching_path,
            metadata_path
        )

        assert status == "valid"

        print("\nAll mock regeneration checks passed!")
        print("Original analysis archived successfully.")
        print("New analysis saved successfully.")
        print("Analysis fingerprint stored correctly.")
        print("Fingerprint matches the new coaching analysis.")
        print("Fingerprint differs from the previous analysis.")
        print("Cache status: valid")
        print("Gemini API calls: 0")



# verify failed regeneration behavior and expected failure handling.
def test_failed_regeneration():
    """Reject a generated recommendation that cites unsupported evidence without changing the original cache or creating an archive."""
    from coaching_schema import CoachingRecommendation

    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        old_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="low"
        )

        coaching_path.write_text(
            old_analysis.model_dump_json(indent=2),
            encoding="utf-8"
        )

        original_contents = coaching_path.read_bytes()

        # simulate a recommendation referencing unsupported evidence.
        invalid_analysis = CoachingAnalysis(
            movement_summary="Invalid coaching analysis",
            recommendations=[
                CoachingRecommendation(
                    observation="The climber lost balance.",
                    suggested_adjustment="Move hips closer to the wall.",
                    supporting_evidence=[
                        "Nonexistent evidence claim"
                    ],
                    uncertainty="Movement cannot be verified."
                )
            ],
            needs_review=[],
            overall_confidence="medium"
        )

        with patch(
            "generate_evidence_coaching.generate_coaching_analysis",
            return_value=invalid_analysis
        ) as mocked_gemini:

            try:
                regenerate_coaching_analysis(
                    prompt,
                    evidence,
                    categorized,
                    coaching_path,
                    metadata_path
                )
            except ValueError as error:
                print("Expected validation error:", error)
            else:
                raise AssertionError(
                    "Invalid analysis was incorrectly accepted."
                )

        mocked_gemini.assert_called_once_with(prompt)

        # original coaching must remain unchanged.
        assert coaching_path.read_bytes() == original_contents

        # failed regeneration must not create an archive.
        assert not (test_dir / "archive").exists()

        # failed regeneration must not create metadata.
        assert not metadata_path.exists()

        print("\nFailed regeneration handled correctly!")
        print("Original coaching analysis preserved.")
        print("No archive created.")
        print("No new metadata written.")
        print("Gemini API calls: 0")

# verify atomic json write behavior and expected failure handling.
def test_atomic_json_write():
    """Verify atomic_write_json creates a JSON file and replaces its contents on a subsequent successful write."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.json"

        atomic_write_json(
            {"version": 1, "status": "original"},
            output_path
        )

        assert json.loads(
            output_path.read_text(encoding="utf-8")
        )["version"] == 1

        atomic_write_json(
            {"version": 2, "status": "updated"},
            output_path
        )

        result = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        assert result["version"] == 2
        assert result["status"] == "updated"

        print("\nAtomic JSON write test passed!")

# verify atomic write failure behavior and expected failure handling.
def test_atomic_write_failure():
    """Ensure JSON serialization errors preserve the destination bytes and clean up temporary files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.json"

        original_data = {
            "version": 1,
            "status": "original"
        }

        atomic_write_json(original_data, output_path)

        original_contents = output_path.read_bytes()

        # a set cannot be serialized to json by default.
        invalid_data = {
            "version": 2,
            "unsupported_value": {1, 2, 3}
        }

        try:
            atomic_write_json(invalid_data, output_path)
        except TypeError:
            print("Expected JSON serialization error caught.")
        else:
            raise AssertionError(
                "Invalid JSON was incorrectly accepted."
            )

        # the original file must remain unchanged.
        assert output_path.read_bytes() == original_contents

        # temporary files should be cleaned up.
        assert not list(Path(temp_dir).glob("*.tmp"))

        print("Original JSON preserved.")
        print("Temporary files cleaned up.")
        print("Atomic write failure test passed!")


# verify metadata write failure behavior and expected failure handling.
def test_metadata_write_failure():
    """Simulate metadata-save failure after coaching is written; verify archived originals are restored and the error propagates."""
    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # create an existing coaching analysis.
        old_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="low"
        )

        coaching_path.write_text(
            old_analysis.model_dump_json(indent=2),
            encoding="utf-8"
        )

        # create matching metadata for the original analysis.
        save_coaching_metadata(
            evidence,
            prompt,
            metadata_path,
            old_analysis
        )

        original_coaching = coaching_path.read_bytes()
        original_metadata = metadata_path.read_bytes()

        # simulate a new gemini response.
        new_analysis = CoachingAnalysis(
            movement_summary="New coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # simulate a failure specifically during metadata saving.
        with patch(
            "generate_evidence_coaching.generate_coaching_analysis",
            return_value=new_analysis
        ) as mocked_gemini, patch(
            "generate_evidence_coaching.save_coaching_metadata",
            side_effect=OSError("Simulated metadata write failure")
        ):

            try:
                regenerate_coaching_analysis(
                    prompt,
                    evidence,
                    categorized,
                    coaching_path,
                    metadata_path
                )
            except OSError as error:
                print("Expected write failure:", error)
            else:
                raise AssertionError(
                    "Metadata write failure was not detected."
                )

        mocked_gemini.assert_called_once_with(prompt)

        # confirm that both original files were archived.
        archive_dir = test_dir / "archive" / "version_1"

        assert (
            archive_dir / "coaching_analysis_v2.json"
        ).read_bytes() == original_coaching

        assert (
            archive_dir / "coaching_metadata_v2.json"
        ).read_bytes() == original_metadata

        # regeneration wrote new coaching before the metadata save failed.

        # verify both original files were restored after failure.
        assert json.loads(
            coaching_path.read_text(encoding="utf-8")
        ) == json.loads(
            original_coaching.decode("utf-8")
        )

        assert json.loads(
            metadata_path.read_text(encoding="utf-8")
        ) == json.loads(
            original_metadata.decode("utf-8")
        )

        print("\nMetadata failure recovery test passed!")
        print("Original coaching and metadata preserved in archive.")
        print("Original coaching and metadata restored successfully.")
        print("Gemini API calls: 0")

# verify coaching write failure behavior and expected failure handling.
def test_coaching_write_failure():
    """Simulate coaching-save failure; verify rollback restores both files and metadata saving is never attempted."""
    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # create an existing coaching analysis.
        old_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="low"
        )

        coaching_path.write_text(
            old_analysis.model_dump_json(indent=2),
            encoding="utf-8"
        )

        save_coaching_metadata(evidence, prompt, metadata_path, old_analysis)

        original_coaching = coaching_path.read_bytes()
        original_metadata = metadata_path.read_bytes()

        # simulate a new gemini response.
        new_analysis = CoachingAnalysis(
            movement_summary="New coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # simulate failure during coaching analysis saving.
        with patch(
                "generate_evidence_coaching.generate_coaching_analysis",
                return_value=new_analysis
        ) as mocked_gemini, patch(
            "generate_evidence_coaching.save_coaching_analysis",
            side_effect=OSError("Simulated coaching write failure")
        ), patch(
            "generate_evidence_coaching.save_coaching_metadata",
            wraps=save_coaching_metadata
        ) as mocked_metadata_save:

            try:
                regenerate_coaching_analysis(
                    prompt,
                    evidence,
                    categorized,
                    coaching_path,
                    metadata_path
                )
            except OSError as error:
                print("Expected write failure:", error)
            else:
                raise AssertionError(
                    "Coaching write failure was not detected."
                )

        # verify gemini was mocked.
        mocked_gemini.assert_called_once_with(prompt)

        # metadata saving should never be attempted.
        mocked_metadata_save.assert_not_called()

        # verify both original files were restored.
        assert json.loads(
            coaching_path.read_text(encoding="utf-8")
        ) == json.loads(original_coaching.decode("utf-8"))

        assert json.loads(
            metadata_path.read_text(encoding="utf-8")
        ) == json.loads(original_metadata.decode("utf-8"))

        # verify the archive contains the original files.
        archive_dir = test_dir / "archive" / "version_1"

        assert (
                       archive_dir / coaching_path.name
               ).read_bytes() == original_coaching

        assert (
                       archive_dir / metadata_path.name
               ).read_bytes() == original_metadata

        print("\nCoaching write failure recovery test passed!")
        print("Original coaching and metadata restored.")
        print("Archived files preserved.")
        print("Metadata saving was skipped.")
        print("Gemini API calls: 0")


# verify analysis fingerprint behavior and expected failure handling.
def test_analysis_fingerprint():
    """Confirm identical analyses hash identically, changed analyses hash differently, and the SHA-256 digest is 64 hex characters."""
    analysis_a = CoachingAnalysis(
        movement_summary="Climber bends left elbow.",
        recommendations=[],
        needs_review=[],
        overall_confidence="medium"
    )

    analysis_b = CoachingAnalysis(
        movement_summary="Climber bends left elbow.",
        recommendations=[],
        needs_review=[],
        overall_confidence="medium"
    )

    analysis_c = CoachingAnalysis(
        movement_summary="Climber extends left elbow.",
        recommendations=[],
        needs_review=[],
        overall_confidence="medium"
    )

    fingerprint_a = calculate_analysis_fingerprint(analysis_a)
    fingerprint_b = calculate_analysis_fingerprint(analysis_b)
    fingerprint_c = calculate_analysis_fingerprint(analysis_c)

    assert fingerprint_a == fingerprint_b
    assert fingerprint_a != fingerprint_c
    assert len(fingerprint_a) == 64

    print("Analysis fingerprint test passed!")
    print("Identical analyses produce matching fingerprints.")
    print("Different analyses produce different fingerprints.")
    print("Gemini API calls: 0")



# verify analysis fingerprint mismatch behavior and expected failure handling.
def test_analysis_fingerprint_mismatch():
    """Check the low-level cache validator rejects coaching modified without a corresponding metadata fingerprint update."""

    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        original_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # save a matching coaching analysis and metadata pair.
        save_coaching_analysis(original_analysis, coaching_path)

        save_coaching_metadata(
            evidence,
            prompt,
            metadata_path,
            original_analysis
        )

        # initially, both fingerprints should match.
        initial_status = check_coaching_cache(
            evidence,
            prompt,
            coaching_path,
            metadata_path
        )

        assert initial_status == "valid"

        # simulate an interrupted regeneration:
        # coaching changes, but metadata remains unchanged.
        modified_analysis = CoachingAnalysis(
            movement_summary="Modified coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(modified_analysis, coaching_path)

        # the cache should now detect the mismatch.
        modified_status = check_coaching_cache(
            evidence,
            prompt,
            coaching_path,
            metadata_path
        )

        assert modified_status == "invalid"

        print("\nAnalysis fingerprint mismatch test passed!")
        print("Matching coaching and metadata: valid")
        print("Modified coaching with old metadata: invalid")
        print("Gemini API calls: 0")



# verify reject mismatched coaching cache behavior and expected failure handling.
def test_reject_mismatched_coaching_cache():
    """Check higher-level cache status rejects mismatched coaching and the loader neither returns it nor invokes Gemini."""
    from generate_evidence_coaching import (
        get_cached_analysis_status,
        get_coaching_analysis
    )

    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # create a valid coaching analysis.
        original_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # save matching coaching and metadata.
        save_coaching_analysis(original_analysis, coaching_path)

        save_coaching_metadata(
            evidence,
            prompt,
            metadata_path,
            original_analysis
        )

        # verify the higher-level cache initially accepts it.
        initial_status = get_cached_analysis_status(
            evidence,
            categorized,
            prompt,
            coaching_path,
            metadata_path
        )

        assert initial_status == "verified"

        # simulate an interrupted regeneration.
        modified_analysis = CoachingAnalysis(
            movement_summary="Modified coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        # replace coaching without updating metadata.
        save_coaching_analysis(modified_analysis, coaching_path)

        invalid_status = get_cached_analysis_status(
            evidence,
            categorized,
            prompt,
            coaching_path,
            metadata_path
        )

        raw_cache_status = check_coaching_cache(
            evidence,
            prompt,
            coaching_path,
            metadata_path
        )

        print("Raw cache status:", raw_cache_status)
        print("Higher-level cache status:", invalid_status)

        assert raw_cache_status == "invalid"
        assert invalid_status == "invalid"

        # verify the loading function refuses the invalid cache.
        with patch(
            "generate_evidence_coaching.generate_coaching_analysis"
        ) as mocked_gemini:

            result = get_coaching_analysis(
                prompt,
                evidence,
                categorized,
                coaching_path,
                metadata_path
            )

        # invalid cache must not be returned.
        assert result is None

        # loading must not trigger gemini automatically.
        mocked_gemini.assert_not_called()

        print("\nMismatched coaching cache rejection test passed!")
        print("Original matching cache: verified")
        print("Modified coaching with old metadata: invalid")
        print("Invalid coaching was not loaded.")
        print("Gemini API calls: 0")


# verify legacy metadata behavior and expected failure handling.
def test_legacy_metadata():
    """Check metadata lacking an analysis fingerprint remains locally validated but unverified and cannot be auto-loaded."""
    from generate_evidence_coaching import (
        get_cached_analysis_status,
        get_coaching_analysis
    )

    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        analysis = CoachingAnalysis(
            movement_summary="Legacy coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(analysis, coaching_path)

        # simulate metadata created before analysis fingerprints existed.
        legacy_metadata = {
            "evidence_fingerprint": calculate_evidence_fingerprint(evidence),
            "coaching_fingerprint": calculate_coaching_fingerprint(
                evidence, prompt
            ),
            "schema_version": evidence["schema_version"]
        }

        atomic_write_json(legacy_metadata, metadata_path)

        raw_status = check_coaching_cache(
            evidence, prompt, coaching_path, metadata_path
        )

        detailed_status = get_cached_analysis_status(
            evidence, categorized, prompt, coaching_path, metadata_path
        )

        assert raw_status == "unverified"
        assert detailed_status == "locally_validated_unverified"

        with patch(
            "generate_evidence_coaching.generate_coaching_analysis"
        ) as mocked_gemini:
            result = get_coaching_analysis(
                prompt, evidence, categorized, coaching_path, metadata_path
            )

        assert result is None
        mocked_gemini.assert_not_called()

        print("\nLegacy metadata test passed!")
        print("Raw cache status: unverified")
        print("Detailed status: locally_validated_unverified")
        print("Unverified analysis was not loaded.")
        print("Gemini API calls: 0")

# simulates the on-disk state left by an interrupted regeneration
# verify interrupted regeneration recovery behavior and expected failure handling.
def test_interrupted_regeneration_recovery():
    """Simulate a crash between coaching and metadata writes, detect the mismatch, then restore the verified archive."""
    from generate_evidence_coaching import (
        archive_cached_coaching,
        restore_cached_coaching,
        get_cached_analysis_status
    )

    evidence = load_evidence_report(EVIDENCE_PATH)
    categorized = categorize_evidence(evidence)
    prompt = build_coaching_prompt(evidence, categorized)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # step 1: create an original verified cache.
        original_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(original_analysis, coaching_path)
        save_coaching_metadata(
            evidence,
            prompt,
            metadata_path,
            original_analysis
        )

        original_coaching = json.loads(
            coaching_path.read_text(encoding="utf-8")
        )
        original_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        initial_status = get_cached_analysis_status(
            evidence,
            categorized,
            prompt,
            coaching_path,
            metadata_path
        )

        assert initial_status == "verified"

        # step 2: archive the original cache.
        archive_dir = archive_cached_coaching(
            coaching_path,
            metadata_path
        )

        assert archive_dir.is_dir()

        # step 3: simulate interrupted regeneration.
        # new coaching is written, but metadata is not.
        new_analysis = CoachingAnalysis(
            movement_summary="New interrupted coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="high"
        )

        save_coaching_analysis(new_analysis, coaching_path)

        # step 4: verify that the cache is now invalid.
        interrupted_status = get_cached_analysis_status(
            evidence,
            categorized,
            prompt,
            coaching_path,
            metadata_path
        )

        assert interrupted_status == "invalid"

        # step 5: verify that archived files are intact.
        archived_coaching = json.loads(
            (archive_dir / coaching_path.name).read_text(
                encoding="utf-8"
            )
        )

        archived_metadata = json.loads(
            (archive_dir / metadata_path.name).read_text(
                encoding="utf-8"
            )
        )

        assert archived_coaching == original_coaching
        assert archived_metadata == original_metadata

        # step 6: restore the original cache.
        restore_cached_coaching(
            archive_dir,
            coaching_path,
            metadata_path
        )

        # step 7: verify the restored contents.
        restored_coaching = json.loads(
            coaching_path.read_text(encoding="utf-8")
        )

        restored_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        assert restored_coaching == original_coaching
        assert restored_metadata == original_metadata

        # step 8: verify the cache is trusted again.
        restored_status = get_cached_analysis_status(
            evidence,
            categorized,
            prompt,
            coaching_path,
            metadata_path
        )

        assert restored_status == "verified"

        print("\nInterrupted regeneration recovery test passed!")
        print("Original cache: verified")
        print("Interrupted cache: invalid")
        print("Archived files preserved.")
        print("Restored cache: verified")
        print("Gemini API calls: 0")


# verify restore missing archived metadata behavior and expected failure handling.
def test_restore_missing_archived_metadata():
    """Verify restoring legacy coaching without archived metadata removes any newer active metadata."""
    from generate_evidence_coaching import (
        archive_cached_coaching,
        restore_cached_coaching
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # create an older coaching analysis without metadata.
        original_analysis = CoachingAnalysis(
            movement_summary="Legacy coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(original_analysis, coaching_path)

        assert not metadata_path.exists()

        # archive the coaching analysis.
        archive_dir = archive_cached_coaching(
            coaching_path,
            metadata_path
        )

        assert (archive_dir / coaching_path.name).is_file()
        assert not (archive_dir / metadata_path.name).exists()

        # simulate an interrupted update.
        modified_analysis = CoachingAnalysis(
            movement_summary="Modified coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="high"
        )

        save_coaching_analysis(modified_analysis, coaching_path)

        # simulate metadata created by the interrupted update.
        atomic_write_json(
            {"temporary": True},
            metadata_path
        )

        # restore the original legacy cache.
        restore_cached_coaching(
            archive_dir,
            coaching_path,
            metadata_path
        )

        # verify the original coaching analysis was restored.
        restored = CoachingAnalysis.model_validate_json(
            coaching_path.read_text(encoding="utf-8")
        )

        assert restored == original_analysis

        # since the original cache had no metadata,
        # the temporary metadata should be removed.
        assert not metadata_path.exists()

        print("\nMissing archived metadata recovery test passed!")
        print("Original legacy coaching restored.")
        print("Temporary metadata removed.")
        print("Gemini API calls: 0")

# corrupted archive protection

# verify restore corrupted archive behavior and expected failure handling.
def test_restore_corrupted_archive():
    """Ensure structurally invalid archived coaching raises a Pydantic validation error before active files change."""
    from generate_evidence_coaching import (
        archive_cached_coaching,
        restore_cached_coaching
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        original_analysis = CoachingAnalysis(
            movement_summary="Original analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(original_analysis, coaching_path)
        atomic_write_json({"version": "original"}, metadata_path)

        # archive the original files.
        archive_dir = archive_cached_coaching(
            coaching_path,
            metadata_path
        )

        # simulate a newer active cache.
        new_analysis = CoachingAnalysis(
            movement_summary="Current active analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="high"
        )

        save_coaching_analysis(new_analysis, coaching_path)
        atomic_write_json({"version": "current"}, metadata_path)

        # record active files before attempting restoration.
        coaching_before = coaching_path.read_bytes()
        metadata_before = metadata_path.read_bytes()

        # corrupt the archived coaching structure.
        archived_coaching = archive_dir / coaching_path.name

        atomic_write_json(
            {"movement_summary": "Missing required fields"},
            archived_coaching
        )

        # restoration must reject the corrupted archive.
        try:
            restore_cached_coaching(
                archive_dir,
                coaching_path,
                metadata_path
            )
        except ValidationError as error:
            print(f"Expected restoration failure: {type(error).__name__}")
        else:
            raise AssertionError(
                "Corrupted archive was incorrectly restored."
            )

        # neither active file should have changed.
        assert coaching_path.read_bytes() == coaching_before
        assert metadata_path.read_bytes() == metadata_before

        print("\nCorrupted archive protection test passed!")
        print("Invalid archived coaching was rejected.")
        print("Active coaching and metadata were preserved.")
        print("Gemini API calls: 0")


# verify restore corrupted metadata behavior and expected failure handling.
def test_restore_corrupted_metadata():
    """Ensure malformed archived metadata raises JSONDecodeError before either active file changes."""
    from generate_evidence_coaching import (
        archive_cached_coaching,
        restore_cached_coaching
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        coaching_path = test_dir / "coaching_analysis_v2.json"
        metadata_path = test_dir / "coaching_metadata_v2.json"

        # step 1: create an original cache.
        original_analysis = CoachingAnalysis(
            movement_summary="Original coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="medium"
        )

        save_coaching_analysis(original_analysis, coaching_path)
        atomic_write_json({"version": "original"}, metadata_path)

        # step 2: archive the original files.
        archive_dir = archive_cached_coaching(
            coaching_path,
            metadata_path
        )

        # step 3: simulate newer active files.
        current_analysis = CoachingAnalysis(
            movement_summary="Current coaching analysis",
            recommendations=[],
            needs_review=[],
            overall_confidence="high"
        )

        save_coaching_analysis(current_analysis, coaching_path)
        atomic_write_json({"version": "current"}, metadata_path)

        # record the active files before restoration.
        coaching_before = coaching_path.read_bytes()
        metadata_before = metadata_path.read_bytes()

        # step 4: corrupt archived metadata with malformed json.
        archived_metadata = archive_dir / metadata_path.name

        archived_metadata.write_text(
            '{"version": "incomplete"',
            encoding="utf-8"
        )

        # step 5: attempt restoration.
        try:
            restore_cached_coaching(
                archive_dir,
                coaching_path,
                metadata_path
            )
        except json.JSONDecodeError as error:
            print(
                f"Expected metadata restoration failure: "
                f"{type(error).__name__}"
            )
        else:
            raise AssertionError(
                "Corrupted archived metadata was incorrectly restored."
            )

        # step 6: verify neither active file changed.
        assert coaching_path.read_bytes() == coaching_before
        assert metadata_path.read_bytes() == metadata_before

        print("\nCorrupted archived metadata test passed!")
        print("Malformed archived metadata was rejected.")
        print("Active coaching and metadata were preserved.")
        print("Gemini API calls: 0")

if __name__ == "__main__":
    test_mock_regeneration()
    test_failed_regeneration()
    test_atomic_json_write()
    test_atomic_write_failure()
    test_metadata_write_failure()
    test_coaching_write_failure()
    test_analysis_fingerprint()
    test_analysis_fingerprint_mismatch()
    test_reject_mismatched_coaching_cache()
    test_legacy_metadata()
    test_interrupted_regeneration_recovery()
    test_restore_missing_archived_metadata()
    test_restore_corrupted_archive()
    test_restore_corrupted_metadata()

