# instead of calling gemini again, im building a rule-based coaching layer that reads the evidence report
# 1st responsibility is to
"""
Distinguish between:
Supported Observations --> can contribute to feedback
Contradicted Observations --> must be flagged for review
Inconclusive Observations --> Should be described as uncertain/omitted from specific coaching claims
"""

import json
from pathlib import Path
import os
import hashlib
import argparse
import shutil
import tempfile
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from coaching_schema import CoachingAnalysis, CoachingRecommendation


GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 4096

BASE_DIR = Path(__file__).resolve().parent

EVIDENCE_PATH = BASE_DIR / "evidence_report_v2.json"
COACHING_OUTPUT_PATH = BASE_DIR / "coaching_analysis_v2.json"
COACHING_METADATA_PATH = BASE_DIR / "coaching_metadata_v2.json"


def load_evidence_report(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Evidence report not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    required_fields = [
        "schema_version",
        "video_file",
        "movement_type",
        "reconciliations"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in report
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required evidence fields: {missing_fields}"
        )

    if report["schema_version"] != 2:
        raise ValueError(
            "Expected evidence schema version 2."
        )


    if not isinstance(report["reconciliations"], list):
        raise ValueError(
            "Reconciliations must be a list."
        )

    return report

# I want to separate observations into three categories before sending it ot Gemini for coaching reccs
# supported, contradicted, inconclusive
def categorize_evidence(report):
    categories = {
        "supported": [],
        "contradicted": [],
        "inconclusive": [],
        "insufficient_evidence": []
    }

    for reconciliation in report["reconciliations"]:
        status = reconciliation.get("status")

        if status in categories:
            categories[status].append(reconciliation)
        else:
            raise ValueError(
                f"Unexpected reconciliation status: {status}"
            )

    return categories



def build_coaching_prompt(evidence, categorized):
    evidence_json = json.dumps(
        categorized,
        indent=2,
        ensure_ascii=False
    )

    visual_context_json = json.dumps(
        evidence.get("visual_observations", []),
        indent=2,
        ensure_ascii=False
    )


    prompt = f"""
You are an AI climbing coach analyzing an indoor bouldering movement.

Your task is to generate actionable, evidence-grounded
coaching recommendations for the climber.

VIDEO:
{evidence["video_file"]}

MOVEMENT:
{evidence["movement_type"]}

RECONCILED EVIDENCE:
{evidence_json}

FRAME-LEVEL VISUAL CONTEXT:
{visual_context_json}

VISUAL CONTEXT RULES:

1. Frame-level visual observations come from Gemini and
   have not been independently verified.

2. Use these observations to identify possible movement
   phases and provide timing context for coaching experiments.

3. Do not treat visual observations as independently
   supported evidence claims.

4. Do not infer that visible foot contact establishes
   secure heel engagement, weight transfer, or stability.

5. If timing depends on an unverified visual observation,
   phrase the instruction conditionally.

6. Every coaching recommendation must still reference
   at least one supported reconciliation claim using
   its exact "claim" text.

COACHING RULES:

1. Use supported evidence to identify observed movement patterns.

2. Do not assume that an observed movement is a technique mistake.
   A hip swing, knee flexion, or bent elbow may be intentional.

3. Treat contradicted evidence as unresolved.
   Do not choose one model's interpretation as correct.

4. Treat inconclusive evidence as uncertain.

5. Treat insufficient_evidence as unavailable or unreliable
   measurement data.

   Do not interpret missing MediaPipe measurements as
   confirmation or contradiction of Gemini's observations.

   You may describe a movement suggested by visual evidence,
   but explicitly state that it was not independently verified.

   Do not invent joint angles or claim that missing measurements
   establish a specific technique problem.
   
   Every recommendation must reference at least one supported
   evidence claim using its exact "claim" text.

6. Do not invent hold positions, joint angles, body movements,
   or climbing intentions that are not supported by the evidence.

7. Distinguish observations from suggested technique adjustments.

8. Make recommendations actionable and specific, but phrase
   uncertain benefits as possibilities rather than guarantees.

9. Include conflicting observations in needs_review.

10. If the evidence does not justify a useful technique
    recommendation, return an empty recommendations list.


ACTIONABLE COACHING REQUIREMENTS:

For each suggested_adjustment:

1. Provide a concrete movement cue or controlled experiment
   that the climber can realistically try.

2. When the evidence provides identifiable movement phases,
   specify when the adjustment should be applied relative
   to those phases.

   For example, distinguish between:
   - Preparing for a foot repositioning
   - Moving the foot toward a new hold
   - Establishing contact with the new hold
   - Adjusting body position after contact

   Only reference phases that are supported by the
   provided evidence.

   Do not invent exact timing, contact confirmation,
   or weight transfer that cannot be verified.

   If precise timing is uncertain, describe the
   adjustment as a conditional experiment rather
   than a verified correction.

3. Explain what the climber should observe or compare to
   determine whether the adjustment is helpful.

4. Distinguish clearly between a measured movement problem
   and a technique hypothesis worth testing.

5. Do not invent hold positions, wall angles, foot contact,
   movement timing, or other missing context.

6. Do not recommend a technique solely because it is a
   common climbing principle, such as keeping arms straight
   or maintaining core tension.

7. If the evidence is insufficient to justify a specific
   correction, propose a cautious experiment or identify
   what additional information would be needed.

8. Prioritize one clear, practical adjustment over several
   vague or speculative suggestions.

Return a structured coaching analysis matching the provided schema.
"""
    return prompt
# rn get_coaching_analysis checks whether a specific coaching analysis exists, it doesn't check
# the coaching analysis that was generated using the latest evidence report.
# idea with this function is to use an evidence fingerprint (SHA-256 hash lf the evidence report)
def calculate_evidence_fingerprint(evidence):
    serialized = json.dumps(
        evidence,
        sort_keys = True,
        separators=(",", ":"),
        ensure_ascii= False
    )
    # generates a reproducible fingerprint based on the evidence report's contents
    # if there are any changes in evidence values, the fingerprint also changes
    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def calculate_coaching_fingerprint(evidence, prompt):
    fingerprint_data = {
        "evidence_fingerprint": calculate_evidence_fingerprint(evidence),
        "coaching_prompt": prompt,
        "gemini_model": GEMINI_MODEL,
        "temperature": GEMINI_TEMPERATURE,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "coaching_schema": CoachingAnalysis.model_json_schema()
    }

    serialized = json.dumps(
        fingerprint_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

# converts pydantic coaching analysis into a json compatible dict
# serializes consistently using sorted keys
# calculates its SHA256 fingerprint
# returns 64 char hexadecimal string
def calculate_analysis_fingerprint(analysis):
    analysis_data = analysis.model_dump(mode="json")
    serialized = json.dumps(
        analysis_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

# fun used to check if saved coaching reccs are outdated
# checks different conditions
# 1) generation configuration fingerprint differs, 2) analysis fingerprint differs
# 3) analysis fingerprint missing, 4) both fingerprints match
def check_coaching_cache(evidence, prompt, coaching_path, metadata_path):
    # Check whether the coaching analysis exists.
    if not coaching_path.is_file():
        return "missing"

    # Without metadata, the cache cannot be verified.
    if not metadata_path.is_file():
        return "unverified"

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError, TypeError):
        return "unverified"

    # Verify the generation configuration fingerprint.
    stored_coaching_fingerprint = metadata.get("coaching_fingerprint")

    if not isinstance(stored_coaching_fingerprint, str):
        return "unverified"

    expected_coaching_fingerprint = calculate_coaching_fingerprint(
        evidence,
        prompt
    )

    if stored_coaching_fingerprint != expected_coaching_fingerprint:
        return "outdated"

    # Verify the actual coaching analysis fingerprint.
    stored_analysis_fingerprint = metadata.get("analysis_fingerprint")

    if not isinstance(stored_analysis_fingerprint, str):
        return "unverified"

    try:
        analysis = CoachingAnalysis.model_validate_json(
            coaching_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "invalid"

    current_analysis_fingerprint = calculate_analysis_fingerprint(
        analysis
    )

    if stored_analysis_fingerprint != current_analysis_fingerprint:
        return "invalid"

    return "valid"


# loads existing coaching analysis
# calls gemini only when the saved file is missing
# validates the analysis before using/saving it

def get_coaching_analysis(prompt,evidence,categorized,coaching_path,metadata_path):
    status = get_cached_analysis_status(
        evidence,
        categorized,
        prompt,
        coaching_path,
        metadata_path
    )

    print(f"Detailed cache status: {status}")

    if status == "verified":
        print("Loading verified coaching analysis...")
        return load_coaching_analysis(coaching_path)

    print("No verified coaching analysis is available.")
    print("Gemini will NOT be called automatically.")

    return None

def generate_coaching_analysis(prompt):
    project_root = BASE_DIR.parent

    load_dotenv(project_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_schema=CoachingAnalysis
            )
        )

    except errors.ServerError as error:
        print(f"Gemini server error: {error}")
        raise SystemExit(1)

    except errors.ClientError as error:
        if error.code == 402:
            print(
                "Gemini prepaid credits are depleted. "
                "Check your AI Studio billing."
            )
        elif error.code == 429:
            print(
                "Gemini rate limit or quota exceeded."
            )
        else:
            print(f"Gemini client error: {error}")

        raise SystemExit(1)

    analysis = response.parsed

    if analysis is None:
        raise ValueError(
            "Gemini did not return a valid coaching analysis."
        )

    return analysis


def validate_coaching_analysis(analysis, categorized):
    supported_claims = {
        record["claim"]
        for record in categorized["supported"]
    }

    errors = []

    for index, recommendation in enumerate(analysis.recommendations, start=1):
        if not recommendation.supporting_evidence:
            errors.append(
                f"Recommendation {index} has no supporting evidence."
            )
            continue
        for claim in recommendation.supporting_evidence:
            if claim not in supported_claims:
                errors.append(
                    f"Recommendation {index} references "
                    f"unsupported evidence: {claim}"
                )
    return errors


# if writing failed before it could finish in the destination file, the file could be incomplete
# fun creates a temp file within the same directory, close the writing before replacing destination
# uses os.replace() to replace the file atomically on same filesystem
def atomic_write_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

            json.dump(data,temp_file,indent=2,ensure_ascii=False,
            )

        os.replace(temp_path, output_path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

# analysis.model_dump converts pydantic model into json-compatible python dict
def save_coaching_analysis(analysis, output_path):
    # atomic write func handles sserialization, writing to a temp file and replaces the destination
    atomic_write_json(
        analysis.model_dump(mode="json"),
        output_path
    )

    print(f"\nCoaching analysis saved to: {output_path}")


# reads our saved recommendations json file

def load_coaching_analysis(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Coaching analysis file not foudn: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # checks if the data matches the pydantic schema
    analysis = CoachingAnalysis.model_validate(data)

    return analysis


def save_coaching_metadata(evidence, prompt, metadata_path, analysis):
    metadata = {
        "evidence_fingerprint": calculate_evidence_fingerprint(evidence),
        "coaching_fingerprint": calculate_coaching_fingerprint(
            evidence,
            prompt
        ),
        "analysis_fingerprint": calculate_analysis_fingerprint(analysis),
        "schema_version": evidence["schema_version"]
    }

    atomic_write_json(metadata, metadata_path)

    print(f"Coaching metadata saved to: {metadata_path}")

def register_existing_analysis(evidence, categorized, prompt, coaching_path, metadata_path):
    if not coaching_path.is_file():
        raise FileNotFoundError(
            f"No existing coaching analysis: {coaching_path}"
        )

    if metadata_path.is_file():
        raise ValueError(
            "Metadata already exists. Registration was not performed"
        )

    analysis = load_coaching_analysis(coaching_path)
    validation_errors = validate_coaching_analysis(analysis, categorized)

    if validation_errors:
        raise ValueError(
            "Existing coaching analysis failed validation:\n"
            + "\n".join(validation_errors)
        )

    save_coaching_metadata(evidence, prompt, metadata_path)

    print("Existing coaching analysis registered successfully!")

# fun that generates fresh reccs when you deliberately request them
# this function makes a billable gemini API request! --> if validation fails, the newly generated analysis won't
# overwrite your saved coaching json
# workflow: req new analysis -> validate schema + video references ->
# if validation passes -> archive existing coaching files
# if validation fails -> stop without modifying.saved coaching
# save new analysis and metadata
def regenerate_coaching_analysis(prompt, evidence, categorized, coaching_path, metadata_path):
    print("Requesting a NEW coaching analysis from Gemini... ")

    analysis = generate_coaching_analysis(prompt)

    validation_errors = validate_coaching_analysis(analysis, categorized)
    if validation_errors:
        raise ValueError(
            "Generated coaching analysis failed validation:\n"
            + "\n".join(validation_errors)
        )

    archive_dir = None

    # Archive the previous analysis before replacing it.
    if coaching_path.is_file():
        print("Existing coaching analysis found. Archiving...")

        archive_dir = archive_cached_coaching(
            coaching_path,
            metadata_path
        )

    try:
        save_coaching_analysis(analysis, coaching_path)
        save_coaching_metadata(evidence, prompt, metadata_path, analysis)

    except Exception:
        print("Saving new coaching analysis failed.")

        if archive_dir is not None:
            print("Restoring previous coaching cache...")

            restore_cached_coaching(
                archive_dir,
                coaching_path,
                metadata_path
            )
        else:
            # No previous cache existed.
            coaching_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

        raise

    return analysis


# create backup analysis that preserves existing analysis instead of overwriting it
def backup_existing_metadata(metadata_path):
    if not metadata_path.is_file():
        return

    backup_path = metadata_path.with_suffix(".backup.jon")

    if backup_path.exists():
        print("Metadata backup already exists.")
        return

    shutil.copy2(metadata_path, backup_path)

    print(f"Metadata backup created: {backup_path}")


def get_event_identifier(evidence):
    frame_numbers = evidence.get("analyzed_frames", [])

    if not frame_numbers:
        raise ValueError(
            "Evidence report contains no analyzed frames."
        )

    if not all(
        isinstance(number, int) and not isinstance(number, bool)
        for number in frame_numbers
    ):
        raise ValueError(
            "Invalid frame numbers in evidence report."
        )

    start_frame = min(frame_numbers)
    end_frame = max(frame_numbers)

    return f"frames_{start_frame}_{end_frame}"

# fun to generate video-specific paths
def get_video_output_paths(evidence):
    video_name = Path(evidence["video_file"]).stem
    movement_type = evidence["movement_type"]

    event_identifier = get_event_identifier(evidence)

    output_dir = (
        BASE_DIR
        / "coaching_outputs"
        / video_name
        / movement_type
        / event_identifier
    )

    return {
        "directory": output_dir,
        "coaching": output_dir / "coaching_analysis_v2.json",
        "metadata": output_dir / "coaching_metadata_v2.json"
    }

# migration function to copy files
# creates new files, keeps original files intact
def migrate_existing_coaching_files(video_paths):
    video_dir = video_paths["directory"]

    files_to_copy = {
        COACHING_OUTPUT_PATH: video_paths["coaching"],
        COACHING_METADATA_PATH: video_paths["metadata"]
    }

    # ensure original coaching analysis exists
    if not COACHING_OUTPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Original coaching analysis not found: {COACHING_OUTPUT_PATH}"
        )

    # prevent overwriting prev migrated files
    for source, destination in files_to_copy.items():
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

    video_dir.mkdir(parents=True, exist_ok=True)

    for source, destination in files_to_copy.items():
        if source.is_file():
            shutil.copy2(source, destination)
            print(f"Copied: {source.name} -> {destination}")

    print("\nMigration completed successfully!")



def inspect_cached_coaching(
    evidence,
    categorized,
    prompt,
    coaching_path,
    metadata_path
):
    cache_status = check_coaching_cache(
        evidence,
        prompt,
        coaching_path,
        metadata_path
    )

    print(f"Coaching cache status: {cache_status}")

    if not coaching_path.is_file():
        print("No cached coaching analysis found.")
        return

    try:
        analysis = load_coaching_analysis(coaching_path)
    except (ValueError, OSError) as error:
        print(f"Cached coaching analysis failed schema validation: {error}")
        return

    print("Schema validation: PASSED")

    validation_errors = validate_coaching_analysis(
        analysis,
        categorized
    )

    if validation_errors:
        print("Evidence validation: FAILED")

        for error in validation_errors:
            print(f"- {error}")
    else:
        print("Evidence validation: PASSED")

    print(f"Recommendations: {len(analysis.recommendations)}")
    print(f"Needs review: {len(analysis.needs_review)}")
    print(f"Overall confidence: {analysis.overall_confidence}")

    if cache_status != "valid":
        print(
            "WARNING: This analysis is not verified against "
            "the current coaching configuration."
        )


def get_cached_analysis_status(
    evidence,
    categorized,
    prompt,
    coaching_path,
    metadata_path
):
    cache_status = check_coaching_cache(
        evidence,
        prompt,
        coaching_path,
        metadata_path
    )

    if cache_status == "missing":
        return "missing"

    try:
        analysis = load_coaching_analysis(coaching_path)
    except (ValueError, OSError):
        return "invalid"

    validation_errors = validate_coaching_analysis(
        analysis,
        categorized
    )

    if validation_errors:
        return "invalid"

    if cache_status == "valid":
        return "verified"

    if cache_status == "invalid":
        return "invalid"

    if cache_status == "unverified":
        return "locally_validated_unverified"

    return "outdated"

# safely archive outdated coaching analyses
def archive_cached_coaching(coaching_path, metadata_path):
    if not coaching_path.is_file():
        raise FileNotFoundError(
            f"No coaching analysis to archive: {coaching_path}"
        )
    archive_root = coaching_path.parent / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    # find next available archive version
    version = 1

    while(archive_root / f"version_{version}").exists():
        version +=1

    archive_dir = archive_root / f"version_{version}"
    archive_dir.mkdir()

    shutil.copy2(
        coaching_path,
        archive_dir / coaching_path.name
    )

    if metadata_path.is_file():
        shutil.copy2(
            metadata_path,
            archive_dir / metadata_path.name
        )

    print(f"Coaching cache archived to: {archive_dir}")

    return archive_dir

# validates against coaching analysis + reads metadata before overwriting
# corrupted metadata is detected before restoration
# restores only strucutrally valid coaching analyses, not just any JSON
def restore_cached_coaching(archive_dir, coaching_path, metadata_path):
    archived_coaching = archive_dir / coaching_path.name
    archived_metadata = archive_dir / metadata_path.name

    if not archived_coaching.is_file():
        raise FileNotFoundError(
            f"Archived coaching analysis not found: {archived_coaching}"
        )

    # STEP 1: Read and validate archived coaching BEFORE restoring.
    coaching_data = json.loads(
        archived_coaching.read_text(encoding="utf-8")
    )

    validated_analysis = CoachingAnalysis.model_validate(coaching_data)

    # STEP 2: Read archived metadata BEFORE modifying active files.
    metadata_data = None

    if archived_metadata.is_file():
        metadata_data = json.loads(
            archived_metadata.read_text(encoding="utf-8")
        )

        if not isinstance(metadata_data, dict):
            raise ValueError("Archived metadata must be a JSON object.")

    # STEP 3: Restore coaching only after archive validation succeeds.
    atomic_write_json(
        validated_analysis.model_dump(mode="json"),
        coaching_path
    )

    # STEP 4: Restore metadata or remove it if none existed.
    if metadata_data is not None:
        atomic_write_json(metadata_data, metadata_path)
    else:
        metadata_path.unlink(missing_ok=True)

    print("Previous coaching cache restored successfully!")



# 4 different modes once file runes
# generate_evidence_coaching.py --> loads verified cached coaching
# generated_evidence_coaching.py --mode register --> registers an existing analysis
# generated_evidence_coaching.py --mode regenerate --> generates + saves a new coaching analysis (API req)
# inspect
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and manage evidence-grounded climbing coaching."
    )

    parser.add_argument(
        "--mode",
        choices=["load", "register", "regenerate", "inspect"],
        default="load",
        help="Choose whether to load, register, or regenerate coaching."
    )

    parser.add_argument(
        "--evidence",
        type=Path,
        default=EVIDENCE_PATH,
        help="Path to the evidence report JSON file."
    )

    args = parser.parse_args()

    # Step 1: Load and categorize evidence
    evidence = load_evidence_report(args.evidence)
    categorized = categorize_evidence(evidence)

    print("Evidence report loaded successfully!")
    print(f"Video: {evidence['video_file']}")

    # Step 2: Build coaching prompt
    coaching_prompt = build_coaching_prompt(
        evidence,
        categorized
    )

    # Step 3: Determine video-specific paths
    video_paths = get_video_output_paths(evidence)
    coaching_path = video_paths["coaching"]
    metadata_path = video_paths["metadata"]


    print(
        "Detailed cache status:",
        get_cached_analysis_status(
            evidence,
            categorized,
            coaching_prompt,
            coaching_path,
            metadata_path
        )
    )
    print(f"Coaching directory: {video_paths['directory']}")

    # Step 4: Execute requested operation
    if args.mode == "register":
        register_existing_analysis(
            evidence,
            categorized,
            coaching_prompt,
            coaching_path,
            metadata_path
        )

    elif args.mode == "regenerate":
        regenerate_coaching_analysis(
            coaching_prompt,
            evidence,
            categorized,
            coaching_path,
            metadata_path
        )
    elif args.mode == "inspect":
        inspect_cached_coaching(
            evidence,
            categorized,
            coaching_prompt,
            coaching_path,
            metadata_path
        )
    else:
        analysis = get_coaching_analysis(
            coaching_prompt,
            evidence,
            categorized,
            coaching_path,
            metadata_path
        )

        if analysis is not None:
            validation_errors = validate_coaching_analysis(
                analysis,
                categorized
            )

            if validation_errors:
                print("\nCoaching analysis failed validation:")

                for error in validation_errors:
                    print(f"- {error}")

            else:
                print("\nCoaching analysis passed validation!")
                print(
                    f"Recommendations: {len(analysis.recommendations)}"
                )
                print(
                    f"Needs review: {len(analysis.needs_review)}"
                )
                print(
                    f"Overall confidence: {analysis.overall_confidence}"
                )
