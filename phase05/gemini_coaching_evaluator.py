
from google import genai
from google.genai import types

from coaching_evaluation_schema import CoachingQualityEvaluation
from gemini_evaluation_prompt import build_gemini_evaluation_prompt
import hashlib
import json
import os
import tempfile
from pathlib import Path

def generate_gemini_coaching_evaluation(
    evidence_report,
    coaching_analysis,
    *,
    client,
    model_name="gemini-3.8-flash"
):
    """
    Evaluate existing climbing coaching using Gemini.

    Makes one explicit Gemini API request when called.
    Validates the structured response with Pydantic.

    Does not save files or manage caching.
    """

    prompt = build_gemini_evaluation_prompt(
        evidence_report,
        coaching_analysis
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CoachingQualityEvaluation,
            temperature=0.2
        )
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty evaluation response."
        )

    evaluation = CoachingQualityEvaluation.model_validate_json(
        response.text
    )

    expected_indices = list(
        range(len(coaching_analysis.get("recommendations", [])))
    )

    actual_indices = sorted(
        item.recommendation_index
        for item in evaluation.recommendation_evaluations
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "Gemini did not evaluate each recommendation exactly once. "
            f"Expected indices: {expected_indices}. "
            f"Received indices: {actual_indices}."
        )

    return evaluation


def calculate_evaluation_fingerprint(evidence_report, coaching_analysis, model_name="gemini-3.8-flash",
                                     evaluator_version="1"):
    """
        Create a fingerprint identifying the exact inputs
        and evaluator configuration used for an evaluation.
        """

    payload = {
        "evidence_report": evidence_report,
        "coaching_analysis": coaching_analysis,
        "model_name": model_name,
        "evaluator_version": evaluator_version
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def load_cached_evaluation(cache_path, expected_fingerprint):
    """
    Load a previously validated Gemini evaluation.

    Return None if the cache is missing, malformed,
    or does not match the current inputs.
    """

    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            cached = json.load(file)

        if cached.get("fingerprint") != expected_fingerprint:
            return None

        return CoachingQualityEvaluation.model_validate(
            cached["evaluation"]
        )

    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached_evaluation(
    cache_path,
    fingerprint,
    evaluation
):
    """
    Atomically save a validated coaching evaluation
    alongside its input fingerprint.
    """

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    validated = CoachingQualityEvaluation.model_validate(
        evaluation
    )

    payload = {
        "fingerprint": fingerprint,
        "evaluation": validated.model_dump()
    }

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            json.dump(payload, temporary_file, indent=2)

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, cache_path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

