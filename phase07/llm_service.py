
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from phase07.metrics_summary_service import summarize_metrics


# Load the existing Gemini API key.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
GEMINI_MODEL = "gemini-3.8-flash"

def generate_llm_feedback(metrics: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    # Prepare a compact summary of the climbing data.
    summary = summarize_metrics(metrics)

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an AI climbing coach analyzing movement data
extracted from a rock climbing video using MediaPipe.

Analyze the following metrics:

{json.dumps(summary, indent=2)}

Important guidelines:
- Base your observations on the supplied metrics.
- Do not claim you watched the video.
- Do not assume that a bent elbow indicates poor technique.
- Distinguish measured observations from possible interpretations.
- Do not invent movements, timestamps, or measurements.
- Explain limitations when the data is insufficient.
- Provide constructive, actionable climbing advice.

Return a JSON object with:
- overall_assessment: A brief summary of the movement data.
- strengths: A list of positive observations.
- improvements: A list of possible improvements.
- limitations: A list of important analysis limitations.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return json.loads(response.text)


if __name__ == "__main__":
    results_directory = PROJECT_ROOT / "phase06" / "backend" / "results"
    metrics_files = sorted(
        results_directory.glob("*/metrics.json"),
        key=lambda path: path.stat().st_mtime
    )

    if not metrics_files:
        raise FileNotFoundError("No metrics.json files found.")

    with open(metrics_files[-1], "r", encoding="utf-8") as file:
        metrics = json.load(file)

    feedback = generate_llm_feedback(metrics)

    print(json.dumps(feedback, indent=4))
