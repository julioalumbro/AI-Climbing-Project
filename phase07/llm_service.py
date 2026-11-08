
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from phase07.metrics_summary_service import summarize_metrics


# load the existing gemini api key.
# resolve the project root so environment settings can be loaded consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
GEMINI_MODEL = "gemini-3.8-flash"

# turn pose-derived metrics into a structured coaching response from gemini.
def generate_llm_feedback(metrics: dict) -> dict:
    # require an api key before attempting the model request.
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    # prepare a compact summary of the climbing data.
    summary = summarize_metrics(metrics)

    # initialize the gemini client for the coaching request.
    client = genai.Client(api_key=api_key)

    # constrain the model to metric-supported observations and explicit limitations.
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

    # request json output so the frontend can consume named feedback sections.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # stop if the model did not provide a usable response body.
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    # parse the generated json into a python dictionary.
    return json.loads(response.text)


# allow this module to be run directly against the latest saved metrics file.
if __name__ == "__main__":
    results_directory = PROJECT_ROOT / "phase06" / "backend" / "results"
    # select the most recently modified metrics file from saved video analyses.
    metrics_files = sorted(
        results_directory.glob("*/metrics.json"),
        key=lambda path: path.stat().st_mtime
    )

    if not metrics_files:
        raise FileNotFoundError("No metrics.json files found.")

    with open(metrics_files[-1], "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # generate coaching and print it in a readable json format.
    feedback = generate_llm_feedback(metrics)

    print(json.dumps(feedback, indent=4))
