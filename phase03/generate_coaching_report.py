import json
from pathlib import Path
from traceback import format_tb

BASE_DIR = Path(__file__).resolve().parent
TECHNIQUE_FILE = BASE_DIR / "technique_feedback.json"

# open the feedback file
def loadTechniqueFeedback(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["technique_feedback"]

# generate human rep
def generateMovementReport(movement):
    foot = movement["foot"].capitalize()
    movement_type = movement["movement_type"].replace("_", " ").title()

    first_event = movement["frame_ranges"]["first_event"]
    second_event = movement["frame_ranges"]["second_event"]

    foot_feedback = movement["foot_analysis"]["coaching_feedback"]
    upper_feedback = movement["upper_body_analysis"]["coaching_feedback"]

    report = (
        f"Movement: {foot} Foot - {movement_type}\n"
        f"Source: {movement['source']}\n"
        f"First_Event: Frames {first_event['start']}-{first_event['end']}\n"
        f"Second_Event: {second_event['start']}-{second_event['end']}\n"
        f"\n"
        f"FOOT MOVEMENT\n"
        f"{foot_feedback['suggestion']}\n"
        f"\n"
        f"UPPER_BODY COORDINATION\n"
        f"{upper_feedback['suggestion']}\n"
        f"\n"
        f"Note: These observations require video review and "
        f"do not confirm technique errors."
    )

    return report

def saveCoachingReports(movements, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("AI CLIMBING COACH - MOVEMENT ANALYSIS REPORT\n")
        file.write("=" * 55 + "\n\n")

        for index, movement in enumerate(movements, start=1):
            report = generateMovementReport(movement)

            file.write(f"MOVEMENT: {index}\n")
            file.write("-" * 55 + "\n")
            file.write(report + "\n\n")
            file.write("=" * 55 + "\n\n")
    print(f"Coaching reports saved to: {output_path}")


if __name__ == "__main__":
    movements = loadTechniqueFeedback(TECHNIQUE_FILE)

    OUTPUT_FILE = BASE_DIR / "coaching_report.txt"

    saveCoachingReports(movements, OUTPUT_FILE)

    print(f"Total movements analyzed: {len(movements)}")