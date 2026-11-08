
# compare analysis output structures to detect changes between processing paths.
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

original_path = BASE_DIR / "combined_movement_analysis_v2.json"
test_path = BASE_DIR / "combined_movement_analysis_test.json"

with open(original_path, "r", encoding="utf-8") as file:
    original = json.load(file)

with open(test_path, "r", encoding="utf-8") as file:
    test = json.load(file)


# recursively compare two results and report differing values.
def find_differences(original, test, path="root"):
    if type(original) is not type(test):
        print(f"{path}: Different types")
        print(f"  Original: {type(original).__name__}")
        print(f"  Test: {type(test).__name__}")
        return

    if isinstance(original, dict):
        for key in sorted(original.keys() | test.keys()):
            current_path = f"{path}.{key}"

            if key not in original:
                print(f"{current_path}: Missing from original")
            elif key not in test:
                print(f"{current_path}: Missing from test")
            else:
                find_differences(
                    original[key],
                    test[key],
                    current_path
                )

    elif isinstance(original, list):
        if len(original) != len(test):
            print(
                f"{path}: Different lengths "
                f"({len(original)} vs {len(test)})"
            )

        for index, (old, new) in enumerate(zip(original, test)):
            find_differences(
                old,
                new,
                f"{path}[{index}]"
            )

    elif original != test:
        print(f"{path}:")
        print(f"  Original: {original}")
        print(f"  Test: {test}")


if original == test:
    print("Combined analysis reproducibility test passed!")
else:
    print("Combined analyses differ:\n")
    find_differences(original, test)

gemini_path = BASE_DIR / "gemini_structured_analysis_v2.json"

with open(gemini_path, "r", encoding="utf-8") as file:
    gemini_data = json.load(file)

current_analysis = gemini_data["analysis"]

assert (
    test["gemini_left_elbow_movement"]
    == current_analysis["left_elbow_movement"]
)

assert (
    test["gemini_right_elbow_movement"]
    == current_analysis["right_elbow_movement"]
)

assert (
    test["gemini_movement_summary"]
    == current_analysis["movement_summary"]
)

print("Current Gemini input matches combined output!")
