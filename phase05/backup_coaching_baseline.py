
import hashlib
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COACHING_DIR = (
    PROJECT_ROOT
    / "phase04/coaching_outputs/climbingTestDeepBentElbow"
    / "left_release_return/frames_332_373"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "phase05/evaluation_outputs/climbingTestDeepBentElbow"
    / "left_release_return/frames_332_373"
)

BACKUP_DIR = (
    PROJECT_ROOT
    / "phase05/comparison_outputs/climbingTestDeepBentElbow"
    / "left_release_return/frames_332_373"
)

# Adjust the metadata filename if yours differs.
FILES_TO_BACKUP = {
    COACHING_DIR / "coaching_analysis_v2.json":
        BACKUP_DIR / "baseline_coaching.json",

    COACHING_DIR / "coaching_metadata_v2.json":
        BACKUP_DIR / "baseline_coaching_metadata.json",

    EVALUATION_DIR / "coaching_quality_evaluation.json":
        BACKUP_DIR / "baseline_evaluation.json",
}


def calculate_file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Verify all source files exist before copying anything.
    for source in FILES_TO_BACKUP:
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing source file: {source}"
            )

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for source, destination in FILES_TO_BACKUP.items():
        # Do not overwrite an existing baseline.
        if destination.exists():
            if calculate_file_hash(source) != calculate_file_hash(destination):
                raise FileExistsError(
                    f"Existing baseline differs: {destination}"
                )
        else:
            shutil.copy2(source, destination)

        # Verify copied contents are identical.
        original_hash = calculate_file_hash(source)
        backup_hash = calculate_file_hash(destination)

        assert original_hash == backup_hash, (
            f"Backup verification failed: {destination}"
        )

        print(f"VERIFIED: {destination.name}")

    print("\nAll baseline files are safely backed up!")


if __name__ == "__main__":
    main()
