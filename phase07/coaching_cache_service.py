import hashlib
import json
from pathlib import Path

CACHE_VERSION = "phase07-coaching-v1"

def create_coaching_fingerprint(
    metrics: dict,
    model_name: str,
) -> str:
    """Create a stable fingerprint for coaching inputs."""

    payload = {
        "cache_version": CACHE_VERSION,
        "model": model_name,
        "metrics": metrics,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def load_cached_coaching(
    cache_path: Path,
    fingerprint: str,
) -> dict | None:
    """Return cached coaching if its fingerprint matches."""

    if not cache_path.is_file():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            cached = json.load(file)

        if cached.get("fingerprint") != fingerprint:
            return None

        coaching = cached.get("coaching")

        if not isinstance(coaching, dict):
            return None

        return coaching

    except (OSError, ValueError, TypeError):
        return None


def save_cached_coaching(
    cache_path: Path,
    fingerprint: str,
    coaching: dict,
) -> None:
    """Save coaching atomically to avoid partial JSON files."""

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = cache_path.with_name(
        cache_path.name + ".tmp"
    )

    payload = {
        "fingerprint": fingerprint,
        "coaching": coaching,
    }

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    temp_path.replace(cache_path)
