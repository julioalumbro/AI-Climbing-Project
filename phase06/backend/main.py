
from pathlib import Path
from uuid import UUID, uuid4
import json
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from phase06.backend.analysis_service import analyze_uploaded_video
from phase07.llm_service import (
    generate_llm_feedback,
    GEMINI_MODEL,
)

from phase07.coaching_cache_service import (
    create_coaching_fingerprint,
    load_cached_coaching,
    save_cached_coaching,
)
app = FastAPI(title="AI Climbing Coach API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# File storage directories
BACKEND_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


def validate_upload_id(upload_id: str) -> str:
    """
    Validate an upload ID and return its normalized UUID.
    """
    try:
        return UUID(hex=upload_id).hex
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid upload ID"
        )


# Step 1: Health check
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AI Climbing Coach",
        "phase": "6.3"
    }


# Step 2: Upload video
@app.post("/api/upload")
def upload_video(file: UploadFile = File(...)):
    original_filename = file.filename or "unnamed"

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    upload_id = uuid4().hex
    saved_filename = f"{upload_id}{extension}"
    destination = UPLOAD_DIR / saved_filename

    try:
        file_size = 0

        with destination.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                file_size += len(chunk)

                if file_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 200 MB limit"
                    )

                output.write(chunk)

        return {
            "upload_id": upload_id,
            "filename": original_filename,
            "saved_filename": saved_filename,
            "size_bytes": file_size,
            "status": "uploaded"
        }

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    except OSError:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to save uploaded video"
        )

    finally:
        file.file.close()


# Step 3: Analyze an uploaded video
@app.post("/api/analyze/{upload_id}")
def analyze_video(upload_id: str):
    validated_id = validate_upload_id(upload_id)

    try:
        return analyze_uploaded_video(validated_id)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Uploaded video not found"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        # Log the underlying error for debugging
        import logging
        logging.exception("Video analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Video analysis failed"
        )


# Step 4: Retrieve JSON movement metrics
@app.get("/api/results/{upload_id}/metrics")
def get_analysis_metrics(upload_id: str):
    validated_id = validate_upload_id(upload_id)

    metrics_path = (
        RESULTS_DIR
        / validated_id
        / "metrics.json"
    )

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Analysis metrics not found"
        )

    return FileResponse(
        path=metrics_path,
        media_type="application/json",
        filename="metrics.json",
        content_disposition_type="inline"
    )


# Step 5: Retrieve the annotated video
@app.get("/api/results/{upload_id}/video")
def get_annotated_video(upload_id: str):
    validated_id = validate_upload_id(upload_id)

    video_path = (
        RESULTS_DIR
        / validated_id
        / "annotated.mp4"
    )

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Annotated video not found"
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="annotated.mp4",
        content_disposition_type="inline"
    )


@app.post("/api/coaching/{upload_id}")
def generate_coaching(upload_id: str):
    import uuid

    try:
        parsed_id = uuid.UUID(upload_id)
        if parsed_id.version != 4 or parsed_id.hex != upload_id:
            raise ValueError("Invalid UUID")
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid upload ID."
        )

    results_dir = Path(__file__).resolve().parent / "results"
    result_dir = results_dir / upload_id

    metrics_path = result_dir / "metrics.json"
    cache_path = result_dir / "gemini_coaching.json"

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="No analysis metrics found for this video."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    fingerprint = create_coaching_fingerprint(
        metrics,
        GEMINI_MODEL,
    )

    # Check the cache before making an API call.
    cached_coaching = load_cached_coaching(
        cache_path,
        fingerprint,
    )

    if cached_coaching is not None:
        return {
            "upload_id": upload_id,
            "coaching": cached_coaching,
            "cached": True,
        }

    # Only call Gemini when no matching cache exists.
    coaching = generate_llm_feedback(metrics)

    save_cached_coaching(
        cache_path,
        fingerprint,
        coaching,
    )

    return {
        "upload_id": upload_id,
        "coaching": coaching,
        "cached": False,
    }


