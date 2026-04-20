"""
NexDrive KYC Microservice — FastAPI application.

Gated pipeline for Tunisian user identity verification:
  1. Extract CIN number from ID card (barcode + OCR fallback)
  2. Extract CIN number from Permis de Conduire (OCR)
  3. Cross-match CIN numbers (hard gate)
  4. Extract face from CIN front (hard gate)
  5. Verify face against selfie (hard gate)
  6. Liveness check (hard gate)
  7. Return final result

All images processed 100% in RAM — never written to disk.
Only the extracted CIN number is returned.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import document_engine
import face_engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kyc.main")


# ---------------------------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------------------------
class KYCResponse(BaseModel):
    success: bool
    verified: bool
    stage_reached: str  # extract_id|extract_permis|document_match|face_extraction|face_match|liveness|complete
    failure_reasons: list[str]
    extracted_cin: Optional[str] = None
    face_match: Optional[dict] = None


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 NexDrive KYC Service starting up...")
    logger.info("Models will be loaded lazily on first request")
    yield
    logger.info("🛑 NexDrive KYC Service shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NexDrive KYC Service",
    description="Self-hosted identity verification for Tunisian users",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FILE_SIZE = 1000  # 1 KB


async def _read_and_validate_image(
    upload: UploadFile, field_name: str
) -> bytes:
    """
    Read upload file bytes and validate:
    - Size within limits
    - Decodes as a valid image via cv2
    Returns raw bytes.
    """
    data = await upload.read()

    if len(data) < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File '{field_name}' is too small ({len(data)} bytes, "
                   f"minimum {MIN_FILE_SIZE} bytes). "
                   f"Please upload a clear photo of the document.",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File '{field_name}' is too large ({len(data)} bytes, "
                   f"maximum {MAX_FILE_SIZE // (1024*1024)} MB).",
        )

    # Validate it's a decodable image
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"File '{field_name}' is not a valid image. "
                   f"Supported formats: JPEG, PNG.",
        )

    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Service info."""
    return {
        "service": "NexDrive KYC",
        "version": "1.0.0",
        "description": "Self-hosted identity verification for Tunisian users",
        "endpoints": {
            "health": "GET /health",
            "kyc_process": "POST /kyc/process",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "kyc",
        "ts": time.time(),
    }


@app.post("/kyc/process", response_model=KYCResponse)
async def kyc_process(
    cin_front: UploadFile = File(..., description="Front of CIN card (face photo & Arabic text)"),
    cin_back: UploadFile = File(..., description="Back of CIN card (1D barcode side)"),
    permis: UploadFile = File(..., description="Permis de Conduire"),
    selfie: UploadFile = File(..., description="Live selfie from front camera"),
):
    """
    Full KYC verification pipeline.

    Accepts 4 images as multipart/form-data:
    - cin_front: front of CIN card (face photo + Arabic text side)
    - cin_back: back of CIN card (1D barcode side)
    - permis: Permis de Conduire (driving license)
    - selfie: live selfie from front camera

    Returns verification result with extracted CIN number.
    """
    t0 = time.time()

    # ── Read and validate all images ──────────────────────────
    logger.info("Reading and validating uploaded images...")
    cin_front_bytes = await _read_and_validate_image(cin_front, "cin_front")
    cin_back_bytes = await _read_and_validate_image(cin_back, "cin_back")
    permis_bytes = await _read_and_validate_image(permis, "permis")
    selfie_bytes = await _read_and_validate_image(selfie, "selfie")
    logger.info("All 4 images validated successfully")

    # ── STEP 1: Extract CIN from ID card (HARD GATE) ─────────
    logger.info("=== STEP 1: Extract CIN from ID Card ===")
    cin_from_id = document_engine.extract_cin_from_id(cin_front_bytes, cin_back_bytes)

    if cin_from_id is None:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("Pipeline stopped at extract_id (%dms)", elapsed)
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="extract_id",
            failure_reasons=["Could not read ID card — ensure the barcode on the back is clearly visible"],
            extracted_cin=None,
        )

    # ── STEP 2: Extract CIN from Permis (HARD GATE) ──────────
    logger.info("=== STEP 2: Extract CIN from Permis ===")
    cin_from_permis = document_engine.extract_cin_from_permis(permis_bytes)

    if cin_from_permis is None:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("Pipeline stopped at extract_permis (%dms)", elapsed)
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="extract_permis",
            failure_reasons=["Could not read Driver's License — ensure the CIN number is clearly visible"],
            extracted_cin=None,
        )

    # ── STEP 3: Cross-match CIN numbers (HARD GATE) ──────────
    logger.info("=== STEP 3: Cross-Match Documents ===")
    logger.info("CIN from ID: %s, CIN from Permis: %s", cin_from_id, cin_from_permis)

    if cin_from_id != cin_from_permis:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning(
            "CIN mismatch: ID=%s, Permis=%s (%dms)",
            cin_from_id, cin_from_permis, elapsed
        )
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="document_match",
            failure_reasons=[
                f"Documents do not belong to the same person — "
                f"ID card CIN '{cin_from_id}' does not match "
                f"Driver's License CIN '{cin_from_permis}'"
            ],
            extracted_cin=None,
        )

    logger.info("CIN numbers match: %s ✓", cin_from_id)

    # ── STEP 4: Extract face from documents (Dual-Face) ──────
    logger.info("=== STEP 4: Dual-Face Extraction ===")
    cin_face = face_engine.extract_face_from_document(cin_front_bytes)
    permis_face = face_engine.extract_face_from_document(permis_bytes)

    if cin_face is None and permis_face is None:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("Pipeline stopped at face_extraction (%dms)", elapsed)
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="face_extraction",
            failure_reasons=["Could not detect a clear face on either the ID card or the Driver's License."],
            extracted_cin=cin_from_id,
        )

    # ── STEP 5: Face verification (OR GATE) ──────────────────
    logger.info("=== STEP 5: Face Verification (OR Gate) ===")
    best_face_result = None

    if cin_face is not None:
        logger.info("Verifying CIN face...")
        cin_result = face_engine.verify_faces(cin_face, selfie_bytes, threshold=0.70)
        if best_face_result is None or cin_result.get("confidence", 0) > best_face_result.get("confidence", 0):
            best_face_result = cin_result

    if permis_face is not None:
        logger.info("Verifying Permis face...")
        permis_result = face_engine.verify_faces(permis_face, selfie_bytes, threshold=0.70)
        if best_face_result is None or permis_result.get("confidence", 0) > best_face_result.get("confidence", 0):
            best_face_result = permis_result

    if not best_face_result or not best_face_result["match"]:
        elapsed = int((time.time() - t0) * 1000)
        reason = best_face_result["reason"] if best_face_result else "Face verification failed"
        logger.warning("Pipeline stopped at face_match (%dms): %s", elapsed, reason)
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="face_match",
            failure_reasons=["Face does not match the photo on the ID card or Driver's License."],
            extracted_cin=cin_from_id,
            face_match=best_face_result,
        )

    # ── STEP 6: Liveness check (HARD GATE) ───────────────────
    logger.info("=== STEP 6: Liveness Check ===")
    live_result = face_engine.check_liveness(selfie_bytes)

    if not live_result["is_live"]:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("Pipeline stopped at liveness (%dms): %s", elapsed, live_result["reason"])
        return KYCResponse(
            success=True,
            verified=False,
            stage_reached="liveness",
            failure_reasons=[f"Liveness check failed — {live_result['reason']}"],
            extracted_cin=cin_from_id,
            face_match=best_face_result,
        )

    # ── STEP 7: All passed ───────────────────────────────────
    elapsed = int((time.time() - t0) * 1000)
    logger.info("✅ KYC VERIFIED — CIN: %s (%dms)", cin_from_id, elapsed)

    return KYCResponse(
        success=True,
        verified=True,
        stage_reached="complete",
        failure_reasons=[],
        extracted_cin=cin_from_id,
        face_match=best_face_result,
    )
