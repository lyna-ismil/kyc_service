"""
Document Engine — NeuroParser Approach
Advanced OpenCV preprocessing + EasyOCR for extracting the 8-digit CIN
number from Tunisian documents.

Handles:
  - Tunisian CIN card front (Arabic text with CIN number — primary)
  - Tunisian CIN card back (reserved for future demographic extraction)
  - Tunisian Permis de Conduire (bilingual, CIN in field #6)

All image processing happens 100% in RAM — no disk writes.
"""

import re
import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("kyc.document_engine")

# ---------------------------------------------------------------------------
# Lazy-loaded EasyOCR reader
# ---------------------------------------------------------------------------
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy-load EasyOCR reader with Arabic + French support."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        logger.info("Loading EasyOCR reader (ar + fr)...")
        t0 = time.time()
        _easyocr_reader = easyocr.Reader(["ar", "fr"], gpu=False, verbose=False)
        logger.info("EasyOCR reader loaded in %.1fs", time.time() - t0)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image helpers (all in RAM)
# ---------------------------------------------------------------------------
def _bytes_to_cv2(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to OpenCV BGR array (in RAM, no disk)."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error("Failed to decode image bytes: %s", e)
        return None


def _extract_8_digits(text: str) -> Optional[str]:
    """Find the first 8-consecutive-digit number in text."""
    match = re.search(r"\b(\d{8})\b", text)
    if match:
        return match.group(1)
    # Fallback: find any 8 consecutive digits even without word boundaries
    match = re.search(r"(\d{8})", text)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Advanced OpenCV preprocessing (NeuroParser pipeline)
# ---------------------------------------------------------------------------
def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    NeuroParser preprocessing pipeline to defeat guilloche background noise.

    Steps:
      1. Convert to grayscale
      2. Non-Local Means Denoising (h=10) — removes high-frequency noise
         while preserving text edges
      3. CLAHE (clipLimit=2.0, 8×8 grid) — boosts local contrast so black
         text pops against the washed-out security background

    Returns:
        Enhanced grayscale image as a numpy array.
    """
    logger.debug("NeuroParser: converting to grayscale")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    logger.debug("NeuroParser: applying Non-Local Means Denoising (h=10)")
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    logger.debug("NeuroParser: applying CLAHE (clipLimit=2.0, grid=8x8)")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


def _binarize_otsu(enhanced: np.ndarray) -> np.ndarray:
    """
    Otsu binarization fallback — converts the enhanced grayscale image to
    pure black & white for a second OCR pass.

    Returns:
        Binary (black text on white) image as a numpy array.
    """
    logger.debug("NeuroParser fallback: applying Otsu thresholding")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# EasyOCR extraction (with NeuroParser preprocessing)
# ---------------------------------------------------------------------------
def _ocr_extract_cin(img: np.ndarray) -> Optional[str]:
    """
    Run EasyOCR on an image with advanced preprocessing to extract
    the first 8-digit CIN number.

    Pipeline:
      1. Upscale small images to ≥1200px wide
      2. NeuroParser preprocessing (grayscale → denoise → CLAHE)
      3. EasyOCR pass #1 on the enhanced image
      4. If no CIN found, binarize via Otsu and run EasyOCR pass #2
    """
    try:
        reader = _get_easyocr_reader()

        # Upscale small images for better OCR accuracy
        h, w = img.shape[:2]
        processed = img.copy()
        if w < 1200:
            scale = 1200 / w
            processed = cv2.resize(processed, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
            logger.debug("Upscaled image from %dpx to %dpx wide", w, int(w * scale))

        # --- Pass 1: NeuroParser enhanced image ---
        logger.info("OCR Pass 1: running EasyOCR on NeuroParser-enhanced image")
        enhanced = _preprocess_for_ocr(processed)

        results = reader.readtext(enhanced, detail=0)
        all_text = " ".join(results)
        logger.debug("OCR Pass 1 extracted text: %s", all_text[:200])

        cin = _extract_8_digits(all_text)
        if cin:
            logger.info("CIN found on Pass 1 (enhanced): %s", cin)
            return cin

        # --- Pass 2: Otsu binarization fallback ---
        logger.info("OCR Pass 1 found no CIN — running Pass 2 with Otsu binarization")
        binary = _binarize_otsu(enhanced)

        results = reader.readtext(binary, detail=0)
        all_text = " ".join(results)
        logger.debug("OCR Pass 2 extracted text: %s", all_text[:200])

        cin = _extract_8_digits(all_text)
        if cin:
            logger.info("CIN found on Pass 2 (Otsu binary): %s", cin)
            return cin

        logger.warning("Both OCR passes failed to find an 8-digit CIN")
        return None

    except Exception as e:
        logger.exception("EasyOCR extraction error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_cin_from_id(cin_front_bytes: bytes, cin_back_bytes: bytes) -> Optional[str]:
    """
    Extract the 8-digit CIN number from the Tunisian ID card.

    Primary method: Run NeuroParser-enhanced OCR on the front of the CIN.
    The back image (cin_back_bytes) is accepted but reserved for future
    demographic extraction (address, date of birth, etc.).

    Args:
        cin_front_bytes: JPEG/PNG bytes of CIN front (face + Arabic text)
        cin_back_bytes: JPEG/PNG bytes of CIN back (reserved for future use)

    Returns:
        8-digit CIN number string, or None if extraction fails.
    """
    t0 = time.time()
    logger.info("Extracting CIN number from ID card (front-only OCR mode)...")

    # Note: cin_back_bytes is kept in the signature for API compatibility.
    # It will be used in a future release for demographic field extraction.
    if cin_back_bytes:
        logger.debug(
            "CIN back image received (%d bytes) — reserved for future "
            "demographic extraction, not used for CIN number.",
            len(cin_back_bytes),
        )

    # --- Primary: NeuroParser OCR on CIN front ---
    img_front = _bytes_to_cv2(cin_front_bytes)
    if img_front is not None:
        logger.info("Running NeuroParser OCR on CIN front...")
        cin = _ocr_extract_cin(img_front)
        if cin:
            elapsed = int((time.time() - t0) * 1000)
            logger.info("CIN from ID front OCR: %s (%dms)", cin, elapsed)
            return cin
    else:
        logger.error("Failed to decode CIN front image")

    elapsed = int((time.time() - t0) * 1000)
    logger.warning("Could not extract CIN from ID card (%dms)", elapsed)
    return None


def extract_cin_from_permis(permis_bytes: bytes) -> Optional[str]:
    """
    Extract the 8-digit CIN number from the Tunisian Permis de Conduire.

    Uses NeuroParser-enhanced EasyOCR (French + Arabic) and regex to find
    the CIN number. The CIN is typically in field #6 of the Permis.

    Args:
        permis_bytes: JPEG/PNG bytes of Permis de Conduire

    Returns:
        8-digit CIN number string, or None if extraction fails.
    """
    t0 = time.time()
    logger.info("Extracting CIN number from Permis de Conduire...")

    img = _bytes_to_cv2(permis_bytes)
    if img is None:
        logger.error("Failed to decode Permis image")
        return None

    cin = _ocr_extract_cin(img)

    elapsed = int((time.time() - t0) * 1000)
    if cin:
        logger.info("CIN from Permis: %s (%dms)", cin, elapsed)
    else:
        logger.warning("Could not extract CIN from Permis (%dms)", elapsed)

    return cin
