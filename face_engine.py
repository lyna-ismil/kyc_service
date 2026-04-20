"""
Face Engine — MTCNN face extraction, DeepFace ArcFace verification,
and MediaPipe liveness detection.

All image processing is 100% in RAM — no disk writes, no file paths.
All models are lazy-loaded and cached in module-level globals.
"""

import logging
import time
import math
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("kyc.face_engine")

# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------
_mtcnn = None
_mp_face_mesh = None


def _get_mtcnn():
    """Lazy-load MTCNN face detector."""
    global _mtcnn
    if _mtcnn is None:
        logger.info("Loading MTCNN face detector...")
        t0 = time.time()
        from mtcnn import MTCNN
        _mtcnn = MTCNN(min_face_size=20)
        logger.info("MTCNN loaded in %.1fs", time.time() - t0)
    return _mtcnn


def _get_face_mesh():
    """Lazy-load MediaPipe FaceMesh for liveness detection."""
    global _mp_face_mesh
    if _mp_face_mesh is None:
        logger.info("Loading MediaPipe FaceMesh...")
        t0 = time.time()
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        logger.info("MediaPipe FaceMesh loaded in %.1fs", time.time() - t0)
    return _mp_face_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bytes_to_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to RGB numpy array (in RAM)."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception as e:
        logger.error("Failed to decode image to RGB: %s", e)
        return None


def _rgb_to_jpeg_bytes(rgb: np.ndarray) -> bytes:
    """Encode RGB numpy array to JPEG bytes (in RAM)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Face extraction from documents
# ---------------------------------------------------------------------------
def extract_face_from_document(image_bytes: bytes) -> Optional[bytes]:
    """
    Extract face from a document image (CIN front or Permis).

    Uses MTCNN to detect faces, picks the face with highest confidence
    (>= 0.60), adds 25% padding, and upscales to min 112×112 for ArcFace.

    Args:
        image_bytes: Raw JPEG/PNG bytes of document

    Returns:
        JPEG bytes of cropped face, or None if no face detected.
    """
    t0 = time.time()
    logger.info("Extracting face from document...")

    try:
        rgb = _bytes_to_rgb(image_bytes)
        if rgb is None:
            logger.error("Failed to decode CIN front image")
            return None

        detector = _get_mtcnn()
        detections = detector.detect_faces(rgb)

        if not detections:
            logger.warning("No faces detected in CIN front")
            return None

        logger.info("MTCNN found %d face(s) in CIN front", len(detections))

        # Pick face with highest confidence >= 0.60
        valid = [d for d in detections if d["confidence"] >= 0.60]
        if not valid:
            logger.warning(
                "All detected faces below confidence threshold (0.60). "
                "Best: %.2f", max(d["confidence"] for d in detections)
            )
            return None

        best = max(valid, key=lambda d: d["confidence"])

        x, y, w, h = best["box"]
        # Clamp negative coordinates (MTCNN can return negatives)
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        img_h, img_w = rgb.shape[:2]

        # Add 25% padding on each side
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        face_crop = rgb[y1:y2, x1:x2]

        # Upscale to minimum 112×112 for ArcFace
        crop_h, crop_w = face_crop.shape[:2]
        if crop_h < 112 or crop_w < 112:
            scale = max(112 / crop_w, 112 / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            face_crop = cv2.resize(
                face_crop, (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
            logger.info("Upscaled face from %dx%d to %dx%d",
                        crop_w, crop_h, new_w, new_h)

        face_bytes = _rgb_to_jpeg_bytes(face_crop)
        elapsed = int((time.time() - t0) * 1000)
        logger.info(
            "Face extracted: %dx%d, confidence=%.2f (%dms)",
            face_crop.shape[1], face_crop.shape[0],
            best["confidence"], elapsed
        )
        return face_bytes

    except Exception as e:
        logger.exception("Error extracting face from CIN front: %s", e)
        return None


# ---------------------------------------------------------------------------
# Face verification (ArcFace)
# ---------------------------------------------------------------------------
def verify_faces(
    id_face_bytes: bytes,
    selfie_bytes: bytes,
    threshold: float = 0.70,
) -> dict:
    """
    Verify that the face on the CIN card matches the selfie.

    Uses DeepFace with ArcFace model and cosine distance.
    All comparison uses numpy arrays — no file paths, no disk writes.

    Args:
        id_face_bytes: JPEG bytes of cropped face from CIN
        selfie_bytes: JPEG bytes of selfie
        threshold: minimum confidence to consider a match (default 0.70)

    Returns:
        dict with match, confidence, distance, threshold_used, reason
    """
    t0 = time.time()
    logger.info("Verifying faces (ArcFace, threshold=%.2f)...", threshold)

    result = {
        "match": False,
        "confidence": 0.0,
        "distance": 1.0,
        "threshold_used": threshold,
        "reason": "Face verification failed",
    }

    try:
        id_rgb = _bytes_to_rgb(id_face_bytes)
        selfie_rgb = _bytes_to_rgb(selfie_bytes)

        if id_rgb is None:
            result["reason"] = "Failed to decode ID card face image"
            return result
        if selfie_rgb is None:
            result["reason"] = "Failed to decode selfie image"
            return result

        from deepface import DeepFace

        # DeepFace.verify accepts numpy arrays directly (no disk)
        verification = DeepFace.verify(
            img1_path=id_rgb,
            img2_path=selfie_rgb,
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=False,
            silent=True,
        )

        distance = verification.get("distance", 1.0)
        confidence = max(0.0, min(1.0, 1.0 - distance))
        match = confidence >= threshold

        result["match"] = match
        result["confidence"] = round(confidence, 3)
        result["distance"] = round(distance, 3)

        if match:
            result["reason"] = "Face matches ID card photo"
        else:
            result["reason"] = (
                f"Face does not match ID card photo "
                f"(confidence {confidence:.0%}, required {threshold:.0%})"
            )

        elapsed = int((time.time() - t0) * 1000)
        logger.info(
            "Face verification: match=%s, confidence=%.3f, distance=%.3f (%dms)",
            match, confidence, distance, elapsed
        )

    except Exception as e:
        logger.exception("Face verification error: %s", e)
        result["reason"] = f"Face verification error: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Liveness check (MediaPipe)
# ---------------------------------------------------------------------------
def _landmark_dist(lm, idx1: int, idx2: int) -> float:
    """Euclidean distance between two MediaPipe landmarks."""
    p1 = lm[idx1]
    p2 = lm[idx2]
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def _compute_ear(lm, eye_indices: list[int]) -> float:
    """
    Compute Eye Aspect Ratio (EAR).
    eye_indices = [p0, p1, p2, p3, p4, p5]
    EAR = (dist(p1,p5) + dist(p2,p4)) / (2 * dist(p0,p3))
    """
    v1 = _landmark_dist(lm, eye_indices[1], eye_indices[5])
    v2 = _landmark_dist(lm, eye_indices[2], eye_indices[4])
    h = _landmark_dist(lm, eye_indices[0], eye_indices[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def check_liveness(selfie_bytes: bytes) -> dict:
    """
    Check liveness of a selfie using MediaPipe FaceMesh.

    Checks:
    - Eye Aspect Ratio (EAR) > 0.15 — eyes must be open
    - Yaw ratio < 0.22 — face must be roughly frontal
    - Overall confidence >= 0.45

    Args:
        selfie_bytes: Raw JPEG/PNG bytes of selfie

    Returns:
        dict with is_live, confidence, ear_left, ear_right, face_detected, reason
    """
    t0 = time.time()
    logger.info("Running liveness check...")

    result = {
        "is_live": False,
        "confidence": 0.0,
        "ear_left": 0.0,
        "ear_right": 0.0,
        "face_detected": False,
        "reason": "Liveness check failed",
    }

    try:
        rgb = _bytes_to_rgb(selfie_bytes)
        if rgb is None:
            result["reason"] = "Failed to decode selfie image"
            return result

        face_mesh = _get_face_mesh()
        mesh_results = face_mesh.process(rgb)

        if not mesh_results.multi_face_landmarks:
            result["reason"] = (
                "No face detected in selfie — "
                "ensure your face is clearly visible and well-lit"
            )
            return result

        result["face_detected"] = True
        lm = mesh_results.multi_face_landmarks[0].landmark

        # Left eye landmarks: [362, 385, 387, 263, 373, 380]
        left_eye = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks: [33, 160, 158, 133, 153, 144]
        right_eye = [33, 160, 158, 133, 153, 144]

        ear_left = _compute_ear(lm, left_eye)
        ear_right = _compute_ear(lm, right_eye)
        avg_ear = (ear_left + ear_right) / 2.0

        result["ear_left"] = round(ear_left, 3)
        result["ear_right"] = round(ear_right, 3)

        # Compute yaw (face frontal check)
        nose_tip = lm[1]
        left_corner = lm[33]
        right_corner = lm[263]
        face_center_x = (left_corner.x + right_corner.x) / 2.0
        face_width = abs(right_corner.x - left_corner.x)
        if face_width < 1e-6:
            face_width = 0.01
        yaw = abs(nose_tip.x - face_center_x) / face_width

        # Score computation
        ear_score = min(1.0, avg_ear / 0.22)
        yaw_score = max(0.0, 1.0 - (yaw / 0.25))
        confidence = ear_score * 0.65 + yaw_score * 0.35

        result["confidence"] = round(confidence, 3)
        result["is_live"] = confidence >= 0.45

        if result["is_live"]:
            result["reason"] = "Liveness check passed"
        else:
            reasons = []
            if avg_ear <= 0.15:
                reasons.append("eyes may be closed")
            if yaw >= 0.22:
                reasons.append("face is not frontal")
            if confidence < 0.45:
                reasons.append(f"liveness confidence too low ({confidence:.0%})")
            result["reason"] = (
                "Liveness check failed: " + ", ".join(reasons)
                if reasons else "Liveness check failed"
            )

        elapsed = int((time.time() - t0) * 1000)
        logger.info(
            "Liveness: live=%s, confidence=%.3f, ear=%.3f/%.3f, yaw=%.3f (%dms)",
            result["is_live"], confidence, ear_left, ear_right, yaw, elapsed
        )

    except Exception as e:
        logger.exception("Liveness check error: %s", e)
        result["reason"] = f"Liveness check error: {str(e)}"

    return result
