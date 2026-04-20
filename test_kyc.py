"""
Integration tests for NexDrive KYC Service.

Uses synthesized test images (no real ID cards needed).
Run with: python -m pytest test_kyc.py -v
"""

import re
import pytest
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Test image generators (all in RAM — no disk writes)
# ---------------------------------------------------------------------------

def _make_text_image(lines: list[str], width: int = 1500, line_height: int = 50) -> bytes:
    """Create a white image with monospace text lines. Returns JPEG bytes."""
    height = max(200, line_height * (len(lines) + 4))
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for i, line in enumerate(lines):
        y = (i + 2) * line_height
        cv2.putText(img, line, (30, y), font, font_scale, (0, 0, 0),
                    thickness, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _make_face_image(width: int = 200, height: int = 250) -> bytes:
    """Create a simple synthetic face-like image. Returns JPEG bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 220

    # Face oval
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (60, 80), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (width // 2 - 20, height // 2 - 15), 6, (40, 40, 40), -1)
    cv2.circle(img, (width // 2 + 20, height // 2 - 15), 6, (40, 40, 40), -1)
    # Nose
    cv2.line(img, (width // 2, height // 2 - 5),
             (width // 2, height // 2 + 10), (120, 100, 90), 2)
    # Mouth
    cv2.ellipse(img, (width // 2, height // 2 + 25), (15, 5),
                0, 0, 180, (100, 60, 60), 2)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _make_cin_front_image() -> bytes:
    """Create a synthetic CIN front: face region + text with CIN number."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240

    # Face area (left side, like real Tunisian CIN)
    face_region = img[30:180, 30:170]
    cv2.ellipse(face_region, (70, 75), (45, 60), 0, 0, 360,
                (180, 150, 130), -1)
    cv2.circle(face_region, (55, 60), 5, (40, 40, 40), -1)
    cv2.circle(face_region, (85, 60), 5, (40, 40, 40), -1)
    cv2.ellipse(face_region, (70, 90), (12, 4), 0, 0, 180,
                (100, 60, 60), 2)

    # Add CIN number text (for OCR fallback testing)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "12345678", (200, 60), font, 1.0,
                (0, 0, 0), 2, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _make_cin_back_image() -> bytes:
    """Create a synthetic CIN back with text (no real barcode)."""
    lines = [
        "REPUBLIQUE TUNISIENNE",
        "CARTE NATIONALE D'IDENTITE",
        "12345678",
    ]
    return _make_text_image(lines)


def _make_permis_image(cin: str = "12345678") -> bytes:
    """Create a synthetic Permis de Conduire with CIN number."""
    lines = [
        "PERMIS DE CONDUIRE",
        "Nom: BEN ALI",
        "Prenom: AHMED",
        f"6. {cin}",
        "Categorie: B",
    ]
    return _make_text_image(lines)


def _make_selfie_image() -> bytes:
    """Create a synthetic selfie image."""
    return _make_face_image(300, 400)


def _make_tiny_file() -> bytes:
    """Create a file too small to be valid."""
    return b"tiny"


def _make_corrupt_file() -> bytes:
    """Create a file that isn't a valid image."""
    return b"x" * 2000


# ---------------------------------------------------------------------------
# Document engine tests
# ---------------------------------------------------------------------------
class TestDocumentEngine:
    """Tests for document_engine module."""

    def test_extract_8_digits(self):
        """Test 8-digit CIN regex extraction."""
        from document_engine import _extract_8_digits

        assert _extract_8_digits("CIN: 12345678") == "12345678"
        assert _extract_8_digits("12345678") == "12345678"
        assert _extract_8_digits("abc12345678xyz") == "12345678"
        assert _extract_8_digits("1234567") is None  # only 7 digits
        assert _extract_8_digits("no digits here") is None
        assert _extract_8_digits("") is None

    def test_extract_cin_from_id_returns_type(self):
        """Test extract_cin_from_id returns str or None."""
        from document_engine import extract_cin_from_id

        front = _make_cin_front_image()
        back = _make_cin_back_image()
        result = extract_cin_from_id(front, back)

        assert result is None or isinstance(result, str)
        if result is not None:
            assert len(result) == 8
            assert result.isdigit()

    def test_extract_cin_from_id_corrupt_image(self):
        """Test that corrupt images don't crash."""
        from document_engine import extract_cin_from_id

        result = extract_cin_from_id(b"not_an_image", b"not_an_image")
        assert result is None

    def test_extract_cin_from_permis_returns_type(self):
        """Test extract_cin_from_permis returns str or None."""
        from document_engine import extract_cin_from_permis

        permis = _make_permis_image()
        result = extract_cin_from_permis(permis)

        assert result is None or isinstance(result, str)
        if result is not None:
            assert len(result) == 8
            assert result.isdigit()

    def test_extract_cin_from_permis_corrupt(self):
        """Test that corrupt Permis image doesn't crash."""
        from document_engine import extract_cin_from_permis

        result = extract_cin_from_permis(b"not_an_image")
        assert result is None


# ---------------------------------------------------------------------------
# Face engine tests
# ---------------------------------------------------------------------------
class TestFaceEngine:
    """Tests for face_engine module."""

    def test_extract_face_blank_image_returns_none(self):
        """Test that a blank white image returns None (no face)."""
        from face_engine import extract_face_from_document

        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".jpg", img)
        result = extract_face_from_document(buf.tobytes())
        assert result is None

    def test_check_liveness_returns_dict(self):
        """Test liveness check returns expected structure."""
        from face_engine import check_liveness

        selfie = _make_selfie_image()
        result = check_liveness(selfie)

        assert isinstance(result, dict)
        assert "is_live" in result
        assert "confidence" in result
        assert "face_detected" in result
        assert "reason" in result

    def test_verify_faces_returns_dict(self):
        """Test face verification returns expected structure."""
        from face_engine import verify_faces

        face1 = _make_face_image(112, 112)
        face2 = _make_face_image(112, 112)
        result = verify_faces(face1, face2)

        assert isinstance(result, dict)
        assert "match" in result
        assert "confidence" in result
        assert "distance" in result
        assert "threshold_used" in result
        assert "reason" in result

    def test_verify_faces_corrupt_input(self):
        """Test face verification handles corrupt input gracefully."""
        from face_engine import verify_faces

        result = verify_faces(b"not_an_image", b"not_an_image")
        assert isinstance(result, dict)
        assert result["match"] is False


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------
class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints via TestClient."""

    def _get_client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns service info."""
        client = self._get_client()
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "NexDrive KYC"
        assert "version" in data

    def test_health_endpoint(self):
        """Test GET /health returns ok status."""
        client = self._get_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ts" in data

    def test_kyc_process_missing_files_returns_422(self):
        """Test POST /kyc/process with no files returns 422."""
        client = self._get_client()
        resp = client.post("/kyc/process")
        assert resp.status_code == 422

    def test_kyc_process_partial_files_returns_422(self):
        """Test POST /kyc/process with incomplete files returns 422."""
        client = self._get_client()
        cin_front = _make_cin_front_image()
        resp = client.post(
            "/kyc/process",
            files={"cin_front": ("cin_front.jpg", cin_front, "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_kyc_process_invalid_image_returns_400(self):
        """Test POST /kyc/process with corrupt image returns 400."""
        client = self._get_client()
        valid = _make_cin_front_image()
        corrupt = _make_corrupt_file()

        resp = client.post(
            "/kyc/process",
            files={
                "cin_front": ("cin_front.jpg", valid, "image/jpeg"),
                "cin_back": ("cin_back.jpg", corrupt, "image/jpeg"),
                "permis": ("permis.jpg", valid, "image/jpeg"),
                "selfie": ("selfie.jpg", valid, "image/jpeg"),
            },
        )
        assert resp.status_code == 400

    def test_kyc_process_too_small_file_returns_400(self):
        """Test POST /kyc/process with tiny file returns 400."""
        client = self._get_client()
        valid = _make_cin_front_image()
        tiny = _make_tiny_file()

        resp = client.post(
            "/kyc/process",
            files={
                "cin_front": ("cin_front.jpg", tiny, "image/jpeg"),
                "cin_back": ("cin_back.jpg", valid, "image/jpeg"),
                "permis": ("permis.jpg", valid, "image/jpeg"),
                "selfie": ("selfie.jpg", valid, "image/jpeg"),
            },
        )
        assert resp.status_code == 400

    def test_kyc_process_returns_valid_structure(self):
        """
        Test POST /kyc/process with synthetic images returns
        a valid KYCResponse structure.
        """
        client = self._get_client()
        cin_front = _make_cin_front_image()
        cin_back = _make_cin_back_image()
        permis = _make_permis_image()
        selfie = _make_selfie_image()

        resp = client.post(
            "/kyc/process",
            files={
                "cin_front": ("cin_front.jpg", cin_front, "image/jpeg"),
                "cin_back": ("cin_back.jpg", cin_back, "image/jpeg"),
                "permis": ("permis.jpg", permis, "image/jpeg"),
                "selfie": ("selfie.jpg", selfie, "image/jpeg"),
            },
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify response structure matches KYCResponse
        assert "success" in data
        assert "verified" in data
        assert "stage_reached" in data
        assert "failure_reasons" in data
        assert "extracted_cin" in data

        assert data["success"] is True
        assert isinstance(data["failure_reasons"], list)
        assert data["stage_reached"] in (
            "extract_id", "extract_permis", "document_match",
            "face_extraction", "face_match", "liveness", "complete"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
