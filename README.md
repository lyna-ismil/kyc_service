NexDrive KYC Service
A self-hosted FastAPI microservice for Tunisian identity verification (KYC). The service performs a multi-stage gated pipeline that extracts CIN (Carte d’Identite Nationale) numbers from official documents, cross-matches them for consistency, verifies the user’s face against their ID photo, and performs a liveness check to prevent spoofing attacks.
All image processing is performed 100% in RAM — no files are ever written to disk.
________________________________________
Table of Contents
•	Architecture Overview
•	Verification Pipeline
•	API Reference
•	Technology Stack
•	Getting Started
•	Docker Deployment
•	Testing
•	Project Structure
•	Environment Variables
•	Logging
________________________________________
Architecture Overview
┌─────────────┐     ┌─────────────────────────────────────────────────────────┐
│   Client    │────>│              NexDrive KYC Service (Port 8001)           │
│ (4 images)  │     │                                                         │
└─────────────┘     │  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
                    │  │  Document   │   │    Face     │   │   Liveness   │  │
                    │  │   Engine    │   │   Engine    │   │    Engine    │  │
                    │  │             │   │             │   │              │  │
                    │  │ - EasyOCR   │   │ - MTCNN     │   │ - MediaPipe  │  │
                    │  │ - OpenCV    │   │ - DeepFace  │   │   FaceMesh   │  │
                    │  │ - NeuroParser│  │ - ArcFace   │   │ - EAR/Yaw    │  │
                    │  └─────────────┘   └─────────────┘   └──────────────┘  │
                    └─────────────────────────────────────────────────────────┘
The service accepts four images via a single multipart endpoint and executes a gated pipeline — each stage must pass before the next begins. Failure at any hard gate immediately returns a structured error response indicating the stage reached and the reason for failure.
________________________________________
Verification Pipeline
Step	Stage	Description	Gate Type
1	ID OCR	Extract 8-digit CIN from CIN card front using EasyOCR with NeuroParser preprocessing (grayscale, NLM denoising, CLAHE contrast enhancement, Otsu binarization fallback)	Hard
2	Permis OCR	Extract 8-digit CIN from Permis de Conduire using the same OCR pipeline	Hard
3	Document Match	Cross-validate that CIN from ID card matches CIN from driver’s license	Hard
4	Face Extraction	Detect and crop faces from CIN front and/or Permis using MTCNN (confidence >= 0.60, 25% padding, upscaled to 112x112 min)	Hard
5	Face Verification	Compare extracted document face(s) against live selfie using DeepFace ArcFace with cosine distance (threshold: 0.70). OR-gate: either CIN face or Permis face may match.	Hard
6	Liveness Check	Validate the selfie is a live human using MediaPipe FaceMesh — checks eye aspect ratio (EAR > 0.15) and frontal pose (yaw < 0.22)	Hard
If all six stages pass, the user is verified and their CIN number is returned.
________________________________________
API Reference
GET /
Service information and available endpoints.
Response:
{
  "service": "NexDrive KYC",
  "version": "1.0.0",
  "description": "Self-hosted identity verification for Tunisian users",
  "endpoints": {
    "health": "GET /health",
    "kyc_process": "POST /kyc/process"
  }
}
________________________________________
GET /health
Health check endpoint for load balancers and monitoring.
Response:
{
  "status": "ok",
  "service": "kyc",
  "ts": 1713600000.0
}
________________________________________
POST /kyc/process
Main KYC verification endpoint. Accepts 4 images as multipart/form-data.
Request:
Field	Type	Description
cin_front	File	Front of CIN card (face photo + Arabic text side)
cin_back	File	Back of CIN card (barcode side — reserved for future use)
permis	File	Permis de Conduire (driver’s license)
selfie	File	Live selfie from front camera
Constraints: - Min file size: 1 KB - Max file size: 10 MB - Accepted formats: JPEG, PNG
Response (KYCResponse):
Field	Type	Description
success	bool	Whether the request was processed without server errors
verified	bool	Whether the user passed all verification stages
stage_reached	string	Last pipeline stage reached: extract_id, extract_permis, document_match, face_extraction, face_match, liveness, or complete
failure_reasons	list[str]	Human-readable explanations if verification failed
extracted_cin	string?	The 8-digit CIN number (only returned if ID extraction succeeded)
face_match	dict?	Face verification details including confidence and distance
Success Example:
{
  "success": true,
  "verified": true,
  "stage_reached": "complete",
  "failure_reasons": [],
  "extracted_cin": "12345678",
  "face_match": {
    "match": true,
    "confidence": 0.923,
    "distance": 0.077,
    "threshold_used": 0.70,
    "reason": "Face matches ID card photo"
  }
}
Failure Example (document mismatch):
{
  "success": true,
  "verified": false,
  "stage_reached": "document_match",
  "failure_reasons": [
    "Documents do not belong to the same person — ID card CIN '12345678' does not match Driver's License CIN '87654321'"
  ],
  "extracted_cin": null
}
________________________________________
Technology Stack
Component	Technology	Purpose
Web Framework	FastAPI + Uvicorn	HTTP API, async request handling
Image Processing	OpenCV (NeuroParser)	Grayscale conversion, NLM denoising, CLAHE contrast, Otsu binarization
OCR	EasyOCR (Arabic + French)	Text extraction from Tunisian documents
Face Detection	MTCNN	Face localization in document images
Face Recognition	DeepFace + ArcFace	512-dimensional face embedding comparison with cosine distance
Liveness Detection	MediaPipe FaceMesh	Eye aspect ratio and head pose analysis
Language	Python 3.11	Runtime
NeuroParser Preprocessing Pipeline
The document engine implements a specialized preprocessing chain designed to defeat guilloche security patterns on official Tunisian documents:
1.	Grayscale conversion — removes color channel noise
2.	Non-Local Means Denoising (h=10) — removes high-frequency security background patterns while preserving text edges
3.	CLAHE (clipLimit=2.0, 8x8 grid) — boosts local contrast so black text stands out against washed-out security backgrounds
4.	Otsu Binarization (fallback) — converts to pure black & white for a second OCR pass if the first fails
________________________________________
Getting Started
Prerequisites
•	Python 3.11+
•	System dependencies: libgl1, libzbar0, libglib2.0-0, libsm6, libxext6, libxrender-dev, libgomp1
Local Installation
# Clone the repository
git clone <repository-url>
cd kyc_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Pre-download ArcFace model (optional, avoids first-request delay)
python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2
The API will be available at http://localhost:8001. Interactive documentation (Swagger UI) is available at http://localhost:8001/docs.
________________________________________
Docker Deployment
Build
docker build -t nexdrive-kyc:latest .
The Dockerfile pre-downloads ArcFace model weights at build time to eliminate the ~500MB cold-start delay on first request.
Run
docker run -d \
  --name kyc-service \
  -p 8001:8001 \
  --memory=4g \
  nexdrive-kyc:latest
Docker Compose Example
version: "3.8"
services:
  kyc:
    build: .
    container_name: kyc-service
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
Resource Recommendations: - Minimum: 2 GB RAM - Recommended: 4 GB RAM (accounts for multiple ML models loaded simultaneously) - CPU: 2+ cores for parallel workers
________________________________________
Testing
The project includes a comprehensive test suite using synthetic images — no real ID documents required.
# Run all tests
pytest test_kyc.py -v

# Run specific test class
pytest test_kyc.py::TestDocumentEngine -v
pytest test_kyc.py::TestFaceEngine -v
pytest test_kyc.py::TestFastAPIEndpoints -v
Test Coverage
Module	Tests
document_engine	Regex extraction, OCR return types, corrupt image handling
face_engine	Blank image rejection, liveness response structure, face verification structure, corrupt input handling
main (FastAPI)	Root/health endpoints, missing/partial file validation (422), invalid image rejection (400), tiny file rejection (400), response structure validation
________________________________________
Project Structure
kyc_service/
|
|-- main.py                  # FastAPI app, request validation, pipeline orchestration
|-- document_engine.py       # OCR pipeline: NeuroParser preprocessing + EasyOCR
|-- face_engine.py           # Face extraction, verification (ArcFace), liveness (MediaPipe)
|-- test_kyc.py             # Integration tests with synthetic image generators
|-- requirements.txt        # Python dependencies
|-- Dockerfile              # Production container image
|-- README.md               # This file
________________________________________
Environment Variables
Variable	Default	Description
LOG_LEVEL	INFO	Logging verbosity (DEBUG, INFO, WARNING, ERROR)
MAX_FILE_SIZE	10 MB	Maximum uploaded file size (bytes)
MIN_FILE_SIZE	1 KB	Minimum uploaded file size (bytes)
FACE_MATCH_THRESHOLD	0.70	Minimum confidence for face match (0.0-1.0)
LIVENESS_CONFIDENCE_MIN	0.45	Minimum liveness confidence score
MTCNN_MIN_CONFIDENCE	0.60	Minimum face detection confidence
CORS_ORIGINS	*	Allowed CORS origins (comma-separated)
________________________________________
Logging
The service uses structured logging with timestamps, log levels, and component names:
2024-01-15 09:23:45 [INFO] kyc.main: === STEP 1: Extract CIN from ID Card ===
2024-01-15 09:23:46 [INFO] kyc.document_engine: EasyOCR reader loaded in 3.2s
2024-01-15 09:23:47 [INFO] kyc.document_engine: CIN found on Pass 1 (enhanced): 12345678
2024-01-15 09:23:47 [INFO] kyc.main: === STEP 3: Cross-Match Documents ===
2024-01-15 09:23:47 [INFO] kyc.main: CIN numbers match: 12345678 ✓
All models are lazily loaded on first request and cached in module-level globals for subsequent requests. First-request latency is higher due to model initialization; subsequent requests are significantly faster.
________________________________________
Security & Privacy Notes
•	No persistent storage: All images are processed in memory and discarded after the request completes. No files are written to disk.
•	No biometric storage: Face embeddings are computed on-the-fly and never stored or logged.
•	CIN-only output: Only the extracted CIN number is returned in the API response. No personal demographic data is extracted or returned.
•	Image validation: All uploads are validated for size, format, and decodability before processing begins.
________________________________________
License
[Your License Here]
