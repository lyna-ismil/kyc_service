FROM python:3.11-slim

# Bypass deb.debian.org CDN DNS issues by switching to a direct mirror
RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list 2>/dev/null || true
# Fix the security mirror which does not exist on the ftp mirror
RUN sed -i 's/ftp.us.debian.org\/debian-security/security.debian.org\/debian-security/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/ftp.us.debian.org\/debian-security/security.debian.org\/debian-security/g' /etc/apt/sources.list 2>/dev/null || true

# Install system deps: libzbar0 for pyzbar, GL libs for OpenCV/ML
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 cmake build-essential \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Pre-download ArcFace model weights at build time
# avoids ~500MB download on first request in production
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')" || true

COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
