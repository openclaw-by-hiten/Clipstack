FROM python:3.11-slim

# Install system dependencies including FFmpeg, certs, and caption fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fonts-dejavu-core \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code and assets
COPY . .

# Environment configuration
ENV YTDLP_NO_PLUGINS=1
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
