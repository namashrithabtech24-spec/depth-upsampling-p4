FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THIS IS THE CRITICAL FIX ---
# Pre-download the MiDaS model and its dependencies during the build.
# This prevents any network calls or trust prompts when the container starts.
RUN python -c "import torch; torch.hub.load('isl-org/MiDaS', 'MiDaS_small', trust_repo=True)"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 8080
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} python app.py"]