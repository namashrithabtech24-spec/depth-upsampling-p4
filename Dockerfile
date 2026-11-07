FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set a consistent cache directory for PyTorch Hub
ENV TORCH_HOME=/app/torch_cache

# --- CRITICAL FIX ---
# Pre-download the MiDaS model AND its problematic dependency during the build.
# This bypasses the interactive "y/N" prompt that occurs at runtime.
RUN python -c "import torch; torch.hub.load('rwightman/gen-efficientnet-pytorch', 'geffnet', trust_repo=True)"
RUN python -c "import torch; torch.hub.load('isl-org/MiDaS', 'MiDaS_small', trust_repo=True)"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 8080
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} python app.py"]