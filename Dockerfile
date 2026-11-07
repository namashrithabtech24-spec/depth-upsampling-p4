FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for OpenCV AND git (required for torch.hub)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set a consistent cache directory for PyTorch Hub
ENV TORCH_HOME=/app/torch_cache

# (Pre-download step removed because app.py handles trust and caching programmatically at startup)

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 8080
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} python app.py"]
