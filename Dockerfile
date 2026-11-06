FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gradio needs to listen on 0.0.0.0
ENV GRADIO_SERVER_NAME=0.0.0.0

# Expose the port Cloud Run expects
EXPOSE 8080

# Dynamically use Cloud Run's PORT variable, fallback to 8080
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} python app.py"]