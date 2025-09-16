FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers logs to stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime deps
COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy source
COPY app ./app

EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8000)); s.close()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
