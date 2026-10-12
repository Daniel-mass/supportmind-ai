
# ---------------------------------------------------------
# Base image
# Python runtime for the SupportMind FastAPI application.
# ---------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing bytecode files inside the image.
ENV PYTHONDONTWRITEBYTECODE=1

# Send application logs directly to the container output.
ENV PYTHONUNBUFFERED=1

# All subsequent commands execute from the backend directory
# inside the image.
WORKDIR /app


# ---------------------------------------------------------
# Install Python dependencies
# Copy requirements separately to improve Docker build caching.
# ---------------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------
# Copy application source
# Includes FastAPI routes, Graph-RAG, retrieval providers,
# vector-store integrations, graph integrations, and other
# Python modules contained within backend/app.
# ---------------------------------------------------------
COPY app ./app


# ---------------------------------------------------------
# FastAPI service
# Expose the application port used by Docker Compose.
# ---------------------------------------------------------
EXPOSE 8000

# Start the API on all container interfaces so Docker port
# forwarding can expose it on the Windows host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
