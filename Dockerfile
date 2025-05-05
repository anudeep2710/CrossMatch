FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p raw_data processed_data models results static/css static/js templates logs

# Make scripts executable
RUN chmod +x run_production.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=main.py \
    FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "main:app"]
