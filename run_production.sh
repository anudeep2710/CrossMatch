#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Set production environment
export FLASK_ENV=production
export CONFIG_FILE=config.production.yaml

# Run database migrations
python -m flask db upgrade

# Initialize models if needed
python init_models.py

# Start the application with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 main:app
