.PHONY: setup install test lint run clean migrate docker-build docker-run

# Default target
all: setup

# Setup the project
setup: install migrate

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest

# Run linting
lint:
	flake8 .

# Run the application in development mode
run:
	python main.py

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Run database migrations
migrate:
	flask db upgrade

# Create a new migration
migration:
	flask db migrate -m "$(message)"

# Build Docker image
docker-build:
	docker-compose build

# Run with Docker
docker-run:
	docker-compose up

# Initialize the database
init-db:
	python init_db.py

# Initialize the models
init-models:
	python init_models.py

# Initialize everything
init: init-db init-models

# Run in production mode
production:
	bash run_production.sh
