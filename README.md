# Cross-Platform User Matching System

A system for identifying and matching users across different social media platforms using machine learning techniques.

## Features

- User matching across Instagram, Twitter, and Facebook
- Web interface for user interaction
- Machine learning models for user matching
- Privacy-preserving techniques
- API for programmatic access

## Requirements

- Python 3.11+
- PostgreSQL
- Redis (optional, for rate limiting and caching)

## Installation

### Local Development

1. Clone the repository:
   ```
   git clone <repository-url>
   cd crossmatch_project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```
   make init-db
   ```

6. Run the application:
   ```
   make run
   ```

### Docker Deployment

1. Build and run with Docker Compose:
   ```
   docker-compose up -d
   ```

2. Initialize the database:
   ```
   docker-compose exec web python init_db.py
   ```

## Configuration

The application can be configured using environment variables or configuration files:

- `config.yaml`: Development configuration
- `config.production.yaml`: Production configuration

## API Endpoints

### Authentication

- `POST /login`: User login
- `POST /register`: User registration
- `GET /logout`: User logout

### User Matching

- `POST /api/match`: Match users across platforms
- `GET /match/results`: View match results

## Development

### Running Tests

```
make test
```

### Linting

```
make lint
```

### Database Migrations

Create a new migration:
```
make migration message="Migration description"
```

Apply migrations:
```
make migrate
```

## Production Deployment

1. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with production configuration
   ```

2. Run in production mode:
   ```
   make production
   ```

## License

[MIT License](LICENSE)
