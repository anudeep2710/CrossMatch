# Cross-Platform User Matching System

A system for identifying and matching users across different social media platforms using machine learning techniques.

## Features

- User matching across Instagram, Twitter, and Facebook
- Web interface for user interaction
- Machine learning models for user matching
- Privacy-preserving techniques
- API for programmatic access
- Sample data generation for testing

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

### API Credentials

To use the application with real social media platforms, you need to configure API credentials in your `.env` file:

#### Instagram API Credentials

1. Create a Facebook Developer account at [developers.facebook.com](https://developers.facebook.com/)
2. Create a new app and add the Instagram Basic Display product
3. Configure your app settings and obtain the following credentials:
   - Client ID (App ID)
   - Client Secret (App Secret)
   - Access Token (generated through the OAuth flow)
4. Add these credentials to your `.env` file:
   ```
   INSTAGRAM_CLIENT_ID=your_client_id
   INSTAGRAM_CLIENT_SECRET=your_client_secret
   INSTAGRAM_ACCESS_TOKEN=your_access_token
   ```

#### Twitter and Facebook

Similar steps apply for configuring Twitter and Facebook API credentials. Refer to their respective developer documentation for details.

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

### Testing with Sample Data

The project includes a script to generate sample data for testing the user matching functionality without requiring real API access:

```
python3 generate_test_data.py
```

This script will:
1. Generate sample user profiles for Instagram, Twitter, and Facebook
2. Create realistic post data for each user
3. Establish ground truth mappings between users across platforms
4. Configure platform credentials for the admin user
5. Add sample match results to the database

After running this script, you can log in with the admin account (username: admin, password: admin) to test the matching functionality with the generated sample data.

#### Sample Data Structure

- **User Profiles**: 5 sample users per platform with different interests (photography, fitness, tech, food, travel)
- **Posts**: 20 posts per user with content relevant to their interests
- **Match Results**: Pre-computed matches with confidence scores

#### Testing the Matching Algorithm

To test the matching algorithm with the sample data:

1. Log in as the admin user
2. Navigate to the Match Users section
3. Select a source platform (e.g., Instagram) and enter a sample username (e.g., "photo_enthusiast")
4. View the match results showing corresponding users on other platforms

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

## Troubleshooting

### Database Connection Issues

If you encounter database connection errors:

1. Ensure PostgreSQL is running: `pg_isready`
2. Check that the database exists: `psql -l | grep crossmatch`
3. Verify your DATABASE_URL environment variable is set correctly in `.env`
4. Run the database initialization script: `python3 init_db.py`

### API Authentication Errors

If you encounter API authentication errors:

1. Verify your API credentials in the `.env` file
2. Check that your Facebook Developer app is properly configured
3. For testing without API access, use the sample data generation script

### Application Startup Issues

If the application fails to start:

1. Check the logs for specific error messages
2. Ensure all required directories exist (logs, raw_data, processed_data, etc.)
3. Verify that all dependencies are installed: `pip install -r requirements.txt`

## License

[MIT License](LICENSE)
