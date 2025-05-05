"""
Cross-Platform User Matching System - Web Interface

This Flask application provides a web interface for the cross-platform
user matching system.
"""

import logging
import os
import json
import sys
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import utils

# Load configuration
config_file = os.environ.get("CONFIG_FILE", "config.yaml")
if not os.path.exists(config_file):
    print(f"Configuration file {config_file} not found.")
    sys.exit(1)

config = utils.load_config(config_file)

# Configure logging
log_level = getattr(logging, config["logging"]["level"])
log_format = config["logging"]["format"]
log_file = config["logging"]["file"]

if log_file:
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only try to create directory if there is a path component
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
else:
    logging.basicConfig(
        level=log_level,
        format=log_format
    )

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Set debug mode based on environment
app.debug = os.environ.get("FLASK_ENV", "production") == "development"

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=config.get("rate_limit", {}).get("default_limits", ["200 per day", "50 per hour"]),
    storage_uri=config.get("rate_limit", {}).get("storage_url", "memory://"),
    strategy="fixed-window"
)

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Platform model to store API credentials
class Platform(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    platform_name = db.Column(db.String(50), nullable=False)
    client_id = db.Column(db.String(255))
    client_secret = db.Column(db.String(255))
    access_token = db.Column(db.String(255))
    api_key = db.Column(db.String(255))
    api_secret = db.Column(db.String(255))
    bearer_token = db.Column(db.String(255))

    user = db.relationship('User', backref=db.backref('platforms', lazy=True))

# MatchResult model to store matching results
class MatchResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    platform1 = db.Column(db.String(50), nullable=False)
    platform1_user_id = db.Column(db.String(255), nullable=False)
    platform2 = db.Column(db.String(50), nullable=False)
    platform2_user_id = db.Column(db.String(255), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('match_results', lazy=True))

# Setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if username or email already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))

        flash('Invalid username or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard route."""
    # Get user's platforms
    platforms = Platform.query.filter_by(user_id=current_user.id).all()
    platform_names = [p.platform_name for p in platforms]

    # Get recent match results
    matches = MatchResult.query.filter_by(user_id=current_user.id).order_by(MatchResult.timestamp.desc()).limit(10).all()

    return render_template('dashboard.html',
                          platforms=platform_names,
                          has_platforms=len(platform_names) > 0,
                          matches=matches)

@app.route('/platform/configure', methods=['GET', 'POST'])
@login_required
def configure_platform():
    """Platform configuration route."""
    if request.method == 'POST':
        platform_name = request.form.get('platform_name')

        # Check if platform already exists for this user
        existing = Platform.query.filter_by(user_id=current_user.id, platform_name=platform_name).first()
        if existing:
            # Update existing platform
            existing.client_id = request.form.get('client_id', '')
            existing.client_secret = request.form.get('client_secret', '')
            existing.access_token = request.form.get('access_token', '')
            existing.api_key = request.form.get('api_key', '')
            existing.api_secret = request.form.get('api_secret', '')
            existing.bearer_token = request.form.get('bearer_token', '')
        else:
            # Create new platform
            platform = Platform(
                user_id=current_user.id,
                platform_name=platform_name,
                client_id=request.form.get('client_id', ''),
                client_secret=request.form.get('client_secret', ''),
                access_token=request.form.get('access_token', ''),
                api_key=request.form.get('api_key', ''),
                api_secret=request.form.get('api_secret', ''),
                bearer_token=request.form.get('bearer_token', '')
            )
            db.session.add(platform)

        db.session.commit()
        flash(f'{platform_name} configuration saved successfully!')
        return redirect(url_for('dashboard'))

    # GET request
    platform_name = request.args.get('platform', '')
    platform = None
    if platform_name:
        platform = Platform.query.filter_by(user_id=current_user.id, platform_name=platform_name).first()

    return render_template('configure_platform.html', platform=platform, platform_name=platform_name)

@app.route('/api/match', methods=['POST'])
@login_required
@limiter.limit("5 per minute")  # Add rate limiting
def api_match():
    """API route for matching users."""
    try:
        data = request.json
        platform1 = data.get('platform1')
        platform1_user = data.get('platform1_user')
        platform2 = data.get('platform2')
        top_k = data.get('top_k', 5)
        threshold = data.get('threshold', 0.5)

        # Check if required parameters are provided
        if not all([platform1, platform1_user, platform2]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        # Check if platforms are configured
        platforms = Platform.query.filter_by(user_id=current_user.id).all()
        platform_dict = {p.platform_name: p for p in platforms}
        platform_names = list(platform_dict.keys())

        if platform1 not in platform_names or platform2 not in platform_names:
            return jsonify({
                'success': False,
                'error': 'One or more platforms not configured'
            }), 400

        # Get platform credentials
        platform1_obj = platform_dict[platform1]
        platform2_obj = platform_dict[platform2]

        platform1_credentials = {
            'client_id': platform1_obj.client_id,
            'client_secret': platform1_obj.client_secret,
            'access_token': platform1_obj.access_token,
            'api_key': platform1_obj.api_key,
            'api_secret': platform1_obj.api_secret,
            'bearer_token': platform1_obj.bearer_token
        }

        platform2_credentials = {
            'client_id': platform2_obj.client_id,
            'client_secret': platform2_obj.client_secret,
            'access_token': platform2_obj.access_token,
            'api_key': platform2_obj.api_key,
            'api_secret': platform2_obj.api_secret,
            'bearer_token': platform2_obj.bearer_token
        }

        # Import matching service here to avoid circular imports
        import matching_service

        # Get matching service instance
        service = matching_service.get_instance(config)

        # Perform matching
        result = service.match_users(
            platform1=platform1,
            platform1_user=platform1_user,
            platform2=platform2,
            platform1_credentials=platform1_credentials,
            platform2_credentials=platform2_credentials,
            top_k=top_k,
            threshold=threshold
        )

        # If matching was successful, save results to database
        if result['success'] and 'matches' in result:
            for match in result['matches']:
                match_result = MatchResult(
                    user_id=current_user.id,
                    platform1=platform1,
                    platform1_user_id=platform1_user,
                    platform2=platform2,
                    platform2_user_id=match['platform2_user'],
                    confidence_score=match['confidence_score']
                )
                db.session.add(match_result)

            db.session.commit()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in match API: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/match', methods=['GET', 'POST'])
@login_required
def match():
    """Match users route."""
    # Get user's platforms
    platforms = Platform.query.filter_by(user_id=current_user.id).all()
    platform_names = [p.platform_name for p in platforms]
    platform_dict = {p.platform_name: p for p in platforms}

    if len(platform_names) < 2:
        flash('You need to configure at least two platforms to perform matching.')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        platform1 = request.form.get('platform1')
        platform1_user = request.form.get('platform1_user')
        platform2 = request.form.get('platform2')

        # Check if platforms are configured
        if platform1 not in platform_names or platform2 not in platform_names:
            flash('One or more platforms not configured.')
            return redirect(url_for('match'))

        # Get platform credentials
        platform1_obj = platform_dict[platform1]
        platform2_obj = platform_dict[platform2]

        platform1_credentials = {
            'client_id': platform1_obj.client_id,
            'client_secret': platform1_obj.client_secret,
            'access_token': platform1_obj.access_token,
            'api_key': platform1_obj.api_key,
            'api_secret': platform1_obj.api_secret,
            'bearer_token': platform1_obj.bearer_token
        }

        platform2_credentials = {
            'client_id': platform2_obj.client_id,
            'client_secret': platform2_obj.client_secret,
            'access_token': platform2_obj.access_token,
            'api_key': platform2_obj.api_key,
            'api_secret': platform2_obj.api_secret,
            'bearer_token': platform2_obj.bearer_token
        }

        # Import matching service here to avoid circular imports
        import matching_service

        # Get matching service instance
        service = matching_service.get_instance(config)

        # Perform matching
        result = service.match_users(
            platform1=platform1,
            platform1_user=platform1_user,
            platform2=platform2,
            platform1_credentials=platform1_credentials,
            platform2_credentials=platform2_credentials,
            top_k=5,
            threshold=0.5
        )

        if result['success'] and 'matches' in result:
            # Save match result to session for display
            session['match_result'] = {
                'platform1': platform1,
                'platform1_user': platform1_user,
                'platform2': platform2,
                'matches': result['matches']
            }

            # Save to database
            for match in result['matches']:
                result_entry = MatchResult(
                    user_id=current_user.id,
                    platform1=platform1,
                    platform1_user_id=platform1_user,
                    platform2=platform2,
                    platform2_user_id=match['platform2_user'],
                    confidence_score=match['confidence_score']
                )
                db.session.add(result_entry)

            db.session.commit()

            return redirect(url_for('match_results'))
        else:
            # If matching failed, show error
            error_message = result.get('error', 'Matching failed. Please try again.')
            flash(error_message)
            return redirect(url_for('match'))

    return render_template('match.html', platforms=platform_names)

@app.route('/match/results')
@login_required
def match_results():
    """Match results route."""
    result = session.get('match_result')
    if not result:
        flash('No matching results found.')
        return redirect(url_for('match'))

    return render_template('match_results.html', result=result)

@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler."""
    return render_template('500.html'), 500

# Create necessary directories
def create_directories():
    """Create necessary directories for the application."""
    # Ensure template directory exists
    os.makedirs('templates', exist_ok=True)

    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # Ensure data directories exist
    for directory in config["directories"].values():
        os.makedirs(directory, exist_ok=True)

    # Ensure logs directory exists
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only try to create directory if there is a path component
            os.makedirs(log_dir, exist_ok=True)

# Main entry point
if __name__ == '__main__':
    create_directories()

    # Get server configuration
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 5000)

    # Run the application
    app.run(host=host, port=port)
