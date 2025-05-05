"""
Database initialization script.

This script initializes the database with the required tables and creates
an admin user if one doesn't exist.
"""

import os
import sys
import logging
from main import app, db, User
from flask_migrate import upgrade

def init_db():
    """Initialize the database."""
    print("Initializing database...")
    
    # Create database tables
    with app.app_context():
        # Run migrations
        try:
            upgrade()
            print("Database migrations applied successfully.")
        except Exception as e:
            print(f"Error applying migrations: {e}")
            print("Creating tables directly...")
            db.create_all()
            print("Database tables created successfully.")
        
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("Creating admin user...")
            admin = User(username='admin', email='admin@example.com')
            admin.set_password(os.environ.get('ADMIN_PASSWORD', 'admin'))
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully.")
        else:
            print("Admin user already exists.")
    
    print("Database initialization complete.")

if __name__ == '__main__':
    init_db()
