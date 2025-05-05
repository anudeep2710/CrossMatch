"""
Script to check users in the PostgreSQL database.
"""

import os
import sys

# Set DATABASE_URL environment variable before importing Flask app
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/crossmatch'

# Now import Flask app
from main import app, db, User

def check_users():
    """Check users in the database."""
    print("Checking users in the database...")
    
    # Query users
    with app.app_context():
        users = User.query.all()
        
        if not users:
            print("No users found in the database.")
            return
        
        print("\nUsers in the database:")
        print("-" * 80)
        print(f"{'ID':<5} {'Username':<20} {'Email':<30}")
        print("-" * 80)
        
        for user in users:
            print(f"{user.id:<5} {user.username:<20} {user.email:<30}")
        
        print("-" * 80)
        print(f"Total users: {len(users)}")

if __name__ == '__main__':
    check_users()
