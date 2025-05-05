#!/usr/bin/env python3
"""
Check if Instagram credentials are properly set in the .env file
without displaying sensitive information.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Instagram credentials
instagram_client_id = os.getenv('INSTAGRAM_CLIENT_ID')
instagram_client_secret = os.getenv('INSTAGRAM_CLIENT_SECRET')
instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')

# Check if credentials are set
print("Instagram Credentials Status:")
print("-" * 30)
print(f"Client ID: {'✓ Set' if instagram_client_id else '✗ Not set'}")
print(f"Client Secret: {'✓ Set' if instagram_client_secret else '✗ Not set'}")
print(f"Access Token: {'✓ Set' if instagram_access_token else '✗ Not set'}")
print("-" * 30)
