"""
Tests for the data_ingest module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import data_ingest


def test_create_session_with_retries():
    """Test creating session with retries."""
    session = data_ingest.create_session_with_retries()
    assert session is not None
    # Check if retries are configured
    adapter = session.adapters['https://']
    assert adapter.max_retries.total == 5


@patch('data_ingest.time.sleep')
@patch('data_ingest.time.time')
def test_handle_rate_limit(mock_time, mock_sleep):
    """Test handling rate limiting for API requests."""
    mock_time.return_value = 100
    
    # Mock response with rate limit headers
    mock_response = MagicMock()
    mock_response.headers = {
        'X-Rate-Limit-Remaining': '0',
        'X-Rate-Limit-Reset': '110'
    }
    
    data_ingest.handle_rate_limit(mock_response, "test_platform", 100)
    
    # Should sleep for reset_time - current_time + 1
    mock_sleep.assert_called_once_with(11)


def test_save_raw_data(tmpdir):
    """Test saving raw API response data to file."""
    data = {"test": "data"}
    platform = "test_platform"
    data_type = "test_type"
    user_id = "test_user"
    output_dir = str(tmpdir)
    
    filepath = data_ingest.save_raw_data(data, platform, data_type, user_id, output_dir)
    
    # Check if file was created
    assert os.path.exists(filepath)
    
    # Check if data was saved correctly
    with open(filepath, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert saved_data == data


@patch('data_ingest.requests.Session')
def test_fetch_instagram_data(mock_session, tmpdir):
    """Test fetching data from Instagram API."""
    # Mock session and response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "123",
        "username": "test_user",
        "account_type": "personal",
        "media_count": 10
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value = mock_response
    mock_session.return_value = mock_session_instance
    
    # Mock config
    config = {
        "access_token": "test_token",
        "rate_limit": 200
    }
    
    # Call function
    with patch('data_ingest.create_session_with_retries', return_value=mock_session_instance):
        files = data_ingest.fetch_instagram_data(config, str(tmpdir))
    
    # Check if API was called
    mock_session_instance.get.assert_called()
    
    # Check if files were saved
    assert len(files) > 0


@patch('data_ingest.requests.Session')
def test_fetch_twitter_data(mock_session, tmpdir):
    """Test fetching data from Twitter API."""
    # Mock session and response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "id": "123",
            "name": "Test User",
            "username": "test_user"
        }
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value = mock_response
    mock_session.return_value = mock_session_instance
    
    # Mock config
    config = {
        "bearer_token": "test_token",
        "rate_limit": 450
    }
    
    # Call function
    with patch('data_ingest.create_session_with_retries', return_value=mock_session_instance):
        files = data_ingest.fetch_twitter_data(config, str(tmpdir))
    
    # Check if API was called
    mock_session_instance.get.assert_called()
    
    # Check if files were saved
    assert len(files) > 0


@patch('data_ingest.requests.Session')
def test_fetch_facebook_data(mock_session, tmpdir):
    """Test fetching data from Facebook API."""
    # Mock session and response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "123",
        "name": "Test User",
        "about": "Test bio"
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value = mock_response
    mock_session.return_value = mock_session_instance
    
    # Mock config
    config = {
        "access_token": "test_token",
        "rate_limit": 200
    }
    
    # Call function
    with patch('data_ingest.create_session_with_retries', return_value=mock_session_instance):
        files = data_ingest.fetch_facebook_data(config, str(tmpdir))
    
    # Check if API was called
    mock_session_instance.get.assert_called()
    
    # Check if files were saved
    assert len(files) > 0


@patch('data_ingest.fetch_instagram_data')
@patch('data_ingest.fetch_twitter_data')
@patch('data_ingest.fetch_facebook_data')
def test_fetch_platform_data(mock_facebook, mock_twitter, mock_instagram, tmpdir):
    """Test fetching data from a specified platform."""
    # Set up mocks
    mock_instagram.return_value = ["file1.json"]
    mock_twitter.return_value = ["file2.json"]
    mock_facebook.return_value = ["file3.json"]
    
    # Test Instagram
    files = data_ingest.fetch_platform_data("instagram", {"test": "config"}, str(tmpdir))
    assert files == ["file1.json"]
    mock_instagram.assert_called_once()
    
    # Test Twitter
    files = data_ingest.fetch_platform_data("twitter", {"test": "config"}, str(tmpdir))
    assert files == ["file2.json"]
    mock_twitter.assert_called_once()
    
    # Test Facebook
    files = data_ingest.fetch_platform_data("facebook", {"test": "config"}, str(tmpdir))
    assert files == ["file3.json"]
    mock_facebook.assert_called_once()
    
    # Test unsupported platform
    files = data_ingest.fetch_platform_data("unsupported", {"test": "config"}, str(tmpdir))
    assert files == []


def test_list_available_data(tmpdir):
    """Test listing available data files."""
    # Create test files
    platform_dir = os.path.join(tmpdir, "instagram")
    os.makedirs(platform_dir, exist_ok=True)
    
    with open(os.path.join(platform_dir, "user1_profile_123.json"), 'w') as f:
        f.write("{}")
    
    with open(os.path.join(platform_dir, "user1_posts_123.json"), 'w') as f:
        f.write("{}")
    
    # Call function
    data_files = data_ingest.list_available_data(str(tmpdir))
    
    # Check results
    assert "instagram" in data_files
    assert "profile" in data_files["instagram"]
    assert "posts" in data_files["instagram"]
    assert len(data_files["instagram"]["profile"]) == 1
    assert len(data_files["instagram"]["posts"]) == 1
