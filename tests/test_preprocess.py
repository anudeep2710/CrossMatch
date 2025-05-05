"""
Tests for the preprocess module.
"""

import os
import json
import datetime
import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock
import preprocess


def test_anonymize_user_id():
    """Test creating a consistent anonymized ID for a user."""
    # Test consistency
    user_id = "12345"
    platform = "instagram"
    
    anon_id_1 = preprocess.anonymize_user_id(user_id, platform)
    anon_id_2 = preprocess.anonymize_user_id(user_id, platform)
    
    assert anon_id_1 == anon_id_2
    assert len(anon_id_1) == 16  # Check length
    
    # Test different platforms produce different IDs
    anon_id_twitter = preprocess.anonymize_user_id(user_id, "twitter")
    assert anon_id_1 != anon_id_twitter
    
    # Test different salts produce different IDs
    anon_id_salt = preprocess.anonymize_user_id(user_id, platform, salt="different_salt")
    assert anon_id_1 != anon_id_salt


def test_clean_text():
    """Test cleaning text content based on configuration."""
    # Test with default config
    text = "Check out this #awesome link: https://example.com! @friend"
    default_config = {
        "lowercase": True,
        "remove_urls": True,
        "remove_mentions": True,
        "remove_hashtags": False,
        "remove_punctuation": True,
        "lemmatize": True
    }
    
    cleaned = preprocess.clean_text(text, default_config)
    
    # Should convert to lowercase, remove URL and mention, keep hashtag text, remove punctuation
    assert "https://example.com" not in cleaned
    assert "@friend" not in cleaned
    assert "awesome" in cleaned
    assert "!" not in cleaned
    assert cleaned.islower()
    
    # Test with different config
    custom_config = {
        "lowercase": False,
        "remove_urls": False,
        "remove_mentions": False,
        "remove_hashtags": True,
        "remove_punctuation": False,
        "lemmatize": False
    }
    
    cleaned = preprocess.clean_text(text, custom_config)
    
    # Should keep URL and mention, remove hashtag, keep punctuation, preserve case
    assert "https://example.com" in cleaned
    assert "@friend" in cleaned
    assert "#awesome" not in cleaned
    assert "awesome" not in cleaned
    assert "!" in cleaned
    assert "Check" in cleaned  # Case preserved
    
    # Test with empty text
    assert preprocess.clean_text("", default_config) == ""
    assert preprocess.clean_text(None, default_config) == ""


def test_normalize_timestamp():
    """Test normalizing timestamps from different formats."""
    # Test ISO format with timezone
    iso_timestamp = "2023-01-15T14:30:45+0000"
    dt = preprocess.normalize_timestamp(iso_timestamp)
    assert isinstance(dt, datetime.datetime)
    assert dt.year == 2023
    assert dt.month == 1
    assert dt.day == 15
    assert dt.hour == 14
    assert dt.minute == 30
    assert dt.second == 45
    
    # Test ISO format with milliseconds
    iso_ms_timestamp = "2023-01-15T14:30:45.123+0000"
    dt = preprocess.normalize_timestamp(iso_ms_timestamp)
    assert isinstance(dt, datetime.datetime)
    assert dt.year == 2023
    
    # Test simple format
    simple_timestamp = "2023-01-15 14:30:45"
    dt = preprocess.normalize_timestamp(simple_timestamp)
    assert isinstance(dt, datetime.datetime)
    assert dt.year == 2023
    
    # Test invalid format
    invalid_timestamp = "invalid"
    dt = preprocess.normalize_timestamp(invalid_timestamp)
    assert isinstance(dt, datetime.datetime)
    # Should return current time for invalid format


def test_process_instagram_data(tmpdir):
    """Test processing raw Instagram data."""
    # Create test directories
    raw_dir = os.path.join(tmpdir, "raw")
    instagram_dir = os.path.join(raw_dir, "instagram")
    processed_dir = os.path.join(tmpdir, "processed")
    
    os.makedirs(instagram_dir, exist_ok=True)
    
    # Create sample profile data
    profile_data = {
        "id": "12345",
        "username": "test_user",
        "account_type": "personal",
        "media_count": 10
    }
    
    with open(os.path.join(instagram_dir, "12345_profile_123.json"), 'w') as f:
        json.dump(profile_data, f)
    
    # Create sample post data
    post_data = {
        "data": [
            {
                "id": "post1",
                "caption": "This is a test post",
                "timestamp": "2023-01-15T14:30:45+0000",
                "like_count": 10,
                "comments_count": 2
            },
            {
                "id": "post2",
                "caption": "Another test post",
                "timestamp": "2023-01-16T10:15:30+0000",
                "like_count": 5,
                "comments_count": 1
            }
        ]
    }
    
    with open(os.path.join(instagram_dir, "12345_posts_123.json"), 'w') as f:
        json.dump(post_data, f)
    
    # Define test config
    config = {
        "min_posts_per_user": 1,
        "max_posts_per_user": 100,
        "text_cleaning": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "remove_punctuation": True,
            "lemmatize": True
        }
    }
    
    # Process data
    processed_data = preprocess.process_instagram_data(raw_dir, processed_dir, config)
    
    # Check results
    assert processed_data is not None
    assert "users" in processed_data
    assert "posts" in processed_data
    
    # Check if output file was created
    assert os.path.exists(os.path.join(processed_dir, "instagram", "processed_data.json"))
    
    # Check user data
    anon_user_id = preprocess.anonymize_user_id("12345", "instagram")
    assert anon_user_id in processed_data["users"]
    assert processed_data["users"][anon_user_id]["username"] == "test_user"
    
    # Check post data
    assert anon_user_id in processed_data["posts"]
    assert len(processed_data["posts"][anon_user_id]) == 2


def test_process_twitter_data(tmpdir):
    """Test processing raw Twitter data."""
    # Create test directories
    raw_dir = os.path.join(tmpdir, "raw")
    twitter_dir = os.path.join(raw_dir, "twitter")
    processed_dir = os.path.join(tmpdir, "processed")
    
    os.makedirs(twitter_dir, exist_ok=True)
    
    # Create sample profile data
    profile_data = {
        "data": {
            "id": "12345",
            "name": "Test User",
            "username": "test_user",
            "created_at": "2020-01-01T00:00:00+0000",
            "public_metrics": {
                "followers_count": 100,
                "following_count": 50,
                "tweet_count": 200
            }
        }
    }
    
    with open(os.path.join(twitter_dir, "12345_profile_123.json"), 'w') as f:
        json.dump(profile_data, f)
    
    # Create sample tweet data
    tweet_data = {
        "data": [
            {
                "id": "tweet1",
                "text": "This is a test tweet",
                "created_at": "2023-01-15T14:30:45+0000",
                "public_metrics": {
                    "like_count": 10,
                    "retweet_count": 2,
                    "reply_count": 1,
                    "quote_count": 0
                }
            },
            {
                "id": "tweet2",
                "text": "Another test tweet",
                "created_at": "2023-01-16T10:15:30+0000",
                "public_metrics": {
                    "like_count": 5,
                    "retweet_count": 1,
                    "reply_count": 0,
                    "quote_count": 0
                }
            }
        ]
    }
    
    with open(os.path.join(twitter_dir, "12345_tweets_123.json"), 'w') as f:
        json.dump(tweet_data, f)
    
    # Create sample followers data
    followers_data = {
        "data": [
            {
                "id": "67890",
                "username": "follower1"
            },
            {
                "id": "54321",
                "username": "follower2"
            }
        ]
    }
    
    with open(os.path.join(twitter_dir, "12345_followers_123.json"), 'w') as f:
        json.dump(followers_data, f)
    
    # Create sample following data
    following_data = {
        "data": [
            {
                "id": "11111",
                "username": "following1"
            }
        ]
    }
    
    with open(os.path.join(twitter_dir, "12345_following_123.json"), 'w') as f:
        json.dump(following_data, f)
    
    # Define test config
    config = {
        "min_posts_per_user": 1,
        "max_posts_per_user": 100,
        "max_followers": 1000,
        "text_cleaning": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "remove_punctuation": True,
            "lemmatize": True
        }
    }
    
    # Process data
    processed_data = preprocess.process_twitter_data(raw_dir, processed_dir, config)
    
    # Check results
    assert processed_data is not None
    assert "users" in processed_data
    assert "posts" in processed_data
    assert "networks" in processed_data
    
    # Check if output files were created
    assert os.path.exists(os.path.join(processed_dir, "twitter", "processed_data.json"))
    assert os.path.exists(os.path.join(processed_dir, "twitter", "network.edgelist"))
    
    # Check user data
    anon_user_id = preprocess.anonymize_user_id("12345", "twitter")
    assert anon_user_id in processed_data["users"]
    assert processed_data["users"][anon_user_id]["username"] == "test_user"
    
    # Check post data
    assert anon_user_id in processed_data["posts"]
    assert len(processed_data["posts"][anon_user_id]) == 2
    
    # Check network data
    assert "graph" in processed_data["networks"]
    graph = processed_data["networks"]["graph"]
    assert isinstance(graph, nx.DiGraph)
    assert anon_user_id in graph.nodes()
    assert graph.number_of_nodes() > 1  # Should have user and followers/following


def test_process_facebook_data(tmpdir):
    """Test processing raw Facebook data."""
    # Create test directories
    raw_dir = os.path.join(tmpdir, "raw")
    facebook_dir = os.path.join(raw_dir, "facebook")
    processed_dir = os.path.join(tmpdir, "processed")
    
    os.makedirs(facebook_dir, exist_ok=True)
    
    # Create sample profile data
    profile_data = {
        "id": "12345",
        "name": "Test User",
        "about": "Test bio",
        "gender": "male",
        "hometown": {
            "name": "Test City"
        }
    }
    
    with open(os.path.join(facebook_dir, "12345_profile_123.json"), 'w') as f:
        json.dump(profile_data, f)
    
    # Create sample post data
    post_data = {
        "data": [
            {
                "id": "post1",
                "message": "This is a test post",
                "created_time": "2023-01-15T14:30:45+0000",
                "reactions": {
                    "summary": {
                        "total_count": 10
                    }
                },
                "comments": {
                    "summary": {
                        "total_count": 2
                    }
                },
                "type": "status"
            },
            {
                "id": "post2",
                "message": "Another test post",
                "created_time": "2023-01-16T10:15:30+0000",
                "reactions": {
                    "summary": {
                        "total_count": 5
                    }
                },
                "comments": {
                    "summary": {
                        "total_count": 1
                    }
                },
                "type": "photo"
            }
        ]
    }
    
    with open(os.path.join(facebook_dir, "12345_posts_123.json"), 'w') as f:
        json.dump(post_data, f)
    
    # Create sample friends data
    friends_data = {
        "data": [
            {
                "id": "67890",
                "name": "Friend 1"
            },
            {
                "id": "54321",
                "name": "Friend 2"
            }
        ]
    }
    
    with open(os.path.join(facebook_dir, "12345_friends_123.json"), 'w') as f:
        json.dump(friends_data, f)
    
    # Define test config
    config = {
        "min_posts_per_user": 1,
        "max_posts_per_user": 100,
        "max_followers": 1000,
        "text_cleaning": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "remove_punctuation": True,
            "lemmatize": True
        }
    }
    
    # Process data
    processed_data = preprocess.process_facebook_data(raw_dir, processed_dir, config)
    
    # Check results
    assert processed_data is not None
    assert "users" in processed_data
    assert "posts" in processed_data
    assert "networks" in processed_data
    
    # Check if output files were created
    assert os.path.exists(os.path.join(processed_dir, "facebook", "processed_data.json"))
    assert os.path.exists(os.path.join(processed_dir, "facebook", "network.edgelist"))
    
    # Check user data
    anon_user_id = preprocess.anonymize_user_id("12345", "facebook")
    assert anon_user_id in processed_data["users"]
    assert processed_data["users"][anon_user_id]["name"] == "Test User"
    
    # Check post data
    assert anon_user_id in processed_data["posts"]
    assert len(processed_data["posts"][anon_user_id]) == 2
    
    # Check network data
    assert "graph" in processed_data["networks"]
    graph = processed_data["networks"]["graph"]
    assert isinstance(graph, nx.Graph)  # Facebook graph is undirected
    assert anon_user_id in graph.nodes()
    assert graph.number_of_nodes() > 1  # Should have user and friends


def test_process_all_platforms(tmpdir):
    """Test processing data from all platforms."""
    # Create test directories
    raw_dir = os.path.join(tmpdir, "raw")
    processed_dir = os.path.join(tmpdir, "processed")
    
    os.makedirs(os.path.join(raw_dir, "instagram"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "twitter"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "facebook"), exist_ok=True)
    
    # Define test config
    config = {
        "min_posts_per_user": 1,
        "max_posts_per_user": 100,
        "max_followers": 1000,
        "text_cleaning": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "remove_punctuation": True,
            "lemmatize": True
        }
    }
    
    # Mock platform processing functions
    with patch('preprocess.process_instagram_data') as mock_instagram, \
         patch('preprocess.process_twitter_data') as mock_twitter, \
         patch('preprocess.process_facebook_data') as mock_facebook:
        
        # Set mock return values
        mock_instagram.return_value = {"users": {"user1": {}}, "posts": {"user1": []}}
        mock_twitter.return_value = {"users": {"user2": {}}, "posts": {"user2": []}, "networks": {"graph": nx.DiGraph()}}
        mock_facebook.return_value = {"users": {"user3": {}}, "posts": {"user3": []}, "networks": {"graph": nx.Graph()}}
        
        # Process data
        processed_data = preprocess.process_all_platforms(raw_dir, processed_dir, config)
        
        # Check if all platforms were processed
        mock_instagram.assert_called_once()
        mock_twitter.assert_called_once()
        mock_facebook.assert_called_once()
        
        # Check results
        assert processed_data is not None
        assert "instagram" in processed_data
        assert "twitter" in processed_data
        assert "facebook" in processed_data
        
        # Check if summary file was created
        assert os.path.exists(os.path.join(processed_dir, "summary_stats.json"))
