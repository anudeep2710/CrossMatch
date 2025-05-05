"""
Tests for the behavior_features module.
"""

import os
import json
import datetime
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import behavior_features


def test_create_temporal_features():
    """Test extraction of temporal features from user posts."""
    # Create test posts
    posts = [
        {
            "id": "post1",
            "text": "Test post 1",
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
            "engagement": {"likes": 10, "comments": 2}
        },
        {
            "id": "post2",
            "text": "Test post 2",
            "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
            "engagement": {"likes": 5, "comments": 1}
        },
        {
            "id": "post3",
            "text": "Test post 3",
            "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
            "engagement": {"likes": 20, "comments": 5}
        }
    ]
    
    config = {
        "temporal_granularity": "hour",
        "time_window": 90
    }
    
    # Extract features
    features = behavior_features.create_temporal_features(posts, config)
    
    # Check results
    assert features is not None
    assert "posting_frequency" in features
    assert "hourly_histogram" in features
    assert "daily_histogram" in features
    assert "monthly_histogram" in features
    assert "regularity" in features
    
    # Check posting frequency (3 posts in 3 days)
    assert features["posting_frequency"] == 1.0
    
    # Check hourly histogram
    assert len(features["hourly_histogram"]) == 24
    assert features["hourly_histogram"][10] > 0  # 2 posts at 10 AM
    assert features["hourly_histogram"][15] > 0  # 1 post at 3:30 PM
    
    # Check daily histogram
    assert len(features["daily_histogram"]) == 7
    
    # Check regularity
    assert "mean_hours_between_posts" in features["regularity"]
    assert "std_hours_between_posts" in features["regularity"]
    
    # Test with empty posts
    features = behavior_features.create_temporal_features([], config)
    assert features == {}


def test_create_engagement_features():
    """Test extraction of engagement features from user posts."""
    # Create test posts
    posts = [
        {
            "id": "post1",
            "text": "Test post 1",
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
            "engagement": {"likes": 10, "comments": 2}
        },
        {
            "id": "post2",
            "text": "Test post 2",
            "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
            "engagement": {"likes": 5, "comments": 1}
        },
        {
            "id": "post3",
            "text": "Test post 3",
            "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
            "engagement": {"likes": 20, "comments": 5}
        }
    ]
    
    config = {
        "activity_types": ["post", "comment", "like", "share"]
    }
    
    # Extract features
    features = behavior_features.create_engagement_features(posts, config)
    
    # Check results
    assert features is not None
    assert "stats" in features
    assert "ratios" in features
    
    # Check like statistics
    assert "likes" in features["stats"]
    assert features["stats"]["likes"]["total"] == 35  # 10 + 5 + 20
    assert features["stats"]["likes"]["mean"] == 35 / 3
    
    # Check comment statistics
    assert "comments" in features["stats"]
    assert features["stats"]["comments"]["total"] == 8  # 2 + 1 + 5
    
    # Check engagement ratios
    assert "likes_per_post" in features["ratios"]
    assert features["ratios"]["likes_per_post"] == 35 / 3
    assert "comments_per_post" in features["ratios"]
    assert features["ratios"]["comments_per_post"] == 8 / 3
    assert "likes_to_comments_ratio" in features["ratios"]
    assert features["ratios"]["likes_to_comments_ratio"] == 35 / 8
    
    # Check histograms
    assert "like_histogram" in features["stats"]
    assert "comment_histogram" in features["stats"]
    
    # Test with empty posts
    features = behavior_features.create_engagement_features([], config)
    assert features == {}


def test_create_content_type_features():
    """Test extraction of content type features from user posts."""
    # Create test posts
    posts = [
        {
            "id": "post1",
            "text": "This is a short post",
            "media_type": "photo",
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0)
        },
        {
            "id": "post2",
            "text": "This is a longer post with more content",
            "media_type": "photo",
            "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0)
        },
        {
            "id": "post3",
            "text": "Video post",
            "media_type": "video",
            "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0)
        }
    ]
    
    config = {}
    
    # Extract features
    features = behavior_features.create_content_type_features(posts, config)
    
    # Check results
    assert features is not None
    assert "content_type_distribution" in features
    assert "text_length_stats" in features
    assert "word_count_stats" in features
    
    # Check content type distribution
    assert "photo" in features["content_type_distribution"]
    assert features["content_type_distribution"]["photo"] == 2/3
    assert "video" in features["content_type_distribution"]
    assert features["content_type_distribution"]["video"] == 1/3
    
    # Check text length statistics
    assert "mean" in features["text_length_stats"]
    assert "median" in features["text_length_stats"]
    assert "std" in features["text_length_stats"]
    
    # Check word count statistics
    assert "mean" in features["word_count_stats"]
    assert "median" in features["word_count_stats"]
    assert "std" in features["word_count_stats"]
    
    # Test with empty posts
    features = behavior_features.create_content_type_features([], config)
    assert features == {}


def test_extract_behavioral_features_for_user():
    """Test extraction of all behavioral features for a single user."""
    user_id = "test_user"
    
    # Create test posts
    posts = [
        {
            "id": "post1",
            "text": "Test post 1",
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
            "engagement": {"likes": 10, "comments": 2},
            "media_type": "photo"
        },
        {
            "id": "post2",
            "text": "Test post 2",
            "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
            "engagement": {"likes": 5, "comments": 1},
            "media_type": "photo"
        },
        {
            "id": "post3",
            "text": "Test post 3",
            "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
            "engagement": {"likes": 20, "comments": 5},
            "media_type": "video"
        }
    ]
    
    config = {
        "temporal_granularity": "hour",
        "time_window": 90,
        "activity_types": ["post", "comment", "like", "share"]
    }
    
    # Extract features
    with patch('behavior_features.create_temporal_features') as mock_temporal, \
         patch('behavior_features.create_engagement_features') as mock_engagement, \
         patch('behavior_features.create_content_type_features') as mock_content:
        
        # Set mock return values
        mock_temporal.return_value = {"posting_frequency": 1.0}
        mock_engagement.return_value = {"stats": {}, "ratios": {}}
        mock_content.return_value = {"content_type_distribution": {}}
        
        features = behavior_features.extract_behavioral_features_for_user(user_id, posts, config)
        
        # Check if all feature types were extracted
        mock_temporal.assert_called_once()
        mock_engagement.assert_called_once()
        mock_content.assert_called_once()
        
        # Check results
        assert features is not None
        assert features["user_id"] == user_id
        assert features["post_count"] == 3
        assert "temporal" in features
        assert "engagement" in features
        assert "content_types" in features


def test_extract_behavioral_features_for_platform():
    """Test extraction of behavioral features for all users on a platform."""
    # Create test platform data
    platform_data = {
        "posts": {
            "user1": [
                {
                    "id": "post1",
                    "text": "Test post 1",
                    "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
                    "engagement": {"likes": 10, "comments": 2}
                }
            ],
            "user2": [
                {
                    "id": "post2",
                    "text": "Test post 2",
                    "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
                    "engagement": {"likes": 5, "comments": 1}
                }
            ]
        }
    }
    
    config = {
        "temporal_granularity": "hour",
        "time_window": 90,
        "activity_types": ["post", "comment", "like", "share"]
    }
    
    # Extract features
    with patch('behavior_features.extract_behavioral_features_for_user') as mock_user_features:
        # Set mock return value
        mock_user_features.return_value = {"user_id": "test", "post_count": 1}
        
        features = behavior_features.extract_behavioral_features_for_platform(platform_data, config)
        
        # Check if extract_behavioral_features_for_user was called for each user
        assert mock_user_features.call_count == 2
        
        # Check results
        assert features is not None
        assert len(features) == 2
        assert "user1" in features
        assert "user2" in features
    
    # Test with empty platform data
    features = behavior_features.extract_behavioral_features_for_platform({}, config)
    assert features == {}
    
    # Test with platform data without posts
    features = behavior_features.extract_behavioral_features_for_platform({"users": {}}, config)
    assert features == {}


def test_extract_all_behavioral_features():
    """Test extraction of behavioral features for all users across all platforms."""
    # Create test processed data
    processed_data = {
        "instagram": {
            "posts": {
                "user1": [
                    {
                        "id": "post1",
                        "text": "Test post 1",
                        "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
                        "engagement": {"likes": 10, "comments": 2}
                    }
                ]
            }
        },
        "twitter": {
            "posts": {
                "user2": [
                    {
                        "id": "post2",
                        "text": "Test post 2",
                        "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
                        "engagement": {"likes": 5, "comments": 1}
                    }
                ]
            }
        }
    }
    
    config = {
        "temporal_granularity": "hour",
        "time_window": 90,
        "activity_types": ["post", "comment", "like", "share"]
    }
    
    # Extract features
    with patch('behavior_features.extract_behavioral_features_for_platform') as mock_platform_features, \
         patch('behavior_features.vectorize_behavioral_features') as mock_vectorize, \
         patch('pickle.dump') as mock_pickle:
        
        # Set mock return values
        mock_platform_features.return_value = {"user1": {"user_id": "user1", "post_count": 1}}
        mock_vectorize.return_value = {"instagram": {"user1": np.array([1.0, 2.0])}}
        
        features = behavior_features.extract_all_behavioral_features(processed_data, config)
        
        # Check if extract_behavioral_features_for_platform was called for each platform
        assert mock_platform_features.call_count == 2
        
        # Check if vectorize_behavioral_features was called
        mock_vectorize.assert_called_once()
        
        # Check if results were pickled
        assert mock_pickle.call_count == 2
        
        # Check results
        assert features is not None
        assert "instagram" in features
        assert "twitter" in features


def test_vectorize_behavioral_features():
    """Test conversion of behavioral features to fixed-length vectors."""
    # Create test behavioral features
    all_features = {
        "instagram": {
            "user1": {
                "user_id": "user1",
                "post_count": 3,
                "temporal": {
                    "posting_frequency": 1.0,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "regularity": {
                        "mean_hours_between_posts": 24.0,
                        "std_hours_between_posts": 2.0
                    }
                },
                "engagement": {
                    "ratios": {
                        "likes_per_post": 10.0,
                        "comments_per_post": 2.0,
                        "likes_to_comments_ratio": 5.0
                    },
                    "stats": {
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    }
                },
                "content_types": {
                    "text_length_stats": {
                        "mean": 50.0,
                        "std": 10.0
                    },
                    "word_count_stats": {
                        "mean": 10.0,
                        "std": 2.0
                    }
                }
            }
        }
    }
    
    # Vectorize features
    vector_features = behavior_features.vectorize_behavioral_features(all_features)
    
    # Check results
    assert vector_features is not None
    assert "instagram" in vector_features
    assert "user1" in vector_features["instagram"]
    
    # Check vector dimensions
    vector = vector_features["instagram"]["user1"]
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    
    # The vector should contain:
    # - posting_frequency (1)
    # - hourly_histogram (24)
    # - daily_histogram (7)
    # - regularity features (2)
    # - engagement ratios (3)
    # - engagement histograms (12)
    # - text/word stats (4)
    # Total: 53 features
    assert vector.shape[0] > 50  # Exact number might vary based on implementation
