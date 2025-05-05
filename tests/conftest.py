"""
Pytest configuration file with shared fixtures.
"""

import os
import json
import datetime
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import torch
from unittest.mock import MagicMock


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "api": {
            "instagram": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "access_token": "test_access_token",
                "rate_limit": 200
            },
            "twitter": {
                "api_key": "test_api_key",
                "api_secret": "test_api_secret",
                "bearer_token": "test_bearer_token",
                "access_token": "test_access_token",
                "access_secret": "test_access_secret",
                "rate_limit": 450
            },
            "facebook": {
                "app_id": "test_app_id",
                "app_secret": "test_app_secret",
                "access_token": "test_access_token",
                "rate_limit": 200
            }
        },
        "directories": {
            "raw_data": "./raw_data",
            "processed_data": "./processed_data",
            "models": "./models",
            "results": "./results"
        },
        "preprocessing": {
            "min_posts_per_user": 5,
            "max_posts_per_user": 1000,
            "language": "en",
            "text_cleaning": {
                "remove_urls": True,
                "remove_mentions": True,
                "remove_hashtags": False,
                "lowercase": True,
                "remove_punctuation": True,
                "lemmatize": True
            }
        },
        "features": {
            "behavior": {
                "time_window": 90,
                "temporal_granularity": "hour",
                "activity_types": ["post", "comment", "like", "share"]
            },
            "content": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_tokens": 512,
                "lda_topics": 20
            },
            "network": {
                "max_followers": 5000,
                "community_algorithm": "louvain",
                "centrality_metrics": ["degree", "betweenness", "closeness", "eigenvector"]
            }
        },
        "fusion": {
            "dimensionality_reduction": "pca",
            "explained_variance": 0.95,
            "embedding_dim": 128
        },
        "models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 20,
                "random_state": 42
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "random_state": 42
            },
            "gcn": {
                "hidden_channels": [128, 64],
                "dropout": 0.2,
                "learning_rate": 0.01,
                "epochs": 200,
                "patience": 20,
                "batch_size": 64
            }
        },
        "privacy": {
            "enable_differential_privacy": True,
            "epsilon": 1.0,
            "delta": 1e-5,
            "k_anonymity": 5,
            "l_diversity": 3
        },
        "evaluation": {
            "train_test_split": 0.8,
            "cross_validation_folds": 5,
            "metrics": ["accuracy", "precision", "recall", "f1"]
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "user_matching.log"
        }
    }


@pytest.fixture
def sample_posts():
    """Sample posts for testing feature extraction."""
    return [
        {
            "id": "post1",
            "text": "This is a test post #testing with @user mention",
            "original_text": "This is a test post #testing with @user mention",
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
            "engagement": {"likes": 10, "comments": 2, "shares": 1},
            "media_type": "photo"
        },
        {
            "id": "post2",
            "text": "Another test post about #ai and #ml",
            "original_text": "Another test post about #ai and #ml",
            "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
            "engagement": {"likes": 5, "comments": 1, "shares": 0},
            "media_type": "photo"
        },
        {
            "id": "post3",
            "text": "Video post with https://example.com link",
            "original_text": "Video post with https://example.com link",
            "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
            "engagement": {"likes": 20, "comments": 5, "shares": 3},
            "media_type": "video"
        }
    ]


@pytest.fixture
def sample_instagram_processed_data():
    """Sample processed Instagram data."""
    user1_id = "anon_user1_instagram"
    user2_id = "anon_user2_instagram"
    
    return {
        "users": {
            user1_id: {
                "platform": "instagram",
                "username": "user1",
                "account_type": "personal",
                "media_count": 50,
                "original_id": "123456"
            },
            user2_id: {
                "platform": "instagram",
                "username": "user2",
                "account_type": "business",
                "media_count": 100,
                "original_id": "789012"
            }
        },
        "posts": {
            user1_id: [
                {
                    "id": "post1",
                    "text": "This is a test post",
                    "original_text": "This is a test post #testing",
                    "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
                    "engagement": {"likes": 10, "comments": 2},
                    "media_type": "photo"
                },
                {
                    "id": "post2",
                    "text": "Another post",
                    "original_text": "Another post about #ai",
                    "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
                    "engagement": {"likes": 5, "comments": 1},
                    "media_type": "photo"
                }
            ],
            user2_id: [
                {
                    "id": "post3",
                    "text": "Business post",
                    "original_text": "Business post about #marketing",
                    "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
                    "engagement": {"likes": 20, "comments": 5},
                    "media_type": "video"
                }
            ]
        },
        "networks": {}
    }


@pytest.fixture
def sample_twitter_processed_data():
    """Sample processed Twitter data."""
    user1_id = "anon_user1_twitter"
    user2_id = "anon_user2_twitter"
    
    # Create a graph
    G = nx.DiGraph()
    G.add_node(user1_id, platform="twitter")
    G.add_node(user2_id, platform="twitter")
    G.add_node("follower1", platform="twitter")
    G.add_node("follower2", platform="twitter")
    G.add_edge("follower1", user1_id)  # follower1 follows user1
    G.add_edge("follower2", user1_id)  # follower2 follows user1
    G.add_edge(user1_id, user2_id)     # user1 follows user2
    
    return {
        "users": {
            user1_id: {
                "platform": "twitter",
                "username": "user1",
                "name": "User One",
                "created_at": "2020-01-01T00:00:00+0000",
                "public_metrics": {
                    "followers_count": 100,
                    "following_count": 50,
                    "tweet_count": 200
                },
                "original_id": "123456"
            },
            user2_id: {
                "platform": "twitter",
                "username": "user2",
                "name": "User Two",
                "created_at": "2020-02-01T00:00:00+0000",
                "public_metrics": {
                    "followers_count": 200,
                    "following_count": 100,
                    "tweet_count": 300
                },
                "original_id": "789012"
            }
        },
        "posts": {
            user1_id: [
                {
                    "id": "tweet1",
                    "text": "This is a test tweet",
                    "original_text": "This is a test tweet #testing",
                    "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
                    "engagement": {
                        "likes": 10,
                        "retweets": 2,
                        "replies": 1,
                        "quotes": 0
                    },
                    "type": "original"
                },
                {
                    "id": "tweet2",
                    "text": "Another tweet",
                    "original_text": "Another tweet about #ai",
                    "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
                    "engagement": {
                        "likes": 5,
                        "retweets": 1,
                        "replies": 0,
                        "quotes": 0
                    },
                    "type": "original"
                }
            ],
            user2_id: [
                {
                    "id": "tweet3",
                    "text": "Professional tweet",
                    "original_text": "Professional tweet about #marketing",
                    "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
                    "engagement": {
                        "likes": 20,
                        "retweets": 5,
                        "replies": 2,
                        "quotes": 1
                    },
                    "type": "original"
                }
            ]
        },
        "networks": {
            "graph": G,
            "connections": {
                user1_id: {
                    "followers": ["follower1", "follower2"],
                    "following": [user2_id]
                },
                user2_id: {
                    "followers": [user1_id],
                    "following": []
                }
            }
        }
    }


@pytest.fixture
def sample_facebook_processed_data():
    """Sample processed Facebook data."""
    user1_id = "anon_user1_facebook"
    user2_id = "anon_user2_facebook"
    
    # Create a graph
    G = nx.Graph()
    G.add_node(user1_id, platform="facebook")
    G.add_node(user2_id, platform="facebook")
    G.add_node("friend1", platform="facebook")
    G.add_node("friend2", platform="facebook")
    G.add_edge(user1_id, "friend1")  # user1 is friends with friend1
    G.add_edge(user1_id, "friend2")  # user1 is friends with friend2
    G.add_edge(user1_id, user2_id)   # user1 is friends with user2
    
    return {
        "users": {
            user1_id: {
                "platform": "facebook",
                "name": "User One",
                "about": "Test bio",
                "gender": "male",
                "hometown": "Test City",
                "original_id": "123456"
            },
            user2_id: {
                "platform": "facebook",
                "name": "User Two",
                "about": "Professional bio",
                "gender": "female",
                "hometown": "Another City",
                "original_id": "789012"
            }
        },
        "posts": {
            user1_id: [
                {
                    "id": "fb_post1",
                    "text": "This is a test post",
                    "original_text": "This is a test post #testing",
                    "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0),
                    "engagement": {
                        "likes": 10,
                        "comments": 2
                    },
                    "type": "status"
                },
                {
                    "id": "fb_post2",
                    "text": "Another post",
                    "original_text": "Another post about #ai",
                    "timestamp": datetime.datetime(2023, 1, 2, 15, 30, 0),
                    "engagement": {
                        "likes": 5,
                        "comments": 1
                    },
                    "type": "photo"
                }
            ],
            user2_id: [
                {
                    "id": "fb_post3",
                    "text": "Professional post",
                    "original_text": "Professional post about #marketing",
                    "timestamp": datetime.datetime(2023, 1, 3, 10, 15, 0),
                    "engagement": {
                        "likes": 20,
                        "comments": 5
                    },
                    "type": "link"
                }
            ]
        },
        "networks": {
            "graph": G
        }
    }


@pytest.fixture
def sample_processed_data(sample_instagram_processed_data, sample_twitter_processed_data, sample_facebook_processed_data):
    """Combined sample processed data from all platforms."""
    return {
        "instagram": sample_instagram_processed_data,
        "twitter": sample_twitter_processed_data,
        "facebook": sample_facebook_processed_data
    }


@pytest.fixture
def sample_behavior_features():
    """Sample behavioral features for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    return {
        "instagram": {
            user1_instagram_id: {
                "user_id": user1_instagram_id,
                "post_count": 2,
                "temporal": {
                    "posting_frequency": 1.0,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 24.0,
                        "median_hours_between_posts": 24.0,
                        "std_hours_between_posts": 2.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 15,
                            "mean": 7.5,
                            "median": 7.5,
                            "std": 3.5,
                            "max": 10
                        },
                        "comments": {
                            "total": 3,
                            "mean": 1.5,
                            "median": 1.5,
                            "std": 0.7,
                            "max": 2
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 7.5,
                        "comments_per_post": 1.5,
                        "likes_to_comments_ratio": 5.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "photo": 1.0
                    },
                    "text_length_stats": {
                        "mean": 15.0,
                        "median": 15.0,
                        "std": 5.0,
                        "max": 18,
                        "min": 12
                    },
                    "word_count_stats": {
                        "mean": 3.0,
                        "median": 3.0,
                        "std": 1.0,
                        "max": 4,
                        "min": 2
                    }
                }
            },
            user2_instagram_id: {
                "user_id": user2_instagram_id,
                "post_count": 1,
                "temporal": {
                    "posting_frequency": 0.5,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 0.0,
                        "median_hours_between_posts": 0.0,
                        "std_hours_between_posts": 0.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 20,
                            "mean": 20.0,
                            "median": 20.0,
                            "std": 0.0,
                            "max": 20
                        },
                        "comments": {
                            "total": 5,
                            "mean": 5.0,
                            "median": 5.0,
                            "std": 0.0,
                            "max": 5
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 20.0,
                        "comments_per_post": 5.0,
                        "likes_to_comments_ratio": 4.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "video": 1.0
                    },
                    "text_length_stats": {
                        "mean": 13.0,
                        "median": 13.0,
                        "std": 0.0,
                        "max": 13,
                        "min": 13
                    },
                    "word_count_stats": {
                        "mean": 2.0,
                        "median": 2.0,
                        "std": 0.0,
                        "max": 2,
                        "min": 2
                    }
                }
            }
        },
        "twitter": {
            user1_twitter_id: {
                "user_id": user1_twitter_id,
                "post_count": 2,
                "temporal": {
                    "posting_frequency": 1.0,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 24.0,
                        "median_hours_between_posts": 24.0,
                        "std_hours_between_posts": 2.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 15,
                            "mean": 7.5,
                            "median": 7.5,
                            "std": 3.5,
                            "max": 10
                        },
                        "retweets": {
                            "total": 3,
                            "mean": 1.5,
                            "median": 1.5,
                            "std": 0.7,
                            "max": 2
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 7.5,
                        "comments_per_post": 0.5,
                        "likes_to_comments_ratio": 15.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "original": 1.0
                    },
                    "text_length_stats": {
                        "mean": 18.0,
                        "median": 18.0,
                        "std": 5.0,
                        "max": 21,
                        "min": 15
                    },
                    "word_count_stats": {
                        "mean": 3.5,
                        "median": 3.5,
                        "std": 0.7,
                        "max": 4,
                        "min": 3
                    }
                }
            },
            user2_twitter_id: {
                "user_id": user2_twitter_id,
                "post_count": 1,
                "temporal": {
                    "posting_frequency": 0.5,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 0.0,
                        "median_hours_between_posts": 0.0,
                        "std_hours_between_posts": 0.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 20,
                            "mean": 20.0,
                            "median": 20.0,
                            "std": 0.0,
                            "max": 20
                        },
                        "retweets": {
                            "total": 5,
                            "mean": 5.0,
                            "median": 5.0,
                            "std": 0.0,
                            "max": 5
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 20.0,
                        "comments_per_post": 2.0,
                        "likes_to_comments_ratio": 10.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "original": 1.0
                    },
                    "text_length_stats": {
                        "mean": 17.0,
                        "median": 17.0,
                        "std": 0.0,
                        "max": 17,
                        "min": 17
                    },
                    "word_count_stats": {
                        "mean": 3.0,
                        "median": 3.0,
                        "std": 0.0,
                        "max": 3,
                        "min": 3
                    }
                }
            }
        },
        "facebook": {
            user1_facebook_id: {
                "user_id": user1_facebook_id,
                "post_count": 2,
                "temporal": {
                    "posting_frequency": 1.0,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 24.0,
                        "median_hours_between_posts": 24.0,
                        "std_hours_between_posts": 2.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 15,
                            "mean": 7.5,
                            "median": 7.5,
                            "std": 3.5,
                            "max": 10
                        },
                        "comments": {
                            "total": 3,
                            "mean": 1.5,
                            "median": 1.5,
                            "std": 0.7,
                            "max": 2
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 7.5,
                        "comments_per_post": 1.5,
                        "likes_to_comments_ratio": 5.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "status": 0.5,
                        "photo": 0.5
                    },
                    "text_length_stats": {
                        "mean": 15.0,
                        "median": 15.0,
                        "std": 5.0,
                        "max": 18,
                        "min": 12
                    },
                    "word_count_stats": {
                        "mean": 3.0,
                        "median": 3.0,
                        "std": 1.0,
                        "max": 4,
                        "min": 2
                    }
                }
            },
            user2_facebook_id: {
                "user_id": user2_facebook_id,
                "post_count": 1,
                "temporal": {
                    "posting_frequency": 0.5,
                    "hourly_histogram": [0.1] * 24,
                    "daily_histogram": [0.1] * 7,
                    "monthly_histogram": [0.1] * 12,
                    "regularity": {
                        "mean_hours_between_posts": 0.0,
                        "median_hours_between_posts": 0.0,
                        "std_hours_between_posts": 0.0
                    }
                },
                "engagement": {
                    "stats": {
                        "likes": {
                            "total": 20,
                            "mean": 20.0,
                            "median": 20.0,
                            "std": 0.0,
                            "max": 20
                        },
                        "comments": {
                            "total": 5,
                            "mean": 5.0,
                            "median": 5.0,
                            "std": 0.0,
                            "max": 5
                        },
                        "like_histogram": [0.1] * 6,
                        "comment_histogram": [0.1] * 6
                    },
                    "ratios": {
                        "likes_per_post": 20.0,
                        "comments_per_post": 5.0,
                        "likes_to_comments_ratio": 4.0
                    }
                },
                "content_types": {
                    "content_type_distribution": {
                        "link": 1.0
                    },
                    "text_length_stats": {
                        "mean": 17.0,
                        "median": 17.0,
                        "std": 0.0,
                        "max": 17,
                        "min": 17
                    },
                    "word_count_stats": {
                        "mean": 3.0,
                        "median": 3.0,
                        "std": 0.0,
                        "max": 3,
                        "min": 3
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_content_features():
    """Sample content features for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    # Create sample embeddings
    embedding_dim = 384  # Default for "all-MiniLM-L6-v2"
    topic_dim = 20  # Default topics
    
    return {
        "instagram": {
            user1_instagram_id: {
                "user_id": user1_instagram_id,
                "post_count": 2,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            },
            user2_instagram_id: {
                "user_id": user2_instagram_id,
                "post_count": 1,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            }
        },
        "twitter": {
            user1_twitter_id: {
                "user_id": user1_twitter_id,
                "post_count": 2,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            },
            user2_twitter_id: {
                "user_id": user2_twitter_id,
                "post_count": 1,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            }
        },
        "facebook": {
            user1_facebook_id: {
                "user_id": user1_facebook_id,
                "post_count": 2,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            },
            user2_facebook_id: {
                "user_id": user2_facebook_id,
                "post_count": 1,
                "has_content": True,
                "avg_embedding": np.random.randn(embedding_dim).astype(np.float32),
                "avg_topic_distribution": np.random.rand(topic_dim).astype(np.float32)
            }
        }
    }


@pytest.fixture
def sample_network_features():
    """Sample network features for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    return {
        "twitter": {
            user1_twitter_id: {
                "user_id": user1_twitter_id,
                "in_graph": True,
                "centrality": {
                    "degree": 3,
                    "in_degree": 2,
                    "out_degree": 1,
                    "betweenness": 0.5,
                    "closeness": 0.7,
                    "eigenvector": 0.8,
                    "pagerank": 0.3
                },
                "community_id": 0,
                "ego_network": {
                    "size": 4,
                    "density": 0.5,
                    "clustering": 0.3,
                    "avg_path_length": 1.5
                },
                "neighbor_count": 3,
                "in_neighbor_count": 2,
                "out_neighbor_count": 1,
                "community_diversity": 1,
                "reciprocity": 0.0
            },
            user2_twitter_id: {
                "user_id": user2_twitter_id,
                "in_graph": True,
                "centrality": {
                    "degree": 1,
                    "in_degree": 1,
                    "out_degree": 0,
                    "betweenness": 0.0,
                    "closeness": 0.5,
                    "eigenvector": 0.5,
                    "pagerank": 0.2
                },
                "community_id": 0,
                "ego_network": {
                    "size": 2,
                    "density": 1.0,
                    "clustering": 0.0,
                    "avg_path_length": 1.0
                },
                "neighbor_count": 1,
                "in_neighbor_count": 1,
                "out_neighbor_count": 0,
                "community_diversity": 1,
                "reciprocity": 0.0
            }
        },
        "facebook": {
            user1_facebook_id: {
                "user_id": user1_facebook_id,
                "in_graph": True,
                "centrality": {
                    "degree": 3,
                    "betweenness": 0.5,
                    "closeness": 0.7,
                    "eigenvector": 0.8
                },
                "community_id": 0,
                "ego_network": {
                    "size": 4,
                    "density": 0.5,
                    "clustering": 0.3,
                    "avg_path_length": 1.5
                },
                "neighbor_count": 3,
                "in_neighbor_count": 3,
                "out_neighbor_count": 3,
                "community_diversity": 1,
                "reciprocity": 1.0
            },
            user2_facebook_id: {
                "user_id": user2_facebook_id,
                "in_graph": True,
                "centrality": {
                    "degree": 1,
                    "betweenness": 0.0,
                    "closeness": 0.5,
                    "eigenvector": 0.5
                },
                "community_id": 0,
                "ego_network": {
                    "size": 2,
                    "density": 1.0,
                    "clustering": 0.0,
                    "avg_path_length": 1.0
                },
                "neighbor_count": 1,
                "in_neighbor_count": 1,
                "out_neighbor_count": 1,
                "community_diversity": 1,
                "reciprocity": 1.0
            }
        }
    }


@pytest.fixture
def sample_feature_vectors():
    """Sample feature vectors for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    # Create sample feature vectors
    dim = 50
    
    behavior_vectors = {
        "instagram": {
            user1_instagram_id: np.random.randn(dim).astype(np.float32),
            user2_instagram_id: np.random.randn(dim).astype(np.float32)
        },
        "twitter": {
            user1_twitter_id: np.random.randn(dim).astype(np.float32),
            user2_twitter_id: np.random.randn(dim).astype(np.float32)
        },
        "facebook": {
            user1_facebook_id: np.random.randn(dim).astype(np.float32),
            user2_facebook_id: np.random.randn(dim).astype(np.float32)
        }
    }
    
    content_vectors = {
        "instagram": {
            user1_instagram_id: np.random.randn(dim).astype(np.float32),
            user2_instagram_id: np.random.randn(dim).astype(np.float32)
        },
        "twitter": {
            user1_twitter_id: np.random.randn(dim).astype(np.float32),
            user2_twitter_id: np.random.randn(dim).astype(np.float32)
        },
        "facebook": {
            user1_facebook_id: np.random.randn(dim).astype(np.float32),
            user2_facebook_id: np.random.randn(dim).astype(np.float32)
        }
    }
    
    network_vectors = {
        "twitter": {
            user1_twitter_id: np.random.randn(dim).astype(np.float32),
            user2_twitter_id: np.random.randn(dim).astype(np.float32)
        },
        "facebook": {
            user1_facebook_id: np.random.randn(dim).astype(np.float32),
            user2_facebook_id: np.random.randn(dim).astype(np.float32)
        }
    }
    
    return behavior_vectors, content_vectors, network_vectors


@pytest.fixture
def sample_fused_features():
    """Sample fused features for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    # Create sample fused vectors
    dim = 128
    
    reduced_vectors = {
        "instagram": {
            user1_instagram_id: np.random.randn(dim).astype(np.float32),
            user2_instagram_id: np.random.randn(dim).astype(np.float32)
        },
        "twitter": {
            user1_twitter_id: np.random.randn(dim).astype(np.float32),
            user2_twitter_id: np.random.randn(dim).astype(np.float32)
        },
        "facebook": {
            user1_facebook_id: np.random.randn(dim).astype(np.float32),
            user2_facebook_id: np.random.randn(dim).astype(np.float32)
        }
    }
    
    cross_platform_features = {
        "instagram_twitter": {
            "instagram": {
                user1_instagram_id: np.random.randn(dim).astype(np.float32),
                user2_instagram_id: np.random.randn(dim).astype(np.float32)
            },
            "twitter": {
                user1_twitter_id: np.random.randn(dim).astype(np.float32),
                user2_twitter_id: np.random.randn(dim).astype(np.float32)
            }
        },
        "instagram_facebook": {
            "instagram": {
                user1_instagram_id: np.random.randn(dim).astype(np.float32),
                user2_instagram_id: np.random.randn(dim).astype(np.float32)
            },
            "facebook": {
                user1_facebook_id: np.random.randn(dim).astype(np.float32),
                user2_facebook_id: np.random.randn(dim).astype(np.float32)
            }
        },
        "twitter_facebook": {
            "twitter": {
                user1_twitter_id: np.random.randn(dim).astype(np.float32),
                user2_twitter_id: np.random.randn(dim).astype(np.float32)
            },
            "facebook": {
                user1_facebook_id: np.random.randn(dim).astype(np.float32),
                user2_facebook_id: np.random.randn(dim).astype(np.float32)
            }
        }
    }
    
    return reduced_vectors, cross_platform_features


@pytest.fixture
def sample_ground_truth_mappings():
    """Sample ground truth mappings for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    return {
        "instagram_twitter": {
            user1_instagram_id: user1_twitter_id,
            user2_instagram_id: user2_twitter_id
        },
        "instagram_facebook": {
            user1_instagram_id: user1_facebook_id,
            user2_instagram_id: user2_facebook_id
        },
        "twitter_facebook": {
            user1_twitter_id: user1_facebook_id,
            user2_twitter_id: user2_facebook_id
        }
    }


@pytest.fixture
def sample_model_predictions():
    """Sample model predictions for testing."""
    user1_instagram_id = "anon_user1_instagram"
    user2_instagram_id = "anon_user2_instagram"
    user1_twitter_id = "anon_user1_twitter"
    user2_twitter_id = "anon_user2_twitter"
    user1_facebook_id = "anon_user1_facebook"
    user2_facebook_id = "anon_user2_facebook"
    
    return {
        "instagram_twitter": {
            "random_forest": {
                "predictions": {
                    user1_instagram_id: {
                        user1_twitter_id: 0.9,
                        user2_twitter_id: 0.1
                    },
                    user2_instagram_id: {
                        user1_twitter_id: 0.2,
                        user2_twitter_id: 0.8
                    }
                }
            },
            "svm": {
                "predictions": {
                    user1_instagram_id: {
                        user1_twitter_id: 0.85,
                        user2_twitter_id: 0.15
                    },
                    user2_instagram_id: {
                        user1_twitter_id: 0.25,
                        user2_twitter_id: 0.75
                    }
                }
            },
            "gcn": {
                "predictions": {
                    user1_instagram_id: {
                        user1_twitter_id: 0.95,
                        user2_twitter_id: 0.05
                    },
                    user2_instagram_id: {
                        user1_twitter_id: 0.1,
                        user2_twitter_id: 0.9
                    }
                }
            },
            "ground_truth": {
                user1_instagram_id: user1_twitter_id,
                user2_instagram_id: user2_twitter_id
            }
        },
        "instagram_facebook": {
            "random_forest": {
                "predictions": {
                    user1_instagram_id: {
                        user1_facebook_id: 0.8,
                        user2_facebook_id: 0.2
                    },
                    user2_instagram_id: {
                        user1_facebook_id: 0.3,
                        user2_facebook_id: 0.7
                    }
                }
            },
            "svm": {
                "predictions": {
                    user1_instagram_id: {
                        user1_facebook_id: 0.75,
                        user2_facebook_id: 0.25
                    },
                    user2_instagram_id: {
                        user1_facebook_id: 0.35,
                        user2_facebook_id: 0.65
                    }
                }
            },
            "gcn": {
                "predictions": {
                    user1_instagram_id: {
                        user1_facebook_id: 0.85,
                        user2_facebook_id: 0.15
                    },
                    user2_instagram_id: {
                        user1_facebook_id: 0.2,
                        user2_facebook_id: 0.8
                    }
                }
            },
            "ground_truth": {
                user1_instagram_id: user1_facebook_id,
                user2_instagram_id: user2_facebook_id
            }
        },
        "twitter_facebook": {
            "random_forest": {
                "predictions": {
                    user1_twitter_id: {
                        user1_facebook_id: 0.85,
                        user2_facebook_id: 0.15
                    },
                    user2_twitter_id: {
                        user1_facebook_id: 0.25,
                        user2_facebook_id: 0.75
                    }
                }
            },
            "svm": {
                "predictions": {
                    user1_twitter_id: {
                        user1_facebook_id: 0.8,
                        user2_facebook_id: 0.2
                    },
                    user2_twitter_id: {
                        user1_facebook_id: 0.3,
                        user2_facebook_id: 0.7
                    }
                }
            },
            "gcn": {
                "predictions": {
                    user1_twitter_id: {
                        user1_facebook_id: 0.9,
                        user2_facebook_id: 0.1
                    },
                    user2_twitter_id: {
                        user1_facebook_id: 0.2,
                        user2_facebook_id: 0.8
                    }
                }
            },
            "ground_truth": {
                user1_twitter_id: user1_facebook_id,
                user2_twitter_id: user2_facebook_id
            }
        }
    }


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
    return mock_model
