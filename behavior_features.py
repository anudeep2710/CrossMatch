"""
Behavior Features Module

This module extracts behavioral features from user activity data,
including temporal patterns, engagement metrics, and activity distributions.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Union, Optional
from collections import Counter
import pickle

logger = logging.getLogger(__name__)


def create_temporal_features(posts: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract temporal features from user posts.
    
    Args:
        posts: List of processed user posts
        config: Behavior feature configuration
        
    Returns:
        Dict: Temporal features including activity patterns
    """
    if not posts:
        return {}
    
    # Convert timestamps to pandas datetime
    timestamps = [p["timestamp"] for p in posts if isinstance(p["timestamp"], datetime)]
    
    if not timestamps:
        return {}
    
    df = pd.DataFrame({"timestamp": timestamps})
    df = df.sort_values("timestamp")
    
    # Get the granularity from config
    granularity = config.get("temporal_granularity", "hour")
    
    # Extract hour of day distribution
    df["hour"] = df["timestamp"].dt.hour
    hourly_activity = df["hour"].value_counts().sort_index().to_dict()
    
    # Normalize hourly activity (0-1 scale)
    total_posts = sum(hourly_activity.values())
    normalized_hourly = {h: count/total_posts for h, count in hourly_activity.items()}
    
    # Extract day of week distribution
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    daily_activity = df["day_of_week"].value_counts().sort_index().to_dict()
    
    # Normalize daily activity
    total_posts = sum(daily_activity.values())
    normalized_daily = {d: count/total_posts for d, count in daily_activity.items()}
    
    # Posting frequency features
    first_post = df["timestamp"].min()
    last_post = df["timestamp"].max()
    date_range = (last_post - first_post).days + 1  # +1 to include both first and last day
    
    # Convert date range to specified time window if needed
    time_window = config.get("time_window", 90)  # default 90 days
    date_range = min(date_range, time_window)
    
    # Calculate overall posting frequency (posts per day)
    if date_range > 0:
        posting_frequency = len(posts) / date_range
    else:
        posting_frequency = 0
    
    # Calculate posting regularity (variance in time between posts)
    if len(df) > 1:
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600  # hours
        time_diffs = df["time_diff"].dropna().tolist()
        regularity = {
            "mean_hours_between_posts": np.mean(time_diffs),
            "median_hours_between_posts": np.median(time_diffs),
            "std_hours_between_posts": np.std(time_diffs)
        }
    else:
        regularity = {
            "mean_hours_between_posts": 0,
            "median_hours_between_posts": 0,
            "std_hours_between_posts": 0
        }
    
    # Create hourly activity histogram (24 features, one per hour)
    hourly_histogram = [normalized_hourly.get(h, 0) for h in range(24)]
    
    # Create daily activity histogram (7 features, one per day)
    daily_histogram = [normalized_daily.get(d, 0) for d in range(7)]
    
    # Create monthly activity histogram if we have sufficient data
    df["month"] = df["timestamp"].dt.month
    monthly_activity = df["month"].value_counts().sort_index().to_dict()
    total_months = sum(monthly_activity.values())
    normalized_monthly = {m: count/total_months for m, count in monthly_activity.items()}
    monthly_histogram = [normalized_monthly.get(m, 0) for m in range(1, 13)]
    
    # Return combined temporal features
    return {
        "posting_frequency": posting_frequency,
        "regularity": regularity,
        "hourly_histogram": hourly_histogram,
        "daily_histogram": daily_histogram,
        "monthly_histogram": monthly_histogram,
        "first_post_time": first_post,
        "last_post_time": last_post,
        "active_days": date_range
    }


def create_engagement_features(posts: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract engagement features from user posts.
    
    Args:
        posts: List of processed user posts
        config: Behavior feature configuration
        
    Returns:
        Dict: Engagement features including like/comment ratios
    """
    if not posts:
        return {}
    
    # Extract engagement metrics from each post
    likes = []
    comments = []
    shares = []
    retweets = []
    
    # Get activity types from config
    activity_types = config.get("activity_types", ["post", "comment", "like", "share"])
    
    for post in posts:
        engagement = post.get("engagement", {})
        
        if "likes" in activity_types and "likes" in engagement:
            likes.append(engagement["likes"])
        
        if "comments" in activity_types and "comments" in engagement:
            comments.append(engagement["comments"])
        
        if "share" in activity_types and "shares" in engagement:
            shares.append(engagement["shares"])
        
        if "retweet" in activity_types and "retweets" in engagement:
            retweets.append(engagement["retweets"])
    
    # Calculate engagement statistics
    engagement_stats = {}
    
    # Likes statistics
    if likes:
        engagement_stats["likes"] = {
            "total": sum(likes),
            "mean": np.mean(likes),
            "median": np.median(likes),
            "std": np.std(likes),
            "max": max(likes)
        }
    
    # Comments statistics
    if comments:
        engagement_stats["comments"] = {
            "total": sum(comments),
            "mean": np.mean(comments),
            "median": np.median(comments),
            "std": np.std(comments),
            "max": max(comments)
        }
    
    # Shares statistics
    if shares:
        engagement_stats["shares"] = {
            "total": sum(shares),
            "mean": np.mean(shares),
            "median": np.median(shares),
            "std": np.std(shares),
            "max": max(shares)
        }
    
    # Retweets statistics
    if retweets:
        engagement_stats["retweets"] = {
            "total": sum(retweets),
            "mean": np.mean(retweets),
            "median": np.median(retweets),
            "std": np.std(retweets),
            "max": max(retweets)
        }
    
    # Calculate engagement ratios
    engagement_ratios = {}
    
    # Likes per post
    if likes:
        engagement_ratios["likes_per_post"] = sum(likes) / len(posts)
    
    # Comments per post
    if comments:
        engagement_ratios["comments_per_post"] = sum(comments) / len(posts)
    
    # Likes to comments ratio
    if likes and comments and sum(comments) > 0:
        engagement_ratios["likes_to_comments_ratio"] = sum(likes) / sum(comments)
    
    # Create engagement distribution histogram (binned like counts)
    if likes:
        like_bins = [0, 1, 5, 10, 50, 100, float('inf')]
        like_hist, _ = np.histogram(likes, bins=like_bins)
        like_hist = like_hist / sum(like_hist)  # Normalize
        engagement_stats["like_histogram"] = like_hist.tolist()
    
    # Create engagement distribution histogram (binned comment counts)
    if comments:
        comment_bins = [0, 1, 3, 5, 10, 20, float('inf')]
        comment_hist, _ = np.histogram(comments, bins=comment_bins)
        comment_hist = comment_hist / sum(comment_hist)  # Normalize
        engagement_stats["comment_histogram"] = comment_hist.tolist()
    
    return {
        "stats": engagement_stats,
        "ratios": engagement_ratios
    }


def create_content_type_features(posts: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features related to content types and posting behavior.
    
    Args:
        posts: List of processed user posts
        config: Behavior feature configuration
        
    Returns:
        Dict: Content type features
    """
    if not posts:
        return {}
    
    # Count different content types
    content_types = Counter([post.get("media_type", post.get("type", "unknown")) for post in posts])
    total_posts = len(posts)
    
    # Normalize counts
    content_type_distribution = {k: v/total_posts for k, v in content_types.items()}
    
    # Calculate text length statistics
    text_lengths = [len(post.get("text", "")) for post in posts]
    
    text_length_stats = {
        "mean": np.mean(text_lengths),
        "median": np.median(text_lengths),
        "std": np.std(text_lengths),
        "max": max(text_lengths),
        "min": min(text_lengths)
    }
    
    # Calculate word count statistics
    word_counts = [len(post.get("text", "").split()) for post in posts]
    
    word_count_stats = {
        "mean": np.mean(word_counts),
        "median": np.median(word_counts),
        "std": np.std(word_counts),
        "max": max(word_counts),
        "min": min(word_counts)
    }
    
    return {
        "content_type_distribution": content_type_distribution,
        "text_length_stats": text_length_stats,
        "word_count_stats": word_count_stats
    }


def extract_behavioral_features_for_user(
    user_id: str, 
    posts: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract all behavioral features for a single user.
    
    Args:
        user_id: Anonymized user ID
        posts: List of processed user posts
        config: Behavior feature configuration
        
    Returns:
        Dict: All behavioral features for the user
    """
    logger.info(f"Extracting behavioral features for user {user_id}")
    
    # Extract different feature categories
    temporal_features = create_temporal_features(posts, config)
    engagement_features = create_engagement_features(posts, config)
    content_type_features = create_content_type_features(posts, config)
    
    # Combine all features
    all_features = {
        "user_id": user_id,
        "post_count": len(posts),
        "temporal": temporal_features,
        "engagement": engagement_features,
        "content_types": content_type_features
    }
    
    return all_features


def extract_behavioral_features_for_platform(
    platform_data: Dict[str, Any], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract behavioral features for all users on a platform.
    
    Args:
        platform_data: Processed data for a platform
        config: Behavior feature configuration
        
    Returns:
        Dict: Behavioral features for all users on the platform
    """
    users_features = {}
    
    if not platform_data or "posts" not in platform_data:
        return users_features
    
    for user_id, posts in platform_data["posts"].items():
        # Skip users with no posts
        if not posts:
            continue
            
        users_features[user_id] = extract_behavioral_features_for_user(
            user_id, posts, config
        )
    
    return users_features


def extract_all_behavioral_features(
    processed_data: Dict[str, Dict[str, Any]], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Extract behavioral features for all users across all platforms.
    
    Args:
        processed_data: Dictionary of processed data by platform
        config: Behavior feature configuration
        
    Returns:
        Dict: Behavioral features for all users across all platforms
    """
    all_features = {}
    
    for platform, data in processed_data.items():
        logger.info(f"Extracting behavioral features for {platform}")
        all_features[platform] = extract_behavioral_features_for_platform(data, config)
    
    # Create vector representations
    vectorized_features = vectorize_behavioral_features(all_features)
    
    # Save all features and vectors
    os.makedirs("processed_data", exist_ok=True)
    
    with open("processed_data/behavioral_features.pkl", "wb") as f:
        pickle.dump(all_features, f)
    
    with open("processed_data/behavioral_vectors.pkl", "wb") as f:
        pickle.dump(vectorized_features, f)
    
    logger.info(f"Extracted behavioral features for {sum(len(users) for users in all_features.values())} users")
    return all_features


def vectorize_behavioral_features(all_features: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert behavioral features to fixed-length vectors for machine learning.
    
    Args:
        all_features: Behavioral features for all users across all platforms
        
    Returns:
        Dict: Fixed-length feature vectors for all users
    """
    vector_features = {}
    
    for platform, users in all_features.items():
        vector_features[platform] = {}
        
        for user_id, features in users.items():
            # Create a fixed-length vector for each user
            feature_vector = []
            
            # Add temporal features
            temporal = features.get("temporal", {})
            feature_vector.append(temporal.get("posting_frequency", 0))
            
            # Add hourly activity histogram (24 features)
            feature_vector.extend(temporal.get("hourly_histogram", [0] * 24))
            
            # Add daily activity histogram (7 features)
            feature_vector.extend(temporal.get("daily_histogram", [0] * 7))
            
            # Add regularity features
            regularity = temporal.get("regularity", {})
            feature_vector.append(regularity.get("mean_hours_between_posts", 0))
            feature_vector.append(regularity.get("std_hours_between_posts", 0))
            
            # Add engagement features
            engagement = features.get("engagement", {})
            ratios = engagement.get("ratios", {})
            feature_vector.append(ratios.get("likes_per_post", 0))
            feature_vector.append(ratios.get("comments_per_post", 0))
            feature_vector.append(ratios.get("likes_to_comments_ratio", 0))
            
            # Add engagement histograms if available
            stats = engagement.get("stats", {})
            if "like_histogram" in stats:
                feature_vector.extend(stats["like_histogram"])
            else:
                feature_vector.extend([0] * 6)  # Default histogram size
                
            if "comment_histogram" in stats:
                feature_vector.extend(stats["comment_histogram"])
            else:
                feature_vector.extend([0] * 6)  # Default histogram size
            
            # Add content type features
            content_types = features.get("content_types", {})
            text_stats = content_types.get("text_length_stats", {})
            feature_vector.append(text_stats.get("mean", 0))
            feature_vector.append(text_stats.get("std", 0))
            
            word_stats = content_types.get("word_count_stats", {})
            feature_vector.append(word_stats.get("mean", 0))
            feature_vector.append(word_stats.get("std", 0))
            
            # Convert to numpy array
            vector_features[platform][user_id] = np.array(feature_vector, dtype=np.float32)
    
    return vector_features


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    
    # Load processed data
    processed_data = {}
    for platform in ["instagram", "twitter", "facebook"]:
        platform_file = os.path.join(
            config["directories"]["processed_data"], 
            platform, 
            "processed_data.json"
        )
        if os.path.exists(platform_file):
            with open(platform_file, 'r', encoding='utf-8') as f:
                processed_data[platform] = json.load(f)
    
    # Extract behavioral features
    extract_all_behavioral_features(processed_data, config["features"]["behavior"])
