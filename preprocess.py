"""
Preprocessing Module

This module handles cleaning and normalizing raw social media data,
anonymizing user IDs, and preparing data for feature extraction.
"""

import json
import logging
import os
import re
import string
import hashlib
import datetime
from typing import Dict, Any, List, Tuple, Union, Set, Optional

import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from pathlib import Path

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


def anonymize_user_id(user_id: str, platform: str, salt: str = "matching_system") -> str:
    """
    Create a consistent anonymized ID for a user.
    
    Args:
        user_id: Original user ID
        platform: Social platform name
        salt: Salt for hashing
        
    Returns:
        str: Anonymized user ID
    """
    input_string = f"{user_id}:{platform}:{salt}"
    return hashlib.sha256(input_string.encode()).hexdigest()[:16]


def clean_text(text: str, config: Dict[str, bool]) -> str:
    """
    Clean text content based on configuration.
    
    Args:
        text: Text to clean
        config: Configuration for text cleaning
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase if specified
    if config.get("lowercase", True):
        text = text.lower()
    
    # Remove URLs if specified
    if config.get("remove_urls", True):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions if specified
    if config.get("remove_mentions", True):
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags if specified (keep the text after #)
    if config.get("remove_hashtags", False):
        text = re.sub(r'#\w+', '', text)
    else:
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation if specified
    if config.get("remove_punctuation", True):
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Lemmatize if specified
    if config.get("lemmatize", True):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(word) for word in words])
    
    return text


def normalize_timestamp(timestamp_str: str) -> datetime.datetime:
    """
    Normalize timestamps from different formats to datetime objects.
    
    Args:
        timestamp_str: Timestamp string from API
        
    Returns:
        datetime.datetime: Normalized datetime object
    """
    # Try different formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO format with timezone
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO format with milliseconds and timezone
        "%Y-%m-%dT%H:%M:%S+0000",  # Twitter-like format
        "%Y-%m-%d %H:%M:%S",  # Simple format
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, log error and return current time
    logger.error(f"Could not parse timestamp: {timestamp_str}")
    return datetime.datetime.now()


def process_instagram_data(raw_data_dir: str, processed_data_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw Instagram data.
    
    Args:
        raw_data_dir: Directory containing raw data
        processed_data_dir: Directory to save processed data
        config: Preprocessing configuration
        
    Returns:
        Dict: Processed Instagram data
    """
    logger.info("Processing Instagram data")
    instagram_dir = os.path.join(raw_data_dir, "instagram")
    if not os.path.exists(instagram_dir):
        logger.warning("No Instagram data found")
        return {}
    
    users = {}
    posts = {}
    user_networks = {}
    
    # Process profile data
    profile_files = [f for f in os.listdir(instagram_dir) if "profile" in f and f.endswith(".json")]
    for filename in profile_files:
        file_path = os.path.join(instagram_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        orig_user_id = profile_data.get("id")
        if not orig_user_id:
            continue
            
        anon_user_id = anonymize_user_id(orig_user_id, "instagram")
        
        users[anon_user_id] = {
            "platform": "instagram",
            "username": profile_data.get("username", ""),
            "account_type": profile_data.get("account_type", ""),
            "media_count": profile_data.get("media_count", 0),
            "original_id": orig_user_id
        }
    
    # Process post data
    post_files = [f for f in os.listdir(instagram_dir) if "posts" in f and f.endswith(".json")]
    for filename in post_files:
        file_path = os.path.join(instagram_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            post_data = json.load(f)
        
        if "data" not in post_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "instagram")
        
        if anon_user_id not in posts:
            posts[anon_user_id] = []
        
        for post in post_data["data"]:
            post_id = post.get("id", "")
            caption = post.get("caption", "")
            
            processed_post = {
                "id": post_id,
                "text": clean_text(caption, config["text_cleaning"]),
                "original_text": caption,
                "timestamp": normalize_timestamp(post.get("timestamp", "")),
                "engagement": {
                    "likes": post.get("like_count", 0),
                    "comments": post.get("comments_count", 0)
                },
                "media_type": post.get("media_type", "")
            }
            
            posts[anon_user_id].append(processed_post)
    
    # Enforce min/max posts per user
    for user_id, user_posts in list(posts.items()):
        if len(user_posts) < config["min_posts_per_user"]:
            logger.warning(f"User {user_id} has fewer than minimum posts, skipping")
            del posts[user_id]
            continue
            
        if len(user_posts) > config["max_posts_per_user"]:
            logger.info(f"User {user_id} has {len(user_posts)} posts, limiting to {config['max_posts_per_user']}")
            posts[user_id] = sorted(user_posts, key=lambda x: x["timestamp"], reverse=True)[:config["max_posts_per_user"]]
    
    # Save processed data
    processed_data = {
        "users": users,
        "posts": posts,
        "networks": user_networks
    }
    
    os.makedirs(os.path.join(processed_data_dir, "instagram"), exist_ok=True)
    with open(os.path.join(processed_data_dir, "instagram", "processed_data.json"), 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, default=str, ensure_ascii=False, indent=2)
    
    return processed_data


def process_twitter_data(raw_data_dir: str, processed_data_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw Twitter data.
    
    Args:
        raw_data_dir: Directory containing raw data
        processed_data_dir: Directory to save processed data
        config: Preprocessing configuration
        
    Returns:
        Dict: Processed Twitter data
    """
    logger.info("Processing Twitter data")
    twitter_dir = os.path.join(raw_data_dir, "twitter")
    if not os.path.exists(twitter_dir):
        logger.warning("No Twitter data found")
        return {}
    
    users = {}
    posts = {}
    user_networks = {}
    
    # Process profile data
    profile_files = [f for f in os.listdir(twitter_dir) if "profile" in f and f.endswith(".json")]
    for filename in profile_files:
        file_path = os.path.join(twitter_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        if "data" not in profile_data:
            continue
            
        user_data = profile_data["data"]
        orig_user_id = user_data.get("id")
        if not orig_user_id:
            continue
            
        anon_user_id = anonymize_user_id(orig_user_id, "twitter")
        
        users[anon_user_id] = {
            "platform": "twitter",
            "username": user_data.get("username", ""),
            "name": user_data.get("name", ""),
            "created_at": user_data.get("created_at", ""),
            "public_metrics": user_data.get("public_metrics", {}),
            "original_id": orig_user_id
        }
    
    # Process tweet data
    tweet_files = [f for f in os.listdir(twitter_dir) if "tweets" in f and f.endswith(".json")]
    for filename in tweet_files:
        file_path = os.path.join(twitter_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            tweet_data = json.load(f)
        
        if "data" not in tweet_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "twitter")
        
        if anon_user_id not in posts:
            posts[anon_user_id] = []
        
        for tweet in tweet_data["data"]:
            tweet_id = tweet.get("id", "")
            text = tweet.get("text", "")
            
            processed_tweet = {
                "id": tweet_id,
                "text": clean_text(text, config["text_cleaning"]),
                "original_text": text,
                "timestamp": normalize_timestamp(tweet.get("created_at", "")),
                "engagement": {
                    "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                    "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                    "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
                    "quotes": tweet.get("public_metrics", {}).get("quote_count", 0)
                },
                "type": "retweet" if tweet.get("referenced_tweets") else "original"
            }
            
            posts[anon_user_id].append(processed_tweet)
    
    # Process followers data to build network
    followers_files = [f for f in os.listdir(twitter_dir) if "followers" in f and f.endswith(".json")]
    following_files = [f for f in os.listdir(twitter_dir) if "following" in f and f.endswith(".json")]
    
    network_data = {}
    
    # Process followers
    for filename in followers_files:
        file_path = os.path.join(twitter_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            followers_data = json.load(f)
        
        if "data" not in followers_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "twitter")
        
        if anon_user_id not in network_data:
            network_data[anon_user_id] = {"followers": set(), "following": set()}
        
        for follower in followers_data["data"]:
            follower_id = follower.get("id")
            if follower_id:
                anon_follower_id = anonymize_user_id(follower_id, "twitter")
                network_data[anon_user_id]["followers"].add(anon_follower_id)
    
    # Process following
    for filename in following_files:
        file_path = os.path.join(twitter_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            following_data = json.load(f)
        
        if "data" not in following_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "twitter")
        
        if anon_user_id not in network_data:
            network_data[anon_user_id] = {"followers": set(), "following": set()}
        
        for follows in following_data["data"]:
            follows_id = follows.get("id")
            if follows_id:
                anon_follows_id = anonymize_user_id(follows_id, "twitter")
                network_data[anon_user_id]["following"].add(anon_follows_id)
    
    # Build networkx graph
    G = nx.DiGraph()
    
    for user_id, connections in network_data.items():
        G.add_node(user_id, platform="twitter")
        
        # Add max_followers edges to keep graph manageable
        max_followers = config.get("max_followers", 5000)
        
        # Add follower edges
        followers_list = list(connections["followers"])[:max_followers]
        for follower_id in followers_list:
            G.add_node(follower_id, platform="twitter")
            G.add_edge(follower_id, user_id)  # follower -> user
        
        # Add following edges
        following_list = list(connections["following"])[:max_followers]
        for following_id in following_list:
            G.add_node(following_id, platform="twitter")
            G.add_edge(user_id, following_id)  # user -> following
    
    user_networks["graph"] = G
    
    # Enforce min/max posts per user
    for user_id, user_posts in list(posts.items()):
        if len(user_posts) < config["min_posts_per_user"]:
            logger.warning(f"User {user_id} has fewer than minimum posts, skipping")
            del posts[user_id]
            continue
            
        if len(user_posts) > config["max_posts_per_user"]:
            logger.info(f"User {user_id} has {len(user_posts)} posts, limiting to {config['max_posts_per_user']}")
            posts[user_id] = sorted(user_posts, key=lambda x: x["timestamp"], reverse=True)[:config["max_posts_per_user"]]
    
    # Save processed data
    processed_data = {
        "users": users,
        "posts": posts,
        "networks": user_networks
    }
    
    os.makedirs(os.path.join(processed_data_dir, "twitter"), exist_ok=True)
    
    # Save graph as edgelist
    nx.write_edgelist(
        G, 
        os.path.join(processed_data_dir, "twitter", "network.edgelist"), 
        data=False
    )
    
    # Save other data as JSON
    with open(os.path.join(processed_data_dir, "twitter", "processed_data.json"), 'w', encoding='utf-8') as f:
        # Can't serialize graph or sets, so remove them for JSON saving
        serializable_data = {
            "users": users,
            "posts": posts,
            "networks": {
                k: v for k, v in user_networks.items() if k != "graph"
            }
        }
        # Convert sets to lists for JSON serialization
        for user_id, data in network_data.items():
            data["followers"] = list(data["followers"])
            data["following"] = list(data["following"])
        
        serializable_data["networks"]["connections"] = network_data
        
        json.dump(serializable_data, f, default=str, ensure_ascii=False, indent=2)
    
    return processed_data


def process_facebook_data(raw_data_dir: str, processed_data_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw Facebook data.
    
    Args:
        raw_data_dir: Directory containing raw data
        processed_data_dir: Directory to save processed data
        config: Preprocessing configuration
        
    Returns:
        Dict: Processed Facebook data
    """
    logger.info("Processing Facebook data")
    facebook_dir = os.path.join(raw_data_dir, "facebook")
    if not os.path.exists(facebook_dir):
        logger.warning("No Facebook data found")
        return {}
    
    users = {}
    posts = {}
    user_networks = {}
    
    # Process profile data
    profile_files = [f for f in os.listdir(facebook_dir) if "profile" in f and f.endswith(".json")]
    for filename in profile_files:
        file_path = os.path.join(facebook_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        orig_user_id = profile_data.get("id")
        if not orig_user_id:
            continue
            
        anon_user_id = anonymize_user_id(orig_user_id, "facebook")
        
        users[anon_user_id] = {
            "platform": "facebook",
            "name": profile_data.get("name", ""),
            "about": profile_data.get("about", ""),
            "gender": profile_data.get("gender", ""),
            "hometown": profile_data.get("hometown", {}).get("name", "") if profile_data.get("hometown") else "",
            "original_id": orig_user_id
        }
    
    # Process post data
    post_files = [f for f in os.listdir(facebook_dir) if "posts" in f and f.endswith(".json")]
    for filename in post_files:
        file_path = os.path.join(facebook_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            post_data = json.load(f)
        
        if "data" not in post_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "facebook")
        
        if anon_user_id not in posts:
            posts[anon_user_id] = []
        
        for post in post_data["data"]:
            post_id = post.get("id", "")
            message = post.get("message", "")
            
            processed_post = {
                "id": post_id,
                "text": clean_text(message, config["text_cleaning"]),
                "original_text": message,
                "timestamp": normalize_timestamp(post.get("created_time", "")),
                "engagement": {
                    "likes": post.get("reactions", {}).get("summary", {}).get("total_count", 0) if "reactions" in post else 0,
                    "comments": post.get("comments", {}).get("summary", {}).get("total_count", 0) if "comments" in post else 0
                },
                "type": post.get("type", "")
            }
            
            posts[anon_user_id].append(processed_post)
    
    # Process friends data to build network
    friends_files = [f for f in os.listdir(facebook_dir) if "friends" in f and f.endswith(".json")]
    
    # Build network graph
    G = nx.Graph()  # Undirected graph for Facebook (mutual friendship)
    
    for filename in friends_files:
        file_path = os.path.join(facebook_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            friends_data = json.load(f)
        
        if "data" not in friends_data:
            continue
            
        # Extract user ID from filename
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        orig_user_id = parts[0]
        anon_user_id = anonymize_user_id(orig_user_id, "facebook")
        
        G.add_node(anon_user_id, platform="facebook")
        
        # Add max_followers edges to keep graph manageable
        max_friends = config.get("max_followers", 5000)
        friends_list = friends_data["data"][:max_friends]
        
        for friend in friends_list:
            friend_id = friend.get("id")
            if friend_id:
                anon_friend_id = anonymize_user_id(friend_id, "facebook")
                G.add_node(anon_friend_id, platform="facebook")
                G.add_edge(anon_user_id, anon_friend_id)
    
    user_networks["graph"] = G
    
    # Enforce min/max posts per user
    for user_id, user_posts in list(posts.items()):
        if len(user_posts) < config["min_posts_per_user"]:
            logger.warning(f"User {user_id} has fewer than minimum posts, skipping")
            del posts[user_id]
            continue
            
        if len(user_posts) > config["max_posts_per_user"]:
            logger.info(f"User {user_id} has {len(user_posts)} posts, limiting to {config['max_posts_per_user']}")
            posts[user_id] = sorted(user_posts, key=lambda x: x["timestamp"], reverse=True)[:config["max_posts_per_user"]]
    
    # Save processed data
    processed_data = {
        "users": users,
        "posts": posts,
        "networks": user_networks
    }
    
    os.makedirs(os.path.join(processed_data_dir, "facebook"), exist_ok=True)
    
    # Save graph as edgelist
    nx.write_edgelist(
        G, 
        os.path.join(processed_data_dir, "facebook", "network.edgelist"), 
        data=False
    )
    
    # Save other data as JSON
    with open(os.path.join(processed_data_dir, "facebook", "processed_data.json"), 'w', encoding='utf-8') as f:
        # Can't serialize graph
        serializable_data = {
            "users": users,
            "posts": posts,
            "networks": {
                k: v for k, v in user_networks.items() if k != "graph"
            }
        }
        json.dump(serializable_data, f, default=str, ensure_ascii=False, indent=2)
    
    return processed_data


def process_all_platforms(raw_data_dir: str, processed_data_dir: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Process data from all platforms.
    
    Args:
        raw_data_dir: Directory containing raw data
        processed_data_dir: Directory to save processed data
        config: Preprocessing configuration
        
    Returns:
        Dict: Dictionary of processed data for each platform
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    
    processed_data = {}
    
    # Process each platform
    processed_data["instagram"] = process_instagram_data(raw_data_dir, processed_data_dir, config)
    processed_data["twitter"] = process_twitter_data(raw_data_dir, processed_data_dir, config)
    processed_data["facebook"] = process_facebook_data(raw_data_dir, processed_data_dir, config)
    
    # Create combined graph from all platforms if needed
    try:
        combined_graph = nx.DiGraph()
        
        # Add Twitter graph (directed)
        if "twitter" in processed_data and "networks" in processed_data["twitter"] and "graph" in processed_data["twitter"]["networks"]:
            twitter_graph = processed_data["twitter"]["networks"]["graph"]
            combined_graph.add_nodes_from(twitter_graph.nodes(data=True))
            combined_graph.add_edges_from(twitter_graph.edges(data=True))
        
        # Add Facebook graph (undirected -> directed)
        if "facebook" in processed_data and "networks" in processed_data["facebook"] and "graph" in processed_data["facebook"]["networks"]:
            facebook_graph = processed_data["facebook"]["networks"]["graph"]
            combined_graph.add_nodes_from(facebook_graph.nodes(data=True))
            # For each undirected edge, add both directions
            for u, v in facebook_graph.edges():
                combined_graph.add_edge(u, v)
                combined_graph.add_edge(v, u)
        
        # Save combined graph
        nx.write_edgelist(
            combined_graph, 
            os.path.join(processed_data_dir, "combined_network.edgelist"), 
            data=False
        )
        
        logger.info(f"Combined graph has {combined_graph.number_of_nodes()} nodes and {combined_graph.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Failed to create combined graph: {e}")
    
    # Write summary statistics
    platform_stats = {}
    for platform, data in processed_data.items():
        user_count = len(data.get("users", {}))
        post_count = sum(len(posts) for posts in data.get("posts", {}).values())
        
        if "networks" in data and "graph" in data["networks"]:
            network_stats = {
                "nodes": data["networks"]["graph"].number_of_nodes(),
                "edges": data["networks"]["graph"].number_of_edges()
            }
        else:
            network_stats = {"nodes": 0, "edges": 0}
        
        platform_stats[platform] = {
            "users": user_count,
            "posts": post_count,
            "network": network_stats
        }
    
    with open(os.path.join(processed_data_dir, "summary_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(platform_stats, f, ensure_ascii=False, indent=2)
    
    logger.info("Preprocessing completed successfully")
    return processed_data


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    process_all_platforms(
        config["directories"]["raw_data"],
        config["directories"]["processed_data"],
        config["preprocessing"]
    )
