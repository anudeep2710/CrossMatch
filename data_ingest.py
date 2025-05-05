"""
Data Ingestion Module

This module handles fetching data from various social media APIs,
including handling rate limits, retries, and storing the raw data.
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import utils


logger = logging.getLogger(__name__)


def create_session_with_retries() -> requests.Session:
    """
    Create a requests session with retry capability.
    
    Returns:
        requests.Session: Session configured with retries
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session


def handle_rate_limit(response: requests.Response, platform: str, rate_limit: int) -> None:
    """
    Handle rate limiting for API requests.
    
    Args:
        response: The API response
        platform: Social media platform name
        rate_limit: Rate limit configuration
    """
    remaining = int(response.headers.get('X-Rate-Limit-Remaining', 1))
    reset_time = int(response.headers.get('X-Rate-Limit-Reset', 0))
    
    if remaining < 1:
        wait_time = max(reset_time - time.time(), 0)
        logger.warning(f"{platform} rate limit reached. Waiting {wait_time:.2f} seconds.")
        time.sleep(wait_time + 1)  # Add 1 second as a buffer


def save_raw_data(data: Dict[str, Any], platform: str, data_type: str, 
                 user_id: str, output_dir: str) -> str:
    """
    Save raw API response data to file.
    
    Args:
        data: API response data
        platform: Social media platform name
        data_type: Type of data (posts, followers, etc.)
        user_id: User identifier
        output_dir: Directory to save data
        
    Returns:
        str: Path to saved file
    """
    # Create platform-specific directory if it doesn't exist
    platform_dir = os.path.join(output_dir, platform)
    os.makedirs(platform_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = int(time.time())
    filename = f"{user_id}_{data_type}_{timestamp}.json"
    filepath = os.path.join(platform_dir, filename)
    
    # Save data to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {platform} {data_type} data for user {user_id} to {filepath}")
    return filepath


def fetch_instagram_data(config: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Fetch data from Instagram Graph API.
    
    Args:
        config: Instagram API configuration
        output_dir: Directory to save raw data
        
    Returns:
        List[str]: List of file paths where data is saved
    """
    logger.info("Fetching Instagram data")
    session = create_session_with_retries()
    saved_files = []
    
    base_url = "https://graph.instagram.com/v17.0"
    headers = {
        "Authorization": f"Bearer {config['access_token']}",
    }
    
    # Get user profile
    try:
        response = session.get(
            f"{base_url}/me",
            headers=headers,
            params={"fields": "id,username,account_type,media_count"}
        )
        response.raise_for_status()
        profile_data = response.json()
        user_id = profile_data["id"]
        
        # Save profile data
        saved_files.append(save_raw_data(
            profile_data, "instagram", "profile", user_id, output_dir
        ))
        
        # Get user media (posts)
        response = session.get(
            f"{base_url}/me/media",
            headers=headers,
            params={
                "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,comments_count,like_count",
                "limit": 50
            }
        )
        response.raise_for_status()
        media_data = response.json()
        
        # Save media data
        saved_files.append(save_raw_data(
            media_data, "instagram", "posts", user_id, output_dir
        ))
        
        # Paginate through all media
        while "paging" in media_data and "next" in media_data["paging"]:
            handle_rate_limit(response, "instagram", config["rate_limit"])
            response = session.get(media_data["paging"]["next"])
            response.raise_for_status()
            media_data = response.json()
            saved_files.append(save_raw_data(
                media_data, "instagram", "posts", user_id, output_dir
            ))
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Instagram data: {e}")
    
    return saved_files


def fetch_twitter_data(config: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Fetch data from Twitter API v2.
    
    Args:
        config: Twitter API configuration
        output_dir: Directory to save raw data
        
    Returns:
        List[str]: List of file paths where data is saved
    """
    logger.info("Fetching Twitter data")
    session = create_session_with_retries()
    saved_files = []
    
    base_url = "https://api.twitter.com/2"
    headers = {
        "Authorization": f"Bearer {config['bearer_token']}",
    }
    
    try:
        # Get user info
        response = session.get(
            f"{base_url}/users/me",
            headers=headers,
            params={"user.fields": "id,name,username,public_metrics,created_at,description"}
        )
        response.raise_for_status()
        user_data = response.json()
        user_id = user_data["data"]["id"]
        
        # Save user data
        saved_files.append(save_raw_data(
            user_data, "twitter", "profile", user_id, output_dir
        ))
        
        # Get user tweets
        response = session.get(
            f"{base_url}/users/{user_id}/tweets",
            headers=headers,
            params={
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,context_annotations,entities",
                "expansions": "author_id,referenced_tweets.id"
            }
        )
        response.raise_for_status()
        tweets_data = response.json()
        
        # Save tweets data
        saved_files.append(save_raw_data(
            tweets_data, "twitter", "tweets", user_id, output_dir
        ))
        
        # Paginate through all tweets
        while "meta" in tweets_data and "next_token" in tweets_data["meta"]:
            handle_rate_limit(response, "twitter", config["rate_limit"])
            response = session.get(
                f"{base_url}/users/{user_id}/tweets",
                headers=headers,
                params={
                    "max_results": 100,
                    "tweet.fields": "created_at,public_metrics,context_annotations,entities",
                    "expansions": "author_id,referenced_tweets.id",
                    "pagination_token": tweets_data["meta"]["next_token"]
                }
            )
            response.raise_for_status()
            tweets_data = response.json()
            saved_files.append(save_raw_data(
                tweets_data, "twitter", "tweets", user_id, output_dir
            ))
        
        # Get user followers
        response = session.get(
            f"{base_url}/users/{user_id}/followers",
            headers=headers,
            params={
                "max_results": 100,
                "user.fields": "id,username"
            }
        )
        response.raise_for_status()
        followers_data = response.json()
        
        # Save followers data
        saved_files.append(save_raw_data(
            followers_data, "twitter", "followers", user_id, output_dir
        ))
        
        # Paginate through all followers
        while "meta" in followers_data and "next_token" in followers_data["meta"]:
            handle_rate_limit(response, "twitter", config["rate_limit"])
            response = session.get(
                f"{base_url}/users/{user_id}/followers",
                headers=headers,
                params={
                    "max_results": 100,
                    "user.fields": "id,username",
                    "pagination_token": followers_data["meta"]["next_token"]
                }
            )
            response.raise_for_status()
            followers_data = response.json()
            saved_files.append(save_raw_data(
                followers_data, "twitter", "followers", user_id, output_dir
            ))
        
        # Get user following
        response = session.get(
            f"{base_url}/users/{user_id}/following",
            headers=headers,
            params={
                "max_results": 100,
                "user.fields": "id,username"
            }
        )
        response.raise_for_status()
        following_data = response.json()
        
        # Save following data
        saved_files.append(save_raw_data(
            following_data, "twitter", "following", user_id, output_dir
        ))
        
        # Paginate through all following
        while "meta" in following_data and "next_token" in following_data["meta"]:
            handle_rate_limit(response, "twitter", config["rate_limit"])
            response = session.get(
                f"{base_url}/users/{user_id}/following",
                headers=headers,
                params={
                    "max_results": 100,
                    "user.fields": "id,username",
                    "pagination_token": following_data["meta"]["next_token"]
                }
            )
            response.raise_for_status()
            following_data = response.json()
            saved_files.append(save_raw_data(
                following_data, "twitter", "following", user_id, output_dir
            ))
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Twitter data: {e}")
    
    return saved_files


def fetch_facebook_data(config: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Fetch data from Facebook Graph API.
    
    Args:
        config: Facebook API configuration
        output_dir: Directory to save raw data
        
    Returns:
        List[str]: List of file paths where data is saved
    """
    logger.info("Fetching Facebook data")
    session = create_session_with_retries()
    saved_files = []
    
    base_url = "https://graph.facebook.com/v17.0"
    
    try:
        # Get user profile
        response = session.get(
            f"{base_url}/me",
            params={
                "fields": "id,name,about,birthday,education,gender,hometown,languages,link",
                "access_token": config["access_token"]
            }
        )
        response.raise_for_status()
        profile_data = response.json()
        user_id = profile_data["id"]
        
        # Save profile data
        saved_files.append(save_raw_data(
            profile_data, "facebook", "profile", user_id, output_dir
        ))
        
        # Get user posts
        response = session.get(
            f"{base_url}/me/posts",
            params={
                "fields": "id,message,created_time,type,permalink_url,reactions.summary(total_count),comments.summary(total_count)",
                "limit": 50,
                "access_token": config["access_token"]
            }
        )
        response.raise_for_status()
        posts_data = response.json()
        
        # Save posts data
        saved_files.append(save_raw_data(
            posts_data, "facebook", "posts", user_id, output_dir
        ))
        
        # Paginate through all posts
        while "paging" in posts_data and "next" in posts_data["paging"]:
            handle_rate_limit(response, "facebook", config["rate_limit"])
            response = session.get(posts_data["paging"]["next"])
            response.raise_for_status()
            posts_data = response.json()
            saved_files.append(save_raw_data(
                posts_data, "facebook", "posts", user_id, output_dir
            ))
        
        # Get user friends
        response = session.get(
            f"{base_url}/me/friends",
            params={
                "fields": "id,name",
                "limit": 50,
                "access_token": config["access_token"]
            }
        )
        response.raise_for_status()
        friends_data = response.json()
        
        # Save friends data
        saved_files.append(save_raw_data(
            friends_data, "facebook", "friends", user_id, output_dir
        ))
        
        # Paginate through all friends
        while "paging" in friends_data and "next" in friends_data["paging"]:
            handle_rate_limit(response, "facebook", config["rate_limit"])
            response = session.get(friends_data["paging"]["next"])
            response.raise_for_status()
            friends_data = response.json()
            saved_files.append(save_raw_data(
                friends_data, "facebook", "friends", user_id, output_dir
            ))
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Facebook data: {e}")
    
    return saved_files


def fetch_platform_data(platform: str, config: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Fetch data from a specified platform.
    
    Args:
        platform: Social media platform to fetch data from
        config: Platform-specific API configuration
        output_dir: Directory to save raw data
        
    Returns:
        List[str]: List of file paths where data is saved
    """
    if platform == "instagram":
        return fetch_instagram_data(config, output_dir)
    elif platform == "twitter":
        return fetch_twitter_data(config, output_dir)
    elif platform == "facebook":
        return fetch_facebook_data(config, output_dir)
    else:
        logger.error(f"Unsupported platform: {platform}")
        return []


def list_available_data(raw_data_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    List all available data files by platform and type.
    
    Args:
        raw_data_dir: Directory containing raw data
        
    Returns:
        Dict: Nested dictionary with platform, data types, and file paths
    """
    data_files = {}
    
    for platform in ["instagram", "twitter", "facebook"]:
        platform_dir = os.path.join(raw_data_dir, platform)
        if not os.path.exists(platform_dir):
            continue
        
        data_files[platform] = {}
        for filename in os.listdir(platform_dir):
            if not filename.endswith('.json'):
                continue
                
            # Parse filename to get data type
            parts = filename.split('_')
            if len(parts) >= 2:
                user_id = parts[0]
                data_type = parts[1]
                
                if data_type not in data_files[platform]:
                    data_files[platform][data_type] = []
                
                data_files[platform][data_type].append(os.path.join(platform_dir, filename))
    
    return data_files


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    for platform in ["instagram", "twitter", "facebook"]:
        fetch_platform_data(platform, config["api"][platform], config["directories"]["raw_data"])
