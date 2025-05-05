"""
Content Features Module

This module extracts content features from user posts,
including text embeddings and topic distributions.
"""

import json
import logging
import os
import numpy as np
import torch
from typing import Dict, Any, List, Union, Tuple, Optional
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load pre-trained SentenceTransformer model.
    
    Args:
        model_name: Name of pretrained model
        
    Returns:
        SentenceTransformer: Loaded model
    """
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise


def generate_text_embeddings(
    texts: List[str], 
    model: SentenceTransformer,
    max_tokens: int = 512
) -> np.ndarray:
    """
    Generate sentence embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: SentenceTransformer model
        max_tokens: Maximum number of tokens to process
        
    Returns:
        np.ndarray: Embeddings for each text
    """
    # Filter out empty texts
    filtered_texts = [text for text in texts if text and isinstance(text, str)]
    
    if not filtered_texts:
        logger.warning("No valid texts to embed")
        return np.array([])
    
    try:
        # Truncate texts if they're too long
        truncated_texts = [text[:max_tokens] for text in filtered_texts]
        
        # Generate embeddings
        embeddings = model.encode(truncated_texts, show_progress_bar=False)
        return embeddings
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.array([])


def train_topic_model(
    texts: List[str], 
    n_topics: int = 20
) -> Tuple[LatentDirichletAllocation, CountVectorizer]:
    """
    Train an LDA topic model on a corpus of texts.
    
    Args:
        texts: List of text strings
        n_topics: Number of topics for LDA
        
    Returns:
        Tuple: (LDA model, Count vectorizer)
    """
    # Filter out empty texts
    filtered_texts = [text for text in texts if text and isinstance(text, str)]
    
    if not filtered_texts:
        logger.warning("No valid texts for topic modeling")
        raise ValueError("No valid texts for topic modeling")
    
    try:
        # Create bag of words representation
        vectorizer = CountVectorizer(
            max_df=0.95,         # Ignore words that appear in >95% of documents
            min_df=2,            # Ignore words that appear in fewer than 2 documents
            stop_words='english',
            max_features=10000    # Limit vocabulary size
        )
        bow = vectorizer.fit_transform(filtered_texts)
        
        # Train LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        lda.fit(bow)
        
        return lda, vectorizer
    
    except Exception as e:
        logger.error(f"Error training topic model: {e}")
        raise


def get_topic_distribution(
    text: str, 
    lda_model: LatentDirichletAllocation, 
    vectorizer: CountVectorizer
) -> np.ndarray:
    """
    Get topic distribution for a text document.
    
    Args:
        text: Input text
        lda_model: Trained LDA model
        vectorizer: Count vectorizer
        
    Returns:
        np.ndarray: Topic distribution
    """
    if not text or not isinstance(text, str):
        return np.zeros(lda_model.n_components)
    
    try:
        # Transform text to bag of words
        bow = vectorizer.transform([text])
        
        # Get topic distribution
        topic_dist = lda_model.transform(bow)[0]
        return topic_dist
    
    except Exception as e:
        logger.error(f"Error getting topic distribution: {e}")
        return np.zeros(lda_model.n_components)


def extract_content_features_for_user(
    user_id: str, 
    posts: List[Dict[str, Any]], 
    embedding_model: SentenceTransformer,
    lda_model: Optional[LatentDirichletAllocation] = None,
    vectorizer: Optional[CountVectorizer] = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extract content features for a single user.
    
    Args:
        user_id: Anonymized user ID
        posts: List of processed user posts
        embedding_model: SentenceTransformer model
        lda_model: Optional LDA model
        vectorizer: Optional Count vectorizer
        config: Content feature configuration
        
    Returns:
        Dict: Content features for the user
    """
    logger.info(f"Extracting content features for user {user_id}")
    
    # Get max tokens from config
    max_tokens = config.get("max_tokens", 512) if config else 512
    
    # Extract text from posts
    texts = [post.get("text", "") for post in posts]
    texts = [text for text in texts if text]  # Filter empty texts
    
    if not texts:
        logger.warning(f"No text content for user {user_id}")
        return {
            "user_id": user_id,
            "post_count": 0,
            "has_content": False
        }
    
    # Generate embeddings for each post
    post_embeddings = generate_text_embeddings(texts, embedding_model, max_tokens)
    
    # Calculate average embedding across all posts
    if len(post_embeddings) > 0:
        avg_embedding = np.mean(post_embeddings, axis=0)
    else:
        avg_embedding = np.array([])
    
    # Get topic distributions if LDA model is provided
    topic_distributions = []
    
    if lda_model is not None and vectorizer is not None:
        for text in texts:
            topic_dist = get_topic_distribution(text, lda_model, vectorizer)
            topic_distributions.append(topic_dist)
        
        # Calculate average topic distribution
        avg_topic_dist = np.mean(topic_distributions, axis=0) if topic_distributions else np.array([])
    else:
        avg_topic_dist = np.array([])
    
    features = {
        "user_id": user_id,
        "post_count": len(texts),
        "has_content": True,
        "avg_embedding": avg_embedding,
        "avg_topic_distribution": avg_topic_dist,
        "post_embeddings": post_embeddings,
        "topic_distributions": topic_distributions
    }
    
    return features


def extract_content_features_for_platform(
    platform_data: Dict[str, Any], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract content features for all users on a platform.
    
    Args:
        platform_data: Processed data for a platform
        config: Content feature configuration
        
    Returns:
        Dict: Content features for all users on the platform
    """
    users_features = {}
    
    if not platform_data or "posts" not in platform_data:
        return users_features
    
    # Load embedding model
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = load_embedding_model(model_name)
    
    # Collect all texts for topic modeling
    all_texts = []
    for user_posts in platform_data["posts"].values():
        for post in user_posts:
            text = post.get("text", "")
            if text:
                all_texts.append(text)
    
    # Train topic model if there are enough texts
    lda_model = None
    vectorizer = None
    
    if len(all_texts) >= 100:  # Arbitrary threshold for minimum corpus size
        try:
            n_topics = config.get("lda_topics", 20)
            lda_model, vectorizer = train_topic_model(all_texts, n_topics)
        except Exception as e:
            logger.error(f"Failed to train topic model: {e}")
    
    # Extract features for each user
    for user_id, posts in platform_data["posts"].items():
        # Skip users with no posts
        if not posts:
            continue
            
        users_features[user_id] = extract_content_features_for_user(
            user_id, posts, embedding_model, lda_model, vectorizer, config
        )
    
    return users_features


def extract_all_content_features(
    processed_data: Dict[str, Dict[str, Any]], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Extract content features for all users across all platforms.
    
    Args:
        processed_data: Dictionary of processed data by platform
        config: Content feature configuration
        
    Returns:
        Dict: Content features for all users across all platforms
    """
    all_features = {}
    
    for platform, data in processed_data.items():
        logger.info(f"Extracting content features for {platform}")
        all_features[platform] = extract_content_features_for_platform(data, config)
    
    # Save all features
    os.makedirs("processed_data", exist_ok=True)
    
    with open("processed_data/content_features.pkl", "wb") as f:
        # Save only the average embeddings and topic distributions for efficiency
        serializable_features = {}
        
        for platform, users in all_features.items():
            serializable_features[platform] = {}
            
            for user_id, features in users.items():
                serializable_features[platform][user_id] = {
                    "user_id": features["user_id"],
                    "post_count": features["post_count"],
                    "has_content": features["has_content"],
                    "avg_embedding": features.get("avg_embedding", np.array([])),
                    "avg_topic_distribution": features.get("avg_topic_distribution", np.array([]))
                }
        
        pickle.dump(serializable_features, f)
    
    logger.info(f"Extracted content features for {sum(len(users) for users in all_features.values())} users")
    return all_features


def get_content_vectors(all_features: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create fixed-length content feature vectors for machine learning.
    
    Args:
        all_features: Content features for all users across all platforms
        
    Returns:
        Dict: Content feature vectors for all users
    """
    vector_features = {}
    
    for platform, users in all_features.items():
        vector_features[platform] = {}
        
        for user_id, features in users.items():
            # Use average embedding as feature vector
            if "avg_embedding" in features and features["avg_embedding"] is not None and len(features["avg_embedding"]) > 0:
                embedding = features["avg_embedding"]
            else:
                # If no embedding, use zeros
                # Determine expected embedding size from config or use default
                embedding = np.zeros(384)  # Default size for all-MiniLM-L6-v2
            
            # Append topic distribution if available
            if "avg_topic_distribution" in features and features["avg_topic_distribution"] is not None and len(features["avg_topic_distribution"]) > 0:
                topic_dist = features["avg_topic_distribution"]
                combined = np.concatenate([embedding, topic_dist])
            else:
                combined = embedding
            
            vector_features[platform][user_id] = combined
    
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
    
    # Extract content features
    extract_all_content_features(processed_data, config["features"]["content"])
