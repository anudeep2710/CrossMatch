"""
Feature Fusion Module

This module fuses different feature types (behavioral, content, network)
and applies dimensionality reduction techniques for user matching.
"""

import json
import logging
import os
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import pandas as pd

from behavior_features import vectorize_behavioral_features
from content_features import get_content_vectors
from network_features import vectorize_network_features

logger = logging.getLogger(__name__)


def load_feature_vectors(processed_data_dir: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], ...]:
    """
    Load feature vectors from pickle files.
    
    Args:
        processed_data_dir: Directory containing processed data
        
    Returns:
        Tuple: (behavior_vectors, content_vectors, network_vectors)
    """
    behavior_vectors = {}
    content_vectors = {}
    network_vectors = {}
    
    # Load behavioral features
    try:
        with open(os.path.join(processed_data_dir, "behavioral_features.pkl"), "rb") as f:
            behavior_features = pickle.load(f)
        behavior_vectors = vectorize_behavioral_features(behavior_features)
        logger.info("Loaded behavioral feature vectors")
    except Exception as e:
        logger.error(f"Error loading behavioral features: {e}")
    
    # Load content features
    try:
        with open(os.path.join(processed_data_dir, "content_features.pkl"), "rb") as f:
            content_features = pickle.load(f)
        content_vectors = get_content_vectors(content_features)
        logger.info("Loaded content feature vectors")
    except Exception as e:
        logger.error(f"Error loading content features: {e}")
    
    # Load network features
    try:
        with open(os.path.join(processed_data_dir, "network_features.pkl"), "rb") as f:
            network_features = pickle.load(f)
        network_vectors = vectorize_network_features(network_features)
        logger.info("Loaded network feature vectors")
    except Exception as e:
        logger.error(f"Error loading network features: {e}")
    
    return behavior_vectors, content_vectors, network_vectors


def concatenate_features(
    behavior_vectors: Dict[str, Dict[str, np.ndarray]],
    content_vectors: Dict[str, Dict[str, np.ndarray]],
    network_vectors: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Concatenate different feature types for each user.
    
    Args:
        behavior_vectors: Behavioral feature vectors by platform and user
        content_vectors: Content feature vectors by platform and user
        network_vectors: Network feature vectors by platform and user
        
    Returns:
        Dict: Concatenated feature vectors by platform and user
    """
    concatenated_vectors = {}
    
    # Get all platforms
    all_platforms = set(behavior_vectors.keys()) | set(content_vectors.keys()) | set(network_vectors.keys())
    
    for platform in all_platforms:
        concatenated_vectors[platform] = {}
        
        # Get all users for this platform
        behavior_users = set(behavior_vectors.get(platform, {}).keys())
        content_users = set(content_vectors.get(platform, {}).keys())
        network_users = set(network_vectors.get(platform, {}).keys())
        
        all_users = behavior_users | content_users | network_users
        
        for user_id in all_users:
            # Get feature vectors for this user (or empty if missing)
            behavior_vector = behavior_vectors.get(platform, {}).get(user_id, np.array([]))
            content_vector = content_vectors.get(platform, {}).get(user_id, np.array([]))
            network_vector = network_vectors.get(platform, {}).get(user_id, np.array([]))
            
            # Skip users with no features
            if len(behavior_vector) == 0 and len(content_vector) == 0 and len(network_vector) == 0:
                continue
            
            # Replace missing vectors with zeros
            if len(behavior_vector) == 0 and any(platform in behavior_vectors for platform in all_platforms):
                # Find a user with behavior features to get vector length
                for p in behavior_vectors:
                    if behavior_vectors[p]:
                        sample_user = next(iter(behavior_vectors[p]))
                        behavior_vector = np.zeros_like(behavior_vectors[p][sample_user])
                        break
                else:
                    behavior_vector = np.array([])
            
            if len(content_vector) == 0 and any(platform in content_vectors for platform in all_platforms):
                # Find a user with content features to get vector length
                for p in content_vectors:
                    if content_vectors[p]:
                        sample_user = next(iter(content_vectors[p]))
                        content_vector = np.zeros_like(content_vectors[p][sample_user])
                        break
                else:
                    content_vector = np.array([])
            
            if len(network_vector) == 0 and any(platform in network_vectors for platform in all_platforms):
                # Find a user with network features to get vector length
                for p in network_vectors:
                    if network_vectors[p]:
                        sample_user = next(iter(network_vectors[p]))
                        network_vector = np.zeros_like(network_vectors[p][sample_user])
                        break
                else:
                    network_vector = np.array([])
            
            # Concatenate vectors
            vectors_to_concatenate = []
            
            if len(behavior_vector) > 0:
                vectors_to_concatenate.append(behavior_vector)
            
            if len(content_vector) > 0:
                vectors_to_concatenate.append(content_vector)
            
            if len(network_vector) > 0:
                vectors_to_concatenate.append(network_vector)
            
            if vectors_to_concatenate:
                concatenated_vector = np.concatenate(vectors_to_concatenate)
                concatenated_vectors[platform][user_id] = concatenated_vector
    
    # Log feature dimensions
    for platform, users in concatenated_vectors.items():
        if users:
            sample_user = next(iter(users))
            logger.info(f"{platform} feature vector dimension: {users[sample_user].shape}")
    
    return concatenated_vectors


def apply_dimensionality_reduction(
    concatenated_vectors: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply dimensionality reduction to feature vectors.
    
    Args:
        concatenated_vectors: Concatenated feature vectors by platform and user
        config: Feature fusion configuration
        
    Returns:
        Dict: Reduced feature vectors by platform and user
    """
    reduced_vectors = {}
    
    # Choose dimensionality reduction method
    method = config.get("dimensionality_reduction", "pca")
    
    # Get target dimension
    target_dim = config.get("embedding_dim", 128)
    
    # Combine all vectors for fitting the transformer
    all_vectors = []
    vector_mapping = []  # [(platform, user_id), ...] to map back
    
    for platform, users in concatenated_vectors.items():
        for user_id, vector in users.items():
            all_vectors.append(vector)
            vector_mapping.append((platform, user_id))
    
    if not all_vectors:
        logger.warning("No feature vectors available for dimensionality reduction")
        return reduced_vectors
    
    # Convert to numpy array
    all_vectors_array = np.vstack(all_vectors)
    
    # Scale data
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(all_vectors_array)
    
    # Apply dimensionality reduction
    if method == "pca":
        # Determine components based on explained variance
        if "explained_variance" in config:
            pca = PCA(n_components=config["explained_variance"], svd_solver='full')
            reduced = pca.fit_transform(scaled_vectors)
            logger.info(f"PCA reduced dimensions from {all_vectors_array.shape[1]} to {reduced.shape[1]} "
                       f"features to explain {config['explained_variance'] * 100:.1f}% of variance")
        else:
            # Use fixed number of components
            pca = PCA(n_components=min(target_dim, scaled_vectors.shape[1]), svd_solver='full')
            reduced = pca.fit_transform(scaled_vectors)
            logger.info(f"PCA reduced dimensions from {all_vectors_array.shape[1]} to {reduced.shape[1]} features")
    
    elif method == "umap":
        # Use UMAP for non-linear dimensionality reduction
        reducer = umap.UMAP(
            n_components=target_dim,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        reduced = reducer.fit_transform(scaled_vectors)
        logger.info(f"UMAP reduced dimensions from {all_vectors_array.shape[1]} to {reduced.shape[1]} features")
    
    else:
        logger.error(f"Unsupported dimensionality reduction method: {method}")
        # Fall back to identity transformation
        reduced = scaled_vectors
        logger.info(f"Using scaled features without dimensionality reduction: {reduced.shape[1]} features")
    
    # Map reduced vectors back to platform/user structure
    for i, (platform, user_id) in enumerate(vector_mapping):
        if platform not in reduced_vectors:
            reduced_vectors[platform] = {}
        
        reduced_vectors[platform][user_id] = reduced[i]
    
    return reduced_vectors


def create_cross_platform_features(
    reduced_vectors: Dict[str, Dict[str, np.ndarray]],
    processed_data_dir: str
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Create features for cross-platform matching.
    
    Args:
        reduced_vectors: Reduced feature vectors by platform and user
        processed_data_dir: Directory to save processed data
        
    Returns:
        Dict: Cross-platform features
    """
    cross_platform_features = {}
    
    # Get all platforms
    platforms = list(reduced_vectors.keys())
    
    # Create cross-platform feature pairs
    for i, platform1 in enumerate(platforms):
        for platform2 in platforms[i+1:]:
            key = f"{platform1}_{platform2}"
            cross_platform_features[key] = {
                platform1: reduced_vectors[platform1],
                platform2: reduced_vectors[platform2]
            }
    
    # Save cross-platform features
    with open(os.path.join(processed_data_dir, "cross_platform_features.pkl"), "wb") as f:
        pickle.dump(cross_platform_features, f)
    
    logger.info(f"Created cross-platform features for {len(cross_platform_features)} platform pairs")
    return cross_platform_features


def fuse_features(
    behavior_data: Dict[str, Dict[str, Dict[str, Any]]],
    content_data: Dict[str, Dict[str, Dict[str, Any]]],
    network_data: Dict[str, Dict[str, Dict[str, Any]]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Fuse features from different sources for user matching.
    
    Args:
        behavior_data: Behavioral features for all users across all platforms
        content_data: Content features for all users across all platforms
        network_data: Network features for all users across all platforms
        config: Feature fusion configuration
        
    Returns:
        Dict: Fused features for cross-platform matching
    """
    processed_data_dir = "processed_data"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Vectorize features if needed, or load pre-vectorized features
    if isinstance(behavior_data, dict) and all(isinstance(p, dict) for p in behavior_data.values()):
        behavior_vectors = vectorize_behavioral_features(behavior_data)
    else:
        behavior_vectors = behavior_data
    
    if isinstance(content_data, dict) and all(isinstance(p, dict) for p in content_data.values()):
        content_vectors = get_content_vectors(content_data)
    else:
        content_vectors = content_data
    
    if isinstance(network_data, dict) and all(isinstance(p, dict) for p in network_data.values()):
        network_vectors = vectorize_network_features(network_data)
    else:
        network_vectors = network_data
    
    # Concatenate features
    logger.info("Concatenating feature vectors")
    concatenated_vectors = concatenate_features(behavior_vectors, content_vectors, network_vectors)
    
    # Apply dimensionality reduction
    logger.info("Applying dimensionality reduction")
    reduced_vectors = apply_dimensionality_reduction(concatenated_vectors, config)
    
    # Save reduced vectors
    with open(os.path.join(processed_data_dir, "reduced_vectors.pkl"), "wb") as f:
        pickle.dump(reduced_vectors, f)
    
    # Create cross-platform features
    cross_platform_features = create_cross_platform_features(reduced_vectors, processed_data_dir)
    
    return cross_platform_features


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    
    processed_data_dir = config["directories"]["processed_data"]
    
    # Load feature vectors
    behavior_vectors, content_vectors, network_vectors = load_feature_vectors(processed_data_dir)
    
    # Fuse features
    fused_features = fuse_features(
        behavior_vectors, 
        content_vectors, 
        network_vectors,
        config["fusion"]
    )
