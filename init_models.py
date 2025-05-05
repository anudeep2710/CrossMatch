"""
Model Initialization Script

This script initializes the machine learning models for user matching.
It trains models on sample data if no models exist.
"""

import os
import logging
import pickle
import json
import numpy as np
import torch
from typing import Dict, Any

import utils
import models
import feature_fusion
from matching_service import MatchingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_features(config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Create sample features for model training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict: Sample features for cross-platform matching
    """
    logger.info("Creating sample features for model training")
    
    # Create sample features for each platform pair
    cross_platform_features = {}
    
    # Define platform pairs
    platform_pairs = ["instagram_twitter", "instagram_facebook", "twitter_facebook"]
    
    for platform_pair in platform_pairs:
        platform1, platform2 = platform_pair.split("_")
        
        # Create sample feature vectors
        features = {
            platform1: {},
            platform2: {}
        }
        
        # Create 10 users for each platform
        for i in range(10):
            # Create random feature vector for platform1
            features[platform1][f"user{i}_{platform1}"] = np.random.rand(100)
            
            # Create random feature vector for platform2
            features[platform2][f"user{i}_{platform2}"] = np.random.rand(100)
        
        cross_platform_features[platform_pair] = features
    
    return cross_platform_features


def create_sample_ground_truth(cross_platform_features: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, Dict[str, str]]:
    """
    Create sample ground truth mappings.
    
    Args:
        cross_platform_features: Sample features
        
    Returns:
        Dict: Sample ground truth mappings
    """
    logger.info("Creating sample ground truth mappings")
    
    ground_truth = {}
    
    for platform_pair, features in cross_platform_features.items():
        platform1, platform2 = platform_pair.split("_")
        
        ground_truth[platform_pair] = {}
        
        # Create mappings for the first 5 users (50% of users have matches)
        for i in range(5):
            user1 = f"user{i}_{platform1}"
            user2 = f"user{i}_{platform2}"
            
            ground_truth[platform_pair][user1] = user2
    
    return ground_truth


def save_sample_data(cross_platform_features: Dict[str, Dict[str, Dict[str, np.ndarray]]], ground_truth: Dict[str, Dict[str, str]], config: Dict[str, Any]) -> None:
    """
    Save sample data to disk.
    
    Args:
        cross_platform_features: Sample features
        ground_truth: Sample ground truth mappings
        config: Configuration dictionary
    """
    logger.info("Saving sample data")
    
    processed_data_dir = config["directories"]["processed_data"]
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Save cross-platform features
    with open(os.path.join(processed_data_dir, "cross_platform_features.pkl"), "wb") as f:
        pickle.dump(cross_platform_features, f)
    
    # Save ground truth mappings
    with open(os.path.join(processed_data_dir, "ground_truth_mappings.json"), "w") as f:
        json.dump(ground_truth, f)
    
    # Save individual platform features
    for platform_pair, features in cross_platform_features.items():
        platform1, platform2 = platform_pair.split("_")
        
        # Create platform directories
        os.makedirs(os.path.join(processed_data_dir, platform1), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, platform2), exist_ok=True)
        
        # Save platform1 features
        with open(os.path.join(processed_data_dir, f"{platform1}_features.pkl"), "wb") as f:
            pickle.dump(features[platform1], f)
        
        # Save platform2 features
        with open(os.path.join(processed_data_dir, f"{platform2}_features.pkl"), "wb") as f:
            pickle.dump(features[platform2], f)
        
        # Create sample processed data
        for platform in [platform1, platform2]:
            processed_data = {
                "posts": {
                    user_id: [
                        {
                            "id": f"post{j}",
                            "text": f"Sample post {j} by {user_id}",
                            "timestamp": "2023-01-01T00:00:00Z",
                            "likes": 10,
                            "comments": 5
                        }
                        for j in range(5)
                    ]
                    for user_id in features[platform].keys()
                }
            }
            
            # Save processed data
            with open(os.path.join(processed_data_dir, platform, "processed_data.json"), "w") as f:
                json.dump(processed_data, f)


def init_models():
    """Initialize machine learning models."""
    logger.info("Initializing machine learning models")
    
    # Load configuration
    config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    config = utils.load_config(config_file)
    
    # Create directories
    models_dir = config["directories"]["models"]
    processed_data_dir = config["directories"]["processed_data"]
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Check if models already exist
    model_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    has_models = any(f.endswith("_rf.pkl") or f.endswith("_svm.pkl") or f.endswith("_gnn.pt") for f in model_files)
    
    if has_models:
        logger.info("Models already exist, skipping initialization")
        return
    
    # Create sample data
    cross_platform_features = create_sample_features(config)
    ground_truth = create_sample_ground_truth(cross_platform_features)
    
    # Save sample data
    save_sample_data(cross_platform_features, ground_truth, config)
    
    # Train models
    logger.info("Training models on sample data")
    results = models.train_and_predict(
        cross_platform_features,
        config["models"],
        ground_truth,
        processed_data_dir=processed_data_dir
    )
    
    logger.info("Models initialized successfully")
    
    # Initialize matching service to verify models
    service = MatchingService(config)
    loaded_models = service._load_trained_models()
    
    logger.info(f"Loaded {len(loaded_models)} trained models")


if __name__ == "__main__":
    init_models()
