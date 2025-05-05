"""
Matching Service Module

This module provides services for matching users across different platforms
by integrating the data ingestion, feature extraction, and model prediction components.
"""

import logging
import os
import pickle
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data

import data_ingest
import preprocess
import behavior_features
import content_features
import network_features
import feature_fusion
import models
import privacy
import utils

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for matching users across platforms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the matching service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_dir = config["directories"]["models"]
        self.processed_data_dir = config["directories"]["processed_data"]
        self.raw_data_dir = config["directories"]["raw_data"]
        self.results_dir = config["directories"]["results"]
        
        # Create directories if they don't exist
        for directory in [self.models_dir, self.processed_data_dir, self.raw_data_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained models
        self.trained_models = self._load_trained_models()
        
        logger.info("Matching service initialized")
    
    def _load_trained_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load trained models for all platform pairs.
        
        Returns:
            Dict: Dictionary of trained models by platform pair and model type
        """
        trained_models = {}
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return trained_models
        
        # Get all platform pairs from model files
        model_files = os.listdir(self.models_dir)
        platform_pairs = set()
        
        for filename in model_files:
            if filename.endswith("_rf.pkl") or filename.endswith("_svm.pkl") or filename.endswith("_gnn.pt"):
                parts = filename.split("_")
                if len(parts) >= 3:
                    platform_pair = f"{parts[0]}_{parts[1]}"
                    platform_pairs.add(platform_pair)
        
        # Load models for each platform pair
        for platform_pair in platform_pairs:
            trained_models[platform_pair] = {}
            
            # Load Random Forest model
            rf_path = os.path.join(self.models_dir, f"{platform_pair}_rf.pkl")
            if os.path.exists(rf_path):
                try:
                    with open(rf_path, "rb") as f:
                        trained_models[platform_pair]["random_forest"] = pickle.load(f)
                    logger.info(f"Loaded Random Forest model for {platform_pair}")
                except Exception as e:
                    logger.error(f"Error loading Random Forest model for {platform_pair}: {e}")
            
            # Load SVM model
            svm_path = os.path.join(self.models_dir, f"{platform_pair}_svm.pkl")
            if os.path.exists(svm_path):
                try:
                    with open(svm_path, "rb") as f:
                        trained_models[platform_pair]["svm"] = pickle.load(f)
                    logger.info(f"Loaded SVM model for {platform_pair}")
                except Exception as e:
                    logger.error(f"Error loading SVM model for {platform_pair}: {e}")
            
            # Load GNN model
            gnn_path = os.path.join(self.models_dir, f"{platform_pair}_gnn.pt")
            if os.path.exists(gnn_path):
                try:
                    # Load graph data
                    graph_data_path = os.path.join(self.processed_data_dir, f"{platform_pair}_graph_data.pkl")
                    if os.path.exists(graph_data_path):
                        with open(graph_data_path, "rb") as f:
                            graph_data = pickle.load(f)
                        
                        # Initialize GNN model
                        gnn_model = models.GCN(
                            in_channels=graph_data.x.size(1),
                            hidden_channels=self.config["models"]["gcn"]["hidden_channels"],
                            dropout=self.config["models"]["gcn"]["dropout"]
                        )
                        
                        # Load state dict
                        gnn_model.load_state_dict(torch.load(gnn_path, map_location=self.device))
                        gnn_model.to(self.device)
                        gnn_model.eval()
                        
                        trained_models[platform_pair]["gcn"] = {
                            "model": gnn_model,
                            "data": graph_data
                        }
                        
                        logger.info(f"Loaded GNN model for {platform_pair}")
                    else:
                        logger.warning(f"Graph data not found for {platform_pair}")
                
                except Exception as e:
                    logger.error(f"Error loading GNN model for {platform_pair}: {e}")
        
        logger.info(f"Loaded models for {len(trained_models)} platform pairs")
        return trained_models
    
    def _extract_features_for_user(
        self, 
        platform: str, 
        user_id: str, 
        api_credentials: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for a user on a specific platform.
        
        Args:
            platform: Platform name
            user_id: User ID on the platform
            api_credentials: API credentials for the platform
            
        Returns:
            Dict: Feature vector for the user
        """
        logger.info(f"Extracting features for user {user_id} on {platform}")
        
        # Fetch data from API
        output_dir = os.path.join(self.raw_data_dir, platform)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Configure API credentials
            platform_config = self.config["api"][platform].copy()
            platform_config.update(api_credentials)
            
            # Fetch data
            data_files = data_ingest.fetch_platform_data(platform, platform_config, output_dir)
            
            if not data_files:
                logger.warning(f"No data fetched for user {user_id} on {platform}")
                return {}
            
            # Preprocess data
            processed_data = preprocess.preprocess_platform_data(
                platform, 
                data_files, 
                self.config["preprocessing"]
            )
            
            # Extract features
            behavior_feats = behavior_features.extract_behavioral_features_for_user(
                user_id, 
                processed_data.get("posts", []), 
                self.config["features"]["behavior"]
            )
            
            # Load embedding model for content features
            embedding_model = content_features.load_embedding_model(
                self.config["features"]["content"]["embedding_model"]
            )
            
            content_feats = content_features.extract_content_features_for_user(
                user_id, 
                processed_data.get("posts", []), 
                embedding_model,
                config=self.config["features"]["content"]
            )
            
            # Combine features
            behavior_vector = behavior_features.vectorize_behavioral_features(
                {user_id: behavior_feats}, 
                platform
            )[platform].get(user_id, np.array([]))
            
            content_vector = content_features.get_content_vectors(
                {user_id: content_feats}, 
                platform
            )[platform].get(user_id, np.array([]))
            
            # Combine all features
            if len(behavior_vector) > 0 and len(content_vector) > 0:
                combined_vector = np.concatenate([behavior_vector, content_vector])
                return {user_id: combined_vector}
            elif len(behavior_vector) > 0:
                return {user_id: behavior_vector}
            elif len(content_vector) > 0:
                return {user_id: content_vector}
            else:
                logger.warning(f"No features extracted for user {user_id} on {platform}")
                return {}
        
        except Exception as e:
            logger.error(f"Error extracting features for user {user_id} on {platform}: {e}")
            return {}
    
    def match_users(
        self, 
        platform1: str, 
        platform1_user: str, 
        platform2: str,
        platform1_credentials: Dict[str, str],
        platform2_credentials: Dict[str, str],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Match a user from one platform to users on another platform.
        
        Args:
            platform1: Source platform
            platform1_user: User ID on source platform
            platform2: Target platform
            platform1_credentials: API credentials for source platform
            platform2_credentials: API credentials for target platform
            top_k: Number of top matches to return
            threshold: Confidence threshold for matches
            
        Returns:
            Dict: Matching results
        """
        start_time = time.time()
        logger.info(f"Matching user {platform1_user} from {platform1} to {platform2}")
        
        # Check if models are available for this platform pair
        platform_pair = f"{platform1}_{platform2}"
        reverse_pair = f"{platform2}_{platform1}"
        
        if platform_pair not in self.trained_models and reverse_pair not in self.trained_models:
            logger.warning(f"No trained models available for {platform1} and {platform2}")
            return {
                "success": False,
                "error": f"No trained models available for {platform1} and {platform2}"
            }
        
        # Use the correct platform pair
        use_reverse = platform_pair not in self.trained_models
        actual_pair = reverse_pair if use_reverse else platform_pair
        
        # Extract features for the source user
        source_features = self._extract_features_for_user(
            platform1, platform1_user, platform1_credentials
        )
        
        if not source_features:
            return {
                "success": False,
                "error": f"Failed to extract features for user {platform1_user} on {platform1}"
            }
        
        # Load potential target users
        target_users = self._load_potential_target_users(platform2)
        
        if not target_users:
            logger.warning(f"No potential target users found for {platform2}")
            # Try to extract features for some sample users
            target_features = self._extract_features_for_sample_users(
                platform2, platform2_credentials
            )
        else:
            # Load features for target users
            target_features = self._load_features_for_users(platform2, target_users)
        
        if not target_features:
            return {
                "success": False,
                "error": f"No features available for users on {platform2}"
            }
        
        # Prepare features for prediction
        if use_reverse:
            # Swap source and target for reverse pair
            features = {
                platform2: target_features,
                platform1: source_features
            }
        else:
            features = {
                platform1: source_features,
                platform2: target_features
            }
        
        # Make predictions with each model
        predictions = {}
        matched_features = []
        
        # Random Forest
        if "random_forest" in self.trained_models[actual_pair]:
            rf_model = self.trained_models[actual_pair]["random_forest"]
            rf_preds = models.predict_matches(
                "random_forest",
                rf_model,
                features,
                actual_pair,
                threshold=threshold,
                top_k=top_k
            )
            
            if use_reverse:
                # Reverse predictions
                rf_preds_reversed = {}
                for target_user, source_users in rf_preds.items():
                    for source_user, score in source_users.items():
                        if source_user == platform1_user:
                            if platform1_user not in rf_preds_reversed:
                                rf_preds_reversed[platform1_user] = {}
                            rf_preds_reversed[platform1_user][target_user] = score
                
                predictions["random_forest"] = rf_preds_reversed.get(platform1_user, {})
            else:
                predictions["random_forest"] = rf_preds.get(platform1_user, {})
            
            matched_features.append("behavioral patterns")
        
        # SVM
        if "svm" in self.trained_models[actual_pair]:
            svm_model = self.trained_models[actual_pair]["svm"]
            svm_preds = models.predict_matches(
                "svm",
                svm_model,
                features,
                actual_pair,
                threshold=threshold,
                top_k=top_k
            )
            
            if use_reverse:
                # Reverse predictions
                svm_preds_reversed = {}
                for target_user, source_users in svm_preds.items():
                    for source_user, score in source_users.items():
                        if source_user == platform1_user:
                            if platform1_user not in svm_preds_reversed:
                                svm_preds_reversed[platform1_user] = {}
                            svm_preds_reversed[platform1_user][target_user] = score
                
                predictions["svm"] = svm_preds_reversed.get(platform1_user, {})
            else:
                predictions["svm"] = svm_preds.get(platform1_user, {})
            
            matched_features.append("content similarity")
        
        # GNN
        if "gcn" in self.trained_models[actual_pair]:
            gnn_data = self.trained_models[actual_pair]["gcn"]
            gnn_model = gnn_data["model"]
            graph_data = gnn_data["data"]
            
            # Update graph data with new users
            # This is a simplified version - in a real system, you would need to
            # properly integrate the new users into the graph
            
            gnn_preds = models.predict_matches(
                "gcn",
                gnn_model,
                features,
                actual_pair,
                data=graph_data,
                threshold=threshold,
                top_k=top_k,
                device=self.device
            )
            
            if use_reverse:
                # Reverse predictions
                gnn_preds_reversed = {}
                for target_user, source_users in gnn_preds.items():
                    for source_user, score in source_users.items():
                        if source_user == platform1_user:
                            if platform1_user not in gnn_preds_reversed:
                                gnn_preds_reversed[platform1_user] = {}
                            gnn_preds_reversed[platform1_user][target_user] = score
                
                predictions["gcn"] = gnn_preds_reversed.get(platform1_user, {})
            else:
                predictions["gcn"] = gnn_preds.get(platform1_user, {})
            
            matched_features.append("network structure")
        
        # Combine predictions from all models
        combined_predictions = self._combine_predictions(predictions)
        
        # Format results
        matches = []
        for target_user, score in combined_predictions.items():
            matches.append({
                "platform2_user": target_user,
                "confidence_score": score,
                "matched_features": matched_features
            })
        
        # Sort by confidence score
        matches.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        # Limit to top K
        matches = matches[:top_k]
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "platform1": platform1,
            "platform1_user": platform1_user,
            "platform2": platform2,
            "matches": matches,
            "processing_time": processing_time
        }
        
        return result
    
    def _combine_predictions(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: Predictions from different models
            
        Returns:
            Dict: Combined predictions
        """
        combined = {}
        
        # Get all target users
        all_users = set()
        for model_preds in predictions.values():
            all_users.update(model_preds.keys())
        
        # Combine scores
        for user in all_users:
            scores = []
            for model_name, model_preds in predictions.items():
                if user in model_preds:
                    scores.append(model_preds[user])
            
            if scores:
                # Use maximum score as the combined score
                # Could also use average, weighted average, etc.
                combined[user] = max(scores)
        
        return combined
    
    def _load_potential_target_users(self, platform: str) -> List[str]:
        """
        Load potential target users for a platform.
        
        Args:
            platform: Platform name
            
        Returns:
            List: List of user IDs
        """
        # Check if processed data directory exists
        platform_dir = os.path.join(self.processed_data_dir, platform)
        if not os.path.exists(platform_dir):
            logger.warning(f"No processed data directory for {platform}")
            return []
        
        # Look for processed data file
        processed_file = os.path.join(platform_dir, "processed_data.json")
        if not os.path.exists(processed_file):
            logger.warning(f"No processed data file for {platform}")
            return []
        
        try:
            with open(processed_file, 'r') as f:
                data = json.load(f)
            
            # Extract user IDs from posts
            if "posts" in data:
                return list(data["posts"].keys())
            else:
                logger.warning(f"No posts found in processed data for {platform}")
                return []
        
        except Exception as e:
            logger.error(f"Error loading potential target users for {platform}: {e}")
            return []
    
    def _load_features_for_users(self, platform: str, users: List[str]) -> Dict[str, np.ndarray]:
        """
        Load features for users on a platform.
        
        Args:
            platform: Platform name
            users: List of user IDs
            
        Returns:
            Dict: Feature vectors by user ID
        """
        # Check if feature vectors file exists
        features_file = os.path.join(self.processed_data_dir, f"{platform}_features.pkl")
        if not os.path.exists(features_file):
            logger.warning(f"No feature vectors file for {platform}")
            return {}
        
        try:
            with open(features_file, 'rb') as f:
                all_features = pickle.load(f)
            
            # Filter to requested users
            user_features = {}
            for user in users:
                if user in all_features:
                    user_features[user] = all_features[user]
            
            return user_features
        
        except Exception as e:
            logger.error(f"Error loading features for users on {platform}: {e}")
            return {}
    
    def _extract_features_for_sample_users(
        self, 
        platform: str, 
        api_credentials: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for sample users on a platform.
        
        Args:
            platform: Platform name
            api_credentials: API credentials for the platform
            
        Returns:
            Dict: Feature vectors by user ID
        """
        logger.info(f"Extracting features for sample users on {platform}")
        
        # Configure API credentials
        platform_config = self.config["api"][platform].copy()
        platform_config.update(api_credentials)
        
        # Fetch data for the platform (this will get data for the authenticated user)
        output_dir = os.path.join(self.raw_data_dir, platform)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Fetch data
            data_files = data_ingest.fetch_platform_data(platform, platform_config, output_dir)
            
            if not data_files:
                logger.warning(f"No data fetched for {platform}")
                return {}
            
            # Preprocess data
            processed_data = preprocess.preprocess_platform_data(
                platform, 
                data_files, 
                self.config["preprocessing"]
            )
            
            # Extract features for all users in the data
            all_features = {}
            
            # Get user IDs from posts
            if "posts" in processed_data:
                user_ids = list(processed_data["posts"].keys())
                
                # Load embedding model for content features
                embedding_model = content_features.load_embedding_model(
                    self.config["features"]["content"]["embedding_model"]
                )
                
                for user_id in user_ids:
                    # Extract behavior features
                    behavior_feats = behavior_features.extract_behavioral_features_for_user(
                        user_id, 
                        processed_data["posts"].get(user_id, []), 
                        self.config["features"]["behavior"]
                    )
                    
                    # Extract content features
                    content_feats = content_features.extract_content_features_for_user(
                        user_id, 
                        processed_data["posts"].get(user_id, []), 
                        embedding_model,
                        config=self.config["features"]["content"]
                    )
                    
                    # Combine features
                    behavior_vector = behavior_features.vectorize_behavioral_features(
                        {user_id: behavior_feats}, 
                        platform
                    )[platform].get(user_id, np.array([]))
                    
                    content_vector = content_features.get_content_vectors(
                        {user_id: content_feats}, 
                        platform
                    )[platform].get(user_id, np.array([]))
                    
                    # Combine all features
                    if len(behavior_vector) > 0 and len(content_vector) > 0:
                        all_features[user_id] = np.concatenate([behavior_vector, content_vector])
                    elif len(behavior_vector) > 0:
                        all_features[user_id] = behavior_vector
                    elif len(content_vector) > 0:
                        all_features[user_id] = content_vector
            
            return all_features
        
        except Exception as e:
            logger.error(f"Error extracting features for sample users on {platform}: {e}")
            return {}


# Singleton instance
_instance = None

def get_instance(config: Dict[str, Any]) -> MatchingService:
    """
    Get the singleton instance of the matching service.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MatchingService: Singleton instance
    """
    global _instance
    if _instance is None:
        _instance = MatchingService(config)
    return _instance
