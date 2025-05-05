"""
Models Module

This module defines machine learning models for user matching,
including scikit-learn pipelines and PyTorch Geometric GNN models.
"""

import json
import logging
import os
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple, Union, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T

logger = logging.getLogger(__name__)


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for link prediction.
    
    Args:
        in_channels: Dimension of input features
        hidden_channels: List of hidden layer dimensions
        dropout: Dropout probability
    """
    def __init__(self, in_channels: int, hidden_channels: List[int], dropout: float = 0.2):
        super(GCN, self).__init__()
        self.dropout = dropout
        
        # Create GCN layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))
        
        # Output layer for link prediction
        self.out = nn.Linear(hidden_channels[-1] * 2, 1)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features using GCN layers.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            torch.Tensor: Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on final layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to predict links.
        
        Args:
            z: Node embeddings
            edge_index: Edges to predict
            
        Returns:
            torch.Tensor: Edge scores
        """
        # Extract features for node pairs
        row, col = edge_index
        z_u = z[row]
        z_v = z[col]
        
        # Concatenate node features
        edge_features = torch.cat([z_u, z_v], dim=1)
        
        # Pass through output layer
        return torch.sigmoid(self.out(edge_features)).squeeze()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                predict_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            predict_edge_index: Edges to predict
            
        Returns:
            torch.Tensor: Edge predictions
        """
        # Get node embeddings
        z = self.encode(x, edge_index)
        
        # Predict links
        return self.decode(z, predict_edge_index)


def load_cross_platform_features(processed_data_dir: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Load cross-platform features for model training.
    
    Args:
        processed_data_dir: Directory containing processed data
        
    Returns:
        Dict: Cross-platform features
    """
    try:
        with open(os.path.join(processed_data_dir, "cross_platform_features.pkl"), "rb") as f:
            cross_platform_features = pickle.load(f)
        logger.info("Loaded cross-platform features")
        return cross_platform_features
    except Exception as e:
        logger.error(f"Error loading cross-platform features: {e}")
        return {}


def load_ground_truth_mappings(processed_data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth mappings between users on different platforms.
    
    Args:
        processed_data_dir: Directory containing processed data
        
    Returns:
        Dict: Ground truth mappings
    """
    try:
        mappings_file = os.path.join(processed_data_dir, "ground_truth_mappings.json")
        
        if not os.path.exists(mappings_file):
            logger.warning("Ground truth mappings file not found")
            # Create empty mappings
            mappings = {}
            for pair in ["instagram_twitter", "instagram_facebook", "twitter_facebook"]:
                mappings[pair] = {}
            return mappings
        
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        
        logger.info(f"Loaded ground truth mappings: {sum(len(m) for m in mappings.values())} total mappings")
        return mappings
    
    except Exception as e:
        logger.error(f"Error loading ground truth mappings: {e}")
        return {}


def prepare_training_data(
    platform_pair: str,
    features: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Dict[str, str],
    negative_ratio: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for supervised learning.
    
    Args:
        platform_pair: Platform pair (e.g., "instagram_twitter")
        features: Feature vectors for users on each platform
        ground_truth: Ground truth mappings
        negative_ratio: Ratio of negative to positive examples
        
    Returns:
        Tuple: (feature_pairs, labels)
    """
    # Split platform names
    platform1, platform2 = platform_pair.split("_")
    
    # Get feature vectors
    vectors1 = features[platform1]
    vectors2 = features[platform2]
    
    # Create positive pairs
    positive_pairs = []
    positive_labels = []
    
    for user1, user2 in ground_truth.items():
        if user1 in vectors1 and user2 in vectors2:
            # Create feature pair (concatenate vectors)
            feature_pair = np.concatenate([vectors1[user1], vectors2[user2]])
            positive_pairs.append(feature_pair)
            positive_labels.append(1)
    
    # Create negative pairs
    negative_pairs = []
    negative_labels = []
    
    # Number of negative examples to generate
    num_negatives = int(len(positive_pairs) * negative_ratio)
    
    # Get all users
    users1 = list(vectors1.keys())
    users2 = list(vectors2.keys())
    
    # Create negative pairs by random sampling
    import random
    random.seed(42)
    
    # Create a set of positive pairs for fast lookup
    positive_set = set(ground_truth.items())
    
    while len(negative_pairs) < num_negatives:
        user1 = random.choice(users1)
        user2 = random.choice(users2)
        
        # Skip if this is a positive pair
        if (user1, user2) in positive_set:
            continue
        
        # Create feature pair
        feature_pair = np.concatenate([vectors1[user1], vectors2[user2]])
        negative_pairs.append(feature_pair)
        negative_labels.append(0)
    
    # Combine positive and negative examples
    all_pairs = np.vstack(positive_pairs + negative_pairs)
    all_labels = np.array(positive_labels + negative_labels)
    
    return all_pairs, all_labels


def create_graph_data(
    platform_pair: str,
    features: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Dict[str, str]
) -> Data:
    """
    Create PyTorch Geometric data for GNN training.
    
    Args:
        platform_pair: Platform pair (e.g., "instagram_twitter")
        features: Feature vectors for users on each platform
        ground_truth: Ground truth mappings
        
    Returns:
        Data: PyTorch Geometric data object
    """
    # Split platform names
    platform1, platform2 = platform_pair.split("_")
    
    # Get feature vectors
    vectors1 = features[platform1]
    vectors2 = features[platform2]
    
    # Map user IDs to node indices
    user_to_idx = {}
    idx = 0
    
    for user in vectors1:
        user_to_idx[f"{platform1}_{user}"] = idx
        idx += 1
    
    for user in vectors2:
        user_to_idx[f"{platform2}_{user}"] = idx
        idx += 1
    
    # Create node features
    all_features = []
    
    for user, vector in vectors1.items():
        all_features.append(vector)
    
    for user, vector in vectors2.items():
        all_features.append(vector)
    
    x = torch.tensor(np.vstack(all_features), dtype=torch.float)
    
    # Create edges from ground truth (training edges)
    edge_index = []
    
    for user1, user2 in ground_truth.items():
        idx1 = user_to_idx.get(f"{platform1}_{user1}")
        idx2 = user_to_idx.get(f"{platform2}_{user2}")
        
        if idx1 is None or idx2 is None:
            continue
        
        edge_index.append([idx1, idx2])
        edge_index.append([idx2, idx1])  # Add in both directions
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    
    # Store mapping for later use
    data.user_to_idx = user_to_idx
    data.platform1 = platform1
    data.platform2 = platform2
    
    return data


def train_sklearn_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any]
) -> Pipeline:
    """
    Train a scikit-learn model for user matching.
    
    Args:
        model_type: Model type ("random_forest" or "svm")
        X_train: Training features
        y_train: Training labels
        config: Model configuration
        
    Returns:
        Pipeline: Trained sklearn pipeline
    """
    # Create pipeline with scaling
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 20),
            random_state=config.get("random_state", 42)
        )
    elif model_type == "svm":
        clf = SVC(
            kernel=config.get("kernel", "rbf"),
            C=config.get("C", 1.0),
            gamma=config.get("gamma", "scale"),
            probability=True,
            random_state=config.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    logger.info(f"Trained {model_type} model")
    
    return pipeline


def train_gnn_model(
    data: Data, 
    config: Dict[str, Any],
    device: torch.device = torch.device('cpu')
) -> GCN:
    """
    Train a GNN model for link prediction.
    
    Args:
        data: PyTorch Geometric data
        config: Model configuration
        device: Torch device
        
    Returns:
        GCN: Trained GNN model
    """
    # Create train/val split
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0
    )
    
    train_data, val_data, _ = transform(data)
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    # Initialize model
    model = GCN(
        in_channels=data.x.size(1),
        hidden_channels=config.get("hidden_channels", [128, 64]),
        dropout=config.get("dropout", 0.2)
    ).to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get("learning_rate", 0.01),
        weight_decay=1e-5
    )
    
    # Training parameters
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 20)
    batch_size = config.get("batch_size", 64)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z = model.encode(train_data.x, train_data.edge_index)
        
        # Create positive and negative examples
        pos_edge_index = train_data.pos_edge_label_index
        neg_edge_index = train_data.neg_edge_label_index
        
        pos_pred = model.decode(z, pos_edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        # Combine predictions and labels
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=device),
            torch.zeros(neg_pred.size(0), device=device)
        ], dim=0)
        
        # Calculate loss
        loss = F.binary_cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x, val_data.edge_index)
            
            pos_edge_index = val_data.pos_edge_label_index
            neg_edge_index = val_data.neg_edge_label_index
            
            pos_pred = model.decode(z, pos_edge_index)
            neg_pred = model.decode(z, neg_edge_index)
            
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            labels = torch.cat([
                torch.ones(pos_pred.size(0), device=device),
                torch.zeros(neg_pred.size(0), device=device)
            ], dim=0)
            
            val_loss = F.binary_cross_entropy(pred, labels)
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Trained GNN model with best validation loss: {best_val_loss:.4f}")
    return model


def predict_matches(
    model_type: str,
    model: Union[Pipeline, GCN],
    features: Dict[str, Dict[str, np.ndarray]],
    platform_pair: str,
    data: Optional[Data] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Dict[str, float]]:
    """
    Predict matches between users on different platforms.
    
    Args:
        model_type: Model type ("random_forest", "svm", or "gcn")
        model: Trained model
        features: Feature vectors for users on each platform
        platform_pair: Platform pair (e.g., "instagram_twitter")
        data: PyTorch Geometric data (for GNN)
        threshold: Probability threshold for matches
        top_k: Return top K matches for each user
        device: Torch device (for GNN)
        
    Returns:
        Dict: Predicted matches with confidence scores
    """
    platform1, platform2 = platform_pair.split("_")
    
    vectors1 = features[platform1]
    vectors2 = features[platform2]
    
    predictions = {}
    
    if model_type in ["random_forest", "svm"]:
        # For sklearn models
        for user1, vec1 in vectors1.items():
            predictions[user1] = {}
            
            for user2, vec2 in vectors2.items():
                # Create feature pair
                feature_pair = np.concatenate([vec1, vec2]).reshape(1, -1)
                
                # Get match probability
                prob = model.predict_proba(feature_pair)[0, 1]
                
                if prob >= threshold:
                    predictions[user1][user2] = float(prob)
            
            # Filter to top K if specified
            if top_k and len(predictions[user1]) > top_k:
                predictions[user1] = dict(
                    sorted(predictions[user1].items(), key=lambda x: x[1], reverse=True)[:top_k]
                )
    
    elif model_type == "gcn":
        # For GNN model
        if data is None:
            logger.error("GNN prediction requires data object")
            return {}
        
        # Prepare data
        data = data.to(device)
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
        
        # Get user-to-index mapping
        user_to_idx = data.user_to_idx
        
        # Predict matches
        for user1, vec1 in vectors1.items():
            predictions[user1] = {}
            idx1 = user_to_idx.get(f"{platform1}_{user1}")
            
            if idx1 is None:
                continue
            
            for user2, vec2 in vectors2.items():
                idx2 = user_to_idx.get(f"{platform2}_{user2}")
                
                if idx2 is None:
                    continue
                
                # Create edge index for this pair
                edge_index = torch.tensor([[idx1, idx2]], dtype=torch.long).t()
                
                # Predict match
                prob = model.decode(z, edge_index).item()
                
                if prob >= threshold:
                    predictions[user1][user2] = float(prob)
            
            # Filter to top K if specified
            if top_k and len(predictions[user1]) > top_k:
                predictions[user1] = dict(
                    sorted(predictions[user1].items(), key=lambda x: x[1], reverse=True)[:top_k]
                )
    
    else:
        logger.error(f"Unsupported model type: {model_type}")
    
    return predictions


def train_and_predict(
    cross_platform_features: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    model_config: Dict[str, Any],
    ground_truth: Optional[Dict[str, Dict[str, str]]] = None,
    processed_data_dir: str = "processed_data"
) -> Dict[str, Dict[str, Any]]:
    """
    Train models and predict matches for all platform pairs.
    
    Args:
        cross_platform_features: Features for cross-platform matching
        model_config: Model configuration
        ground_truth: Ground truth mappings
        processed_data_dir: Directory to save processed data
        
    Returns:
        Dict: Model results for all platform pairs
    """
    # Load ground truth if not provided
    if ground_truth is None:
        ground_truth = load_ground_truth_mappings(processed_data_dir)
    
    # Create directory for models
    models_dir = os.path.join(processed_data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    results = {}
    
    for platform_pair, features in cross_platform_features.items():
        logger.info(f"Training models for {platform_pair}")
        
        pair_ground_truth = ground_truth.get(platform_pair, {})
        
        if not pair_ground_truth:
            logger.warning(f"No ground truth mappings for {platform_pair}")
            # Create synthetic mappings for testing if no real ones exist
            # This is just for development purposes
            platform1, platform2 = platform_pair.split("_")
            users1 = list(features[platform1].keys())[:5]
            users2 = list(features[platform2].keys())[:5]
            pair_ground_truth = {u1: u2 for u1, u2 in zip(users1, users2)}
        
        results[platform_pair] = {}
        
        # Prepare training data for sklearn models
        X, y = prepare_training_data(
            platform_pair,
            features,
            pair_ground_truth,
            negative_ratio=3.0
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf_model = train_sklearn_model(
            "random_forest",
            X_train,
            y_train,
            model_config["random_forest"]
        )
        
        # Save Random Forest model
        with open(os.path.join(models_dir, f"{platform_pair}_rf.pkl"), "wb") as f:
            pickle.dump(rf_model, f)
        
        # Train SVM
        svm_model = train_sklearn_model(
            "svm",
            X_train,
            y_train,
            model_config["svm"]
        )
        
        # Save SVM model
        with open(os.path.join(models_dir, f"{platform_pair}_svm.pkl"), "wb") as f:
            pickle.dump(svm_model, f)
        
        # Prepare data for GNN
        graph_data = create_graph_data(
            platform_pair,
            features,
            pair_ground_truth
        )
        
        # Train GNN
        gnn_model = train_gnn_model(
            graph_data,
            model_config["gcn"],
            device
        )
        
        # Save GNN model
        torch.save(gnn_model.state_dict(), os.path.join(models_dir, f"{platform_pair}_gnn.pt"))
        
        # Make predictions
        rf_preds = predict_matches(
            "random_forest",
            rf_model,
            features,
            platform_pair,
            threshold=0.5,
            top_k=5
        )
        
        svm_preds = predict_matches(
            "svm",
            svm_model,
            features,
            platform_pair,
            threshold=0.5,
            top_k=5
        )
        
        gnn_preds = predict_matches(
            "gcn",
            gnn_model,
            features,
            platform_pair,
            data=graph_data,
            threshold=0.5,
            top_k=5,
            device=device
        )
        
        # Store results
        results[platform_pair] = {
            "random_forest": {
                "model": rf_model,
                "predictions": rf_preds,
                "test_X": X_test,
                "test_y": y_test
            },
            "svm": {
                "model": svm_model,
                "predictions": svm_preds,
                "test_X": X_test,
                "test_y": y_test
            },
            "gcn": {
                "model": gnn_model,
                "predictions": gnn_preds,
                "data": graph_data
            },
            "ground_truth": pair_ground_truth
        }
    
    # Save prediction results
    with open(os.path.join(processed_data_dir, "predictions.pkl"), "wb") as f:
        # Don't save the actual models in predictions.pkl
        serializable_results = {}
        
        for platform_pair, platform_results in results.items():
            serializable_results[platform_pair] = {
                model_name: {
                    "predictions": model_data["predictions"],
                }
                for model_name, model_data in platform_results.items()
                if model_name != "ground_truth"
            }
            serializable_results[platform_pair]["ground_truth"] = platform_results["ground_truth"]
        
        pickle.dump(serializable_results, f)
    
    logger.info(f"Trained models for {len(cross_platform_features)} platform pairs")
    return results


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    
    # Load cross-platform features
    cross_platform_features = load_cross_platform_features(config["directories"]["processed_data"])
    
    # Train models and predict matches
    results = train_and_predict(
        cross_platform_features,
        config["models"],
        processed_data_dir=config["directories"]["processed_data"]
    )
