"""
Tests for the models module.
"""

import os
import json
import pytest
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from unittest.mock import patch, MagicMock
import models


def test_gcn_class():
    """Test the Graph Convolutional Network class for link prediction."""
    # Create test data
    in_channels = 64
    hidden_channels = [128, 64]
    
    # Test initialization
    gcn = models.GCN(in_channels, hidden_channels)
    
    # Check model structure
    assert len(gcn.convs) == 2
    assert isinstance(gcn.convs[0], torch.nn.Module)
    assert isinstance(gcn.convs[1], torch.nn.Module)
    assert isinstance(gcn.out, torch.nn.Linear)
    assert gcn.out.out_features == 1
    
    # Test forward pass
    x = torch.randn(10, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    predict_edge_index = torch.tensor([[0, 5], [6, 7]], dtype=torch.long)
    
    # Test encode method
    z = gcn.encode(x, edge_index)
    assert z.shape == (10, hidden_channels[-1])
    
    # Test decode method
    edge_scores = gcn.decode(z, predict_edge_index)
    assert edge_scores.shape == (2,)
    assert torch.all(edge_scores >= 0) and torch.all(edge_scores <= 1)
    
    # Test forward method
    out = gcn(x, edge_index, predict_edge_index)
    assert out.shape == (2,)
    assert torch.all(out >= 0) and torch.all(out <= 1)


@patch('pickle.load')
def test_load_cross_platform_features(mock_pickle_load):
    """Test loading cross-platform features for model training."""
    # Set up mock
    mock_features = {
        "instagram_twitter": {
            "instagram": {"user1": np.array([1.0, 2.0])},
            "twitter": {"user2": np.array([3.0, 4.0])}
        }
    }
    mock_pickle_load.return_value = mock_features
    
    # Call function
    with patch('os.path.exists', return_value=True):
        features = models.load_cross_platform_features("processed_data")
    
    # Check if pickle.load was called
    mock_pickle_load.assert_called_once()
    
    # Check results
    assert features == mock_features
    
    # Test with missing file
    mock_pickle_load.side_effect = FileNotFoundError
    
    with patch('os.path.exists', return_value=False):
        features = models.load_cross_platform_features("processed_data")
    
    # Should return empty dictionary
    assert features == {}


@patch('json.load')
def test_load_ground_truth_mappings(mock_json_load):
    """Test loading ground truth mappings between users on different platforms."""
    # Set up mock
    mock_mappings = {
        "instagram_twitter": {
            "user1_instagram": "user1_twitter",
            "user2_instagram": "user2_twitter"
        }
    }
    mock_json_load.return_value = mock_mappings
    
    # Call function
    with patch('os.path.exists', return_value=True):
        mappings = models.load_ground_truth_mappings("processed_data")
    
    # Check if json.load was called
    mock_json_load.assert_called_once()
    
    # Check results
    assert mappings == mock_mappings
    
    # Test with missing file
    with patch('os.path.exists', return_value=False):
        mappings = models.load_ground_truth_mappings("processed_data")
    
    # Should return empty dictionaries
    assert "instagram_twitter" in mappings
    assert "instagram_facebook" in mappings
    assert "twitter_facebook" in mappings
    assert mappings["instagram_twitter"] == {}


def test_prepare_training_data(sample_fused_features, sample_ground_truth_mappings):
    """Test preparing training data for supervised learning."""
    # Unpack sample features
    _, cross_platform_features = sample_fused_features
    
    # Call function
    platform_pair = "instagram_twitter"
    features = cross_platform_features[platform_pair]
    ground_truth = sample_ground_truth_mappings[platform_pair]
    
    feature_pairs, labels = models.prepare_training_data(
        platform_pair, features, ground_truth
    )
    
    # Check results
    assert isinstance(feature_pairs, np.ndarray)
    assert isinstance(labels, np.ndarray)
    
    # Should have both positive and negative examples
    assert np.any(labels == 1)
    assert np.any(labels == 0)
    
    # Check feature dimensions
    user1 = next(iter(features["instagram"]))
    user2 = next(iter(features["twitter"]))
    expected_dim = features["instagram"][user1].shape[0] + features["twitter"][user2].shape[0]
    assert feature_pairs.shape[1] == expected_dim
    
    # Test with empty ground truth
    empty_ground_truth = {}
    feature_pairs, labels = models.prepare_training_data(
        platform_pair, features, empty_ground_truth
    )
    
    # Should have no positive examples
    assert len(feature_pairs) == 0
    assert len(labels) == 0


def test_create_graph_data(sample_fused_features, sample_ground_truth_mappings):
    """Test creating PyTorch Geometric data for GNN training."""
    # Unpack sample features
    _, cross_platform_features = sample_fused_features
    
    # Call function
    platform_pair = "instagram_twitter"
    features = cross_platform_features[platform_pair]
    ground_truth = sample_ground_truth_mappings[platform_pair]
    
    data = models.create_graph_data(platform_pair, features, ground_truth)
    
    # Check results
    assert hasattr(data, 'x')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'user_to_idx')
    assert hasattr(data, 'platform1')
    assert hasattr(data, 'platform2')
    
    # Check number of nodes and edges
    total_users = len(features["instagram"]) + len(features["twitter"])
    assert data.x.shape[0] == total_users
    
    # Each ground truth mapping creates 2 edges (bidirectional)
    assert data.edge_index.shape[1] == 2 * len(ground_truth)
    
    # Check if node features have the right dimensions
    user1 = next(iter(features["instagram"]))
    expected_dim = features["instagram"][user1].shape[0]
    assert data.x.shape[1] == expected_dim
    
    # Test with empty ground truth
    empty_ground_truth = {}
    data = models.create_graph_data(platform_pair, features, empty_ground_truth)
    
    # Should have no edges
    assert data.edge_index.shape[1] == 0


def test_train_sklearn_model():
    """Test training a scikit-learn model for user matching."""
    # Create test data
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, 100)
    
    # Test Random Forest
    rf_config = {
        "n_estimators": 100,
        "max_depth": 20,
        "random_state": 42
    }
    
    rf_model = models.train_sklearn_model("random_forest", X_train, y_train, rf_config)
    
    # Check if model is a pipeline
    assert isinstance(rf_model, Pipeline)
    assert isinstance(rf_model.named_steps['classifier'], RandomForestClassifier)
    
    # Check if model has the right parameters
    rf_classifier = rf_model.named_steps['classifier']
    assert rf_classifier.n_estimators == 100
    assert rf_classifier.max_depth == 20
    assert rf_classifier.random_state == 42
    
    # Test SVM
    svm_config = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": 42
    }
    
    svm_model = models.train_sklearn_model("svm", X_train, y_train, svm_config)
    
    # Check if model is a pipeline
    assert isinstance(svm_model, Pipeline)
    assert isinstance(svm_model.named_steps['classifier'], SVC)
    
    # Check if model has the right parameters
    svm_classifier = svm_model.named_steps['classifier']
    assert svm_classifier.kernel == "rbf"
    assert svm_classifier.C == 1.0
    assert svm_classifier.gamma == "scale"
    assert svm_classifier.random_state == 42
    assert svm_classifier.probability == True
    
    # Test unsupported model
    with pytest.raises(ValueError):
        models.train_sklearn_model("unsupported", X_train, y_train, {})


@patch('models.GCN')
@patch('torch.optim.Adam')
def test_train_gnn_model(mock_adam, mock_gcn):
    """Test training a GNN model for link prediction."""
    # Set up mocks
    mock_gcn_instance = MagicMock()
    mock_gcn.return_value = mock_gcn_instance
    
    mock_gcn_instance.encode.return_value = torch.randn(10, 64)
    mock_gcn_instance.decode.return_value = torch.sigmoid(torch.randn(5))
    
    mock_optimizer = MagicMock()
    mock_adam.return_value = mock_optimizer
    
    # Create test data
    data = MagicMock()
    data.x = torch.randn(10, 128)
    data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    # Mock the RandomLinkSplit transform
    train_data = MagicMock()
    train_data.x = data.x
    train_data.edge_index = data.edge_index
    train_data.pos_edge_label_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_data.neg_edge_label_index = torch.tensor([[0, 2], [2, 3]], dtype=torch.long)
    
    val_data = MagicMock()
    val_data.x = data.x
    val_data.edge_index = data.edge_index
    val_data.pos_edge_label_index = torch.tensor([[0, 3], [3, 4]], dtype=torch.long)
    val_data.neg_edge_label_index = torch.tensor([[0, 4], [4, 5]], dtype=torch.long)
    
    test_data = None
    
    with patch('models.T.RandomLinkSplit') as mock_split:
        mock_split_instance = MagicMock()
        mock_split.return_value = mock_split_instance
        mock_split_instance.return_value = (train_data, val_data, test_data)
        
        # Call function
        config = {
            "hidden_channels": [128, 64],
            "dropout": 0.2,
            "learning_rate": 0.01,
            "epochs": 5,  # Reduce for testing
            "patience": 3,
            "batch_size": 64
        }
        
        model = models.train_gnn_model(data, config)
    
    # Check if GCN was initialized correctly
    mock_gcn.assert_called_once()
    assert mock_gcn.call_args[0][0] == 128  # in_channels
    assert mock_gcn.call_args[0][1] == [128, 64]  # hidden_channels
    
    # Check if optimizer was initialized correctly
    mock_adam.assert_called_once()
    assert mock_adam.call_args[0][0] == mock_gcn_instance.parameters()
    assert mock_adam.call_args[1]["lr"] == 0.01
    
    # Check if training was performed
    assert mock_gcn_instance.train.call_count > 0
    assert mock_gcn_instance.eval.call_count > 0
    assert mock_optimizer.zero_grad.call_count > 0
    assert mock_optimizer.step.call_count > 0


def test_predict_matches():
    """Test predicting matches between users on different platforms."""
    # Create test data
    platform_pair = "instagram_twitter"
    features = {
        "instagram": {
            "user1": np.array([0.1, 0.2]),
            "user2": np.array([0.3, 0.4])
        },
        "twitter": {
            "user3": np.array([0.5, 0.6]),
            "user4": np.array([0.7, 0.8])
        }
    }
    
    # Test with sklearn model
    mock_sklearn_model = MagicMock()
    mock_sklearn_model.predict_proba.side_effect = lambda X: np.array([[0.2, 0.8], [0.7, 0.3]])
    
    predictions = models.predict_matches(
        "random_forest", mock_sklearn_model, features, platform_pair,
        threshold=0.5, top_k=1
    )
    
    # Check predictions
    assert "user1" in predictions
    assert "user2" in predictions
    assert len(predictions["user1"]) == 1  # top_k=1
    assert "user3" in predictions["user1"] or "user4" in predictions["user1"]
    
    # Test with GNN model
    mock_gnn_model = MagicMock()
    mock_gnn_model.encode.return_value = torch.randn(4, 64)
    mock_gnn_model.decode.side_effect = lambda z, edge_index: torch.tensor([0.8, 0.3])
    
    mock_data = MagicMock()
    mock_data.x = torch.randn(4, 64)
    mock_data.edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    mock_data.user_to_idx = {
        "instagram_user1": 0,
        "instagram_user2": 1,
        "twitter_user3": 2,
        "twitter_user4": 3
    }
    
    predictions = models.predict_matches(
        "gcn", mock_gnn_model, features, platform_pair,
        data=mock_data, threshold=0.5, top_k=1
    )
    
    # Check predictions
    assert "user1" in predictions
    assert "user2" in predictions
    
    # Test with unsupported model
    with pytest.raises(Exception):
        models.predict_matches(
            "unsupported", None, features, platform_pair
        )


@patch('models.load_ground_truth_mappings')
@patch('models.train_sklearn_model')
@patch('models.train_gnn_model')
@patch('models.prepare_training_data')
@patch('models.create_graph_data')
@patch('models.predict_matches')
@patch('pickle.dump')
@patch('torch.save')
def test_train_and_predict(
    mock_torch_save, mock_pickle, mock_predict, mock_create_graph,
    mock_prepare_data, mock_train_gnn, mock_train_sklearn, mock_load_ground_truth,
    sample_fused_features
):
    """Test training models and predicting matches for all platform pairs."""
    # Unpack sample features
    _, cross_platform_features = sample_fused_features
    
    # Set up mocks
    mock_ground_truth = {
        "instagram_twitter": {
            "user1_instagram": "user1_twitter",
            "user2_instagram": "user2_twitter"
        }
    }
    mock_load_ground_truth.return_value = mock_ground_truth
    
    X = np.random.randn(10, 20)
    y = np.random.randint(0, 2, 10)
    mock_prepare_data.return_value = (X, y)
    
    mock_graph_data = MagicMock()
    mock_create_graph.return_value = mock_graph_data
    
    mock_rf_model = MagicMock()
    mock_svm_model = MagicMock()
    mock_gnn_model = MagicMock()
    mock_train_sklearn.side_effect = [mock_rf_model, mock_svm_model]
    mock_train_gnn.return_value = mock_gnn_model
    
    mock_rf_preds = {"user1_instagram": {"user1_twitter": 0.8}}
    mock_svm_preds = {"user1_instagram": {"user1_twitter": 0.7}}
    mock_gnn_preds = {"user1_instagram": {"user1_twitter": 0.9}}
    mock_predict.side_effect = [mock_rf_preds, mock_svm_preds, mock_gnn_preds]
    
    # Call function
    with patch('os.makedirs') as mock_makedirs, \
         patch('sklearn.model_selection.train_test_split', return_value=(X, X, y, y)) as mock_split:
        
        config = {
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
                "epochs": 5,
                "patience": 3,
                "batch_size": 64
            }
        }
        
        results = models.train_and_predict(
            cross_platform_features, config, mock_ground_truth
        )
    
    # Check if directory was created
    mock_makedirs.assert_called()
    
    # Check if models were trained
    assert mock_train_sklearn.call_count == 2  # RF and SVM
    assert mock_train_gnn.call_count == 1
    
    # Check if predictions were made
    assert mock_predict.call_count == 3  # RF, SVM, and GNN
    
    # Check if models were saved
    assert mock_pickle.call_count >= 1
    assert mock_torch_save.call_count == 1
    
    # Check results
    assert "instagram_twitter" in results
    assert "random_forest" in results["instagram_twitter"]
    assert "svm" in results["instagram_twitter"]
    assert "gcn" in results["instagram_twitter"]
    assert "ground_truth" in results["instagram_twitter"]
    
    # Test without provided ground truth
    mock_load_ground_truth.reset_mock()
    mock_load_ground_truth.return_value = mock_ground_truth
    
    with patch('os.makedirs') as mock_makedirs, \
         patch('sklearn.model_selection.train_test_split', return_value=(X, X, y, y)) as mock_split:
        
        results = models.train_and_predict(
            cross_platform_features, config
        )
    
    # Should load ground truth
    mock_load_ground_truth.assert_called_once()
