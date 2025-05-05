"""
Tests for the feature_fusion module.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import feature_fusion


@patch('feature_fusion.vectorize_behavioral_features')
@patch('feature_fusion.get_content_vectors')
@patch('feature_fusion.vectorize_network_features')
@patch('pickle.load')
def test_load_feature_vectors(
    mock_pickle_load, mock_vectorize_network, mock_get_content, mock_vectorize_behavior
):
    """Test loading feature vectors from pickle files."""
    # Set up mocks
    behavior_features = {"instagram": {"user1": {}}}
    content_features = {"instagram": {"user1": {}}}
    network_features = {"twitter": {"user2": {}}}
    
    mock_pickle_load.side_effect = [behavior_features, content_features, network_features]
    
    behavior_vectors = {"instagram": {"user1": np.array([1.0, 2.0])}}
    content_vectors = {"instagram": {"user1": np.array([3.0, 4.0])}}
    network_vectors = {"twitter": {"user2": np.array([5.0, 6.0])}}
    
    mock_vectorize_behavior.return_value = behavior_vectors
    mock_get_content.return_value = content_vectors
    mock_vectorize_network.return_value = network_vectors
    
    # Call function
    with patch('os.path.exists', return_value=True):
        b_vectors, c_vectors, n_vectors = feature_fusion.load_feature_vectors("processed_data")
    
    # Check if pickle.load was called for each feature type
    assert mock_pickle_load.call_count == 3
    
    # Check if vectorize functions were called
    mock_vectorize_behavior.assert_called_once_with(behavior_features)
    mock_get_content.assert_called_once_with(content_features)
    mock_vectorize_network.assert_called_once_with(network_features)
    
    # Check returned vectors
    assert b_vectors == behavior_vectors
    assert c_vectors == content_vectors
    assert n_vectors == network_vectors
    
    # Test with missing files
    mock_pickle_load.side_effect = [FileNotFoundError, FileNotFoundError, FileNotFoundError]
    
    with patch('os.path.exists', return_value=False):
        b_vectors, c_vectors, n_vectors = feature_fusion.load_feature_vectors("processed_data")
    
    # Should return empty dictionaries
    assert b_vectors == {}
    assert c_vectors == {}
    assert n_vectors == {}


def test_concatenate_features(sample_feature_vectors):
    """Test concatenating different feature types for each user."""
    # Unpack sample vectors
    behavior_vectors, content_vectors, network_vectors = sample_feature_vectors
    
    # Call function
    concatenated = feature_fusion.concatenate_features(
        behavior_vectors, content_vectors, network_vectors
    )
    
    # Check results
    assert "instagram" in concatenated
    assert "twitter" in concatenated
    assert "facebook" in concatenated
    
    # Check if vectors were concatenated for each user
    for platform in ["instagram", "twitter", "facebook"]:
        for user_id in behavior_vectors[platform]:
            assert user_id in concatenated[platform]
            
            # Vector should be concatenation of behavior and content vectors
            # For twitter and facebook, should also include network vectors
            expected_dim = behavior_vectors[platform][user_id].shape[0] + content_vectors[platform][user_id].shape[0]
            if platform in network_vectors and user_id in network_vectors[platform]:
                expected_dim += network_vectors[platform][user_id].shape[0]
            
            assert concatenated[platform][user_id].shape[0] == expected_dim
    
    # Test with missing vectors
    partial_behavior = {"instagram": {"user1": np.array([1.0, 2.0])}}
    partial_content = {"twitter": {"user2": np.array([3.0, 4.0])}}
    partial_network = {"facebook": {"user3": np.array([5.0, 6.0])}}
    
    concatenated = feature_fusion.concatenate_features(
        partial_behavior, partial_content, partial_network
    )
    
    # Check if platforms with any vectors are included
    assert "instagram" in concatenated
    assert "twitter" in concatenated
    assert "facebook" in concatenated
    
    # Check if users are included only if they have at least one vector
    assert "user1" in concatenated["instagram"]
    assert "user2" in concatenated["twitter"]
    assert "user3" in concatenated["facebook"]
    
    # Check if dimensions are correct
    assert concatenated["instagram"]["user1"].shape[0] == 2  # Only behavior
    assert concatenated["twitter"]["user2"].shape[0] == 2    # Only content
    assert concatenated["facebook"]["user3"].shape[0] == 2   # Only network


@patch('feature_fusion.PCA')
@patch('feature_fusion.StandardScaler')
@patch('feature_fusion.umap.UMAP')
def test_apply_dimensionality_reduction(mock_umap, mock_scaler, mock_pca):
    """Test applying dimensionality reduction to feature vectors."""
    # Set up mocks
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance
    mock_scaler_instance.fit_transform.return_value = np.random.randn(6, 100)
    
    mock_pca_instance = MagicMock()
    mock_pca.return_value = mock_pca_instance
    mock_pca_instance.fit_transform.return_value = np.random.randn(6, 50)
    
    mock_umap_instance = MagicMock()
    mock_umap.return_value = mock_umap_instance
    mock_umap_instance.fit_transform.return_value = np.random.randn(6, 20)
    
    # Create test data
    concatenated_vectors = {
        "instagram": {
            "user1": np.random.randn(100),
            "user2": np.random.randn(100)
        },
        "twitter": {
            "user3": np.random.randn(100),
            "user4": np.random.randn(100)
        },
        "facebook": {
            "user5": np.random.randn(100),
            "user6": np.random.randn(100)
        }
    }
    
    # Test PCA with explained variance
    config = {
        "dimensionality_reduction": "pca",
        "explained_variance": 0.95
    }
    
    reduced = feature_fusion.apply_dimensionality_reduction(concatenated_vectors, config)
    
    # Check if scaler was applied
    mock_scaler_instance.fit_transform.assert_called_once()
    
    # Check if PCA was applied with explained variance
    mock_pca.assert_called_once()
    assert mock_pca.call_args[1]["n_components"] == 0.95
    
    # Check results
    assert "instagram" in reduced
    assert "twitter" in reduced
    assert "facebook" in reduced
    assert len(reduced["instagram"]) == 2
    assert len(reduced["twitter"]) == 2
    assert len(reduced["facebook"]) == 2
    
    # Reset mocks
    mock_scaler.reset_mock()
    mock_pca.reset_mock()
    mock_umap.reset_mock()
    mock_scaler_instance.reset_mock()
    
    # Test PCA with fixed dimension
    config = {
        "dimensionality_reduction": "pca",
        "embedding_dim": 20
    }
    
    reduced = feature_fusion.apply_dimensionality_reduction(concatenated_vectors, config)
    
    # Check if PCA was applied with fixed dimension
    mock_pca.assert_called_once()
    assert mock_pca.call_args[1]["n_components"] == 20
    
    # Reset mocks
    mock_scaler.reset_mock()
    mock_pca.reset_mock()
    mock_umap.reset_mock()
    mock_scaler_instance.reset_mock()
    
    # Test UMAP
    config = {
        "dimensionality_reduction": "umap",
        "embedding_dim": 20
    }
    
    reduced = feature_fusion.apply_dimensionality_reduction(concatenated_vectors, config)
    
    # Check if UMAP was applied
    mock_umap.assert_called_once()
    assert mock_umap.call_args[1]["n_components"] == 20
    
    # Reset mocks
    mock_scaler.reset_mock()
    mock_pca.reset_mock()
    mock_umap.reset_mock()
    mock_scaler_instance.reset_mock()
    
    # Test unsupported method
    config = {
        "dimensionality_reduction": "unsupported"
    }
    
    reduced = feature_fusion.apply_dimensionality_reduction(concatenated_vectors, config)
    
    # Should fall back to scaled features
    assert not mock_pca.called
    assert not mock_umap.called


@patch('pickle.dump')
def test_create_cross_platform_features(mock_pickle, sample_fused_features):
    """Test creating features for cross-platform matching."""
    # Unpack sample features
    reduced_vectors, _ = sample_fused_features
    
    # Call function
    with patch('os.makedirs', return_value=None):
        cross_platform = feature_fusion.create_cross_platform_features(
            reduced_vectors, "processed_data"
        )
    
    # Check if pickle.dump was called
    mock_pickle.assert_called_once()
    
    # Check results
    assert "instagram_twitter" in cross_platform
    assert "instagram_facebook" in cross_platform
    assert "twitter_facebook" in cross_platform
    
    # Check if each pair contains the right platforms
    assert "instagram" in cross_platform["instagram_twitter"]
    assert "twitter" in cross_platform["instagram_twitter"]
    assert "instagram" in cross_platform["instagram_facebook"]
    assert "facebook" in cross_platform["instagram_facebook"]
    assert "twitter" in cross_platform["twitter_facebook"]
    assert "facebook" in cross_platform["twitter_facebook"]
    
    # Check if user vectors were preserved
    for platform, users in reduced_vectors.items():
        for user_id, vector in users.items():
            # Find platform pairs containing this platform
            for pair, pair_data in cross_platform.items():
                if platform in pair_data:
                    assert user_id in pair_data[platform]
                    assert np.array_equal(pair_data[platform][user_id], vector)


@patch('feature_fusion.concatenate_features')
@patch('feature_fusion.apply_dimensionality_reduction')
@patch('feature_fusion.create_cross_platform_features')
@patch('pickle.dump')
def test_fuse_features(
    mock_pickle, mock_create_cross, mock_apply_reduction, mock_concatenate,
    sample_behavior_features, sample_content_features, sample_network_features
):
    """Test fusing features from different sources for user matching."""
    # Set up mocks
    mock_concatenate.return_value = {"instagram": {"user1": np.array([1.0, 2.0])}}
    mock_apply_reduction.return_value = {"instagram": {"user1": np.array([0.5, 0.5])}}
    mock_create_cross.return_value = {"instagram_twitter": {"instagram": {"user1": np.array([0.5, 0.5])}}}
    
    # Call function - with feature dictionaries
    config = {
        "dimensionality_reduction": "pca",
        "embedding_dim": 20
    }
    
    with patch('os.makedirs') as mock_makedirs:
        fused = feature_fusion.fuse_features(
            sample_behavior_features,
            sample_content_features,
            sample_network_features,
            config
        )
    
    # Check if vectorization functions were called
    # No need to call vectorize since we're already passing feature dictionaries
    
    # Check if concatenate_features was called
    mock_concatenate.assert_called_once()
    
    # Check if apply_dimensionality_reduction was called
    mock_apply_reduction.assert_called_once()
    assert mock_apply_reduction.call_args[0][1] == config
    
    # Check if create_cross_platform_features was called
    mock_create_cross.assert_called_once()
    
    # Check if results were pickled
    assert mock_pickle.call_count == 1
    
    # Check results
    assert fused == mock_create_cross.return_value
    
    # Reset mocks
    mock_concatenate.reset_mock()
    mock_apply_reduction.reset_mock()
    mock_create_cross.reset_mock()
    mock_pickle.reset_mock()
    
    # Call function - with feature vectors
    behavior_vectors = {"instagram": {"user1": np.array([1.0, 2.0])}}
    content_vectors = {"instagram": {"user1": np.array([3.0, 4.0])}}
    network_vectors = {"twitter": {"user2": np.array([5.0, 6.0])}}
    
    with patch('os.makedirs') as mock_makedirs:
        fused = feature_fusion.fuse_features(
            behavior_vectors,
            content_vectors,
            network_vectors,
            config
        )
    
    # Check if concatenate_features was called with vectors directly
    mock_concatenate.assert_called_once_with(behavior_vectors, content_vectors, network_vectors)
    
    # Check results
    assert fused == mock_create_cross.return_value
