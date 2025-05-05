"""
Tests for the content_features module.
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import content_features


@patch('content_features.SentenceTransformer')
def test_load_embedding_model(mock_sentence_transformer):
    """Test loading pre-trained SentenceTransformer model."""
    # Set up mock
    mock_instance = MagicMock()
    mock_sentence_transformer.return_value = mock_instance
    
    # Call function
    model = content_features.load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if SentenceTransformer was called with the right model name
    mock_sentence_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if model was returned
    assert model == mock_instance


def test_generate_text_embeddings(mock_sentence_transformer):
    """Test generating sentence embeddings for a list of texts."""
    # Set up test data
    texts = ["This is a test text", "Another test text", "A third test text"]
    expected_embeddings = np.random.randn(3, 384).astype(np.float32)
    mock_sentence_transformer.encode.return_value = expected_embeddings
    
    # Call function
    embeddings = content_features.generate_text_embeddings(texts, mock_sentence_transformer, max_tokens=512)
    
    # Check if encoder was called with the right parameters
    mock_sentence_transformer.encode.assert_called_once()
    
    # Check if the correct embeddings were returned
    assert embeddings is expected_embeddings
    
    # Test with empty texts
    embeddings = content_features.generate_text_embeddings([], mock_sentence_transformer)
    assert len(embeddings) == 0
    
    # Test with invalid texts
    embeddings = content_features.generate_text_embeddings([None, 123, ""], mock_sentence_transformer)
    mock_sentence_transformer.encode.assert_called()


@patch('content_features.CountVectorizer')
@patch('content_features.LatentDirichletAllocation')
def test_train_topic_model(mock_lda, mock_vectorizer):
    """Test training an LDA topic model on a corpus of texts."""
    # Set up mocks
    mock_vectorizer_instance = MagicMock()
    mock_vectorizer.return_value = mock_vectorizer_instance
    mock_vectorizer_instance.fit_transform.return_value = "bow_matrix"
    
    mock_lda_instance = MagicMock()
    mock_lda.return_value = mock_lda_instance
    
    # Set up test data
    texts = ["This is a test text", "Another test text", "A third test text"]
    
    # Call function
    lda, vectorizer = content_features.train_topic_model(texts, n_topics=20)
    
    # Check if CountVectorizer was initialized correctly
    mock_vectorizer.assert_called_once()
    assert mock_vectorizer.call_args[1]["max_df"] == 0.95
    assert mock_vectorizer.call_args[1]["min_df"] == 2
    assert mock_vectorizer.call_args[1]["stop_words"] == "english"
    
    # Check if vectorizer was used to transform texts
    mock_vectorizer_instance.fit_transform.assert_called_once_with(texts)
    
    # Check if LDA was initialized correctly
    mock_lda.assert_called_once()
    assert mock_lda.call_args[1]["n_components"] == 20
    assert mock_lda.call_args[1]["random_state"] == 42
    
    # Check if LDA was fitted
    mock_lda_instance.fit.assert_called_once_with("bow_matrix")
    
    # Check if models were returned
    assert lda == mock_lda_instance
    assert vectorizer == mock_vectorizer_instance
    
    # Test with empty texts
    with pytest.raises(ValueError):
        content_features.train_topic_model([], n_topics=20)


def test_get_topic_distribution():
    """Test getting topic distribution for a text document."""
    # Set up mocks
    mock_lda = MagicMock()
    mock_lda.n_components = 20
    mock_lda.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4] + [0.0] * 16])
    
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = "bow_vector"
    
    # Call function
    text = "This is a test text"
    topic_dist = content_features.get_topic_distribution(text, mock_lda, mock_vectorizer)
    
    # Check if vectorizer was used to transform text
    mock_vectorizer.transform.assert_called_once_with([text])
    
    # Check if LDA was used to get topic distribution
    mock_lda.transform.assert_called_once_with("bow_vector")
    
    # Check if topic distribution was returned
    assert np.array_equal(topic_dist, np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 16))
    
    # Test with empty text
    topic_dist = content_features.get_topic_distribution("", mock_lda, mock_vectorizer)
    assert len(topic_dist) == 20
    assert np.all(topic_dist == 0)
    
    # Test with None text
    topic_dist = content_features.get_topic_distribution(None, mock_lda, mock_vectorizer)
    assert len(topic_dist) == 20
    assert np.all(topic_dist == 0)


def test_extract_content_features_for_user(mock_sentence_transformer, sample_posts):
    """Test extracting content features for a single user."""
    # Set up mocks
    user_id = "test_user"
    mock_sentence_transformer.encode.return_value = np.random.randn(3, 384).astype(np.float32)
    
    mock_lda = MagicMock()
    mock_lda.n_components = 20
    mock_lda.transform.return_value = np.random.rand(1, 20)
    
    mock_vectorizer = MagicMock()
    
    config = {"max_tokens": 512}
    
    # Call function
    features = content_features.extract_content_features_for_user(
        user_id, sample_posts, mock_sentence_transformer, mock_lda, mock_vectorizer, config
    )
    
    # Check if features were extracted
    assert features["user_id"] == user_id
    assert features["post_count"] == 3
    assert features["has_content"] == True
    assert "avg_embedding" in features
    assert "avg_topic_distribution" in features
    assert "post_embeddings" in features
    assert "topic_distributions" in features
    
    # Check specific feature properties
    assert isinstance(features["avg_embedding"], np.ndarray)
    assert isinstance(features["avg_topic_distribution"], np.ndarray)
    assert len(features["post_embeddings"]) > 0
    assert len(features["topic_distributions"]) > 0
    
    # Test with empty posts
    features = content_features.extract_content_features_for_user(
        user_id, [], mock_sentence_transformer, mock_lda, mock_vectorizer, config
    )
    assert features["post_count"] == 0
    assert features["has_content"] == False
    
    # Test without LDA model
    features = content_features.extract_content_features_for_user(
        user_id, sample_posts, mock_sentence_transformer, None, None, config
    )
    assert isinstance(features["avg_embedding"], np.ndarray)
    assert isinstance(features["avg_topic_distribution"], np.ndarray)
    assert len(features["avg_topic_distribution"]) == 0


@patch('content_features.load_embedding_model')
@patch('content_features.train_topic_model')
@patch('content_features.extract_content_features_for_user')
def test_extract_content_features_for_platform(
    mock_extract_user, mock_train_topic, mock_load_model, sample_instagram_processed_data
):
    """Test extracting content features for all users on a platform."""
    # Set up mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    mock_lda = MagicMock()
    mock_vectorizer = MagicMock()
    mock_train_topic.return_value = (mock_lda, mock_vectorizer)
    
    mock_extract_user.return_value = {
        "user_id": "test_user",
        "post_count": 2,
        "has_content": True,
        "avg_embedding": np.random.randn(384),
        "avg_topic_distribution": np.random.rand(20)
    }
    
    # Call function
    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_tokens": 512,
        "lda_topics": 20
    }
    
    features = content_features.extract_content_features_for_platform(
        sample_instagram_processed_data, config
    )
    
    # Check if model was loaded
    mock_load_model.assert_called_once_with(config["embedding_model"])
    
    # Check if topic model was trained
    mock_train_topic.assert_called_once()
    
    # Check if features were extracted for each user
    assert mock_extract_user.call_count == 2
    
    # Check results
    assert len(features) == 2
    assert all(u in features for u in sample_instagram_processed_data["posts"])
    
    # Test with empty platform data
    features = content_features.extract_content_features_for_platform({}, config)
    assert features == {}


@patch('content_features.extract_content_features_for_platform')
@patch('pickle.dump')
def test_extract_all_content_features(
    mock_pickle, mock_extract_platform, sample_processed_data
):
    """Test extracting content features for all users across all platforms."""
    # Set up mock
    def mock_extract_platform_func(data, config):
        platform = next(iter(data["users"].values()))["platform"]
        return {
            f"user1_{platform}": {
                "user_id": f"user1_{platform}",
                "post_count": 2,
                "has_content": True,
                "avg_embedding": np.random.randn(384),
                "avg_topic_distribution": np.random.rand(20)
            },
            f"user2_{platform}": {
                "user_id": f"user2_{platform}",
                "post_count": 1,
                "has_content": True,
                "avg_embedding": np.random.randn(384),
                "avg_topic_distribution": np.random.rand(20)
            }
        }
    
    mock_extract_platform.side_effect = mock_extract_platform_func
    
    # Call function
    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_tokens": 512,
        "lda_topics": 20
    }
    
    with patch('os.makedirs') as mock_makedirs:
        features = content_features.extract_all_content_features(
            sample_processed_data, config
        )
    
    # Check if extract_content_features_for_platform was called for each platform
    assert mock_extract_platform.call_count == 3
    
    # Check if features were serialized
    mock_pickle.assert_called_once()
    
    # Check results
    assert "instagram" in features
    assert "twitter" in features
    assert "facebook" in features
    assert len(features["instagram"]) == 2
    assert len(features["twitter"]) == 2
    assert len(features["facebook"]) == 2


def test_get_content_vectors(sample_content_features):
    """Test creating fixed-length content feature vectors for machine learning."""
    # Call function
    vectors = content_features.get_content_vectors(sample_content_features)
    
    # Check results
    assert "instagram" in vectors
    assert "twitter" in vectors
    assert "facebook" in vectors
    
    # Check if vectors were created for each user
    for platform, users in sample_content_features.items():
        for user_id in users:
            assert user_id in vectors[platform]
            # Check vector types and dimensions
            assert isinstance(vectors[platform][user_id], np.ndarray)
            # Vector should be embedding + topic distribution
            expected_dim = 384 + 20  # Default embedding size + topics
            assert vectors[platform][user_id].shape[0] == expected_dim
    
    # Test with missing embeddings
    modified_features = {
        "instagram": {
            "user_no_embedding": {
                "user_id": "user_no_embedding",
                "post_count": 0,
                "has_content": False
            }
        }
    }
    
    vectors = content_features.get_content_vectors(modified_features)
    assert "user_no_embedding" in vectors["instagram"]
    assert vectors["instagram"]["user_no_embedding"].shape[0] == 384  # Default embedding size
