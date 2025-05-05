"""
Tests for the network_features module.
"""

import os
import json
import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock
import network_features


def test_load_network_graph(tmpdir):
    """Test loading network graph from edgelist file."""
    # Create test edgelist file for Twitter (directed)
    twitter_dir = os.path.join(tmpdir, "twitter")
    os.makedirs(twitter_dir, exist_ok=True)
    twitter_edgelist = os.path.join(twitter_dir, "network.edgelist")
    
    with open(twitter_edgelist, 'w') as f:
        f.write("user1 user2\n")
        f.write("user3 user1\n")
        f.write("user2 user4\n")
    
    # Create test edgelist file for Facebook (undirected)
    facebook_dir = os.path.join(tmpdir, "facebook")
    os.makedirs(facebook_dir, exist_ok=True)
    facebook_edgelist = os.path.join(facebook_dir, "network.edgelist")
    
    with open(facebook_edgelist, 'w') as f:
        f.write("user1 user2\n")
        f.write("user1 user3\n")
        f.write("user2 user4\n")
    
    # Test loading Twitter graph
    twitter_graph = network_features.load_network_graph("twitter", str(tmpdir))
    assert isinstance(twitter_graph, nx.DiGraph)
    assert twitter_graph.number_of_nodes() == 4
    assert twitter_graph.number_of_edges() == 3
    assert twitter_graph.has_edge("user1", "user2")
    
    # Test loading Facebook graph
    facebook_graph = network_features.load_network_graph("facebook", str(tmpdir))
    assert isinstance(facebook_graph, nx.Graph)
    assert facebook_graph.number_of_nodes() == 4
    assert facebook_graph.number_of_edges() == 3
    assert facebook_graph.has_edge("user1", "user2")
    
    # Test with non-existent file
    missing_graph = network_features.load_network_graph("missing", str(tmpdir))
    assert missing_graph is None


def test_compute_centrality_metrics():
    """Test computing centrality metrics for nodes in a graph."""
    # Create a test graph
    G = nx.DiGraph()
    G.add_edges_from([
        ("user1", "user2"),
        ("user3", "user1"),
        ("user2", "user4"),
        ("user4", "user3")
    ])
    
    # Test with directed graph and all metrics
    metrics = network_features.compute_centrality_metrics(
        G, ["degree", "eigenvector", "betweenness", "closeness", "pagerank"]
    )
    
    # Check metrics
    assert "in_degree" in metrics
    assert "out_degree" in metrics
    assert "degree" in metrics
    assert "eigenvector" in metrics
    assert "betweenness" in metrics
    assert "closeness" in metrics
    assert "pagerank" in metrics
    
    # Check if metrics were computed for all nodes
    for node in G.nodes():
        assert node in metrics["in_degree"]
        assert node in metrics["out_degree"]
        assert node in metrics["degree"]
        assert node in metrics["eigenvector"]
        assert node in metrics["betweenness"]
        assert node in metrics["closeness"]
        assert node in metrics["pagerank"]
    
    # Check specific metrics
    assert metrics["in_degree"]["user1"] == 1
    assert metrics["out_degree"]["user1"] == 1
    assert metrics["degree"]["user1"] == 2
    
    # Test with undirected graph
    G_undirected = nx.Graph()
    G_undirected.add_edges_from([
        ("user1", "user2"),
        ("user1", "user3"),
        ("user2", "user4")
    ])
    
    metrics_undirected = network_features.compute_centrality_metrics(
        G_undirected, ["degree", "eigenvector"]
    )
    
    # Check metrics
    assert "degree" in metrics_undirected
    assert "eigenvector" in metrics_undirected
    
    # Check if metrics were computed for all nodes
    for node in G_undirected.nodes():
        assert node in metrics_undirected["degree"]
        assert node in metrics_undirected["eigenvector"]
    
    # Check specific metrics
    assert metrics_undirected["degree"]["user1"] == 2
    
    # Test with large graph and skipping expensive metrics
    large_G = nx.complete_graph(1001)
    metrics_large = network_features.compute_centrality_metrics(
        large_G, ["degree", "eigenvector", "betweenness", "closeness"]
    )
    
    # Only degree should be computed for large graph
    assert "degree" in metrics_large
    assert "eigenvector" not in metrics_large
    assert "betweenness" not in metrics_large
    assert "closeness" not in metrics_large


@patch('network_features.community_louvain.best_partition')
def test_detect_communities(mock_best_partition):
    """Test detecting communities in a graph."""
    # Set up mock
    mock_best_partition.return_value = {
        "user1": 0,
        "user2": 0,
        "user3": 1,
        "user4": 1
    }
    
    # Create a test graph
    G = nx.Graph()
    G.add_edges_from([
        ("user1", "user2"),
        ("user3", "user4"),
        ("user2", "user3")
    ])
    
    # Test Louvain algorithm
    partition = network_features.detect_communities(G, algorithm="louvain")
    mock_best_partition.assert_called_once()
    
    # Check partition
    assert partition["user1"] == 0
    assert partition["user2"] == 0
    assert partition["user3"] == 1
    assert partition["user4"] == 1
    
    # Test with directed graph
    G_directed = nx.DiGraph()
    G_directed.add_edges_from([
        ("user1", "user2"),
        ("user3", "user1"),
        ("user2", "user4")
    ])
    
    # Reset mock
    mock_best_partition.reset_mock()
    
    partition = network_features.detect_communities(G_directed, algorithm="louvain")
    mock_best_partition.assert_called_once()
    
    # Test with unsupported algorithm
    partition = network_features.detect_communities(G, algorithm="unknown")
    assert partition == {}


def test_extract_ego_network_stats():
    """Test extracting statistics for a node's ego network."""
    # Create a test graph
    G = nx.Graph()
    G.add_edges_from([
        ("user1", "user2"),
        ("user1", "user3"),
        ("user1", "user4"),
        ("user2", "user3"),
        ("user4", "user5")
    ])
    
    # Test ego network with radius 1
    stats = network_features.extract_ego_network_stats(G, "user1", radius=1)
    
    # Check stats
    assert stats["size"] > 1
    assert stats["edge_count"] > 0
    assert 0 <= stats["density"] <= 1
    assert 0 <= stats["clustering"] <= 1
    assert stats["avg_path_length"] > 0
    
    # Test with isolated node
    G.add_node("isolated")
    stats = network_features.extract_ego_network_stats(G, "isolated", radius=1)
    
    # Should have minimal stats
    assert stats["size"] == 1
    assert stats["density"] == 0.0
    assert stats["clustering"] == 0.0
    assert stats["avg_path_length"] == 0.0
    
    # Test with disconnected ego network
    G = nx.Graph()
    G.add_edges_from([
        ("user1", "user2"),
        ("user1", "user3"),
        ("user4", "user5")  # Disconnected from user1's component
    ])
    
    stats = network_features.extract_ego_network_stats(G, "user1", radius=2)
    
    # Should compute paths only for the largest connected component
    assert stats["avg_path_length"] > 0
    
    # Test with directed graph
    G = nx.DiGraph()
    G.add_edges_from([
        ("user1", "user2"),
        ("user3", "user1"),
        ("user2", "user4")
    ])
    
    stats = network_features.extract_ego_network_stats(G, "user1", radius=1)
    
    # Should work with directed graph
    assert stats["size"] > 1
    assert stats["edge_count"] > 0


def test_extract_network_features_for_user(sample_twitter_processed_data):
    """Test extracting network features for a single user."""
    # Get test data
    G = sample_twitter_processed_data["networks"]["graph"]
    user_id = "anon_user1_twitter"
    
    # Set up mock centrality metrics
    centrality_metrics = {
        "degree": {user_id: 2},
        "in_degree": {user_id: 2},
        "out_degree": {user_id: 1},
        "betweenness": {user_id: 0.5},
        "closeness": {user_id: 0.7},
        "eigenvector": {user_id: 0.8},
        "pagerank": {user_id: 0.3}
    }
    
    # Set up mock communities
    communities = {
        user_id: 0,
        "anon_user2_twitter": 0,
        "follower1": 1,
        "follower2": 1
    }
    
    # Test with valid user
    with patch('network_features.extract_ego_network_stats') as mock_ego_stats:
        mock_ego_stats.return_value = {
            "size": 4,
            "density": 0.5,
            "clustering": 0.3,
            "avg_path_length": 1.5
        }
        
        features = network_features.extract_network_features_for_user(
            user_id, G, centrality_metrics, communities, {}
        )
    
    # Check features
    assert features["user_id"] == user_id
    assert features["in_graph"] == True
    
    # Check centrality metrics
    assert features["centrality"]["degree"] == 2
    assert features["centrality"]["in_degree"] == 2
    assert features["centrality"]["out_degree"] == 1
    assert features["centrality"]["betweenness"] == 0.5
    assert features["centrality"]["closeness"] == 0.7
    assert features["centrality"]["eigenvector"] == 0.8
    assert features["centrality"]["pagerank"] == 0.3
    
    # Check community
    assert features["community_id"] == 0
    
    # Check ego network
    assert features["ego_network"]["size"] == 4
    assert features["ego_network"]["density"] == 0.5
    assert features["ego_network"]["clustering"] == 0.3
    assert features["ego_network"]["avg_path_length"] == 1.5
    
    # Check neighbor counts
    assert features["neighbor_count"] >= 0
    assert features["in_neighbor_count"] >= 0
    assert features["out_neighbor_count"] >= 0
    
    # Test with user not in graph
    features = network_features.extract_network_features_for_user(
        "nonexistent_user", G, centrality_metrics, communities, {}
    )
    
    assert features["user_id"] == "nonexistent_user"
    assert features["in_graph"] == False


@patch('network_features.load_network_graph')
@patch('network_features.compute_centrality_metrics')
@patch('network_features.detect_communities')
@patch('network_features.extract_network_features_for_user')
def test_extract_network_features_for_platform(
    mock_extract_user, mock_detect_communities, mock_compute_centrality, mock_load_graph
):
    """Test extracting network features for all users on a platform."""
    # Set up mocks
    G = nx.DiGraph()
    G.add_nodes_from(["user1", "user2", "user3"])
    G.add_edges_from([("user1", "user2"), ("user3", "user1")])
    
    mock_load_graph.return_value = G
    
    mock_compute_centrality.return_value = {
        "degree": {"user1": 2, "user2": 1, "user3": 1},
        "in_degree": {"user1": 1, "user2": 1, "user3": 0},
        "out_degree": {"user1": 1, "user2": 0, "user3": 1}
    }
    
    mock_detect_communities.return_value = {
        "user1": 0, "user2": 0, "user3": 1
    }
    
    mock_extract_user.side_effect = lambda user_id, G, centrality, communities, config: {
        "user_id": user_id,
        "in_graph": True,
        "centrality": {"degree": centrality["degree"].get(user_id, 0)},
        "community_id": communities.get(user_id, -1)
    }
    
    # Call function
    config = {
        "centrality_metrics": ["degree", "betweenness", "closeness", "eigenvector"],
        "community_algorithm": "louvain"
    }
    
    features = network_features.extract_network_features_for_platform(
        "twitter", "processed_data", config
    )
    
    # Check if graph was loaded
    mock_load_graph.assert_called_once_with("twitter", "processed_data")
    
    # Check if centrality metrics were computed
    mock_compute_centrality.assert_called_once()
    assert mock_compute_centrality.call_args[0][1] == config["centrality_metrics"]
    
    # Check if communities were detected
    mock_detect_communities.assert_called_once()
    assert mock_detect_communities.call_args[0][1] == config["community_algorithm"]
    
    # Check if features were extracted for each user
    assert mock_extract_user.call_count == 3
    
    # Check results
    assert len(features) == 3
    assert "user1" in features
    assert "user2" in features
    assert "user3" in features
    
    # Test with missing graph
    mock_load_graph.return_value = None
    
    features = network_features.extract_network_features_for_platform(
        "missing", "processed_data", config
    )
    
    assert features == {}


@patch('network_features.extract_network_features_for_platform')
@patch('pickle.dump')
def test_extract_all_network_features(mock_pickle, mock_extract_platform, sample_processed_data):
    """Test extracting network features for all users across all platforms."""
    # Set up mock
    def mock_extract_platform_func(platform, processed_data_dir, config):
        if platform == "twitter":
            return {
                "anon_user1_twitter": {
                    "user_id": "anon_user1_twitter",
                    "in_graph": True,
                    "centrality": {"degree": 2}
                },
                "anon_user2_twitter": {
                    "user_id": "anon_user2_twitter",
                    "in_graph": True,
                    "centrality": {"degree": 1}
                }
            }
        elif platform == "facebook":
            return {
                "anon_user1_facebook": {
                    "user_id": "anon_user1_facebook",
                    "in_graph": True,
                    "centrality": {"degree": 3}
                },
                "anon_user2_facebook": {
                    "user_id": "anon_user2_facebook",
                    "in_graph": True,
                    "centrality": {"degree": 1}
                }
            }
        else:
            return {}
    
    mock_extract_platform.side_effect = mock_extract_platform_func
    
    # Call function
    config = {
        "centrality_metrics": ["degree", "betweenness", "closeness", "eigenvector"],
        "community_algorithm": "louvain"
    }
    
    with patch('os.makedirs') as mock_makedirs:
        features = network_features.extract_all_network_features(
            sample_processed_data, config
        )
    
    # Check if extract_network_features_for_platform was called for each platform
    assert mock_extract_platform.call_count == 3
    
    # Check if features were serialized
    mock_pickle.assert_called_once()
    
    # Check results
    assert "instagram" in features  # Should have empty results for Instagram
    assert "twitter" in features
    assert "facebook" in features
    assert features["instagram"] == {}  # Instagram has no graph
    assert len(features["twitter"]) == 2
    assert len(features["facebook"]) == 2


def test_vectorize_network_features(sample_network_features):
    """Test converting network features to fixed-length vectors."""
    # Call function
    vectors = network_features.vectorize_network_features(sample_network_features)
    
    # Check results
    assert "twitter" in vectors
    assert "facebook" in vectors
    
    # Check if vectors were created for each user
    for platform, users in sample_network_features.items():
        for user_id in users:
            assert user_id in vectors[platform]
            assert isinstance(vectors[platform][user_id], np.ndarray)
            assert vectors[platform][user_id].dtype == np.float32
    
    # Check vector dimensions (should be consistent)
    twitter_user = next(iter(vectors["twitter"]))
    facebook_user = next(iter(vectors["facebook"]))
    assert vectors["twitter"][twitter_user].shape == vectors["facebook"][facebook_user].shape
    
    # Test with missing data
    modified_features = {
        "twitter": {
            "user_not_in_graph": {
                "user_id": "user_not_in_graph",
                "in_graph": False
            }
        }
    }
    
    vectors = network_features.vectorize_network_features(modified_features)
    assert "twitter" in vectors
    assert len(vectors["twitter"]) == 0  # User should be skipped
