"""
Network Features Module

This module extracts network features from social graphs,
including centrality metrics, community detection, and ego-network statistics.
"""

import json
import logging
import os
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Union, Optional
import pickle
from pathlib import Path
import community as community_louvain  # python-louvain package

logger = logging.getLogger(__name__)


def load_network_graph(platform: str, processed_data_dir: str) -> Optional[nx.Graph]:
    """
    Load network graph for a platform from edgelist file.
    
    Args:
        platform: Social platform name
        processed_data_dir: Directory containing processed data
        
    Returns:
        nx.Graph: NetworkX graph object or None if file not found
    """
    edgelist_path = os.path.join(processed_data_dir, platform, "network.edgelist")
    
    if not os.path.exists(edgelist_path):
        logger.warning(f"No network edgelist found for {platform}")
        return None
    
    try:
        # Determine if graph is directed based on platform
        directed = platform != "facebook"  # Twitter is directed, Facebook is undirected
        
        if directed:
            G = nx.read_edgelist(edgelist_path, create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(edgelist_path, create_using=nx.Graph())
            
        logger.info(f"Loaded {platform} graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    except Exception as e:
        logger.error(f"Error loading {platform} network graph: {e}")
        return None


def compute_centrality_metrics(
    G: nx.Graph, 
    metrics: List[str], 
    max_nodes: int = 10000
) -> Dict[str, Dict[str, float]]:
    """
    Compute centrality metrics for nodes in a graph.
    
    Args:
        G: NetworkX graph
        metrics: List of centrality metrics to compute
        max_nodes: Maximum number of nodes for computationally expensive metrics
        
    Returns:
        Dict: Dictionary of centrality metrics for each node
    """
    centrality_metrics = {}
    
    # Check if graph is too large for some metrics
    is_large_graph = G.number_of_nodes() > max_nodes
    
    for metric in metrics:
        try:
            if metric == "degree":
                if isinstance(G, nx.DiGraph):
                    # For directed graphs, compute in and out degree
                    in_degree = dict(G.in_degree())
                    out_degree = dict(G.out_degree())
                    centrality_metrics["in_degree"] = in_degree
                    centrality_metrics["out_degree"] = out_degree
                    
                    # Total degree
                    centrality_metrics["degree"] = {
                        node: in_degree.get(node, 0) + out_degree.get(node, 0)
                        for node in G.nodes()
                    }
                else:
                    centrality_metrics["degree"] = dict(G.degree())
            
            elif metric == "eigenvector" and not is_large_graph:
                # Skip eigenvector centrality for large graphs
                centrality_metrics["eigenvector"] = nx.eigenvector_centrality_numpy(G, max_iter=100)
            
            elif metric == "betweenness" and not is_large_graph:
                # For very large graphs, use approximate betweenness
                if G.number_of_nodes() > 1000:
                    # Sample k nodes for approximation
                    k = min(1000, G.number_of_nodes())
                    centrality_metrics["betweenness"] = nx.betweenness_centrality(G, k=k)
                else:
                    centrality_metrics["betweenness"] = nx.betweenness_centrality(G)
            
            elif metric == "closeness" and not is_large_graph:
                # Skip closeness centrality for large graphs
                centrality_metrics["closeness"] = nx.closeness_centrality(G)
            
            elif metric == "pagerank":
                centrality_metrics["pagerank"] = nx.pagerank(G)
            
        except Exception as e:
            logger.error(f"Error computing {metric} centrality: {e}")
    
    return centrality_metrics


def detect_communities(G: nx.Graph, algorithm: str = "louvain") -> Dict[str, int]:
    """
    Detect communities in a graph.
    
    Args:
        G: NetworkX graph
        algorithm: Community detection algorithm
        
    Returns:
        Dict: Node to community mapping
    """
    if algorithm == "louvain":
        try:
            # Convert directed graph to undirected for Louvain
            if isinstance(G, nx.DiGraph):
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
                
            # Apply Louvain community detection
            partition = community_louvain.best_partition(G_undirected)
            return partition
        
        except Exception as e:
            logger.error(f"Error detecting communities with Louvain: {e}")
            return {}
    
    elif algorithm == "label_propagation":
        try:
            communities = nx.algorithms.community.label_propagation_communities(G)
            # Convert to node:community_id format
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            return partition
        
        except Exception as e:
            logger.error(f"Error detecting communities with label propagation: {e}")
            return {}
    
    else:
        logger.error(f"Unsupported community detection algorithm: {algorithm}")
        return {}


def extract_ego_network_stats(
    G: nx.Graph, 
    node: str, 
    radius: int = 1
) -> Dict[str, Any]:
    """
    Extract statistics for a node's ego network.
    
    Args:
        G: NetworkX graph
        node: Target node
        radius: Radius of ego network
        
    Returns:
        Dict: Ego network statistics
    """
    try:
        # Extract the ego network
        ego_network = nx.ego_graph(G, node, radius=radius)
        
        # Skip trivial ego networks
        if ego_network.number_of_nodes() <= 1:
            return {
                "size": 1,
                "density": 0.0,
                "clustering": 0.0,
                "avg_path_length": 0.0
            }
        
        # Calculate basic stats
        stats = {
            "size": ego_network.number_of_nodes(),
            "edge_count": ego_network.number_of_edges(),
            "density": nx.density(ego_network)
        }
        
        # Calculate clustering coefficient
        try:
            stats["clustering"] = nx.average_clustering(ego_network)
        except:
            stats["clustering"] = 0.0
        
        # Calculate average path length (for connected components)
        try:
            # Check if ego network is connected
            if nx.is_connected(ego_network.to_undirected()):
                stats["avg_path_length"] = nx.average_shortest_path_length(ego_network.to_undirected())
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(ego_network.to_undirected()), key=len)
                largest_subgraph = ego_network.subgraph(largest_cc).to_undirected()
                stats["avg_path_length"] = nx.average_shortest_path_length(largest_subgraph)
        except:
            stats["avg_path_length"] = 0.0
        
        return stats
    
    except Exception as e:
        logger.error(f"Error extracting ego network stats for node {node}: {e}")
        return {
            "size": 0,
            "density": 0.0,
            "clustering": 0.0,
            "avg_path_length": 0.0
        }


def extract_network_features_for_user(
    user_id: str, 
    G: nx.Graph, 
    centrality_metrics: Dict[str, Dict[str, float]],
    communities: Dict[str, int],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract network features for a single user.
    
    Args:
        user_id: Anonymized user ID
        G: NetworkX graph
        centrality_metrics: Pre-computed centrality metrics
        communities: Community assignments
        config: Network feature configuration
        
    Returns:
        Dict: Network features for the user
    """
    logger.info(f"Extracting network features for user {user_id}")
    
    # Check if user exists in graph
    if user_id not in G:
        logger.warning(f"User {user_id} not found in graph")
        return {
            "user_id": user_id,
            "in_graph": False
        }
    
    # Extract centrality values for user
    centrality_values = {}
    for metric, values in centrality_metrics.items():
        centrality_values[metric] = values.get(user_id, 0.0)
    
    # Get community assignment
    community_id = communities.get(user_id, -1)
    
    # Extract ego network statistics
    ego_stats = extract_ego_network_stats(G, user_id, radius=1)
    
    # Compute neighbor statistics
    neighbors = list(G.neighbors(user_id))
    neighbor_count = len(neighbors)
    
    # For directed graphs, differentiate between in- and out-neighbors
    if isinstance(G, nx.DiGraph):
        in_neighbors = list(G.predecessors(user_id))
        out_neighbors = list(G.successors(user_id))
        in_neighbor_count = len(in_neighbors)
        out_neighbor_count = len(out_neighbors)
    else:
        in_neighbors = neighbors
        out_neighbors = neighbors
        in_neighbor_count = neighbor_count
        out_neighbor_count = neighbor_count
    
    # Get community diversity of neighbors
    neighbor_communities = [communities.get(n, -1) for n in neighbors]
    unique_communities = set(neighbor_communities) - {-1}  # Exclude unknown
    community_diversity = len(unique_communities)
    
    # Calculate reciprocity for directed graphs
    if isinstance(G, nx.DiGraph):
        # Count reciprocal connections
        reciprocal_count = sum(1 for n in out_neighbors if n in in_neighbors)
        reciprocity = reciprocal_count / out_neighbor_count if out_neighbor_count > 0 else 0.0
    else:
        reciprocity = 1.0  # All connections are reciprocal in undirected graphs
    
    # Combine all features
    features = {
        "user_id": user_id,
        "in_graph": True,
        "centrality": centrality_values,
        "community_id": community_id,
        "ego_network": ego_stats,
        "neighbor_count": neighbor_count,
        "in_neighbor_count": in_neighbor_count,
        "out_neighbor_count": out_neighbor_count,
        "community_diversity": community_diversity,
        "reciprocity": reciprocity
    }
    
    return features


def extract_network_features_for_platform(
    platform: str,
    processed_data_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract network features for all users on a platform.
    
    Args:
        platform: Social platform name
        processed_data_dir: Directory containing processed data
        config: Network feature configuration
        
    Returns:
        Dict: Network features for all users on the platform
    """
    # Load network graph
    G = load_network_graph(platform, processed_data_dir)
    
    if G is None:
        logger.warning(f"No network graph available for {platform}")
        return {}
    
    # Compute centrality metrics
    centrality_metrics = compute_centrality_metrics(
        G, 
        config.get("centrality_metrics", ["degree", "betweenness", "closeness", "eigenvector"])
    )
    
    # Detect communities
    communities = detect_communities(G, config.get("community_algorithm", "louvain"))
    
    # Extract features for each user
    users_features = {}
    
    for user_id in G.nodes():
        users_features[user_id] = extract_network_features_for_user(
            user_id, G, centrality_metrics, communities, config
        )
    
    return users_features


def extract_all_network_features(
    processed_data: Dict[str, Dict[str, Any]], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Extract network features for all users across all platforms.
    
    Args:
        processed_data: Dictionary of processed data by platform
        config: Network feature configuration
        
    Returns:
        Dict: Network features for all users across all platforms
    """
    all_features = {}
    processed_data_dir = "processed_data"  # Default directory
    
    for platform in processed_data.keys():
        logger.info(f"Extracting network features for {platform}")
        all_features[platform] = extract_network_features_for_platform(
            platform, processed_data_dir, config
        )
    
    # Save all features
    os.makedirs(processed_data_dir, exist_ok=True)
    
    with open(os.path.join(processed_data_dir, "network_features.pkl"), "wb") as f:
        pickle.dump(all_features, f)
    
    logger.info(f"Extracted network features for {sum(len(users) for users in all_features.values())} users")
    return all_features


def vectorize_network_features(all_features: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert network features to fixed-length vectors for machine learning.
    
    Args:
        all_features: Network features for all users across all platforms
        
    Returns:
        Dict: Fixed-length feature vectors for all users
    """
    vector_features = {}
    
    for platform, users in all_features.items():
        vector_features[platform] = {}
        
        for user_id, features in users.items():
            # Skip users not in graph
            if not features.get("in_graph", False):
                continue
                
            # Create a fixed-length vector for each user
            feature_vector = []
            
            # Add centrality metrics
            centrality = features.get("centrality", {})
            feature_vector.append(centrality.get("degree", 0.0))
            feature_vector.append(centrality.get("in_degree", 0.0))
            feature_vector.append(centrality.get("out_degree", 0.0))
            feature_vector.append(centrality.get("betweenness", 0.0))
            feature_vector.append(centrality.get("closeness", 0.0))
            feature_vector.append(centrality.get("eigenvector", 0.0))
            feature_vector.append(centrality.get("pagerank", 0.0))
            
            # Add ego network statistics
            ego_network = features.get("ego_network", {})
            feature_vector.append(ego_network.get("size", 0))
            feature_vector.append(ego_network.get("density", 0.0))
            feature_vector.append(ego_network.get("clustering", 0.0))
            feature_vector.append(ego_network.get("avg_path_length", 0.0))
            
            # Add neighbor statistics
            feature_vector.append(features.get("neighbor_count", 0))
            feature_vector.append(features.get("in_neighbor_count", 0))
            feature_vector.append(features.get("out_neighbor_count", 0))
            feature_vector.append(features.get("community_diversity", 0))
            feature_vector.append(features.get("reciprocity", 0.0))
            
            # Add community ID as one-hot feature (simplified, max 10 communities)
            community_id = features.get("community_id", -1)
            if community_id >= 0:
                community_idx = min(community_id, 9)  # Cap at 10 communities
                community_vector = [0] * 10
                community_vector[community_idx] = 1
                feature_vector.extend(community_vector)
            else:
                feature_vector.extend([0] * 10)
            
            # Convert to numpy array
            vector_features[platform][user_id] = np.array(feature_vector, dtype=np.float32)
    
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
    
    # Extract network features
    extract_all_network_features(processed_data, config["features"]["network"])
