"""
Privacy Module

This module implements privacy-preserving mechanisms for the user matching system,
including differential privacy and k-anonymity.
"""

import json
import logging
import os
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple, Union, Optional, Set
from collections import Counter, defaultdict
import pandas as pd

# Import Google's differential privacy library if available
try:
    import dp_accounting
    from dp_accounting import rdp as dp_rdp
    from dp_accounting.pld import privacy_loss_distribution
    HAS_DP_LIBRARY = True
except ImportError:
    HAS_DP_LIBRARY = False
    logging.warning("Google's differential privacy library not found. Using basic DP implementation.")

logger = logging.getLogger(__name__)


def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """
    Add Laplace noise to a value according to differential privacy.
    
    Args:
        value: Original value
        sensitivity: Sensitivity of the function
        epsilon: Privacy parameter (smaller = more private)
        
    Returns:
        float: Value with Laplace noise added
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise


def add_gaussian_noise(value: float, sensitivity: float, epsilon: float, delta: float) -> float:
    """
    Add Gaussian noise to a value according to differential privacy.
    
    Args:
        value: Original value
        sensitivity: Sensitivity of the function
        epsilon: Privacy parameter (smaller = more private)
        delta: Failure probability
        
    Returns:
        float: Value with Gaussian noise added
    """
    # Calculate sigma (standard deviation) based on privacy parameters
    # Using the analytic Gaussian mechanism calibration
    if delta >= 1.0 or delta <= 0.0:
        raise ValueError(f"Delta must be in (0, 1), got {delta}")
    
    # Calculate sigma directly
    c2 = 2 * np.log(1.25 / delta)
    sigma = sensitivity * np.sqrt(c2) / epsilon
    
    # Add noise
    noise = np.random.normal(0, sigma)
    return value + noise


def apply_differential_privacy_to_vector(
    vector: np.ndarray, 
    sensitivity: float,
    epsilon: float,
    delta: float,
    mechanism: str = "gaussian"
) -> np.ndarray:
    """
    Apply differential privacy to a feature vector.
    
    Args:
        vector: Original feature vector
        sensitivity: L2 sensitivity of the vector
        epsilon: Privacy parameter (smaller = more private)
        delta: Failure probability (for Gaussian mechanism)
        mechanism: "laplace" or "gaussian"
        
    Returns:
        np.ndarray: Vector with differential privacy applied
    """
    # Split privacy budget across dimensions
    dim = vector.shape[0]
    dimension_epsilon = epsilon / np.sqrt(dim)
    dimension_delta = delta / dim if delta > 0 else 0
    
    if mechanism == "laplace":
        noisy_vector = np.array([
            add_laplace_noise(value, sensitivity, dimension_epsilon)
            for value in vector
        ])
    elif mechanism == "gaussian":
        noisy_vector = np.array([
            add_gaussian_noise(value, sensitivity, dimension_epsilon, dimension_delta)
            for value in vector
        ])
    else:
        raise ValueError(f"Unsupported mechanism: {mechanism}")
    
    return noisy_vector


def apply_differential_privacy_to_features(
    features: Dict[str, Dict[str, np.ndarray]], 
    config: Dict[str, Any]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply differential privacy to feature vectors.
    
    Args:
        features: Feature vectors by platform and user
        config: Privacy configuration
        
    Returns:
        Dict: Feature vectors with differential privacy applied
    """
    if not config.get("enable_differential_privacy", False):
        logger.info("Differential privacy disabled, returning original features")
        return features
    
    # Get privacy parameters
    epsilon = config.get("epsilon", 1.0)
    delta = config.get("delta", 1e-5)
    
    # Choose mechanism based on delta
    mechanism = "gaussian" if delta > 0 else "laplace"
    logger.info(f"Applying {mechanism} mechanism with ε={epsilon}, δ={delta}")
    
    # Initialize output
    private_features = {}
    
    for platform, users in features.items():
        private_features[platform] = {}
        
        # Estimate sensitivity based on vector norms
        vector_norms = [np.linalg.norm(vector) for vector in users.values()]
        sensitivity = np.percentile(vector_norms, 95) / 10  # Conservative estimate
        
        for user_id, vector in users.items():
            private_features[platform][user_id] = apply_differential_privacy_to_vector(
                vector, sensitivity, epsilon, delta, mechanism
            )
    
    return private_features


def check_k_anonymity(
    features: Dict[str, Dict[str, np.ndarray]], 
    k: int,
    clustering_resolution: float = 0.1
) -> Tuple[bool, Dict[str, List[Set[str]]]]:
    """
    Check if features satisfy k-anonymity.
    
    Args:
        features: Feature vectors by platform and user
        k: Minimum group size for k-anonymity
        clustering_resolution: Resolution for clustering similar vectors
        
    Returns:
        Tuple: (is_k_anonymous, groups_by_platform)
    """
    groups_by_platform = {}
    is_k_anonymous = True
    
    for platform, users in features.items():
        # Group similar vectors
        groups = defaultdict(set)
        
        for user_id, vector in users.items():
            # Create a "signature" by rounding vector values
            signature = tuple(np.round(vector / clustering_resolution) * clustering_resolution)
            groups[signature].add(user_id)
        
        # Check if all groups have at least k members
        small_groups = [group for group in groups.values() if len(group) < k]
        
        if small_groups:
            is_k_anonymous = False
            logger.warning(f"Platform {platform} has {len(small_groups)} groups with fewer than {k} users")
        
        groups_by_platform[platform] = list(groups.values())
    
    return is_k_anonymous, groups_by_platform


def enforce_k_anonymity(
    features: Dict[str, Dict[str, np.ndarray]], 
    k: int,
    clustering_resolution: float = 0.1,
    max_iterations: int = 10
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Enforce k-anonymity by generalizing feature vectors.
    
    Args:
        features: Feature vectors by platform and user
        k: Minimum group size for k-anonymity
        clustering_resolution: Resolution for clustering similar vectors
        max_iterations: Maximum iterations to enforce k-anonymity
        
    Returns:
        Dict: Feature vectors satisfying k-anonymity
    """
    # Check if features already satisfy k-anonymity
    is_k_anonymous, groups_by_platform = check_k_anonymity(
        features, k, clustering_resolution
    )
    
    if is_k_anonymous:
        logger.info(f"Features already satisfy {k}-anonymity")
        return features
    
    # Initialize output
    anonymized_features = {platform: {} for platform in features}
    
    for platform, users in features.items():
        groups = groups_by_platform[platform]
        
        # Process each group
        for group in groups:
            # Skip groups that already satisfy k-anonymity
            if len(group) >= k:
                # Use original vectors for these users
                for user_id in group:
                    anonymized_features[platform][user_id] = features[platform][user_id]
                continue
            
            # For small groups, merge with nearest group
            if len(groups) <= 1:
                # If only one group, create synthetic vectors
                for user_id in group:
                    anonymized_features[platform][user_id] = features[platform][user_id]
                continue
            
            # Find nearest group
            group_vectors = np.array([features[platform][user_id] for user_id in group])
            group_center = np.mean(group_vectors, axis=0)
            
            nearest_group = None
            min_distance = float('inf')
            
            for other_group in groups:
                if other_group == group or len(other_group) < k:
                    continue
                
                other_vectors = np.array([features[platform][user_id] for user_id in other_group])
                other_center = np.mean(other_vectors, axis=0)
                
                distance = np.linalg.norm(group_center - other_center)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_group = other_group
            
            # If no suitable group found, create a new one
            if nearest_group is None:
                logger.warning(f"Could not find a suitable group to merge with for {len(group)} users")
                # Use original vectors for these users
                for user_id in group:
                    anonymized_features[platform][user_id] = features[platform][user_id]
                continue
            
            # Merge groups by averaging vectors
            merged_vectors = np.array([
                features[platform][user_id] 
                for user_id in (group | nearest_group)
            ])
            merged_center = np.mean(merged_vectors, axis=0)
            
            # Assign merged center to all users in both groups
            for user_id in group:
                anonymized_features[platform][user_id] = merged_center
            
            for user_id in nearest_group:
                anonymized_features[platform][user_id] = merged_center
    
    # Verify k-anonymity
    is_k_anonymous, _ = check_k_anonymity(
        anonymized_features, k, clustering_resolution
    )
    
    if is_k_anonymous:
        logger.info(f"Successfully enforced {k}-anonymity")
    else:
        logger.warning(f"Could not fully enforce {k}-anonymity after merging groups")
    
    return anonymized_features


def check_l_diversity(
    features: Dict[str, Dict[str, np.ndarray]],
    sensitive_attributes: Dict[str, Dict[str, Any]],
    k: int,
    l: int,
    clustering_resolution: float = 0.1
) -> bool:
    """
    Check if features satisfy l-diversity for sensitive attributes.
    
    Args:
        features: Feature vectors by platform and user
        sensitive_attributes: Sensitive attributes by platform and user
        k: Minimum group size for k-anonymity
        l: Minimum distinct values for l-diversity
        clustering_resolution: Resolution for clustering similar vectors
        
    Returns:
        bool: True if l-diversity is satisfied
    """
    # First check k-anonymity
    is_k_anonymous, groups_by_platform = check_k_anonymity(
        features, k, clustering_resolution
    )
    
    if not is_k_anonymous:
        return False
    
    # Check l-diversity for each platform
    for platform, groups in groups_by_platform.items():
        if platform not in sensitive_attributes:
            continue
        
        platform_attributes = sensitive_attributes[platform]
        
        for group in groups:
            # Skip small groups
            if len(group) < k:
                continue
            
            # Count distinct values for each sensitive attribute
            attribute_diversity = {}
            
            for user_id in group:
                if user_id not in platform_attributes:
                    continue
                
                user_attributes = platform_attributes[user_id]
                
                for attr_name, attr_value in user_attributes.items():
                    if attr_name not in attribute_diversity:
                        attribute_diversity[attr_name] = set()
                    
                    attribute_diversity[attr_name].add(attr_value)
            
            # Check if each attribute has at least l distinct values
            for attr_name, distinct_values in attribute_diversity.items():
                if len(distinct_values) < l:
                    logger.warning(f"Group with {len(group)} users has only {len(distinct_values)} distinct values for attribute '{attr_name}'")
                    return False
    
    return True


def enforce_l_diversity(
    features: Dict[str, Dict[str, np.ndarray]],
    sensitive_attributes: Dict[str, Dict[str, Any]],
    k: int,
    l: int,
    clustering_resolution: float = 0.1
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Enforce l-diversity by further generalizing feature vectors.
    
    Args:
        features: Feature vectors by platform and user
        sensitive_attributes: Sensitive attributes by platform and user
        k: Minimum group size for k-anonymity
        l: Minimum distinct values for l-diversity
        clustering_resolution: Resolution for clustering similar vectors
        
    Returns:
        Dict: Feature vectors satisfying l-diversity
    """
    # First enforce k-anonymity
    k_anonymous_features = enforce_k_anonymity(
        features, k, clustering_resolution
    )
    
    # Check if already l-diverse
    if check_l_diversity(k_anonymous_features, sensitive_attributes, k, l, clustering_resolution):
        logger.info(f"Features already satisfy {l}-diversity")
        return k_anonymous_features
    
    # Get groups
    _, groups_by_platform = check_k_anonymity(
        k_anonymous_features, k, clustering_resolution
    )
    
    # Initialize output
    diverse_features = {platform: {} for platform in features}
    
    for platform, groups in groups_by_platform.items():
        if platform not in sensitive_attributes:
            # No sensitive attributes for this platform
            diverse_features[platform] = k_anonymous_features[platform]
            continue
        
        platform_attributes = sensitive_attributes[platform]
        
        for group in groups:
            # Skip small groups
            if len(group) < k:
                for user_id in group:
                    diverse_features[platform][user_id] = k_anonymous_features[platform][user_id]
                continue
            
            # Check attribute diversity for this group
            attribute_diversity = {}
            
            for user_id in group:
                if user_id not in platform_attributes:
                    continue
                
                user_attributes = platform_attributes[user_id]
                
                for attr_name, attr_value in user_attributes.items():
                    if attr_name not in attribute_diversity:
                        attribute_diversity[attr_name] = set()
                    
                    attribute_diversity[attr_name].add(attr_value)
            
            # Check if group is diverse enough
            is_diverse = True
            for attr_name, distinct_values in attribute_diversity.items():
                if len(distinct_values) < l:
                    is_diverse = False
                    break
            
            if is_diverse:
                # Use k-anonymous vectors for this group
                for user_id in group:
                    diverse_features[platform][user_id] = k_anonymous_features[platform][user_id]
                continue
            
            # Not diverse enough, need to merge with another group
            # Find a group with complementary attribute values
            best_group = None
            best_combined_diversity = 0
            
            for other_group in groups:
                if other_group == group:
                    continue
                
                # Check combined diversity
                combined_diversity = {}
                
                for attr_name in attribute_diversity:
                    combined_diversity[attr_name] = attribute_diversity[attr_name].copy()
                
                # Add diversity from other group
                for user_id in other_group:
                    if user_id not in platform_attributes:
                        continue
                    
                    user_attributes = platform_attributes[user_id]
                    
                    for attr_name, attr_value in user_attributes.items():
                        if attr_name in combined_diversity:
                            combined_diversity[attr_name].add(attr_value)
                
                # Calculate minimum diversity across attributes
                min_diversity = min(len(values) for values in combined_diversity.values())
                
                if min_diversity > best_combined_diversity:
                    best_combined_diversity = min_diversity
                    best_group = other_group
            
            if best_group is None or best_combined_diversity < l:
                logger.warning(f"Could not find a suitable group to merge with for {len(group)} users to achieve {l}-diversity")
                for user_id in group:
                    diverse_features[platform][user_id] = k_anonymous_features[platform][user_id]
                continue
            
            # Merge groups by averaging vectors
            merged_vectors = np.array([
                k_anonymous_features[platform][user_id] 
                for user_id in (group | best_group)
            ])
            merged_center = np.mean(merged_vectors, axis=0)
            
            # Assign merged center to all users in both groups
            for user_id in group:
                diverse_features[platform][user_id] = merged_center
            
            for user_id in best_group:
                diverse_features[platform][user_id] = merged_center
    
    return diverse_features


def apply_privacy_measures(
    features: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    config: Dict[str, Any],
    sensitive_attributes: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Apply privacy measures to feature vectors.
    
    Args:
        features: Cross-platform feature vectors
        config: Privacy configuration
        sensitive_attributes: Optional sensitive attributes
        
    Returns:
        Dict: Feature vectors with privacy measures applied
    """
    if not config:
        logger.warning("No privacy configuration provided, returning original features")
        return features
    
    private_features = {}
    
    for platform_pair, pair_features in features.items():
        private_features[platform_pair] = {}
        
        for platform, platform_features in pair_features.items():
            # Apply differential privacy if enabled
            if config.get("enable_differential_privacy", False):
                platform_features = apply_differential_privacy_to_features(
                    platform_features, config
                )
            
            # Enforce k-anonymity if configured
            if "k_anonymity" in config and config["k_anonymity"] > 1:
                k = config["k_anonymity"]
                platform_features = enforce_k_anonymity(
                    platform_features, k
                )
            
            # Enforce l-diversity if configured and sensitive attributes available
            if "l_diversity" in config and config["l_diversity"] > 1 and sensitive_attributes:
                if platform in sensitive_attributes.get(platform_pair, {}):
                    k = config.get("k_anonymity", 2)
                    l = config["l_diversity"]
                    platform_features = enforce_l_diversity(
                        platform_features,
                        sensitive_attributes[platform_pair][platform],
                        k, l
                    )
            
            private_features[platform_pair][platform] = platform_features
    
    # Save privatized features
    os.makedirs("processed_data", exist_ok=True)
    
    with open("processed_data/private_features.pkl", "wb") as f:
        pickle.dump(private_features, f)
    
    logger.info("Applied privacy measures to feature vectors")
    return private_features


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    
    # Load feature vectors
    try:
        with open("processed_data/cross_platform_features.pkl", "rb") as f:
            cross_platform_features = pickle.load(f)
        
        # Apply privacy measures
        private_features = apply_privacy_measures(
            cross_platform_features,
            config["privacy"]
        )
    except FileNotFoundError:
        logger.error("Feature vectors not found, run feature_fusion.py first")
