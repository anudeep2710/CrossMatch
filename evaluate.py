"""
Evaluation Module

This module evaluates the performance of user matching models using
standard metrics and visualizations.
"""

import json
import logging
import os
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple, Union, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, str],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute evaluation metrics for user matching.
    
    Args:
        predictions: Predicted matches with confidence scores
        ground_truth: Ground truth mappings
        threshold: Confidence threshold for matches
        
    Returns:
        Dict: Evaluation metrics
    """
    y_true = []
    y_pred = []
    y_score = []
    
    # Process all possible user pairs
    user1_set = set(predictions.keys())
    user2_set = set()
    for matches in predictions.values():
        user2_set.update(matches.keys())
    
    # For each potential pair, check if it's a true match and if it was predicted
    for user1 in user1_set:
        true_user2 = ground_truth.get(user1, None)
        
        for user2 in user2_set:
            # Check if this is a ground truth match
            is_match = (true_user2 == user2)
            y_true.append(int(is_match))
            
            # Check if this pair was predicted
            score = predictions.get(user1, {}).get(user2, 0.0)
            y_score.append(score)
            y_pred.append(int(score >= threshold))
    
    # Handle edge case with no predictions
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc_roc": 0.0,
            "auc_pr": 0.0
        }
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle edge case with no positive predictions
    if sum(y_pred) == 0:
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Handle edge case with no positive ground truth
    if sum(y_true) == 0:
        recall = 0.0
    else:
        recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Handle edge case with zero precision or recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Compute ROC and PR curves
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_roc = auc(fpr, tpr)
    except:
        auc_roc = 0.0
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall_curve, precision_curve)
    except:
        auc_pr = 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }


def compute_top_k_metrics(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[int, Dict[str, float]]:
    """
    Compute top-K accuracy, precision, and recall metrics.
    
    Args:
        predictions: Predicted matches with confidence scores
        ground_truth: Ground truth mappings
        k_values: List of K values to evaluate
        
    Returns:
        Dict: Metrics for each K value
    """
    top_k_metrics = {}
    
    # Make sure predictions are sorted by confidence
    sorted_predictions = {}
    for user1, matches in predictions.items():
        sorted_matches = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
        sorted_predictions[user1] = sorted_matches
    
    for k in k_values:
        correct_at_k = 0
        total_predictions = 0
        
        for user1, true_user2 in ground_truth.items():
            if user1 not in sorted_predictions:
                continue
            
            # Get top-K predictions for this user
            top_k_matches = list(sorted_predictions[user1].keys())[:k]
            
            if true_user2 in top_k_matches:
                correct_at_k += 1
            
            total_predictions += 1
        
        # Calculate metrics
        if total_predictions > 0:
            accuracy_at_k = correct_at_k / total_predictions
            precision_at_k = correct_at_k / (total_predictions * k)
            recall_at_k = correct_at_k / len(ground_truth)
        else:
            accuracy_at_k = 0.0
            precision_at_k = 0.0
            recall_at_k = 0.0
        
        top_k_metrics[k] = {
            f"accuracy@{k}": accuracy_at_k,
            f"precision@{k}": precision_at_k,
            f"recall@{k}": recall_at_k
        }
    
    return top_k_metrics


def plot_roc_curve(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, str],
    title: str = "ROC Curve"
) -> Figure:
    """
    Plot ROC curve for model predictions.
    
    Args:
        predictions: Predicted matches with confidence scores
        ground_truth: Ground truth mappings
        title: Plot title
        
    Returns:
        Figure: Matplotlib figure object
    """
    y_true = []
    y_score = []
    
    # Process all possible user pairs
    user1_set = set(predictions.keys())
    user2_set = set()
    for matches in predictions.values():
        user2_set.update(matches.keys())
    
    # For each potential pair, check if it's a true match and record score
    for user1 in user1_set:
        true_user2 = ground_truth.get(user1, None)
        
        for user2 in user2_set:
            # Check if this is a ground truth match
            is_match = (true_user2 == user2)
            y_true.append(int(is_match))
            
            # Get prediction score
            score = predictions.get(user1, {}).get(user2, 0.0)
            y_score.append(score)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    return fig


def plot_precision_recall_curve(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, str],
    title: str = "Precision-Recall Curve"
) -> Figure:
    """
    Plot precision-recall curve for model predictions.
    
    Args:
        predictions: Predicted matches with confidence scores
        ground_truth: Ground truth mappings
        title: Plot title
        
    Returns:
        Figure: Matplotlib figure object
    """
    y_true = []
    y_score = []
    
    # Process all possible user pairs
    user1_set = set(predictions.keys())
    user2_set = set()
    for matches in predictions.values():
        user2_set.update(matches.keys())
    
    # For each potential pair, check if it's a true match and record score
    for user1 in user1_set:
        true_user2 = ground_truth.get(user1, None)
        
        for user2 in user2_set:
            # Check if this is a ground truth match
            is_match = (true_user2 == user2)
            y_true.append(int(is_match))
            
            # Get prediction score
            score = predictions.get(user1, {}).get(user2, 0.0)
            y_score.append(score)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot precision-recall curve
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="upper right")
    
    return fig


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
) -> Dict[str, pd.DataFrame]:
    """
    Compare performance of different models.
    
    Args:
        model_results: Results from different models
        output_dir: Directory to save comparison plots
        metrics: Metrics to include in comparison
        
    Returns:
        Dict: DataFrames with model comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    comparison_by_platform = {}
    all_metrics = []
    
    # Process results for each platform pair
    for platform_pair, results in model_results.items():
        metrics_by_model = {}
        
        ground_truth = results.get("ground_truth", {})
        
        for model_name, model_data in results.items():
            if model_name == "ground_truth":
                continue
            
            predictions = model_data.get("predictions", {})
            metrics = compute_metrics(predictions, ground_truth)
            top_k = compute_top_k_metrics(predictions, ground_truth)
            
            # Combine all metrics
            combined_metrics = {**metrics}
            for k, k_metrics in top_k.items():
                combined_metrics.update(k_metrics)
            
            metrics_by_model[model_name] = combined_metrics
            
            # Generate plots
            try:
                roc_fig = plot_roc_curve(
                    predictions, ground_truth, 
                    title=f"ROC Curve - {platform_pair} - {model_name}"
                )
                roc_path = os.path.join(output_dir, f"{platform_pair}_{model_name}_roc.png")
                roc_fig.savefig(roc_path)
                plt.close(roc_fig)
                
                pr_fig = plot_precision_recall_curve(
                    predictions, ground_truth, 
                    title=f"Precision-Recall Curve - {platform_pair} - {model_name}"
                )
                pr_path = os.path.join(output_dir, f"{platform_pair}_{model_name}_pr.png")
                pr_fig.savefig(pr_path)
                plt.close(pr_fig)
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics_by_model).T
        comparison_by_platform[platform_pair] = comparison_df
        
        # Save comparison to CSV
        csv_path = os.path.join(output_dir, f"{platform_pair}_comparison.csv")
        comparison_df.to_csv(csv_path)
        
        # Add platform pair to metrics for combined results
        platform_metrics = []
        for model, metrics in metrics_by_model.items():
            metrics_with_platform = {
                "platform_pair": platform_pair,
                "model": model,
                **metrics
            }
            platform_metrics.append(metrics_with_platform)
        
        all_metrics.extend(platform_metrics)
    
    # Create combined comparison DataFrame
    all_metrics_df = pd.DataFrame(all_metrics)
    
    # Save combined comparison to CSV
    csv_path = os.path.join(output_dir, "all_models_comparison.csv")
    all_metrics_df.to_csv(csv_path, index=False)
    
    # Create summary plots
    try:
        # Summary plot for accuracy
        plt.figure(figsize=(12, 8))
        summary_df = all_metrics_df.pivot(index="model", columns="platform_pair", values="accuracy")
        summary_df.plot(kind="bar")
        plt.title("Accuracy Comparison Across Platform Pairs")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()
        
        # Summary plot for F1 score
        plt.figure(figsize=(12, 8))
        summary_df = all_metrics_df.pivot(index="model", columns="platform_pair", values="f1")
        summary_df.plot(kind="bar")
        plt.title("F1 Score Comparison Across Platform Pairs")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_comparison.png"))
        plt.close()
    except Exception as e:
        logger.error(f"Error generating summary plots: {e}")
    
    return {
        "by_platform": comparison_by_platform,
        "all_metrics": all_metrics_df
    }


def evaluate_models(
    model_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate user matching models and generate reports.
    
    Args:
        model_results: Results from different models
        config: Evaluation configuration
        output_dir: Directory to save evaluation results
        
    Returns:
        Dict: Evaluation results
    """
    logger.info("Evaluating model performance")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics to evaluate
    metrics = config.get("metrics", ["accuracy", "precision", "recall", "f1"])
    
    # Compare models
    comparison_results = compare_models(
        model_results,
        output_dir,
        metrics
    )
    
    # Log summary of results
    summary = comparison_results["all_metrics"].groupby("model").mean()
    logger.info(f"Model performance summary:\n{summary[metrics]}")
    
    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(comparison_results, f)
    
    return comparison_results


if __name__ == "__main__":
    # Example usage
    import utils
    logging.basicConfig(level=logging.INFO)
    config = utils.load_config("config.yaml")
    
    try:
        # Load model results
        with open("processed_data/predictions.pkl", "rb") as f:
            model_results = pickle.load(f)
        
        # Evaluate models
        evaluation_results = evaluate_models(
            model_results,
            config["evaluation"],
            config["directories"]["results"]
        )
    except FileNotFoundError:
        logger.error("Prediction results not found, run models.py first")
