#!/usr/bin/env python3
"""
Enhanced Parking Space Detection Model Evaluation Script

This script evaluates the performance of a parking space detection model
by comparing its predictions with ground truth annotations. It can process
multiple pairs of ground truth and prediction files, providing both individual
and aggregated statistics across all evaluated pairs.

Usage:
    # For a single pair
    python multi_model_evaluation.py --pair ground_truth1.json prediction1.json --gt-images /path/to/gt_images/ --pred-images /path/to/pred_images/ [--iou 0.5] [--output ./results]

    # For multiple pairs
    python multi_model_evaluation.py --pair ground_truth1.json prediction1.json --pair ground_truth2.json prediction2.json [--output ./results]

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, box
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import shutil
import glob
from scipy.optimize import linear_sum_assignment
import torch
from torchvision import ops
import re
import copy


GT_COLOR_TOP = 'green'        
PRED_COLOR_TOP = '#196DEB'
AISLE_COLOR_TOP = '#FFCF03'    # Access aisles


# Define all possible parking space classes
ALL_CLASSES = [
    "curbside", 
    "dp_no_aisle", 
    "dp_one_aisle", 
    "dp_two_aisle", 
    "one_aisle", 
    "two_aisle"
]

def load_data(file_path):
    """Load parking space data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle both formats: with and without 'parking_spaces' key
        if 'parking_spaces' in data:
            return data['parking_spaces']
        else:
            return data
            
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def polygon_to_shapely(polygon):
    """Convert polygon coordinates to a Shapely Polygon object"""
    try:
        return Polygon(polygon)
    except Exception as e:
        print(f"Error converting polygon: {e}")
        return None

def get_100pix_away_from_north_west_bounds(ground_truth_data):
    for obj in ground_truth_data['parking_spaces']:
        tile_x, tile_y

def create_shapefile_from_json(json_data, output_shapefile_path):
    """
    Create a GIS shapefile from a JSON file containing objects with bounding boxes.
    
    Parameters:
    json_file_path (str): Path to the JSON file
    output_shapefile_path (str): Path where shapefile will be saved
    
    Returns:
    GeoDataFrame: The created GeoDataFrame with bounding boxes
    # """
    # # Read the JSON file
    # with open(json_file_path, 'r') as f:
    #     data = json.load(f)
    data = json_data
    
    # Create lists to store geometries and attributes
    geometries = []
    attributes = []
    
    # Process each object
    for obj in data:
        # Extract bounding box coordinates
        # Assuming format is [min_x, min_y, max_x, max_y]
        if 'bbox' in obj:
            bbox = obj['bbox']
            # Create a shapely box from the coordinates
            geom = box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
            geometries.append(geom)
            
            # Store all attributes for this object
            obj_attrs = {k: v for k, v in obj.items() if k != 'bbox'}
            attributes.append(obj_attrs)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs="EPSG:3857")
    
    # Save to shapefile
    gdf.to_file(output_shapefile_path)
    
    return gdf

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each bounding box should be in the format [[x1, y1], [x2, y2]] where:
    - (x1, y1) is the top-left corner (note: x1 is smaller, y1 is LARGER)
    - (x2, y2) is the bottom-right corner (note: x2 is larger, y2 is SMALLER)
    
    Note: This function handles a coordinate system where the y-axis is reversed from
    standard image processing, with y increasing as you move upward/northward.
    
    Args:
        bbox1 (list): First bounding box in format [[x1, y1], [x2, y2]]
        bbox2 (list): Second bounding box in format [[x1, y1], [x2, y2]]
        
    Returns:
        float: IoU score between 0 and 1
    """
    # Extract coordinates
    [x1_1, y1_1], [x2_1, y2_1] = bbox1
    [x1_2, y1_2], [x2_2, y2_2] = bbox2

    # flip axes, as higher y means northward in epsg3857
    y1_1, y2_1, y1_2, y2_2 = -y1_1, -y2_1, -y1_2, -y2_2

    box1_tensor = torch.tensor([[x1_1, y1_1, x2_1, y2_1]], dtype=torch.float)
    box2_tensor = torch.tensor([[x1_2, y1_2, x2_2, y2_2]], dtype=torch.float)

    iou = ops.box_iou(box1_tensor, box2_tensor)
    return float(iou)


def match_objects_optimal(ground_truth, predictions, iou_threshold=0.5):
    """
    Match ground truth objects with predictions based on IoU using the Hungarian algorithm
    for globally optimal assignment.
    
    Args:
        ground_truth: List of ground truth objects
        predictions: List of prediction objects
        iou_threshold: Minimum IoU for a match (default: 0.5)
        
    Returns:
        Dictionary containing matches and unmatched objects
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    # Create cost matrix (negative IoU to convert maximization to minimization)
    cost_matrix = np.zeros((len(ground_truth), len(predictions)))
    
    # Populate the cost matrix with IoU values
    for gt_idx, gt_obj in enumerate(ground_truth):
        for pred_idx, pred_obj in enumerate(predictions):
            iou = calculate_iou(gt_obj['bbox'], pred_obj['bbox'])
            # Only consider matches above threshold, otherwise set cost to 0
            cost_matrix[gt_idx, pred_idx] = -iou if iou >= iou_threshold else 0
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Process results
    matches = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for i in range(len(row_ind)):
        gt_idx = row_ind[i]
        pred_idx = col_ind[i]
        
        # Skip assignments with zero cost (below threshold)
        if cost_matrix[gt_idx, pred_idx] < 0:
            iou = -cost_matrix[gt_idx, pred_idx]  # Convert back to positive IoU
            gt_obj = ground_truth[gt_idx]
            pred_obj = predictions[pred_idx]
            
            matches.append({
                'gt_idx': gt_idx,
                'pred_idx': pred_idx,
                'iou': iou,
                'gt_class': gt_obj['class'],
                'pred_class': pred_obj['class'],
                'gt_width': gt_obj.get('total_width', 'Unknown'),
                'pred_width': pred_obj.get('total_width', 'Unknown'),
                'gt_id': gt_obj['id'],
                'pred_id': pred_obj['id']
            })
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
    
    # Identify unmatched ground truth and prediction objects
    unmatched_gt = [idx for idx in range(len(ground_truth)) if idx not in matched_gt_indices]
    unmatched_pred = [idx for idx in range(len(predictions)) if idx not in matched_pred_indices]
    
    return {
        'matches': matches,
        'matched_gt_indices': matched_gt_indices,
        'matched_pred_indices': matched_pred_indices,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred
    }

def create_confusion_matrix(matching_results, ground_truth, predictions, classes):
    """
    Create confusion matrix from matching results
    
    Args:
        matching_results: Output from match_objects function
        ground_truth: List of ground truth objects
        predictions: List of prediction objects
        classes: List of class names
        
    Returns:
        DataFrame containing confusion matrix with background class
    """
    # Get matching information
    matches = matching_results['matches']
    unmatched_gt = matching_results['unmatched_gt']
    unmatched_pred = matching_results['unmatched_pred']
    
    # Create confusion matrix with 'background' class
    all_classes = sorted(classes) + ['background']
    cm = pd.DataFrame(0, index=all_classes, columns=all_classes)
    
    # Fill in matches (true positives and class confusions)
    for match in matches:
        gt_class = match['gt_class']
        pred_class = match['pred_class']
        cm.loc[pred_class, gt_class] += 1
    
    # Add unmatched ground truth (false negatives - predicted as background)
    for idx in unmatched_gt:
        gt_class = ground_truth[idx]['class']
        cm.loc['background', gt_class] += 1
    
    # Add unmatched predictions (false positives - predicted on background)
    for idx in unmatched_pred:
        pred_class = predictions[idx]['class']
        cm.loc[pred_class, 'background'] += 1
    
    return cm

def calculate_metrics(confusion_matrix, classes, gt_classes=None):
    """
    Calculate precision, recall, and F1 score for each class
    
    Args:
        confusion_matrix: DataFrame containing confusion matrix
        classes: List of all class names (excluding 'background')
        gt_classes: List of classes that actually exist in ground truth
        
    Returns:
        Dictionary containing metrics for each class and overall
    """
    metrics = {}
    
    # If gt_classes not provided, calculate metrics for all classes
    if gt_classes is None:
        gt_classes = [cls for cls in classes if cls != 'background']
    
    # For combined "all classes" metrics (treating all classes as one)
    total_gt_all = 0
    total_detected_all = 0
    
    # Calculate metrics for each class
    for cls in classes:
        if cls == 'background':
            continue
            
        # True positives: Objects correctly classified as this class
        tp = confusion_matrix.loc[cls, cls]
        
        # False positives: Objects incorrectly classified as this class (including background)
        fp = confusion_matrix.loc[cls, :].sum() - tp
        
        # False negatives: Objects of this class classified as something else (including background)
        fn = confusion_matrix.loc[:, cls].sum() - tp
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Total ground truth instances of this class
        total_gt = confusion_matrix.loc[:, cls].sum()
        
        # Total detected instances (excluding those classified as background)
        # This counts how many ground truth objects of this class were detected as ANY class
        detected_as_any_class = total_gt - confusion_matrix.loc['background', cls]
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'total_gt': int(total_gt),
            'detected_as_any': int(detected_as_any_class)
        }
        
        # Add to combined stats
        total_gt_all += total_gt
        total_detected_all += detected_as_any_class
    
    # Calculate per-class average metrics (existing "overall" calculation)
    if gt_classes:
        # Only use classes that actually exist in ground truth for overall calculations
        classes_for_overall = [cls for cls in gt_classes if cls != 'background' and cls in metrics]
        
        if classes_for_overall:
            overall = {
                'precision': np.mean([metrics[cls]['precision'] for cls in classes_for_overall]),
                'recall': np.mean([metrics[cls]['recall'] for cls in classes_for_overall]),
                'f1': np.mean([metrics[cls]['f1'] for cls in classes_for_overall])
            }
            metrics['overall'] = overall
    
    # For all_classes_combined metrics (treating all classes as one object class):
    # TP = Objects that were detected as any class (total_detected_all)
    all_tp = total_detected_all
    
    # FN = Objects that weren't detected at all (total_gt_all - total_detected_all)
    all_fn = total_gt_all - total_detected_all
    
    # FP = Total predictions that didn't match any ground truth
    # This is the sum of all entries in the 'background' column except background-background
    all_fp = confusion_matrix.loc[:, 'background'].sum() - confusion_matrix.loc['background', 'background']
    
    # Calculate metrics
    all_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    all_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0
    
    metrics['all_classes_combined'] = {
        'precision': all_precision,
        'recall': all_recall,
        'f1': all_f1,
        'tp': int(all_tp),
        'fp': int(all_fp),
        'fn': int(all_fn),
        'total_gt': int(total_gt_all),
        'detected_as_any': int(total_detected_all)
    }
    
    return metrics
    
    return metrics
    
    # Calculate overall metrics (macro average) ONLY for classes in ground truth
    if gt_classes:
        # Only use classes that actually exist in ground truth for overall calculations
        classes_for_overall = [cls for cls in gt_classes if cls != 'background' and cls in metrics]
        
        if classes_for_overall:
            overall = {
                'precision': np.mean([metrics[cls]['precision'] for cls in classes_for_overall]),
                'recall': np.mean([metrics[cls]['recall'] for cls in classes_for_overall]),
                'f1': np.mean([metrics[cls]['f1'] for cls in classes_for_overall])
            }
            metrics['overall'] = overall
    
    return metrics

def calculate_width_stats(matches, classes):
    """
    Calculate width difference statistics for matched objects
    
    Args:
        matches: List of match dictionaries
        classes: List of class names
        
    Returns:
        Dictionary containing width statistics for each class and overall
    """
    # Initialize statistics dictionaries for each class
    width_stats = {}
    for cls in classes:
        width_stats[cls] = {
            'count': 0,
            'diffs': [],
            'abs_diffs': [],
            'rel_diffs': []
        }
    
    # Add an entry for overall statistics across all classes
    width_stats['overall'] = {
        'count': 0,
        'diffs': [],
        'abs_diffs': [],
        'rel_diffs': []
    }
    
    # Collect width differences for each match
    for match in matches:
        gt_class = match['gt_class']
        
        if match['pred_width'] == 'Unknown' or match['gt_width'] == 'Unknown':
            continue
            
        # Calculate differences
        diff = match['pred_width'] - match['gt_width']
        abs_diff = abs(diff)
        rel_diff = (diff / match['gt_width']) * 100 if match['gt_width'] != 0 else 0
        
        # Store values for this specific class
        if gt_class in width_stats:
            width_stats[gt_class]['count'] += 1
            width_stats[gt_class]['diffs'].append(diff)
            width_stats[gt_class]['abs_diffs'].append(abs_diff)
            width_stats[gt_class]['rel_diffs'].append(rel_diff)
        
        # Also store values for overall statistics
        width_stats['overall']['count'] += 1
        width_stats['overall']['diffs'].append(diff)
        width_stats['overall']['abs_diffs'].append(abs_diff)
        width_stats['overall']['rel_diffs'].append(rel_diff)
    
    # Calculate statistics for each class and overall
    for stat_key in width_stats:
        stats = width_stats[stat_key]
        
        if stats['count'] > 0:
            # Calculate mean, std, min, max
            stats['mean'] = np.mean(stats['diffs'])
            stats['mean_abs'] = np.mean(stats['abs_diffs'])
            stats['mean_rel'] = np.mean(stats['rel_diffs'])
            stats['std_dev'] = np.std(stats['diffs'])
            stats['min'] = min(stats['diffs'])
            stats['max'] = max(stats['diffs'])
            
            # Histogram data for absolute differences
            hist, bin_edges = np.histogram(stats['diffs'], bins=10)
            stats['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
    
    return width_stats

def print_metrics_table(metrics, title=None):
    """
    Print a formatted table of evaluation metrics
    
    Args:
        metrics: Dictionary containing precision, recall, and F1 metrics
        title: Optional title for the metrics table
    """
    # Header
    print("\n" + "="*110)
    if title:
        print(f"{title}")
        print("-"*110)
    
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'GT Total':<10} {'Detected':<10} {'Det %':<8}")
    print("-"*110)
    
    # Print metrics for each class
    for cls, cls_metrics in metrics.items():
        # Skip the overall average metrics for now - we'll print these at the end
        if cls in ['overall', 'all_classes_combined']:
            continue
            
        precision = cls_metrics['precision'] * 100
        recall = cls_metrics['recall'] * 100
        f1 = cls_metrics['f1'] * 100
        
        # Total GT instances of this class
        total_gt = cls_metrics.get('total_gt', '-')
        
        # Total detected as any class 
        detected_as_any = cls_metrics.get('detected_as_any', '-')
        
        # Detection percentage (what percent of this class was detected as any class)
        detection_percentage = '-'
        if isinstance(total_gt, int) and isinstance(detected_as_any, int) and total_gt > 0:
            detection_percentage = f"{100 * detected_as_any / total_gt:.1f}%"
        
        print(f"{cls:<20} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%   {cls_metrics.get('tp', '-'):<8} {cls_metrics.get('fp', '-'):<8} {cls_metrics.get('fn', '-'):<8} {total_gt:<10} {detected_as_any:<10} {detection_percentage:<8}")
    
    print("-"*110)
    
    # Print all_classes_combined metrics (treating all classes as one)
    if 'all_classes_combined' in metrics:
        combined_metrics = metrics['all_classes_combined']
        precision = combined_metrics['precision'] * 100
        recall = combined_metrics['recall'] * 100
        f1 = combined_metrics['f1'] * 100
        
        total_gt = combined_metrics.get('total_gt', '-')
        detected_as_any = combined_metrics.get('detected_as_any', '-')
        
        detection_percentage = '-'
        if isinstance(total_gt, int) and isinstance(detected_as_any, int) and total_gt > 0:
            detection_percentage = f"{100 * detected_as_any / total_gt:.1f}%"
        
        print(f"{'ALL CLASSES COMBINED':<20} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%   {combined_metrics.get('tp', '-'):<8} {combined_metrics.get('fp', '-'):<8} {combined_metrics.get('fn', '-'):<8} {total_gt:<10} {detected_as_any:<10} {detection_percentage:<8}")
    
    # Print class average metrics (mean of per-class metrics)
    if 'overall' in metrics:
        print("-"*110)
        precision = metrics['overall']['precision'] * 100
        recall = metrics['overall']['recall'] * 100
        f1 = metrics['overall']['f1'] * 100
        print(f"{'CLASS AVERAGE':<20} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%")
    
    print("="*110 + "\n")
    
    # Print overall metrics
    if 'overall' in metrics:
        print("-"*90)
        precision = metrics['overall']['precision'] * 100
        recall = metrics['overall']['recall'] * 100
        f1 = metrics['overall']['f1'] * 100
        print(f"{'Overall':<15} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%")
    
    print("="*90 + "\n")

def print_width_stats_table(width_stats, title=None):
    """
    Print a formatted table of width statistics
    
    Args:
        width_stats: Dictionary containing width statistics
        title: Optional title for the width statistics table
    """
    # Header
    print("\n" + "="*100)
    if title:
        print(f"{title}")
        print("-"*100)
        
    print(f"{'Class':<15} {'Count':<8} {'Mean Diff':<12} {'Mean Abs':<12} {'Mean Rel %':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8}")
    print("-"*100)
    
    # Print statistics for each class
    for cls, stats in width_stats.items():
        if stats['count'] == 0:
            continue
            
        print(f"{cls:<15} {stats['count']:<8} {stats['mean']:>8.2f}    {stats['mean_abs']:>8.2f}    {stats['mean_rel']:>8.2f}%    {stats['std_dev']:>8.2f}    {stats['min']:>6.2f}  {stats['max']:>6.2f}")
    
    print("="*100 + "\n")

def plot_width_histograms(width_stats, output_dir, figsize=(10, 6), title_prefix=""):
    """
    Plot histograms of width differences for each class and overall
    
    Args:
        width_stats: Dictionary containing width statistics
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height) tuple
        title_prefix: Optional prefix for histogram titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cls, stats in width_stats.items():
        if stats['count'] < 2:  # Need at least 2 samples for a meaningful histogram
            continue
        
        if 'diffs' in stats:
            # We're working with raw data arrays, not histogram data
            plt.figure(figsize=figsize)
            plt.hist(stats['diffs'], bins=10, alpha=0.7)
            
            # Add statistics as text box
            stats_text = (f"Mean: {np.mean(stats['diffs']):.2f}\n"
                          f"Mean Abs: {np.mean(stats['abs_diffs']):.2f}\n"
                          f"StdDev: {np.std(stats['diffs']):.2f}\n"
                          f"Samples: {stats['count']}")
        elif 'histogram' in stats:
            # We're working with precomputed histogram data
            plt.figure(figsize=figsize)
            # Get histogram data
            counts = stats['histogram']['counts']
            bin_edges = stats['histogram']['bin_edges']
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Plot histogram
            plt.bar(bin_edges[:-1], counts, width=bin_width, alpha=0.7, align='edge')
            
            # Add statistics as text box
            stats_text = (f"Mean: {stats['mean']:.2f}\n"
                          f"Mean Abs: {stats['mean_abs']:.2f}\n"
                          f"StdDev: {stats['std_dev']:.2f}\n"
                          f"Samples: {stats['count']}")
        else:
            continue  # Skip if no histogram data
        
        # Position the text box in the upper right corner
        plt.text(0.80, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Use "Overall" for the overall plot, otherwise use class name
        if cls == 'overall':
            display_name = "Overall (All Classes)"
        else:
            display_name = cls
            
        title = f"Width Difference Distribution for {display_name}"
        if title_prefix:
            title = f"{title_prefix} - {title}"
        
        plt.title(title)
        plt.xlabel('Width Difference')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Add title prefix to filename if provided
        filename_prefix = ""
        if title_prefix:
            # Convert title prefix to a valid filename component
            filename_prefix = re.sub(r'[^\w\-]', '_', title_prefix) + "_"
        
        # Use "overall" for the overall plot filename
        output_filename = f'{filename_prefix}width_hist_{cls}.png'
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()
        
def calculate_width_stats(matches, classes):
    """
    Calculate width difference statistics for matched objects
    
    Args:
        matches: List of match dictionaries
        classes: List of class names
        
    Returns:
        Dictionary containing width statistics for each class and overall
    """
    # Initialize statistics dictionaries for each class
    width_stats = {}
    for cls in classes:
        width_stats[cls] = {
            'count': 0,
            'diffs': [],
            'abs_diffs': [],
            'rel_diffs': []
        }
    
    # Add an entry for overall statistics across all classes
    width_stats['overall'] = {
        'count': 0,
        'diffs': [],
        'abs_diffs': [],
        'rel_diffs': []
    }
    
    # Collect width differences for each match
    for match in matches:
        gt_class = match['gt_class']
        
        if match['pred_width'] == 'Unknown' or match['gt_width'] == 'Unknown':
            continue
            
        # Calculate differences
        diff = match['pred_width'] - match['gt_width']
        abs_diff = abs(diff)
        rel_diff = (diff / match['gt_width']) * 100 if match['gt_width'] != 0 else 0
        
        # Store values for this specific class
        if gt_class in width_stats:
            width_stats[gt_class]['count'] += 1
            width_stats[gt_class]['diffs'].append(diff)
            width_stats[gt_class]['abs_diffs'].append(abs_diff)
            width_stats[gt_class]['rel_diffs'].append(rel_diff)
        
        # Also store values for overall statistics
        width_stats['overall']['count'] += 1
        width_stats['overall']['diffs'].append(diff)
        width_stats['overall']['abs_diffs'].append(abs_diff)
        width_stats['overall']['rel_diffs'].append(rel_diff)
    
    # Calculate statistics for each class and overall
    for stat_key in width_stats:
        stats = width_stats[stat_key]
        
        if stats['count'] > 0:
            # Calculate mean, std, min, max
            stats['mean'] = np.mean(stats['diffs'])
            stats['mean_abs'] = np.mean(stats['abs_diffs'])
            stats['mean_rel'] = np.mean(stats['rel_diffs'])
            stats['std_dev'] = np.std(stats['diffs'])
            stats['min'] = min(stats['diffs'])
            stats['max'] = max(stats['diffs'])
            
            # Histogram data for absolute differences
            hist, bin_edges = np.histogram(stats['diffs'], bins=10)
            stats['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
    
    return width_stats

def merge_confusion_matrices(confusion_matrices):
    """
    Merge multiple confusion matrices into one
    
    Args:
        confusion_matrices: List of confusion matrix DataFrames
        
    Returns:
        DataFrame containing the merged confusion matrix
    """
    if not confusion_matrices:
        return None
    
    # Use the first matrix as a base
    merged_cm = confusion_matrices[0].copy()
    
    # Add values from other matrices
    for cm in confusion_matrices[1:]:
        merged_cm = merged_cm.add(cm, fill_value=0)
    
    return merged_cm

def merge_matches(all_matches):
    """
    Merge matches from multiple evaluation pairs
    
    Args:
        all_matches: List of match lists from different evaluation pairs
        
    Returns:
        List of all matches combined
    """
    merged_matches = []
    for matches in all_matches:
        merged_matches.extend(matches)
    return merged_matches

def plot_confusion_matrix(cm, output_file, figsize=(12, 10), title=None):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        cm: DataFrame containing confusion matrix
        output_file: Path to save the plot
        figsize: Figure size as (width, height) tuple
        title: Optional custom title for the plot
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Add labels and title
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Predicted Class', fontsize=14)
    plt.xlabel('Ground Truth Class', fontsize=14)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_width_histograms(width_stats, output_dir, figsize=(10, 6), title_prefix=""):
    """
    Plot histograms of width differences for each class and overall
    
    Args:
        width_stats: Dictionary containing width statistics
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height) tuple
        title_prefix: Optional prefix for histogram titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cls, stats in width_stats.items():
        if stats['count'] < 2:  # Need at least 2 samples for a meaningful histogram
            continue
        
        if 'diffs' in stats:
            # We're working with raw data arrays, not histogram data
            plt.figure(figsize=figsize)
            plt.hist(stats['diffs'], bins=10, alpha=0.7)
            
            # Add statistics as text box
            stats_text = (f"Mean: {np.mean(stats['diffs']):.2f}\n"
                          f"Mean Abs: {np.mean(stats['abs_diffs']):.2f}\n"
                          f"StdDev: {np.std(stats['diffs']):.2f}\n"
                          f"Samples: {stats['count']}")
        elif 'histogram' in stats:
            # We're working with precomputed histogram data
            plt.figure(figsize=figsize)
            # Get histogram data
            counts = stats['histogram']['counts']
            bin_edges = stats['histogram']['bin_edges']
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Plot histogram
            plt.bar(bin_edges[:-1], counts, width=bin_width, alpha=0.7, align='edge')
            
            # Add statistics as text box
            stats_text = (f"Mean: {stats['mean']:.2f}\n"
                          f"Mean Abs: {stats['mean_abs']:.2f}\n"
                          f"StdDev: {stats['std_dev']:.2f}\n"
                          f"Samples: {stats['count']}")
        else:
            continue  # Skip if no histogram data
        
        # Position the text box in the upper right corner
        plt.text(0.80, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Use "Overall" for the overall plot, otherwise use class name
        if cls == 'overall':
            display_name = "Overall (All Classes)"
        else:
            display_name = cls
            
        title = f"Width Difference Distribution for {display_name}"
        if title_prefix:
            title = f"{title_prefix} - {title}"
        
        plt.title(title)
        plt.xlabel('Width Difference')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Add title prefix to filename if provided
        filename_prefix = ""
        if title_prefix:
            # Convert title prefix to a valid filename component
            filename_prefix = re.sub(r'[^\w\-]', '_', title_prefix) + "_"
        
        # Use "overall" for the overall plot filename
        output_filename = f'{filename_prefix}width_hist_{cls}.png'
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()

def find_image_file(directory, base_id):
    """
    Find image file with various possible extensions
    
    Args:
        directory: Directory to search for the image
        base_id: Base ID of the image file (without extension)
        
    Returns:
        Path to the image file if found, None otherwise
    """
    possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    for ext in possible_extensions:
        image_path = os.path.join(directory, f"{base_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    
    # If no exact match found, try glob to find case-insensitive matches or partial matches
    pattern = os.path.join(directory, f"{base_id}.*")
    matches = glob.glob(pattern)
    
    if matches:
        return matches[0]  # Return the first match
    
    return None


def create_labeled_image(image_path, class_name, output_path, label_prefix="", dpi=300):
    """
    Create a labeled image with class information on a white canvas,
    preserving the original image resolution.
    
    Args:
        image_path: Path to the source image
        class_name: Class name to display on the image
        output_path: Path to save the labeled image
        label_prefix: Optional prefix for the label (e.g., "GT:", "Pred:")
        dpi: Resolution of the output image (dots per inch)
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Keep original dimensions
        original_width, original_height = img.width, img.height
        
        # Create a new white canvas with space for text above
        padding = int(0.05 * original_height)  # Proportional padding
        label_height = int(0.08 * original_height)  # Proportional label area
        canvas_width = original_width + 2 * padding
        canvas_height = original_height + 2 * padding + label_height
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Paste image on canvas at original resolution
        canvas.paste(img, (padding, padding + label_height))
        
        # Add label
        draw = ImageDraw.Draw(canvas)
        
        # Calculate font size proportional to image size
        font_size = max(14, int(0.025 * original_height))
        
        try:
            # Try to load a font - fallback to default if not available
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Try DejaVuSans as a fallback
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                # Use default font if neither is available
                font = ImageFont.load_default()
        
        # Format label
        if label_prefix:
            label = f"{label_prefix} {class_name}"
        else:
            label = f"Class: {class_name}"
        
        # Draw label centered above the image
        try:
            # For PIL >= 9.2.0
            text_width = draw.textlength(label, font=font)
        except AttributeError:
            # For older PIL versions
            text_width = font.getmask(label).getbbox()[2]
            
        text_x = (canvas_width - text_width) // 2
        draw.text((text_x, padding // 2), label, fill="black", font=font)
        
        # Add a separator line
        draw.line([(padding, padding + label_height // 2), 
                  (canvas_width - padding, padding + label_height // 2)], 
                  fill="gray", width=max(1, int(original_height / 400)))
        
        # Save the labeled image with specified DPI
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        canvas.save(output_path, dpi=(dpi, dpi))
        return True
    
    except Exception as e:
        print(f"Error creating labeled image: {e}")
        return False

def create_comparison_image(gt_image_path, pred_image_path, output_path, match_info=None, dpi=300):
    """
    Create a side-by-side comparison image with ground truth and prediction,
    preserving the original image resolution.
    
    Args:
        gt_image_path: Path to ground truth image
        pred_image_path: Path to prediction image
        output_path: Path to save the comparison image
        match_info: Optional dictionary with match information to display
        dpi: Resolution of the output image (dots per inch)
    """
    try:
        # Load images
        gt_img = Image.open(gt_image_path)
        pred_img = Image.open(pred_image_path)
        
        # Keep original dimensions
        gt_width, gt_height = gt_img.width, gt_img.height
        pred_width, pred_height = pred_img.width, pred_img.height
        
        # Scale images proportionally if they have very different sizes
        # to ensure they're visually comparable while maintaining aspect ratios
        scale_factor = 1.0
        if abs(gt_height/pred_height - 1) > 0.2:  # If heights differ by more than 20%
            # Scale to match the smaller image height to preserve details
            if gt_height > pred_height:
                scale_factor = pred_height / gt_height
                new_gt_height = pred_height
                new_gt_width = int(gt_width * scale_factor)
                gt_img = gt_img.resize((new_gt_width, new_gt_height), Image.LANCZOS)
                gt_width, gt_height = new_gt_width, new_gt_height
            else:
                scale_factor = gt_height / pred_height
                new_pred_height = gt_height
                new_pred_width = int(pred_width * scale_factor)
                pred_img = pred_img.resize((new_pred_width, new_pred_height), Image.LANCZOS)
                pred_width, pred_height = new_pred_width, new_pred_height
        
        # Calculate padding proportional to the average image height
        avg_height = (gt_height + pred_height) // 2
        padding = int(0.05 * avg_height)
        
        # Calculate info section height
        info_height = int(0.15 * avg_height) if match_info else padding
        
        # Create a new white canvas
        canvas_width = gt_width + pred_width + 3 * padding
        canvas_height = max(gt_height, pred_height) + 2 * padding + info_height

        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Paste images on canvas
        canvas.paste(gt_img, (padding, padding))
        canvas.paste(pred_img, (gt_width + 2 * padding, padding))
        
        # Add labels and information
        draw = ImageDraw.Draw(canvas)
        
        # Calculate font sizes proportional to image size
        title_font_size = max(14, int(0.025 * avg_height))
        info_font_size = max(12, int(0.02 * avg_height))
        
        try:
            # Try to load a font - fallback to default if not available
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
            info_font = ImageFont.truetype("arial.ttf", info_font_size)
        except IOError:
            try:
                # Try DejaVuSans as a fallback
                title_font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
                info_font = ImageFont.truetype("DejaVuSans.ttf", info_font_size)
            except IOError:
                # Use default font if neither is available
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
        
        # Labels for ground truth and prediction
        gt_label = "Ground Truth"
        pred_label = "Prediction"
        
        draw.text((padding, padding // 2), gt_label, fill="black", font=title_font)
        draw.text((gt_width + 2 * padding, padding // 2), pred_label, fill="black", font=title_font)
        
        # Add match information below images if provided
        if match_info:
            text_y = max(gt_height, pred_height) + padding + 10
            
            # Handle case where width might be a string ('Unknown')
            gt_width_str = str(match_info['gt_width'])
            pred_width_str = str(match_info['pred_width'])
            
            width_diff_str = "N/A"
            if isinstance(match_info['gt_width'], (int, float)) and isinstance(match_info['pred_width'], (int, float)):
                width_diff_str = f"{match_info['pred_width'] - match_info['gt_width']:.2f}"
            
            # Format class match or mismatch with colors
            class_match = match_info['gt_class'] == match_info['pred_class']
            gt_class_color = "black"
            pred_class_color = "green" if class_match else "red"
            
            info_text = [
                f"IoU: {match_info['iou']:.4f}",
                f"GT Class: {match_info['gt_class']}",
                f"Pred Class: {match_info['pred_class']}",
                f"GT Width: {gt_width_str}",
                f"Pred Width: {pred_width_str}",
                f"Width Diff: {width_diff_str}"
            ]
            
            info_colors = [
                "black",             # IoU
                gt_class_color,      # GT Class
                pred_class_color,    # Pred Class
                "black",             # GT Width
                "black",             # Pred Width
                "black"              # Width Diff
            ]
            
            line_height = info_font_size * 1.2
            for i, text in enumerate(info_text):
                draw.text((padding, text_y + int(i * line_height)), 
                          text, fill=info_colors[i], font=info_font)
        
        # Save the comparison image with specified DPI
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        canvas.save(output_path, dpi=(dpi, dpi))
        return True
    
    except Exception as e:
        print(f"Error creating comparison image: {e}")
        return False

def process_visual_comparisons(matching_results, ground_truth, predictions, gt_image_dir, pred_image_dir, output_dir, pair_name=None, dpi=300):
    """
    Create high-resolution visual comparisons for matched objects and organize false positives/negatives
    
    Args:
        matching_results: Output from match_objects function
        ground_truth: List of ground truth objects
        predictions: List of prediction objects
        gt_image_dir: Directory containing ground truth images
        pred_image_dir: Directory containing prediction images
        output_dir: Root output directory
        pair_name: Optional name for this evaluation pair
        dpi: Resolution for output images (dots per inch)
    """
    # Create output directories with pair name if provided
    prefix = ""
    if pair_name:
        prefix = f"{pair_name}_"
        print(f"\nProcessing visual comparisons for pair: {pair_name}")
    
    comparisons_dir = os.path.join(output_dir, f'{prefix}comparisons')
    false_positives_dir = os.path.join(output_dir, f'{prefix}false_positives')
    false_negatives_dir = os.path.join(output_dir, f'{prefix}false_negatives')
    
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(false_positives_dir, exist_ok=True)
    os.makedirs(false_negatives_dir, exist_ok=True)
    
    # Create comparison images for matched objects
    matches = matching_results['matches']
    print(f"Creating {len(matches)} high-resolution comparison images for matched objects...")
    
    for i, match in enumerate(matches):
        gt_obj = ground_truth[match['gt_idx']]
        pred_obj = predictions[match['pred_idx']]
        
        # Get image paths with support for various extensions
        gt_image_path = find_image_file(gt_image_dir, gt_obj['id'])
        pred_image_path = find_image_file(pred_image_dir, pred_obj['id'])
        
        if not gt_image_path:
            print(f"Warning: Ground truth image not found for {gt_obj['id']}")
            continue
                
        if not pred_image_path:
            print(f"Warning: Prediction image not found for {pred_obj['id']}")
            continue
        
        # Create high-resolution comparison image
        output_path = os.path.join(comparisons_dir, f"{gt_obj['id']}_vs_{pred_obj['id']}.png")
        success = create_comparison_image(gt_image_path, pred_image_path, output_path, match, dpi=dpi)
        
        if success and (i+1) % 10 == 0:
            print(f"Progress: {i+1}/{len(matches)} comparison images created")
    
    # Process false negatives (unmatched ground truth)
    unmatched_gt = matching_results['unmatched_gt']
    print(f"\nProcessing {len(unmatched_gt)} false negatives (unmatched ground truth objects)...")
    
    for idx in unmatched_gt:
        gt_obj = ground_truth[idx]
        gt_image_path = find_image_file(gt_image_dir, gt_obj['id'])
        
        if not gt_image_path:
            print(f"Warning: Ground truth image not found for false negative: {gt_obj['id']}")
            continue
        
        # Create high-resolution labeled image
        output_path = os.path.join(false_negatives_dir, f"{gt_obj['id']}.png")
        
        class_name = gt_obj.get('class', 'Unknown')
        success = create_labeled_image(gt_image_path, class_name, output_path, "Ground Truth:", dpi=dpi)
        
        if not success:
            print(f"Error creating labeled image for false negative: {gt_obj['id']}")
    
    # Process false positives (unmatched predictions)
    unmatched_pred = matching_results['unmatched_pred']
    print(f"\nProcessing {len(unmatched_pred)} false positives (unmatched prediction objects)...")
    
    for idx in unmatched_pred:
        pred_obj = predictions[idx]
        pred_image_path = find_image_file(pred_image_dir, pred_obj['id'])
        
        if not pred_image_path:
            print(f"Warning: Prediction image not found for false positive: {pred_obj['id']}")
            continue
        
        # Create high-resolution labeled image
        output_path = os.path.join(false_positives_dir, f"{pred_obj['id']}.png")
        
        class_name = pred_obj.get('class', 'Unknown')
        success = create_labeled_image(pred_image_path, class_name, output_path, "Prediction:", dpi=dpi)
        
        if not success:
            print(f"Error creating labeled image for false positive: {pred_obj['id']}")
    
    print("\nHigh-resolution visual comparison processing complete!")


def save_results_to_json(metrics, width_stats, output_file):
    """
    Save evaluation results to a JSON file
    
    Args:
        metrics: Dictionary containing precision, recall, and F1 metrics
        width_stats: Dictionary containing width statistics
        output_file: Path to save the results
    """
    results = {
        'metrics': metrics,
        'width_stats': width_stats
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

def print_metrics_table(metrics, title=None):
    """
    Print a formatted table of evaluation metrics
    
    Args:
        metrics: Dictionary containing precision, recall, and F1 metrics
        title: Optional title for the metrics table
    """
    # Header
    print("\n" + "="*90)
    if title:
        print(f"{title}")
        print("-"*90)
    
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Total':<8}")
    print("-"*90)
    
    # Print metrics for each class
    for cls, cls_metrics in metrics.items():
        if cls == 'overall':
            continue
            
        precision = cls_metrics['precision'] * 100
        recall = cls_metrics['recall'] * 100
        f1 = cls_metrics['f1'] * 100
        
        # Include total detected instances in output
        total_detected = cls_metrics.get('total_detected', '-')
        
        print(f"{cls:<15} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%   {cls_metrics.get('tp', '-'):<8} {cls_metrics.get('fp', '-'):<8} {cls_metrics.get('fn', '-'):<8} {total_detected:<8}")
    
    # Print overall metrics
    if 'overall' in metrics:
        print("-"*90)
        precision = metrics['overall']['precision'] * 100
        recall = metrics['overall']['recall'] * 100
        f1 = metrics['overall']['f1'] * 100
        print(f"{'Overall':<15} {precision:>8.2f}%   {recall:>8.2f}%   {f1:>8.2f}%")
    
    print("="*90 + "\n")

def print_width_stats_table(width_stats, title=None):
    """
    Print a formatted table of width statistics
    
    Args:
        width_stats: Dictionary containing width statistics
        title: Optional title for the width statistics table
    """
    # Header
    print("\n" + "="*100)
    if title:
        print(f"{title}")
        print("-"*100)
        
    print(f"{'Class':<15} {'Count':<8} {'Mean Diff':<12} {'Mean Abs':<12} {'Mean Rel %':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8}")
    print("-"*100)
    
    # Print statistics for each class
    for cls, stats in width_stats.items():
        if stats['count'] == 0:
            continue
            
        print(f"{cls:<15} {stats['count']:<8} {stats['mean']:>8.2f}    {stats['mean_abs']:>8.2f}    {stats['mean_rel']:>8.2f}%    {stats['std_dev']:>8.2f}    {stats['min']:>6.2f}  {stats['max']:>6.2f}")
    
    print("="*100 + "\n")

def evaluate_pair(gt_path, pred_path, args, pair_name=None):
    """
    Evaluate a single ground truth and prediction pair
    
    Args:
        gt_path: Path to ground truth JSON file
        pred_path: Path to prediction JSON file
        args: Command-line arguments
        pair_name: Optional name for this evaluation pair
        
    Returns:
        Dictionary containing evaluation results
    """
    # Determine output directory for this pair
    if pair_name:
        pair_output_dir = os.path.join(args.output, pair_name)
    else:
        pair_output_dir = args.output
    
    # Add prefix to output filenames if pair_name is provided
    prefix = ""
    if pair_name:
        prefix = f"{pair_name}_"
        print(f"\n{'='*60}")
        print(f"Evaluating pair: {pair_name}")
        print(f"{'='*60}")
    
    # Load data
    print(f"Loading ground truth data from {gt_path}...")
    ground_truth = load_data(gt_path)
    
    print(f"Loading prediction data from {pred_path}...")
    predictions = load_data(pred_path)
    
    print(f"Loaded {len(ground_truth)} ground truth objects and {len(predictions)} prediction objects.")
    
    # Create output directory
    os.makedirs(pair_output_dir, exist_ok=True)
    
    # Determine classes to evaluate
    if args.classes:
        classes_to_evaluate = args.classes
    else:
        classes_to_evaluate = ALL_CLASSES
    
    # Find classes actually present in ground truth
    gt_classes = {obj['class'] for obj in ground_truth if 'class' in obj}
    print(f"Classes found in ground truth: {', '.join(sorted(gt_classes))}")
    
    # Match objects
    print(f"\nMatching ground truth objects with predictions (IoU threshold: {args.iou})...")
    matching_results = match_objects_optimal(ground_truth, predictions, args.iou)
    
    num_matches = len(matching_results['matches'])
    num_unmatched_gt = len(matching_results['unmatched_gt'])
    num_unmatched_pred = len(matching_results['unmatched_pred'])
    
    print(f"Found {num_matches} matches, {num_unmatched_gt} unmatched ground truth objects, "
          f"and {num_unmatched_pred} unmatched predictions.")
    
    # Create confusion matrix
    print("\nCreating confusion matrix...")
    confusion_matrix = create_confusion_matrix(matching_results, ground_truth, predictions, classes_to_evaluate)
    
    # Calculate metrics
    print("\nCalculating precision, recall, and F1 score...")
    metrics = calculate_metrics(confusion_matrix, classes_to_evaluate, gt_classes)
    
    # Calculate width statistics
    print("\nCalculating width prediction statistics...")
    width_stats = calculate_width_stats(matching_results['matches'], classes_to_evaluate)
    
    # Print metrics and statistics
    if pair_name:
        print_metrics_table(metrics, f"Metrics for {pair_name}")
        print_width_stats_table(width_stats, f"Width Statistics for {pair_name}")
    else:
        print_metrics_table(metrics)
        print_width_stats_table(width_stats)
    
    # Save results to JSON
    save_results_to_json(metrics, width_stats, os.path.join(pair_output_dir, f'{prefix}results.json'))
    
    # Create plots
    print("\nCreating plots...")
    title_prefix = pair_name if pair_name else ""
    plot_confusion_matrix(confusion_matrix, 
                        os.path.join(pair_output_dir, f'{prefix}confusion_matrix.png'),
                        title=f"Confusion Matrix{' for ' + pair_name if pair_name else ''}")
    
    plot_width_histograms(width_stats, 
                        os.path.join(pair_output_dir, f'{prefix}width_histograms'),
                        title_prefix=title_prefix)
    
    # Process visual comparisons if image directories are provided
    if not args.no_visuals and args.gt_images and args.pred_images:
        print("\nProcessing visual comparisons...")
        process_visual_comparisons(matching_results, ground_truth, predictions, 
                                  args.gt_images, args.pred_images, pair_output_dir, pair_name,
                                  dpi=args.dpi)

    return {
        'metrics': metrics,
        'width_stats': width_stats,
        'confusion_matrix': confusion_matrix,
        'matching_results': matching_results,
        'ground_truth': ground_truth,
        'predictions': predictions
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate parking space detection model by comparing predictions with ground truth.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments - now using pairs instead of single files
    parser.add_argument('--pair', nargs=2, action='append', required=True,
                        metavar=('ground_truth.json', 'prediction.json'),
                        help='A pair of ground truth and prediction JSON files. Can be specified multiple times.')
    
    # Optional arguments
    parser.add_argument('--gt-images', help='Directory containing ground truth images', default='')
    parser.add_argument('--pred-images', help='Directory containing prediction images', default='')
    parser.add_argument('--iou', type=float, help='IoU threshold for matching objects', default=0.5)
    parser.add_argument('--output', help='Output directory for results', default='./results')
    parser.add_argument('--classes', nargs='+', help='Space-separated list of classes to evaluate (default: all classes)', default=None)
    parser.add_argument('--no-visuals', action='store_true', help='Skip generating visual comparisons')
    parser.add_argument('--pair-names', nargs='+', help='Names for each pair (must match number of pairs)', default=None)
    parser.add_argument('--simple-viz', action='store_true', help='Generate simple visualizations with green/red/orange color scheme')
    parser.add_argument('--dpi', type=int, help='Resolution (DPI) for output images', default=30)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create root output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if pair names are provided and match the number of pairs
    if args.pair_names and len(args.pair_names) != len(args.pair):
        print(f"Error: Number of pair names ({len(args.pair_names)}) doesn't match number of pairs ({len(args.pair)})")
        return

    # Process each pair
    pair_results = []
    all_confusion_matrices = []
    all_matches = []
    
    for i, (gt_path, pred_path) in enumerate(args.pair):
        # Determine pair name
        pair_name = None
        if args.pair_names:
            pair_name = args.pair_names[i]
        elif len(args.pair) > 1:
            # Generate a name based on filenames if multiple pairs but no names provided
            gt_base = os.path.splitext(os.path.basename(gt_path))[0]
            pred_base = os.path.splitext(os.path.basename(pred_path))[0]
            pair_name = f"pair_{i+1}_{gt_base}_vs_{pred_base}"
        
        # Evaluate this pair
        result = evaluate_pair(gt_path, pred_path, args, pair_name)
        pair_results.append(result)
        
        # Collect data for aggregate statistics
        all_confusion_matrices.append(result['confusion_matrix'])
        all_matches.append(result['matching_results']['matches'])
    
    # If we have multiple pairs, calculate aggregate statistics
    if len(pair_results) > 1:
        print("\n\n" + "="*80)
        print(f"AGGREGATE STATISTICS ACROSS ALL {len(pair_results)} PAIRS")
        print("="*80)
        
        # Merge confusion matrices
        merged_cm = merge_confusion_matrices(all_confusion_matrices)
        
        # Determine all classes present across all ground truth files
        all_gt_classes = set()
        for result in pair_results:
            gt_classes = {obj['class'] for obj in result['ground_truth'] if 'class' in obj}
            all_gt_classes.update(gt_classes)
        
        # Calculate metrics from merged confusion matrix
        classes_to_evaluate = args.classes if args.classes else ALL_CLASSES
        aggregate_metrics = calculate_metrics(merged_cm, classes_to_evaluate, all_gt_classes)
        
        # Calculate width statistics from all matches
        merged_matches = merge_matches(all_matches)
        aggregate_width_stats = calculate_width_stats(merged_matches, classes_to_evaluate)
        
        # Print and save aggregate results
        print_metrics_table(aggregate_metrics, "Aggregate Metrics Across All Pairs")
        print_width_stats_table(aggregate_width_stats, "Aggregate Width Statistics Across All Pairs")
        
        save_results_to_json(aggregate_metrics, aggregate_width_stats, 
                           os.path.join(args.output, 'aggregate_results.json'))
        
        # Create aggregate plots
        plot_confusion_matrix(merged_cm, 
                            os.path.join(args.output, 'aggregate_confusion_matrix.png'),
                            title="Aggregate Confusion Matrix Across All Pairs")
        
        plot_width_histograms(aggregate_width_stats, 
                            os.path.join(args.output, 'aggregate_width_histograms'),
                            title_prefix="Aggregate")
    
    print("\nEvaluation complete! Results saved to:", args.output)

if __name__ == "__main__":
    main()