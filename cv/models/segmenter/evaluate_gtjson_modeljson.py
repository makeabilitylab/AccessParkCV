#!/usr/bin/env python3
"""
Object Detection Evaluation Script for Oriented Bounding Boxes

This script evaluates the performance of an object detection model by comparing
ground truth annotations with predictions. It is specifically designed to handle
oriented bounding boxes represented as polygons (four corner points).

Features:
- Supports oriented bounding box evaluation using Intersection over Union (IoU)
- Calculates precision, recall, and F1-score for overall and per-class performance
- Generates and visualizes confusion matrices for class identification
- Allows class name mapping for grouping similar classes or handling inconsistent naming
- Exports comprehensive evaluation results as CSV, JSON, and visualizations
- Visualizes ground truth and predictions on original images with bounding boxes
- Creates dedicated folders for false positive and false negative cases

The expected input format for both ground truth and predictions is JSON:
{
    "image_filename.jpg": [
        {
            "xyxyxyxy": [
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4]
            ],
            "category_id": "class_name"
        },
        ...
    ],
    ...
}

Requirements:
- numpy, matplotlib, seaborn, shapely, scikit-learn, pandas, opencv-python (cv2)

Usage:
    python detection_evaluation.py --gt ground_truth.json --pred predictions.json [options]

Options:
    --gt PATH           Path to ground truth JSON file
    --pred PATH         Path to prediction JSON file
    --iou FLOAT         IoU threshold for matching (default: 0.5)
    --output PATH       Output directory for results (default: './results')
    --class-map PATH    Path to JSON file with class mapping dictionary
    --img-dir PATH      Path to directory containing the original images (for visualization)

Example class mapping JSON:
{
    "car": "vehicle",
    "truck": "vehicle",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler"
}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import cv2
import shutil
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional, Union

parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])


def load_json(file_path: str) -> Dict:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def polygon_iou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """
    Calculate IoU between two oriented bounding boxes represented as polygons.
    
    Args:
        box1: List of 4 [x, y] coordinates defining a polygon
        box2: List of 4 [x, y] coordinates defining a polygon
        
    Returns:
        IoU value
    """
    try:
        polygon1 = Polygon(box1)
        polygon2 = Polygon(box2)
        
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0
        
        # Handle cases where polygons don't intersect
        if not polygon1.intersects(polygon2):
            return 0.0
        
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.area + polygon2.area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0

def match_detections(
    gt_boxes: List[Dict], 
    pred_boxes: List[Dict], 
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match ground truth boxes with predicted boxes based on IoU.
    
    Args:
        gt_boxes: List of ground truth bounding boxes with category_id
        pred_boxes: List of predicted bounding boxes with category_id
        iou_threshold: Minimum IoU for a match
        
    Returns:
        Tuple of (matched pairs (gt_idx, pred_idx), unmatched_gt_indices, unmatched_pred_indices)
    """
    matches = []
    unmatched_gt = list(range(len(gt_boxes)))
    unmatched_pred = list(range(len(pred_boxes)))
    
    # If either list is empty, return early
    if not gt_boxes or not pred_boxes:
        return matches, unmatched_gt, unmatched_pred
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = polygon_iou(gt_box["xyxyxyxy"], pred_box["xyxyxyxy"])
    
    # Match based on IoU in descending order
    while unmatched_gt and unmatched_pred:
        # Find highest IoU
        max_iou = 0
        max_i, max_j = -1, -1
        for i in unmatched_gt:
            for j in unmatched_pred:
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j
        
        # If no matches above threshold, break
        if max_iou < iou_threshold:
            break
            
        # Add match and remove indices
        matches.append((max_i, max_j))
        unmatched_gt.remove(max_i)
        unmatched_pred.remove(max_j)
    
    return matches, unmatched_gt, unmatched_pred

def apply_class_mapping(data: Dict, class_mapping: Dict[str, str]) -> Dict:
    """
    Apply class mapping to detection data.
    
    Args:
        data: Detection data dictionary
        class_mapping: Dictionary mapping original class names to new class names
        
    Returns:
        Updated detection data with mapped class names
    """
    mapped_data = {}
    for image_id, boxes in data.items():
        mapped_boxes = []
        for box in boxes:
            mapped_box = box.copy()
            if box["category_id"] in class_mapping:
                mapped_box["category_id"] = class_mapping[box["category_id"]]
                mapped_box["og_category_id"] = box["category_id"]
            mapped_boxes.append(mapped_box)
        mapped_data[image_id] = mapped_boxes
    return mapped_data

def calculate_width_including_aisles(detections):

    # find parking space
    parking_space = None
    access_aisle_widths = {}
    for detection in detections:
        if detection['category_id'] == 'parking_space':
            parking_space = detection
        else: # detection['category_id'] == 'access_aisle
            # take the max parking width on that side of parking space
            access_aisle_widths[str(detection['closest_parking_edge'])] = max(detection['width'], access_aisle_widths.get(str(detection['closest_parking_edge']), 0))

    if parking_space is None: return 0
    
    total_width = parking_space['width'] + sum(access_aisle_widths.values())
    return total_width

def calculate_width_statistics(widths):
    ground_truth_widths = widths['ground_truth']
    predicted_widths = widths['predicted']
    parking_classes = ['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']
    
    # For individual categories
    category_differences = {key:[] for key in parking_classes}
    category_percent_differences = {key:[] for key in parking_classes}
    
    # For all categories aggregated
    all_differences = []
    all_percent_differences = []
    
    for (cat1, val1), (cat2, val2) in zip(ground_truth_widths, predicted_widths):
        # Changed from GT - Pred to Pred - GT
        difference = val2 - val1
        
        # Add to category-specific lists
        if cat1 in category_differences:
            category_differences[cat1].append(difference)
            
            # Calculate percent difference: (Pred - GT) / GT * 100
            percent_difference = (difference / val1 * 100) if val1 != 0 else 0
            category_percent_differences[cat1].append(percent_difference)
            
            # Add to aggregated lists
            all_differences.append(difference)
            all_percent_differences.append(percent_difference)
    
    results = {}
    
    # Process individual categories
    for category in parking_classes:
        differences = category_differences[category]
        percent_differences = category_percent_differences[category]
        
        # Convert to numpy arrays for calculations
        diff_array = np.array(differences)
        percent_diff_array = np.array(percent_differences)
        
        results[category] = {
            "count": len(differences),
            # Absolute differences (in pixels/units)
            "mean_difference": np.mean(diff_array) if len(diff_array) > 0 else 0,
            "std_difference": np.std(diff_array) if len(diff_array) > 0 else 0,
            "min_difference": np.min(diff_array) if len(diff_array) > 0 else 0,
            "max_difference": np.max(diff_array) if len(diff_array) > 0 else 0,
            "abs_mean_difference": np.mean(np.abs(diff_array)) if len(diff_array) > 0 else 0,
            "abs_std_difference": np.std(np.abs(diff_array)) if len(diff_array) > 0 else 0,
            
            # Percent differences
            "mean_percent_difference": np.mean(percent_diff_array) if len(percent_diff_array) > 0 else 0,
            "std_percent_difference": np.std(percent_diff_array) if len(percent_diff_array) > 0 else 0,
            "min_percent_difference": np.min(percent_diff_array) if len(percent_diff_array) > 0 else 0,
            "max_percent_difference": np.max(percent_diff_array) if len(percent_diff_array) > 0 else 0,
            "abs_mean_percent_difference": np.mean(np.abs(percent_diff_array)) if len(percent_diff_array) > 0 else 0,
            "abs_std_percent_difference": np.std(np.abs(percent_diff_array)) if len(percent_diff_array) > 0 else 0
        }
    
    # Process aggregated statistics for all categories
    all_diff_array = np.array(all_differences)
    all_percent_diff_array = np.array(all_percent_differences)
    
    results["all_categories"] = {
        "count": len(all_differences),
        # Absolute differences (in pixels/units)
        "mean_difference": np.mean(all_diff_array) if len(all_diff_array) > 0 else 0,
        "std_difference": np.std(all_diff_array) if len(all_diff_array) > 0 else 0,
        "min_difference": np.min(all_diff_array) if len(all_diff_array) > 0 else 0,
        "max_difference": np.max(all_diff_array) if len(all_diff_array) > 0 else 0,
        "abs_mean_difference": np.mean(np.abs(all_diff_array)) if len(all_diff_array) > 0 else 0,
        "abs_std_difference": np.std(np.abs(all_diff_array)) if len(all_diff_array) > 0 else 0,
        
        # Percent differences
        "mean_percent_difference": np.mean(all_percent_diff_array) if len(all_percent_diff_array) > 0 else 0,
        "std_percent_difference": np.std(all_percent_diff_array) if len(all_percent_diff_array) > 0 else 0,
        "min_percent_difference": np.min(all_percent_diff_array) if len(all_percent_diff_array) > 0 else 0,
        "max_percent_difference": np.max(all_percent_diff_array) if len(all_percent_diff_array) > 0 else 0,
        "abs_mean_percent_difference": np.mean(np.abs(all_percent_diff_array)) if len(all_percent_diff_array) > 0 else 0,
        "abs_std_percent_difference": np.std(np.abs(all_percent_diff_array)) if len(all_percent_diff_array) > 0 else 0
    }

    # Add all_differences and all_percent_differences to category arrays for plotting
    category_differences["all_categories"] = all_differences
    category_percent_differences["all_categories"] = all_percent_differences

    return results, category_differences, category_percent_differences
        
def transform_keys(original_dict):
    transformed_dict = {}
    
    for key in original_dict:
        # Split the key at '.rf.' and take the first part
        new_key = key.split('.rf.')[0]
        new_key += '.jpg'
        transformed_dict[new_key] = original_dict[key]
    
    return transformed_dict

def evaluate_detections(
    gt_json: str, 
    pred_json: str, 
    iou_threshold: float = 0.5, 
    class_mapping: Optional[Dict[str, str]] = None,
    img_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate detection performance by comparing ground truth with predictions.
    
    Args:
        gt_json: Path to ground truth JSON file
        pred_json: Path to prediction JSON file
        iou_threshold: IoU threshold for counting true positives
        class_mapping: Dictionary mapping original class names to new class names
                       e.g., {"foo": "bar"} will treat all "foo" classes as "bar"
        img_dir: Path to directory containing the original images (for visualization)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading ground truth from {gt_json}")
    gt_data = load_json(gt_json)
    
    print(f"Loading predictions from {pred_json}")
    pred_data = load_json(pred_json)
    

    # gt_data = transform_keys(gt_data)
    # pred_data = transform_keys(pred_data)

    # Apply class mapping if provided
    if class_mapping:
        print(f"Applying class mapping: {class_mapping}")
        gt_data = apply_class_mapping(gt_data, class_mapping)
        pred_data = apply_class_mapping(pred_data, class_mapping)

    # Get all unique class names
    all_classes: Set[str] = set()
    for image_id in gt_data:
        for box in gt_data[image_id]:
            all_classes.add(box["category_id"])
    for image_id in pred_data:
        for box in pred_data[image_id]:
            all_classes.add(box["category_id"])
    
    all_classes_list = sorted(list(all_classes))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes_list)}
    
    print(f"Found {len(all_classes_list)} classes: {all_classes_list}")
    
    # Performance metrics
    total_gt = 0
    total_pred = 0
    true_positives = 0
    class_true_positives = defaultdict(int)
    class_false_positives = defaultdict(int)
    class_false_negatives = defaultdict(int)
    class_gt_count = defaultdict(int)
    class_pred_count = defaultdict(int)
    
    # Width statistics
    # Should look like
    #  widths = {'ground_truth': [(dp_one_aisle, 4), (two_aisle, 10)],
            #   'predicted': [(dp_one_aisle, 5), (one_aisle, 12)]}
    # Indices match objects, first elem in tuple is class, second is width
    widths = {'ground_truth': [],
              'predicted': []}

    # For confusion matrix
    y_true = []
    y_pred = []
    
    # For visualization
    image_matches = {}
    images_with_fp = set()  # Images with false positives
    images_with_fn = set()  # Images with false negatives
    
    # Process each image
    all_image_ids = set(gt_data.keys()) | set(pred_data.keys())
    for image_id in all_image_ids:
        gt_boxes = gt_data.get(image_id, [])
        pred_boxes = pred_data.get(image_id, [])

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # Count GT classes
        for box in gt_boxes:
            class_gt_count[box["category_id"]] += 1
        
        # Count predicted classes
        for box in pred_boxes:
            class_pred_count[box["category_id"]] += 1
        
        # Match detections
        matches, unmatched_gt, unmatched_pred = match_detections(gt_boxes, pred_boxes, iou_threshold)
        
        # Store match results for visualization
        if img_dir:
            image_matches[image_id] = {
                'gt_boxes': gt_boxes,
                'pred_boxes': pred_boxes,
                'matches': matches,
                'unmatched_gt': unmatched_gt,
                'unmatched_pred': unmatched_pred
            }
            
            # Determine if this image has errors (false positives or false negatives)
            has_fp = len(unmatched_pred) > 0
            has_fn = len(unmatched_gt) > 0
            
            if has_fp:
                images_with_fp.add(image_id)
            if has_fn:
                images_with_fn.add(image_id)
        
        parking_space_matched = False
        # Process matches for confusion matrix and class metrics
        for gt_idx, pred_idx in matches:
            gt_class = gt_boxes[gt_idx]["category_id"]
            pred_class = pred_boxes[pred_idx]["category_id"]

            y_true.append(class_to_idx[gt_class])
            y_pred.append(class_to_idx[pred_class])
            
            # Count true positives (both box and class match)
            if gt_class == pred_class:
                true_positives += 1
                class_true_positives[gt_class] += 1

                if gt_class == 'parking_space': # calculate width statistics if its parking space
                    # TODO CALL width
                    parking_width_truth = calculate_width_including_aisles(gt_boxes)
                    parking_width_pred = calculate_width_including_aisles(pred_boxes)
                    widths['ground_truth'].append((gt_boxes[gt_idx]['og_category_id'], parking_width_truth))
                    widths['predicted'].append((pred_boxes[pred_idx]["og_category_id"], parking_width_pred))


        # Add unmatched ground truth boxes to confusion matrix (false negatives)
        for gt_idx in unmatched_gt:
            gt_class = gt_boxes[gt_idx]["category_id"]
            class_false_negatives[gt_class] += 1
            y_true.append(class_to_idx[gt_class])
            # Represent "no detection" as -1 for now
            y_pred.append(-1)
        
        # Add unmatched predictions to confusion matrix (false positives)
        for pred_idx in unmatched_pred:
            pred_class = pred_boxes[pred_idx]["category_id"]
            class_false_positives[pred_class] += 1
            # Represent "no ground truth" as -1 for now
            y_true.append(-1)
            y_pred.append(class_to_idx[pred_class])
    
    # Compute overall metrics
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute class-wise metrics
    class_metrics = {}
    for class_name in all_classes_list:
        tp = class_true_positives[class_name]
        fp = class_false_positives[class_name]
        fn = class_false_negatives[class_name]
        gt_count = class_gt_count[class_name]
        pred_count = class_pred_count[class_name]
        
        class_precision = tp / pred_count if pred_count > 0 else 0
        class_recall = tp / gt_count if gt_count > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        class_metrics[class_name] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
            "support": gt_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    # Create confusion matrix without -1 entries
    valid_indices = [(i, j) for i, j in enumerate(zip(y_true, y_pred)) if j[0] != -1 and j[1] != -1]
    if valid_indices:
        valid_y_true = [y_true[i] for i in range(len(y_true)) if i in [x[0] for x in valid_indices]]
        valid_y_pred = [y_pred[i] for i in range(len(y_pred)) if i in [x[0] for x in valid_indices]]
        cm = confusion_matrix(valid_y_true, valid_y_pred, labels=range(len(all_classes_list)))
    else:
        cm = np.zeros((len(all_classes_list), len(all_classes_list)), dtype=int)
    
    # Calculate overall false positives and negatives
    false_positives = sum(class_false_positives.values())
    false_negatives = sum(class_false_negatives.values())

    # When calculating width statistics, capture both absolute and percent differences
    width_stats, category_width_diffs, category_percent_diffs = calculate_width_statistics(widths)

    results = {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "iou_threshold": iou_threshold
        },
        "width_statistics": width_stats,
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "class_names": all_classes_list
    }
    
    # Include visualization data if image directory was provided
    if img_dir:
        results["visualization"] = {
            "image_matches": image_matches,
            "images_with_fp": sorted(list(images_with_fp)),
            "images_with_fn": sorted(list(images_with_fn))
        }
    
    return results, category_width_diffs, category_percent_diffs

def draw_oriented_bbox(
    image: np.ndarray, 
    box: List[List[float]], 
    color: Tuple[int, int, int], 
    label: Optional[str] = None, 
    thickness: int = 2, 
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw an oriented bounding box on an image.
    
    Args:
        image: Input image
        box: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        color: Color in BGR format (tuple of 3 integers)
        label: Optional text label to display
        thickness: Line thickness
        font_scale: Font scale for label text
        
    Returns:
        Image with bounding box drawn
    """
    try:
        # Convert points to integer numpy array
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(image, [pts], True, color, thickness)
        
        if label:
            # Find top-left corner for text placement
            min_x = min(pt[0] for pt in box)
            min_y = min(pt[1] for pt in box)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            min_x = max(0, min(min_x, w - 1))
            min_y = max(20, min(min_y, h - 1))  # Add padding for text
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw text background - ensure coordinates are integers
            pt1 = (int(min_x), int(min_y - text_height - 5))
            pt2 = (int(min_x + text_width), int(min_y))
            cv2.rectangle(image, pt1, pt2, color, -1)
            
            # Draw text
            cv2.putText(
                image, 
                label, 
                (int(min_x), int(min_y - 5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness, 
                cv2.LINE_AA
            )
    except Exception as e:
        print(f"Error drawing bounding box: {e}")
    
    return image

def visualize_detection_results(
    results: Dict[str, Any], 
    img_dir: str, 
    output_dir: str = "./results"
) -> None:
    """
    Visualize detection results using matplotlib for high-quality figures.
    Creates side-by-side visualizations with ground truth and predictions.
    
    Args:
        results: Results dictionary from evaluate_detections
        img_dir: Directory containing the original images
        output_dir: Directory to save visualization results
    """
    if "visualization" not in results:
        print("No visualization data available. Skipping visualization.")
        return
    
    # Verify image directory exists
    if not os.path.isdir(img_dir):
        print(f"Error: Image directory '{img_dir}' not found. Skipping visualization.")
        return
    
    # Create output directories
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create directories for special cases
    fp_dir = os.path.join(vis_dir, "false_positives")
    fn_dir = os.path.join(vis_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    
    # Define colors for different detection types
    MATCH_COLOR = 'green'      # Correct detections
    FP_COLOR = 'red'           # False positives 
    FN_COLOR = 'blue'          # False negatives
    CLASS_ERROR_COLOR = 'orange'  # Class errors
    
    # Process each image
    total_images = len(results["visualization"]["image_matches"])
    processed_images = 0
    
    # Set up matplotlib to use a non-interactive backend for batch processing
    plt.switch_backend('agg')
    
    for image_id, match_data in results["visualization"]["image_matches"].items():

        processed_images += 1
        if processed_images % 10 == 0:
            print(f"Visualizing image {processed_images}/{total_images}...")
        
        # Construct image path (try multiple extensions if exact filename not found)
        base_name = os.path.splitext(image_id)[0]
        img_path = None
        
        # Try with original extension
        if os.path.exists(os.path.join(img_dir, image_id)):
            img_path = os.path.join(img_dir, image_id)
        else:
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                potential_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
        
        if not img_path:
            print(f"Warning: Image for '{image_id}' not found in {img_dir}. Skipping.")
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        
        # Convert to RGB (matplotlib uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get match data
        gt_boxes = match_data["gt_boxes"]
        pred_boxes = match_data["pred_boxes"]
        matches = match_data["matches"]
        unmatched_gt = match_data["unmatched_gt"]
        unmatched_pred = match_data["unmatched_pred"]
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Detection Results for {image_id}", fontsize=16)
        
        # Display original images
        ax1.imshow(img_rgb)
        ax1.set_title("Ground Truth")
        ax1.axis('off')
        
        ax2.imshow(img_rgb)
        ax2.set_title("Predictions")
        ax2.axis('off')
        
        # Create legend elements
        legend_elements = [
            plt.Line2D([0], [0], color=MATCH_COLOR, lw=4, label='Correct Detection (TP)'),
            plt.Line2D([0], [0], color=FN_COLOR, lw=4, label='Missed Detection (FN)'),
            plt.Line2D([0], [0], color=FP_COLOR, lw=4, label='False Detection (FP)'),
            plt.Line2D([0], [0], color=CLASS_ERROR_COLOR, lw=4, label='Class Error')
        ]
        
        # Add legend at the bottom
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
        
        # Add statistics
        stats_text = f"TP: {len(matches)} | FP: {len(unmatched_pred)} | FN: {len(unmatched_gt)}"
        fig.text(0.5, 0.95, stats_text, ha='center', fontsize=12)
        
        # Draw ground truth boxes
        for i, box in enumerate(gt_boxes):
            box_pts = np.array(box["xyxyxyxy"])
            class_name = box["category_id"]
            
            # Determine color based on match status
            if i in [match[0] for match in matches]:
                color = MATCH_COLOR  # Matched detection
                label = f"{class_name}"
            else:
                color = FN_COLOR  # False negative
                label = f"{class_name} (missed)"
            
            # Draw polygon
            polygon = plt.Polygon(box_pts, fill=False, edgecolor=color, linewidth=2)
            ax1.add_patch(polygon)
            
            # Add label at the top-left corner of the box
            min_x = min(pt[0] for pt in box_pts)
            min_y = min(pt[1] for pt in box_pts)
            ax1.text(min_x, min_y, label, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1), 
                    color='white', fontsize=8, ha='left', va='bottom')
        
        # Draw predicted boxes
        for i, box in enumerate(pred_boxes):
            box_pts = np.array(box["xyxyxyxy"])
            class_name = box["category_id"]
            
            # Determine color based on match status
            if i in [match[1] for match in matches]:
                match_indices = [idx for idx, m in enumerate(matches) if m[1] == i]
                if not match_indices:
                    continue
                    
                gt_idx = matches[match_indices[0]][0]
                gt_class = gt_boxes[gt_idx]["category_id"]
                
                if gt_class == class_name:
                    color = MATCH_COLOR  # Correct class match
                    label = f"{class_name}"
                else:
                    color = CLASS_ERROR_COLOR  # Class error
                    label = f"{class_name} (should be {gt_class})"
            else:
                color = FP_COLOR  # False positive
                label = f"{class_name} (FP)"
            
            # Draw polygon
            polygon = plt.Polygon(box_pts, fill=False, edgecolor=color, linewidth=2)
            ax2.add_patch(polygon)
            
            # Add label at the top-left corner of the box
            min_x = min(pt[0] for pt in box_pts)
            min_y = min(pt[1] for pt in box_pts)
            ax2.text(min_x, min_y, label, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1), 
                    color='white', fontsize=8, ha='left', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
        
        # Save the visualization
        output_path = os.path.join(vis_dir, f"{base_name}_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save copies in special folders if needed
        has_fp = image_id in results["visualization"]["images_with_fp"]
        has_fn = image_id in results["visualization"]["images_with_fn"]
        
        if has_fp:
            fp_path = os.path.join(fp_dir, f"{base_name}_comparison.png")
            shutil.copy(output_path, fp_path)
        
        if has_fn:
            fn_path = os.path.join(fn_dir, f"{base_name}_comparison.png")
            shutil.copy(output_path, fn_path)
    
    print(f"Visualization results saved to {vis_dir}")
    print(f"  - False positives: {fp_dir}")
    print(f"  - False negatives: {fn_dir}")

def visualize_simple_detection_results(
    results: Dict[str, Any], 
    img_dir: str, 
    output_dir: str = "./results"
) -> None:
    """
    Visualize detection results with simpler color coding:
    - Access aisles are always orange
    - Ground truth boxes are green
    - Predicted boxes are red
    - No text labels on boxes
    
    Args:
        results: Results dictionary from evaluate_detections
        img_dir: Directory containing the original images
        output_dir: Directory to save visualization results
    """
    if "visualization" not in results:
        print("No visualization data available. Skipping simple visualization.")
        return
    
    # Verify image directory exists
    if not os.path.isdir(img_dir):
        print(f"Error: Image directory '{img_dir}' not found. Skipping simple visualization.")
        return
    
    # Create output directories
    simple_vis_dir = os.path.join(output_dir, "simple_visualizations")
    os.makedirs(simple_vis_dir, exist_ok=True)
    
    # Create subfolders for false positives and false negatives
    simple_fp_dir = os.path.join(simple_vis_dir, "false_positives")
    simple_fn_dir = os.path.join(simple_vis_dir, "false_negatives")
    os.makedirs(simple_fp_dir, exist_ok=True)
    os.makedirs(simple_fn_dir, exist_ok=True)
    
    # Define colors for different detection types
    GT_COLOR = 'green'        # Ground truth boxes
    PRED_COLOR = '#196DEB'    # Prediction boxes
    AISLE_COLOR = '#FFCF03'   # Access aisles
    
    # Process each image
    total_images = len(results["visualization"]["image_matches"])
    processed_images = 0
    
    # Set up matplotlib to use a non-interactive backend for batch processing
    plt.switch_backend('agg')
    
    for image_id, match_data in results["visualization"]["image_matches"].items():
        processed_images += 1
        if processed_images % 10 == 0:
            print(f"Creating simple visualization for image {processed_images}/{total_images}...")
        
        # Construct image path (try multiple extensions if exact filename not found)
        base_name = os.path.splitext(image_id)[0]
        img_path = None
        
        # Try with original extension
        if os.path.exists(os.path.join(img_dir, image_id)):
            img_path = os.path.join(img_dir, image_id)
        else:
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                potential_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
        
        if not img_path:
            print(f"Warning: Image for '{image_id}' not found in {img_dir}. Skipping.")
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        
        # Convert to RGB (matplotlib uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get match data
        gt_boxes = match_data["gt_boxes"]
        pred_boxes = match_data["pred_boxes"]
        
        # Check for unmatched boxes (false positives and false negatives)
        unmatched_gt = match_data["unmatched_gt"]
        unmatched_pred = match_data["unmatched_pred"]
        has_fp = len(unmatched_pred) > 0
        has_fn = len(unmatched_gt) > 0
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Simple Detection Visualization for {image_id}", fontsize=16)
        
        # Display original images
        ax1.imshow(img_rgb)
        ax1.set_title("Ground Truth")
        ax1.axis('off')
        
        ax2.imshow(img_rgb)
        ax2.set_title("Predictions")
        ax2.axis('off')
        
        # Create legend elements
        legend_elements = [
            plt.Line2D([0], [0], color=GT_COLOR, lw=4, label='Ground Truth'),
            plt.Line2D([0], [0], color=PRED_COLOR, lw=4, label='Prediction'),
            plt.Line2D([0], [0], color=AISLE_COLOR, lw=4, label='Access Aisle')
        ]
        
        # Add legend at the bottom
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
        
        # Draw ground truth boxes
        for box in gt_boxes:
            box_pts = np.array(box["xyxyxyxy"])
            
            # Determine color based on class
            if box["category_id"] == "access_aisle":
                color = AISLE_COLOR
            else:
                color = GT_COLOR
            
            # Draw polygon without label
            polygon = plt.Polygon(box_pts, fill=False, edgecolor=color, linewidth=6)
            ax1.add_patch(polygon)
        
        # Draw predicted boxes
        for box in pred_boxes:
            box_pts = np.array(box["xyxyxyxy"])
            
            # Determine color based on class
            if box["category_id"] == "access_aisle":
                color = AISLE_COLOR
            else:
                color = PRED_COLOR
            
            # Draw polygon without label
            polygon = plt.Polygon(box_pts, fill=False, edgecolor=color, linewidth=6)
            ax2.add_patch(polygon)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
        
        # Save the visualization
        output_path = os.path.join(simple_vis_dir, f"{base_name}_simple.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Save copies in special folders if needed
        if has_fp:
            fp_path = os.path.join(simple_fp_dir, f"{base_name}_simple.png")
            shutil.copy(output_path, fp_path)
        
        if has_fn:
            fn_path = os.path.join(simple_fn_dir, f"{base_name}_simple.png")
            shutil.copy(output_path, fn_path)
            
        plt.close(fig)
    
    print(f"Simple visualization results saved to {simple_vis_dir}")
    print(f"  - Simple false positives: {simple_fp_dir}")
    print(f"  - Simple false negatives: {simple_fn_dir}")


def plot_confusion_matrix(results: Dict[str, Any], output_dir: str = "./results") -> None:
    """
    Plot and save the confusion matrix with false positives and false negatives.
    Predicted classes on the Y-axis, ground truth on the X-axis.
    
    Args:
        results: Results dictionary from evaluate_detections
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the basic confusion matrix
    cm = results["confusion_matrix"]
    class_names = results["class_names"]
    
    # Transpose confusion matrix to put predicted on Y-axis and truth on X-axis
    cm_transposed = cm.T
    
    # Count false negatives for each class
    false_negatives = np.zeros(len(class_names), dtype=int)
    for i, class_name in enumerate(class_names):
        false_negatives[i] = results["class_metrics"][class_name]["false_negatives"]
    
    # Count false positives for each class
    false_positives = np.zeros(len(class_names), dtype=int)
    for j, class_name in enumerate(class_names):
        false_positives[j] = results["class_metrics"][class_name]["false_positives"]
    
    # Create extended confusion matrix with FP/FN
    cm_ext = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=int)
    # Copy original transposed confusion matrix
    cm_ext[:len(class_names), :len(class_names)] = cm_transposed
    # Add false positives column
    cm_ext[:len(class_names), -1] = false_positives
    # Add false negatives row
    cm_ext[-1, :len(class_names)] = false_negatives

    cm_ext = [[int(item) for item in inner_list] for inner_list in cm_ext] # convert float to int in CM
    
    # Create extended labels
    extended_labels_x = class_names + ["FP"]  # Add FP column label
    extended_labels_y = class_names + ["FN"]  # Add FN row label
    
    # Plot the extended confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Blues',
                xticklabels=extended_labels_x,
                yticklabels=extended_labels_y)
    plt.xlabel('Ground Truth (FP = False Positive)')
    plt.ylabel('Predicted (FN = False Negative)')
    plt.title('Confusion Matrix with False Positives and Negatives')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Also save the standard confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_transposed, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title('Standard Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'standard_confusion_matrix.png'), dpi=300)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def plot_width_histograms(data_dict, percent_data_dict, output_dir: str = "./results"):
    """
    Plots histograms for each category with both absolute and percent differences.
    Each category gets two histograms stacked vertically - absolute difference on top,
    percent difference on bottom. Now includes an "all_categories" aggregated view.
    
    Parameters:
    data_dict (dict): Dictionary with category IDs as keys and absolute differences as values
    percent_data_dict (dict): Dictionary with category IDs as keys and percent differences as values
    output_dir (str): Directory to save the output figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all category IDs sorted, with "all_categories" at the end
    standard_categories = sorted([cat for cat in data_dict.keys() if cat != "all_categories"])
    categories = standard_categories + ["all_categories"]  # Put "all_categories" at the end
    
    # Calculate the needed grid size
    n_categories = len(categories)
    n_rows = ((n_categories - 1) // 2 + 1) * 2  # Each category needs 2 rows, calculate pairs needed
    
    # Create figure and grid
    fig = plt.figure(figsize=(12, 3 + (n_rows * 2)))  # Dynamic height based on categories
    gs = GridSpec(n_rows, 2, figure=fig)
    
    # Plot position mapping
    for i, category in enumerate(categories):
        # Skip "all_categories" for now, we'll handle it separately
        if category == "all_categories":
            continue
            
        col = i % 2  # 0 or 1
        row = (i // 2) * 2  # 0, 2, 4, etc. (doubled to make room for two plots per category)
        
        # Get data
        abs_data = data_dict[category]
        pct_data = percent_data_dict[category]
        
        # Skip if no data
        if len(abs_data) == 0:
            continue
            
        # ===== Plot 1: Absolute Differences =====
        ax1 = fig.add_subplot(gs[row, col])
        
        # Calculate bins for absolute data
        abs_margin = (max(abs_data) - min(abs_data)) * 0.05 if len(abs_data) > 1 else 1
        abs_min = min(abs_data) - abs_margin
        abs_max = max(abs_data) + abs_margin
        abs_bins = np.linspace(abs_min, abs_max, 20)
        
        # Statistics for absolute data
        abs_mean = np.mean(abs_data)
        abs_std = np.std(abs_data)
        n = len(abs_data)
        
        # Draw the histogram
        ax1.hist(abs_data, bins=abs_bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Updated title to reflect Pred - GT instead of GT - Pred
        ax1.set_title(f"{category}: Absolute Difference (Pred - GT)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Difference (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.set_xlim(abs_min, abs_max)
        
        # Add statistics to plot
        ax1.axvline(x=abs_mean, color='red', linestyle='--', linewidth=1)
        ax1.text(0.05, 0.95, f"n: {n}\nMean: {abs_mean:.2f}\nStd Dev: {abs_std:.2f}",
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ===== Plot 2: Percent Differences =====
        ax2 = fig.add_subplot(gs[row+1, col])
        
        # Calculate bins for percent data
        pct_margin = (max(pct_data) - min(pct_data)) * 0.05 if len(pct_data) > 1 else 1
        pct_min = min(pct_data) - pct_margin
        pct_max = max(pct_data) + pct_margin
        pct_bins = np.linspace(pct_min, pct_max, 20)
        
        # Statistics for percent data
        pct_mean = np.mean(pct_data)
        pct_std = np.std(pct_data)
        
        # Draw the histogram
        ax2.hist(pct_data, bins=pct_bins, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        ax2.set_title(f"{category}: Percent Difference", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Percent Difference (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_xlim(pct_min, pct_max)
        
        # Add statistics to plot
        ax2.axvline(x=pct_mean, color='red', linestyle='--', linewidth=1)
        ax2.text(0.05, 0.95, f"Mean: {pct_mean:.2f}%\nStd Dev: {pct_std:.2f}%",
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Now handle "all_categories" separately with a dedicated subplot at the bottom
    if "all_categories" in data_dict:
        category = "all_categories"
        abs_data = data_dict[category]
        pct_data = percent_data_dict[category]
        
        if len(abs_data) > 0:
            # Create a new row at the bottom for all_categories
            # This spans both columns
            
            # Make sure we're not exceeding the grid dimensions
            bottom_row = n_rows - 2  # Last two rows
            
            # ===== Plot 1: Absolute Differences for all categories =====
            ax1 = fig.add_subplot(gs[bottom_row, :])  # Span both columns
            
            # Calculate bins for absolute data
            abs_margin = (max(abs_data) - min(abs_data)) * 0.05 if len(abs_data) > 1 else 1
            abs_min = min(abs_data) - abs_margin
            abs_max = max(abs_data) + abs_margin
            abs_bins = np.linspace(abs_min, abs_max, 20)
            
            # Statistics for absolute data
            abs_mean = np.mean(abs_data)
            abs_std = np.std(abs_data)
            n = len(abs_data)
            
            # Draw the histogram
            ax1.hist(abs_data, bins=abs_bins, alpha=0.7, color='navy', edgecolor='black')
            
            # Remove top and right spines
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Updated title to reflect Pred - GT instead of GT - Pred
            ax1.set_title(f"ALL CATEGORIES: Absolute Difference (Pred - GT)", 
                         fontsize=14, fontweight='bold', color='darkblue')
            ax1.set_xlabel("Difference (units)")
            ax1.set_ylabel("Frequency")
            ax1.set_xlim(abs_min, abs_max)
            
            # Add statistics to plot
            ax1.axvline(x=abs_mean, color='red', linestyle='--', linewidth=1)
            ax1.text(0.05, 0.95, f"n: {n}\nMean (pix): {abs_mean:.2f}\nStd Dev: {abs_std:.2f}",
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # ===== Plot 2: Percent Differences for all categories =====
            ax2 = fig.add_subplot(gs[bottom_row+1, :])  # Span both columns
            
            # Calculate bins for percent data
            pct_margin = (max(pct_data) - min(pct_data)) * 0.05 if len(pct_data) > 1 else 1
            pct_min = min(pct_data) - pct_margin
            pct_max = max(pct_data) + pct_margin
            pct_bins = np.linspace(pct_min, pct_max, 20)
            
            # Statistics for percent data
            pct_mean = np.mean(pct_data)
            pct_std = np.std(pct_data)
            
            # Draw the histogram
            ax2.hist(pct_data, bins=pct_bins, alpha=0.7, color='darkgreen', edgecolor='black')
            
            # Remove top and right spines
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            ax2.set_title(f"ALL CATEGORIES: Percent Difference", 
                         fontsize=14, fontweight='bold', color='darkblue')
            ax2.set_xlabel("Percent Difference (%)")
            ax2.set_ylabel("Frequency")
            ax2.set_xlim(pct_min, pct_max)
            
            # Add statistics to plot
            ax2.axvline(x=pct_mean, color='red', linestyle='--', linewidth=1)
            ax2.text(0.05, 0.95, f"Mean: {pct_mean:.2f}%\nStd Dev: {pct_std:.2f}%",
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'width_error_histograms.png'), dpi=300)
    
    # Save a second image with better resolution
    # plt.savefig(os.path.join(output_dir, 'width_error_histograms_highres.png'), dpi=600)
    
    # Save as SVG format
    plt.savefig(os.path.join(output_dir, 'width_error_histograms.svg'), format='svg')
    
    plt.close(fig)
    return fig

def save_results(results: Dict[str, Any], output_dir: str = "./results") -> None:
    """
    Save evaluation results to files.
    
    Args:
        results: Results dictionary from evaluate_detections
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall metrics
    overall_df = pd.DataFrame([results["overall"]])
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"), index=False)
    
    # Save class metrics
    class_metrics = []
    for class_name, metrics in results["class_metrics"].items():
        metrics_dict = {"class": class_name}
        metrics_dict.update(metrics)
        class_metrics.append(metrics_dict)
    
    class_df = pd.DataFrame(class_metrics)
    class_df.to_csv(os.path.join(output_dir, "class_metrics.csv"), index=False)
    
    # Save confusion matrix - transpose to match visualization (pred on Y, truth on X)
    cm_df = pd.DataFrame(
        results["confusion_matrix"].T,  # Transpose to match new orientation
        index=results["class_names"],  # Predicted classes (Y-axis)
        columns=results["class_names"]  # Ground truth classes (X-axis)
    )
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    
    # Save results as JSON (excluding visualization data which can be large)
    json_results = results.copy()
    if "visualization" in json_results:
        del json_results["visualization"]
        
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results["confusion_matrix"] = json_results["confusion_matrix"].tolist()
        json.dump(json_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection performance.')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction JSON file')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--class-map', type=str, help='Path to JSON file with class mapping dictionary')
    parser.add_argument('--img-dir', type=str, help='Path to directory containing the original images (for visualization)')
    parser.add_argument('--simple-viz', action='store_true', help='Generate simple visualization with fixed colors')
    
    args = parser.parse_args()
    
    # Load class mapping if provided
    class_mapping = None
    if args.class_map:
        try:
            with open(args.class_map, 'r') as f:
                class_mapping = json.load(f)
            print(f"Loaded class mapping: {class_mapping}")
        except Exception as e:
            print(f"Error loading class mapping: {e}")
            return
    
    print(f"Evaluating detections with IoU threshold: {args.iou}")
    results, category_width_diffs, category_percent_diffs = evaluate_detections(args.gt, args.pred, args.iou, class_mapping, args.img_dir)
    
    print("\nOverall Metrics:")
    for key, value in results["overall"].items():
        print(f"  {key}: {value}")
    
    print("\nClass-wise Metrics:")
    for class_name, metrics in results["class_metrics"].items():
        print(f"  {class_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")
    
    print("\nWidth Statistics:")
    for category, stats in results["width_statistics"].items():
        print(f"  {category}:")
        print(f"    Count: {stats['count']}")
        # Updated label to indicate Pred - GT instead of GT - Pred
        print(f"    Absolute Difference (Pred - GT) - Mean: {stats['mean_difference']:.2f}, Std Dev: {stats['std_difference']:.2f}")
        print(f"    Percent Difference - Mean: {stats['mean_percent_difference']:.2f}%, Std Dev: {stats['std_percent_difference']:.2f}%")
    
    # Print aggregated statistics separately for emphasis
    if "all_categories" in results["width_statistics"]:
        all_stats = results["width_statistics"]["all_categories"]
        print("\nAGGREGATED Width Statistics (All Categories Combined):")
        print(f"  Total Count: {all_stats['count']}")
        print(f"  Absolute Difference (Pred - GT) - Mean: {all_stats['mean_difference']:.2f}, Std Dev: {all_stats['std_difference']:.2f}")
        print(f"  Absolute Difference (|Pred - GT|) - Mean: {all_stats['abs_mean_difference']:.2f}, Std Dev: {all_stats['abs_std_difference']:.2f}")
        print(f"  Percent Difference - Mean: {all_stats['mean_percent_difference']:.2f}%, Std Dev: {all_stats['std_percent_difference']:.2f}%")
        print(f"  Absolute Percent Difference (|%|) - Mean: {all_stats['abs_mean_percent_difference']:.2f}%, Std Dev: {all_stats['abs_std_percent_difference']:.2f}%")
    
    print("\nSaving results and plots...")
    save_results(results, args.output)
    plot_confusion_matrix(results, args.output)
    plot_width_histograms(category_width_diffs, category_percent_diffs, args.output)

    # Visualize detections if image directory was provided
    if args.img_dir:
        print("\nGenerating detection visualizations...")
        # visualize_detection_results(results, args.img_dir, args.output)
        
        # Generate simple visualization if requested
        visualize_simple_detection_results(results, args.img_dir, args.output)
    
    print(f"Results saved to {args.output}")
if __name__ == "__main__":
    main()