"""
Visualization utilities for object detection evaluation.
This module extends the DetectionEvaluator class with additional
visualization capabilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import random
from collections import defaultdict

GT_COLOR = 'green'        
PRED_COLOR = '#196DEB'

def visualize_box(image, bbox, is_ground_truth=True, label=None, score=None, 
                 bbox_format='xyxy', ax=None, color=None):
    """
    Visualize a bounding box on an image.
    
    Args:
        image: The image (numpy array in RGB format)
        bbox: Bounding box coordinates
        is_ground_truth: If True, box is colored green; otherwise red
        label: Optional label to display
        score: Optional confidence score to display
        bbox_format: 'xyxy' for [x1,y1,x2,y2] or 'xywh' for [x,y,width,height]
        ax: Matplotlib axis to plot on (if None, creates a new figure)
        color: Optional explicit color override (if None, uses green for GT and red for predictions)
        
    Returns:
        Matplotlib axis with the visualization
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
    
    # Set box color: green for ground truth, red for prediction
    if color is None:
        color = GT_COLOR if is_ground_truth else PRED_COLOR
    
    # Set line style: solid for ground truth, dashed for prediction
    linestyle = '-' if is_ground_truth else '--'
    
    # Get box coordinates based on format
    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                edgecolor=color, facecolor='none', linestyle=linestyle)
    else:  # xywh format
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                edgecolor=color, facecolor='none', linestyle=linestyle)
    
    # Add box to the axes
    ax.add_patch(rect)
    
    # Add label and score if provided
    if label or score is not None:
        if bbox_format == 'xyxy':
            x, y = x1, y1 - 5  # Position label above box
            if y < 10:  # If too close to top, place it inside box
                y = y1 + 15
        else:
            x, y = x, y - 5
            if y < 10:
                y = y + 15
        
        text = ""
        if label:
            text = f"{label}"
        if score is not None:
            text += f" ({score:.2f})" if label else f"{score:.2f}"
        
        ax.text(x, y, text, bbox=dict(facecolor=color, alpha=0.5), color='white', fontsize=10)
    
    return ax

def save_blank_crop(crop, save_path):
    """
    Save a blank version of a crop with no annotations or borders.
    
    Args:
        crop: The image crop (numpy array in RGB format)
        save_path: Path to save the side-by-side visualization
    
    Returns:
        Path to the saved blank crop
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create blank crop filename by adding "_blank" before the extension
    base, ext = os.path.splitext(save_path)
    blank_save_path = f"{base}_blank{ext}"
    
    # Create a clean figure with no borders, axes, or padding
    fig = plt.figure(frameon=False)
    fig.set_size_inches(crop.shape[1]/100, crop.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Add the crop with no additional elements
    ax.imshow(crop)
    
    # Save with tight layout and no borders
    plt.savefig(blank_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return blank_save_path

def visualize_polygon(image, coords, is_ground_truth=True, label=None, score=None, ax=None, color=None):
    """
    Visualize a polygon on an image.
    
    Args:
        image: The image (numpy array in RGB format)
        coords: List of (x,y) polygon coordinate pairs
        is_ground_truth: If True, polygon is colored green; otherwise red
        label: Optional label to display
        score: Optional confidence score to display
        ax: Matplotlib axis to plot on (if None, creates a new figure)
        color: Optional explicit color override
        
    Returns:
        Matplotlib axis with the visualization
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
    
    # Set polygon color: green for ground truth, red for prediction
    if color is None:
        color = GT_COLOR if is_ground_truth else PRED_COLOR
    
    # Set line style: solid for ground truth, dashed for prediction
    linestyle = '-' if is_ground_truth else '--'
    
    # Convert to numpy array for easier handling
    coords = np.array(coords)
    
    # Create polygon patch
    polygon = patches.Polygon(coords, linewidth=2, edgecolor=color, 
                            facecolor='none', linestyle=linestyle)
    
    # Add polygon to the axes
    ax.add_patch(polygon)
    
    # Add label and score if provided
    if label or score is not None:
        # Calculate centroid for text placement
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        text = ""
        if label:
            text = f"{label}"
        if score is not None:
            text += f" ({score:.2f})" if label else f"{score:.2f}"
        
        ax.text(centroid_x, centroid_y, text, 
                bbox=dict(facecolor=color, alpha=0.5), 
                color='white', fontsize=10, ha='center')
    
    return ax


def create_overlay_visualization(image, gt_annotations, pred_annotations, 
                              bbox_format='xyxy', figsize=(12, 12),
                              gt_color=GT_COLOR, pred_color=PRED_COLOR, 
                              save_path=None, title=None):
    """
    Create a visualization with both ground truth and predictions overlaid.
    
    Args:
        image: The image (numpy array in RGB format)
        gt_annotations: List of ground truth annotations (each with 'bbox' and 'category_name')
        pred_annotations: List of prediction annotations (each with 'bbox', 'category_name', and 'score')
        bbox_format: 'xyxy' for [x1,y1,x2,y2] or 'xywh' for [x,y,width,height]
        figsize: Figure size
        gt_color: Color for ground truth boxes
        pred_color: Color for prediction boxes
        save_path: Path to save the visualization (if None, figure is shown)
        title: Optional title for the figure
        
    Returns:
        Matplotlib figure with the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    # Draw ground truth boxes
    for gt in gt_annotations:
        visualize_box(image, gt['bbox'], True, gt['category_name'], 
                      None, bbox_format, ax, gt_color)
    
    # Draw prediction boxes
    for pred in pred_annotations:
        visualize_box(image, pred['bbox'], False, pred['category_name'], 
                      pred.get('score', None), bbox_format, ax, pred_color)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add legend
    gt_line = Line2D([0], [0], color=gt_color, linewidth=2, linestyle='-', label='Ground Truth')
    pred_line = Line2D([0], [0], color=pred_color, linewidth=2, linestyle='--', label='Prediction')
    ax.legend(handles=[gt_line, pred_line], loc='upper right')
    
    # Turn off axis
    ax.axis('off')
    
    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig

def visualize_true_positives(self, output_dir, max_samples=None):
    """
    Visualize correctly detected objects (true positives) and save them as 100x100 crops
    centered around the object, using the same approach as for false positives/negatives.
    
    Args:
        output_dir: Directory to save the visualizations
        max_samples: Maximum number of samples to visualize (None for all)
    """
    if not self.images_dir:
        print("Error: Cannot visualize true positives without images directory.")
        return
    
    # Create directory for true positive visualizations
    tp_dir = os.path.join(output_dir, "true_positives")
    os.makedirs(tp_dir, exist_ok=True)
    
    # Create a list to store true positive information
    true_positives_info = []
    
    # Collect all true positive matches
    for image_id in self.gt_by_image.keys():
        gt_boxes = self.gt_by_image[image_id]
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        if not gt_boxes or not pred_boxes:
            continue
        
        # Build matches using IoU threshold
        for gt in gt_boxes:
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        # Try global if available
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                # Check if this is a correct match (same category and sufficient IoU)
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                if iou >= self.iou_threshold and pred_cat_id == gt_cat_id:
                    true_positives_info.append({
                        'image_id': image_id,
                        'gt_bbox': gt_bbox,
                        'pred_bbox': pred_bbox,
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0),
                        'iou': iou
                    })
    
    # Select a random subset if max_samples is provided
    if max_samples is not None and len(true_positives_info) > max_samples:
        import random
        true_positives_info = random.sample(true_positives_info, max_samples)
    
    print(f"Visualizing {len(true_positives_info)} true positive samples...")
    
    # Generate visualizations for each true positive
    for i, tp in enumerate(true_positives_info):
        image_id = tp['image_id']
        image_path = self.get_image_path(image_id)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Could not find image for ID {image_id}")
            continue
        
        try:
            # Read the image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
            
            # Get centroid of the bounding box (using ground truth box for consistency)
            cx, cy = self.get_bbox_centroid(tp['gt_bbox'])
            
            # Create a 100x100 crop centered on the object
            height, width = image.shape[:2]
            
            # Calculate crop boundaries
            crop_size = 100
            x1 = max(0, int(cx - crop_size / 2))
            y1 = max(0, int(cy - crop_size / 2))
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if crop goes beyond image boundaries
            if x2 - x1 < crop_size:
                if x1 == 0:
                    x2 = min(width, crop_size)
                else:
                    x1 = max(0, x2 - crop_size)
            
            if y2 - y1 < crop_size:
                if y1 == 0:
                    y2 = min(height, crop_size)
                else:
                    y1 = max(0, y2 - crop_size)
            
            crop = image[y1:y2, x1:x2]
            
            # Convert ground truth bounding box to crop coordinates
            if self.bbox_format == 'xyxy':
                x1_gt, y1_gt, x2_gt, y2_gt = tp['gt_bbox']
                gt_crop_bbox = [
                    max(0, x1_gt - x1),
                    max(0, y1_gt - y1),
                    min(crop_size, x2_gt - x1),
                    min(crop_size, y2_gt - y1)
                ]
            else:  # xywh
                x_gt, y_gt, w_gt, h_gt = tp['gt_bbox']
                gt_crop_bbox = [
                    max(0, x_gt - x1),
                    max(0, y_gt - y1),
                    min(crop_size, w_gt),
                    min(crop_size, h_gt)
                ]
            
            # Convert prediction bounding box to crop coordinates
            if self.bbox_format == 'xyxy':
                x1_pred, y1_pred, x2_pred, y2_pred = tp['pred_bbox']
                pred_crop_bbox = [
                    max(0, x1_pred - x1),
                    max(0, y1_pred - y1),
                    min(crop_size, x2_pred - x1),
                    min(crop_size, y2_pred - y1)
                ]
            else:  # xywh
                x_pred, y_pred, w_pred, h_pred = tp['pred_bbox']
                pred_crop_bbox = [
                    max(0, x_pred - x1),
                    max(0, y_pred - y1),
                    min(crop_size, w_pred),
                    min(crop_size, h_pred)
                ]
            
            # Create figure with clean visualization
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(crop)
            
            # Draw the ground truth bbox in green
            if self.bbox_format == 'xyxy':
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]), 
                    gt_crop_bbox[2] - gt_crop_bbox[0], 
                    gt_crop_bbox[3] - gt_crop_bbox[1],
                    linewidth=2, 
                    edgecolor=GT_COLOR, 
                    facecolor='none',
                    label='Ground Truth'
                )
            else:  # xywh
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]), 
                    gt_crop_bbox[2], 
                    gt_crop_bbox[3],
                    linewidth=2, 
                    edgecolor=GT_COLOR, 
                    facecolor='none',
                    label='Ground Truth'
                )
            ax.add_patch(rect_gt)
            
            # Draw the prediction bbox in blue (to distinguish from red false positives)
            if self.bbox_format == 'xyxy':
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]), 
                    pred_crop_bbox[2] - pred_crop_bbox[0], 
                    pred_crop_bbox[3] - pred_crop_bbox[1],
                    linewidth=2, 
                    edgecolor='blue', 
                    facecolor='none',
                    linestyle='--',
                    label='Prediction'
                )
            else:  # xywh
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]), 
                    pred_crop_bbox[2], 
                    pred_crop_bbox[3],
                    linewidth=2, 
                    edgecolor='blue', 
                    facecolor='none',
                    linestyle='--',
                    label='Prediction'
                )
            ax.add_patch(rect_pred)
            
            # Add title with details
            title = f"True Positive: {tp['category_name']}\nScore: {tp['score']:.2f}, IoU: {tp['iou']:.2f}"
            ax.set_title(title, fontsize=10)
            
            # Add legend
            ax.legend(loc='lower right', fontsize=8)
            
            ax.axis('off')  # Hide axes for cleaner visualization
            
            # Save the figure
            img_name = os.path.basename(image_path)
            save_path = os.path.join(tp_dir, f"tp_{i}_{img_name}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing true positive {i} from image {image_id}: {str(e)}")
    
    print(f"True positive visualization complete. Results saved to {tp_dir}")

def visualize_false_positives(self, output_dir, max_samples=None, samples_per_class=None):
    """
    Visualize false positive predictions that don't match any ground truth object according to
    the optimal assignment (Hungarian algorithm or greedy fallback).
    This ensures consistency with the evaluation metrics.
    
    Args:
        output_dir: Directory to save the visualizations
        max_samples: Maximum number of samples to visualize across all classes (None for all)
        samples_per_class: Maximum number of samples per class (None for all)
    """
    if not self.images_dir:
        print("Error: Cannot visualize false positives without images directory.")
        return
    
    # Create base directory for false positive visualizations
    fp_base_dir = os.path.join(output_dir, "false_positives")
    os.makedirs(fp_base_dir, exist_ok=True)
    
    # Initialize a dictionary to store false positives by class
    false_positives_by_class = {}
    
    # Process each image using the same matching logic as in evaluate()
    print("Finding false positives using optimal assignment...")
    
    for image_id in set(list(self.gt_by_image.keys()) + list(self.pred_by_image.keys())):
        gt_boxes = self.gt_by_image.get(image_id, [])
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        
        # If no ground truth but has predictions, all are false positives
        if n_gt == 0 and n_pred > 0:
            for pred in pred_boxes:
                pred_cat_id = pred['category_id']
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                category_name = self.categories.get(pred_cat_id, 'unknown')
                
                # Add to the class-specific list
                if category_name not in false_positives_by_class:
                    false_positives_by_class[category_name] = []
                
                false_positives_by_class[category_name].append({
                    'image_id': image_id,
                    'bbox': pred['bbox'],
                    'category_id': pred_cat_id,
                    'category_name': category_name,
                    'score': pred.get('score', 1.0)
                })
            continue
        
        # If no predictions or both lists are empty, continue
        if n_pred == 0 or n_gt == 0:
            continue
        
        # Use Hungarian algorithm to find optimal matching, just like in evaluate()
        # Build cost matrix
        cost_matrix = np.zeros((n_gt, n_pred))
        
        for i, gt in enumerate(gt_boxes):
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for j, pred in enumerate(pred_boxes):
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                # If IoU is below threshold or categories don't match, cost is infinite
                if iou < self.iou_threshold:
                    cost_matrix[i, j] = float('inf')
                else:
                    # Negative IoU because Hungarian algorithm minimizes cost
                    cost_matrix[i, j] = -iou
        
        # Process matched pairs
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        try:
            # Try Hungarian algorithm for optimal matching
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process matched pairs from Hungarian algorithm
            for i, j in zip(row_indices, col_indices):
                # Skip invalid matches (below threshold)
                if cost_matrix[i, j] == float('inf'):
                    continue
                
                matched_gt_indices.add(i)
                matched_pred_indices.add(j)
                
        except (ValueError, ImportError) as e:
            # Fallback to greedy matching if Hungarian algorithm fails
            print(f"Warning: Hungarian algorithm failed on image {image_id}: {e}")
            print("Falling back to greedy matching algorithm...")
            
            # Create a list of (gt_index, pred_index, iou) tuples for all combinations
            all_pairs = []
            for i in range(n_gt):
                for j in range(n_pred):
                    gt_bbox = gt_boxes[i]['bbox']
                    pred_bbox = pred_boxes[j]['bbox']
                    iou = self.calculate_iou(gt_bbox, pred_bbox)
                    
                    if iou >= self.iou_threshold:
                        all_pairs.append((i, j, iou))
            
            # Sort by IoU (highest first)
            all_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Greedily match pairs
            for i, j, iou in all_pairs:
                if i not in matched_gt_indices and j not in matched_pred_indices:
                    matched_gt_indices.add(i)
                    matched_pred_indices.add(j)
        
        # Now, detect unmatched predictions (these are pure false positives)
        for j in range(n_pred):
            if j not in matched_pred_indices:
                # This is a false positive from optimal assignment
                pred = pred_boxes[j]
                pred_cat_id = pred['category_id']
                
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                category_name = self.categories.get(pred_cat_id, 'unknown')
                
                # Add to the class-specific list
                if category_name not in false_positives_by_class:
                    false_positives_by_class[category_name] = []
                
                false_positives_by_class[category_name].append({
                    'image_id': image_id,
                    'bbox': pred['bbox'],
                    'category_id': pred_cat_id,
                    'category_name': category_name,
                    'score': pred.get('score', 1.0)
                })
    
    # Create a safe path function to convert category names to valid folder names
    def safe_path(name):
        return name.replace(' ', '_').replace('/', '-').replace('\\', '-').replace(':', '_')
    
    # Track total samples across all classes
    total_visualized = 0
    total_fps = sum(len(fps) for fps in false_positives_by_class.values())
    
    print(f"Found {total_fps} pure false positives (not matching any ground truth in optimal assignment)")
    
    # Process each class
    for category_name, false_positives_info in false_positives_by_class.items():
        # Create class-specific directory with safe name
        safe_category = safe_path(category_name)
        class_dir = os.path.join(fp_base_dir, safe_category)
        os.makedirs(class_dir, exist_ok=True)
        
        # Limit samples per class if specified
        samples = false_positives_info
        if samples_per_class is not None and len(samples) > samples_per_class:
            import random
            samples = random.sample(samples, samples_per_class)
        
        # Check if we've reached the total max samples
        if max_samples is not None and total_visualized + len(samples) > max_samples:
            samples = samples[:max_samples - total_visualized]
        
        print(f"Visualizing {len(samples)} pure false positives for class '{category_name}'")
        
        # Generate visualizations for each false positive in this class
        for i, fp in enumerate(samples):
            image_id = fp['image_id']
            image_path = self.get_image_path(image_id)
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Could not find image for ID {image_id}")
                continue
            
            try:
                # Read the image
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                
                # Get centroid of the bounding box
                cx, cy = self.get_bbox_centroid(fp['bbox'])
                
                # Create a 100x100 crop centered on the object
                height, width = image.shape[:2]
                
                # Calculate crop boundaries
                crop_size = 100
                x1 = max(0, int(cx - crop_size / 2))
                y1 = max(0, int(cy - crop_size / 2))
                x2 = min(width, x1 + crop_size)
                y2 = min(height, y1 + crop_size)
                
                # Adjust if crop goes beyond image boundaries
                if x2 - x1 < crop_size:
                    if x1 == 0:
                        x2 = min(width, crop_size)
                    else:
                        x1 = max(0, x2 - crop_size)
                
                if y2 - y1 < crop_size:
                    if y1 == 0:
                        y2 = min(height, crop_size)
                    else:
                        y1 = max(0, y2 - crop_size)
                
                crop = image[y1:y2, x1:x2]
                
                # Convert bounding box to crop coordinates
                if self.bbox_format == 'xyxy':
                    x1_pred, y1_pred, x2_pred, y2_pred = fp['bbox']
                    pred_crop_bbox = [
                        max(0, x1_pred - x1),
                        max(0, y1_pred - y1),
                        min(crop_size, x2_pred - x1),
                        min(crop_size, y2_pred - y1)
                    ]
                else:  # xywh
                    x_pred, y_pred, w_pred, h_pred = fp['bbox']
                    pred_crop_bbox = [
                        max(0, x_pred - x1),
                        max(0, y_pred - y1),
                        min(crop_size, w_pred),
                        min(crop_size, h_pred)
                    ]
                
                # Create figure with clean visualization
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(crop)
                
                # Draw the prediction bbox in red (false positive)
                if self.bbox_format == 'xyxy':
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]), 
                        pred_crop_bbox[2] - pred_crop_bbox[0], 
                        pred_crop_bbox[3] - pred_crop_bbox[1],
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none',
                        linestyle='--',
                        label='False Positive'
                    )
                else:  # xywh
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]), 
                        pred_crop_bbox[2], 
                        pred_crop_bbox[3],
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none',
                        linestyle='--',
                        label='False Positive'
                    )
                ax.add_patch(rect_pred)
                
                # Add title with details
                title = f"False Positive: {fp['category_name']}\nScore: {fp['score']:.2f}"
                ax.set_title(title, fontsize=10)
                
                # Add legend
                ax.legend(loc='lower right', fontsize=8)
                
                ax.axis('off')  # Hide axes for cleaner visualization
                
                # Save the figure
                img_name = os.path.basename(image_path)
                save_path = os.path.join(class_dir, f"fp_{i}_{img_name}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Save blank crop with no annotations
                save_blank_crop(crop, save_path)
                
            except Exception as e:
                print(f"Error processing false positive {i} from image {image_id}: {str(e)}")
        
        # Update the total count
        total_visualized += len(samples)
        
        # Break if we've reached the max total samples
        if max_samples is not None and total_visualized >= max_samples:
            break
    
    print(f"Pure false positive visualization complete. Total samples: {total_visualized}")
    print(f"Results organized by class in {fp_base_dir}")

def visualize_false_negatives(self, output_dir, max_samples=None, samples_per_class=None):
    """
    Visualize false negative objects (ground truth without matching predictions according to
    the optimal assignment) and save them as crops.
    Uses the same matching logic as the evaluation function for consistency.
    
    Args:
        output_dir: Directory to save the visualizations
        max_samples: Maximum number of samples to visualize across all classes (None for all)
        samples_per_class: Maximum number of samples per class (None for all)
    """
    if not self.images_dir:
        print("Error: Cannot visualize false negatives without images directory.")
        return
    
    # Create base directory for false negative visualizations
    fn_base_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fn_base_dir, exist_ok=True)
    
    # Initialize a dictionary to store false negatives by class
    false_negatives_by_class = {}
    
    # Process each image using the same matching logic as in evaluate()
    print("Finding false negatives using optimal assignment...")
    
    for image_id in self.gt_by_image.keys():
        gt_boxes = self.gt_by_image[image_id]
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        
        # If has ground truth but no predictions, all are false negatives
        if n_pred == 0 and n_gt > 0:
            for gt in gt_boxes:
                gt_cat_id = gt['category_id']
                category_name = self.categories.get(gt_cat_id, 'unknown')
                
                # Add to the class-specific list
                if category_name not in false_negatives_by_class:
                    false_negatives_by_class[category_name] = []
                
                false_negatives_by_class[category_name].append({
                    'image_id': image_id,
                    'bbox': gt['bbox'],
                    'category_id': gt_cat_id,
                    'category_name': category_name
                })
            continue
        
        # If no ground truth or both lists are empty, continue
        if n_gt == 0 or n_pred == 0:
            continue
        
        # Use Hungarian algorithm to find optimal matching, just like in evaluate()
        # Build cost matrix
        cost_matrix = np.zeros((n_gt, n_pred))
        
        for i, gt in enumerate(gt_boxes):
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for j, pred in enumerate(pred_boxes):
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                # If IoU is below threshold, cost is infinite
                if iou < self.iou_threshold:
                    cost_matrix[i, j] = float('inf')
                else:
                    # Negative IoU because Hungarian algorithm minimizes cost
                    cost_matrix[i, j] = -iou
        
        # Process matched pairs
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        try:
            # Try Hungarian algorithm for optimal matching
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process matched pairs from Hungarian algorithm
            for i, j in zip(row_indices, col_indices):
                # Skip invalid matches (below threshold)
                if cost_matrix[i, j] == float('inf'):
                    continue
                
                matched_gt_indices.add(i)
                matched_pred_indices.add(j)
                
        except (ValueError, ImportError) as e:
            # Fallback to greedy matching if Hungarian algorithm fails
            print(f"Warning: Hungarian algorithm failed on image {image_id}: {e}")
            print("Falling back to greedy matching algorithm...")
            
            # Create a list of (gt_index, pred_index, iou) tuples for all combinations
            all_pairs = []
            for i in range(n_gt):
                for j in range(n_pred):
                    gt_bbox = gt_boxes[i]['bbox']
                    pred_bbox = pred_boxes[j]['bbox']
                    iou = self.calculate_iou(gt_bbox, pred_bbox)
                    
                    if iou >= self.iou_threshold:
                        all_pairs.append((i, j, iou))
            
            # Sort by IoU (highest first)
            all_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Greedily match pairs
            for i, j, iou in all_pairs:
                if i not in matched_gt_indices and j not in matched_pred_indices:
                    matched_gt_indices.add(i)
                    matched_pred_indices.add(j)
        
        # Now, detect unmatched ground truths (these are false negatives)
        for i in range(n_gt):
            if i not in matched_gt_indices:
                # This is a false negative from optimal assignment
                gt = gt_boxes[i]
                gt_cat_id = gt['category_id']
                category_name = self.categories.get(gt_cat_id, 'unknown')
                
                # Add to the class-specific list
                if category_name not in false_negatives_by_class:
                    false_negatives_by_class[category_name] = []
                
                false_negatives_by_class[category_name].append({
                    'image_id': image_id,
                    'bbox': gt['bbox'],
                    'category_id': gt_cat_id,
                    'category_name': category_name
                })
    
    # Create a safe path function to convert category names to valid folder names
    def safe_path(name):
        return name.replace(' ', '_').replace('/', '-').replace('\\', '-').replace(':', '_')
    
    # Track total samples across all classes
    total_visualized = 0
    total_fns = sum(len(fns) for fns in false_negatives_by_class.values())
    
    print(f"Found {total_fns} false negatives (ground truths without matching predictions in optimal assignment)")
    
    # Process each class
    for category_name, false_negatives_info in false_negatives_by_class.items():
        # Create class-specific directory with safe name
        safe_category = safe_path(category_name)
        class_dir = os.path.join(fn_base_dir, safe_category)
        os.makedirs(class_dir, exist_ok=True)
        
        # Limit samples per class if specified
        samples = false_negatives_info
        if samples_per_class is not None and len(samples) > samples_per_class:
            import random
            samples = random.sample(samples, samples_per_class)
        
        # Check if we've reached the total max samples
        if max_samples is not None and total_visualized + len(samples) > max_samples:
            samples = samples[:max_samples - total_visualized]
        
        print(f"Visualizing {len(samples)} false negatives for class '{category_name}'")
        
        # Generate visualizations for each false negative in this class
        for i, fn in enumerate(samples):
            image_id = fn['image_id']
            image_path = self.get_image_path(image_id)
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Could not find image for ID {image_id}")
                continue
            
            try:
                # Read the image
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                
                # Get centroid of the bounding box
                cx, cy = self.get_bbox_centroid(fn['bbox'])
                
                # Create a 100x100 crop centered on the object
                height, width = image.shape[:2]
                
                # Calculate crop boundaries
                crop_size = 100
                x1 = max(0, int(cx - crop_size / 2))
                y1 = max(0, int(cy - crop_size / 2))
                x2 = min(width, x1 + crop_size)
                y2 = min(height, y1 + crop_size)
                
                # Adjust if crop goes beyond image boundaries
                if x2 - x1 < crop_size:
                    if x1 == 0:
                        x2 = min(width, crop_size)
                    else:
                        x1 = max(0, x2 - crop_size)
                
                if y2 - y1 < crop_size:
                    if y1 == 0:
                        y2 = min(height, crop_size)
                    else:
                        y1 = max(0, y2 - crop_size)
                
                crop = image[y1:y2, x1:x2]
                
                # Convert bounding box to crop coordinates
                if self.bbox_format == 'xyxy':
                    x1_gt, y1_gt, x2_gt, y2_gt = fn['bbox']
                    gt_crop_bbox = [
                        max(0, x1_gt - x1),
                        max(0, y1_gt - y1),
                        min(crop_size, x2_gt - x1),
                        min(crop_size, y2_gt - y1)
                    ]
                else:  # xywh
                    x_gt, y_gt, w_gt, h_gt = fn['bbox']
                    gt_crop_bbox = [
                        max(0, x_gt - x1),
                        max(0, y_gt - y1),
                        min(crop_size, w_gt),
                        min(crop_size, h_gt)
                    ]
                
                # Create figure with clean visualization
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(crop)
                
                # Draw the ground truth bbox in green
                if self.bbox_format == 'xyxy':
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]), 
                        gt_crop_bbox[2] - gt_crop_bbox[0], 
                        gt_crop_bbox[3] - gt_crop_bbox[1],
                        linewidth=2, 
                        edgecolor=GT_COLOR, 
                        facecolor='none',
                        label='Missed Ground Truth'
                    )
                else:  # xywh
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]), 
                        gt_crop_bbox[2], 
                        gt_crop_bbox[3],
                        linewidth=2, 
                        edgecolor=GT_COLOR, 
                        facecolor='none',
                        label='Missed Ground Truth'
                    )
                ax.add_patch(rect_gt)
                
                # Add title with details
                title = f"False Negative: {fn['category_name']}\n(Missed Detection)"
                ax.set_title(title, fontsize=10)
                
                # Add legend
                ax.legend(loc='lower right', fontsize=8)
                
                ax.axis('off')  # Hide axes for cleaner visualization
                
                # Save the figure
                img_name = os.path.basename(image_path)
                save_path = os.path.join(class_dir, f"fn_{i}_{img_name}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Save blank crop with no annotations
                save_blank_crop(crop, save_path)
                
            except Exception as e:
                print(f"Error processing false negative {i} from image {image_id}: {str(e)}")
        
        # Update the total count
        total_visualized += len(samples)
        
        # Break if we've reached the max total samples
        if max_samples is not None and total_visualized >= max_samples:
            break
    
    print(f"False negative visualization complete. Total samples: {total_visualized}")
    print(f"Results organized by class in {fn_base_dir}")

def visualize_all_detections(self, output_dir, image_ids=None, max_images=10):
    """
    Visualize all detections (TP, FP, FN) together on original images.
    
    Args:
        output_dir: Directory to save the visualizations
        image_ids: List of specific image IDs to visualize (if None, random selection)
        max_images: Maximum number of images to visualize
    """
    if not self.images_dir:
        print("Error: Cannot visualize detections without images directory.")
        return
    
    # Create directory for full image visualizations
    full_viz_dir = os.path.join(output_dir, "full_visualizations")
    os.makedirs(full_viz_dir, exist_ok=True)
    
    # Get a list of image IDs to visualize
    if image_ids is None:
        # Select random image IDs from those with annotations
        all_image_ids = list(self.gt_by_image.keys())
        if len(all_image_ids) > max_images:
            image_ids = random.sample(all_image_ids, max_images)
        else:
            image_ids = all_image_ids
    
    print(f"Visualizing detections for {len(image_ids)} images...")
    
    # Visualize each image
    for idx, image_id in enumerate(image_ids):
        image_path = self.get_image_path(image_id)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Could not find image for ID {image_id}")
            continue
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
            
            # Get ground truth and predictions for this image
            gt_boxes = self.gt_by_image.get(image_id, [])
            pred_boxes = self.pred_by_image.get(image_id, [])
            
            # Create a figure for this image
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image)
            
            # Track matches to identify TP, FP, FN
            matched_gt_indices = set()
            matched_pred_indices = set()
            
            # First, find matches (true positives)
            for i_gt, gt in enumerate(gt_boxes):
                gt_bbox = gt['bbox']
                gt_cat_id = gt['category_id']
                gt_cat_name = self.categories.get(gt_cat_id, 'unknown')
                
                best_iou = 0
                best_pred_idx = -1
                
                for i_pred, pred in enumerate(pred_boxes):
                    pred_bbox = pred['bbox']
                    pred_cat_id = pred['category_id']
                    
                    # Handle string category IDs if needed
                    if isinstance(pred_cat_id, str):
                        if hasattr(self, 'category_name_to_string'):
                            pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                        else:
                            # Try global if available
                            try:
                                from __main__ import category_name_to_string
                                pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                            except (ImportError, AttributeError):
                                pass
                    
                    iou = self.calculate_iou(gt_bbox, pred_bbox)
                    if iou >= self.iou_threshold and pred_cat_id == gt_cat_id and iou > best_iou:
                        best_iou = iou
                        best_pred_idx = i_pred
                
                if best_pred_idx >= 0:
                    # We found a match - this is a true positive
                    matched_gt_indices.add(i_gt)
                    matched_pred_indices.add(best_pred_idx)
                    
                    # Draw the ground truth box in green
                    visualize_box(image, gt_bbox, True, gt_cat_name, 
                                None, self.bbox_format, ax, color='green')
                    
                    # Draw the prediction box in blue (for TP)
                    pred = pred_boxes[best_pred_idx]
                    visualize_box(image, pred['bbox'], False, None,
                                pred.get('score', None), self.bbox_format, ax, color='blue')
            
            # Draw false negatives (ground truth without match) in green with 'MISSED' label
            for i_gt, gt in enumerate(gt_boxes):
                if i_gt not in matched_gt_indices:
                    gt_cat_id = gt['category_id']
                    gt_cat_name = self.categories.get(gt_cat_id, 'unknown') + " (MISSED)"
                    visualize_box(image, gt['bbox'], True, gt_cat_name, 
                                None, self.bbox_format, ax, color='green')
            
            # Draw false positives (predictions without match) in red
            for i_pred, pred in enumerate(pred_boxes):
                if i_pred not in matched_pred_indices:
                    pred_cat_id = pred['category_id']
                    if isinstance(pred_cat_id, str):
                        if hasattr(self, 'category_name_to_string'):
                            pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                        else:
                            # Try global if available
                            try:
                                from __main__ import category_name_to_string
                                pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                            except (ImportError, AttributeError):
                                pass
                    pred_cat_name = self.categories.get(pred_cat_id, 'unknown') + " (FP)"
                    visualize_box(image, pred['bbox'], False, pred_cat_name,
                                pred.get('score', None), self.bbox_format, ax, color='red')
            
            # Add legend
            gt_line = Line2D([0], [0], color='green', linewidth=2, label='Ground Truth')
            tp_line = Line2D([0], [0], color='blue', linewidth=2, label='True Positive')
            fp_line = Line2D([0], [0], color='red', linewidth=2, label='False Positive')
            ax.legend(handles=[gt_line, tp_line, fp_line], loc='upper right')
            
            # Add title with stats
            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)
            n_tp = len(matched_gt_indices)
            n_fp = n_pred - n_tp
            n_fn = n_gt - n_tp
            
            title = f"Detection Results - Image ID: {image_id}\n"
            title += f"GT: {n_gt}, Pred: {n_pred}, TP: {n_tp}, FP: {n_fp}, FN: {n_fn}"
            ax.set_title(title, fontsize=14)
            
            ax.axis('off')
            
            # Save the figure
            img_name = os.path.basename(image_path)
            save_path = os.path.join(full_viz_dir, f"detections_{idx}_{img_name}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error visualizing detections for image {image_id}: {str(e)}")
    
    print(f"Full detection visualization complete. Results saved to {full_viz_dir}")

    
def create_true_positives_grid(self, output_dir, max_samples=None, grid_size=(4, 5)):
    """
    Create a grid visualization showing multiple true positive examples.
    
    Args:
        output_dir: Directory to save the visualization
        max_samples: Maximum number of samples to include (None for all)
        grid_size: Tuple of (rows, cols) for the grid layout
    """
    if not self.images_dir:
        print("Error: Cannot create true positives grid without images directory.")
        return
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import random
    
    # Create output directory
    tp_dir = os.path.join(output_dir, "true_positives")
    os.makedirs(tp_dir, exist_ok=True)
    
    # Get true positives info
    true_positives_info = []
    
    # Collect all true positive matches
    for image_id in self.gt_by_image.keys():
        gt_boxes = self.gt_by_image[image_id]
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        if not gt_boxes or not pred_boxes:
            continue
        
        # Find matches
        for gt in gt_boxes:
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Handle string category IDs
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                # Check for a match
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                if iou >= self.iou_threshold and pred_cat_id == gt_cat_id:
                    true_positives_info.append({
                        'image_id': image_id,
                        'gt_bbox': gt_bbox,
                        'pred_bbox': pred_bbox,
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0),
                        'iou': iou
                    })
    
    # If no true positives found
    if not true_positives_info:
        print("No true positives found for grid visualization.")
        return
    
    # Group by category
    tp_by_category = {}
    for tp in true_positives_info:
        cat_name = tp['category_name']
        if cat_name not in tp_by_category:
            tp_by_category[cat_name] = []
        tp_by_category[cat_name].append(tp)
    
    # Sort categories by number of examples (most common first)
    sorted_categories = sorted(tp_by_category.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Determine how many to show per category
    rows, cols = grid_size
    grid_cells = rows * cols
    
    # Get samples for grid
    grid_samples = []
    
    # Strategy 1: Try to take examples from each category
    samples_per_category = grid_cells // len(sorted_categories)
    if samples_per_category > 0:
        for cat_name, samples in sorted_categories:
            if len(samples) > samples_per_category:
                selected = random.sample(samples, samples_per_category)
            else:
                selected = samples
            grid_samples.extend(selected)
            if len(grid_samples) >= grid_cells:
                break
    
    # If we still have space, fill with random samples
    if len(grid_samples) < grid_cells:
        remaining_slots = grid_cells - len(grid_samples)
        all_samples = [s for cat_samples in tp_by_category.values() for s in cat_samples 
                       if s not in grid_samples]
        if remaining_slots > len(all_samples):
            remaining_slots = len(all_samples)
        if remaining_slots > 0:
            grid_samples.extend(random.sample(all_samples, remaining_slots))
    
    # If we have too many, trim
    if len(grid_samples) > grid_cells:
        grid_samples = grid_samples[:grid_cells]
    
    # Create the grid figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if grid_cells > 1 else [axes]
    
    # Function to process each sample
    def process_sample(ax, tp):
        image_id = tp['image_id']
        image_path = self.get_image_path(image_id)
        
        if not image_path or not os.path.exists(image_path):
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
            ax.axis('off')
            return
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                ax.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                ax.axis('off')
                return
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get centroid and create crop
            cx, cy = self.get_bbox_centroid(tp['gt_bbox'])
            height, width = image.shape[:2]
            
            crop_size = 100
            x1 = max(0, int(cx - crop_size / 2))
            y1 = max(0, int(cy - crop_size / 2))
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if crop goes beyond boundaries
            if x2 - x1 < crop_size:
                if x1 == 0:
                    x2 = min(width, crop_size)
                else:
                    x1 = max(0, x2 - crop_size)
            
            if y2 - y1 < crop_size:
                if y1 == 0:
                    y2 = min(height, crop_size)
                else:
                    y1 = max(0, y2 - crop_size)
            
            crop = image[y1:y2, x1:x2]
            
            # Convert bounding boxes to crop coordinates
            def convert_bbox_to_crop(bbox):
                if self.bbox_format == 'xyxy':
                    x1_b, y1_b, x2_b, y2_b = bbox
                    return [
                        max(0, x1_b - x1),
                        max(0, y1_b - y1),
                        min(crop_size, x2_b - x1),
                        min(crop_size, y2_b - y1)
                    ]
                else:  # xywh
                    x_b, y_b, w_b, h_b = bbox
                    return [
                        max(0, x_b - x1),
                        max(0, y_b - y1),
                        min(crop_size, w_b),
                        min(crop_size, h_b)
                    ]
            
            gt_crop_bbox = convert_bbox_to_crop(tp['gt_bbox'])
            pred_crop_bbox = convert_bbox_to_crop(tp['pred_bbox'])
            
            # Display the crop
            ax.imshow(crop)
            
            # Draw ground truth box (green)
            if self.bbox_format == 'xyxy':
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]),
                    gt_crop_bbox[2] - gt_crop_bbox[0],
                    gt_crop_bbox[3] - gt_crop_bbox[1],
                    linewidth=1.5,
                    edgecolor=GT_COLOR,
                    facecolor='none'
                )
            else:
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]),
                    gt_crop_bbox[2],
                    gt_crop_bbox[3],
                    linewidth=1.5,
                    edgecolor=GT_COLOR,
                    facecolor='none'
                )
            ax.add_patch(rect_gt)
            
            # Draw prediction box (blue)
            if self.bbox_format == 'xyxy':
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]),
                    pred_crop_bbox[2] - pred_crop_bbox[0],
                    pred_crop_bbox[3] - pred_crop_bbox[1],
                    linewidth=1.5,
                    edgecolor='blue',
                    facecolor='none',
                    linestyle='--'
                )
            else:
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]),
                    pred_crop_bbox[2],
                    pred_crop_bbox[3],
                    linewidth=1.5,
                    edgecolor='blue',
                    facecolor='none',
                    linestyle='--'
                )
            ax.add_patch(rect_pred)
            
            # Add title
            title = f"{tp['category_name']}\nIoU: {tp['iou']:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=6, wrap=True)
            ax.axis('off')
    
    # Process each sample in the grid
    for i, tp in enumerate(grid_samples):
        if i < len(axes):
            process_sample(axes[i], tp)
    
    # Hide any unused axes
    for i in range(len(grid_samples), len(axes)):
        axes[i].axis('off')
    
    # Add a global legend for all subplots
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GT_COLOR, lw=2, label='Ground Truth'),
        Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10)
    
    # Set a main title
    fig.suptitle(f'True Positive Examples (IoU ≥ {self.iou_threshold})', fontsize=16)
    
    # Add some spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.subplots_adjust(top=0.9)  # Adjust for the legend
    
    # Save the grid
    save_path = os.path.join(tp_dir, "true_positives_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"True positives grid visualization saved to {save_path}")

def visualize_true_positives_side_by_side(self, output_dir, max_samples=None):
    """
    Visualize correctly detected objects (true positives) with ground truth and 
    prediction bounding boxes on separate images, shown side by side.
    Also saves a blank version of the crop with no annotations.
    
    Args:
        output_dir: Directory to save the visualizations
        max_samples: Maximum number of samples to visualize (None for all)
    """
    if not self.images_dir:
        print("Error: Cannot visualize true positives without images directory.")
        return
    
    # Create directory for true positive visualizations
    tp_dir = os.path.join(output_dir, "true_positives")
    os.makedirs(tp_dir, exist_ok=True)
    
    # Create a list to store true positive information
    true_positives_info = []
    
    # Collect all true positive matches
    for image_id in self.gt_by_image.keys():
        gt_boxes = self.gt_by_image[image_id]
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        if not gt_boxes or not pred_boxes:
            continue
        
        # Build matches using IoU threshold
        for gt in gt_boxes:
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Convert string-based category ID if needed
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        # Try global if available
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                # Check if this is a correct match (same category and sufficient IoU)
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                if iou >= self.iou_threshold and pred_cat_id == gt_cat_id:
                    true_positives_info.append({
                        'image_id': image_id,
                        'gt_bbox': gt_bbox,
                        'pred_bbox': pred_bbox,
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0),
                        'iou': iou
                    })
    
    # Select a random subset if max_samples is provided
    if max_samples is not None and len(true_positives_info) > max_samples:
        import random
        true_positives_info = random.sample(true_positives_info, max_samples)
    
    print(f"Visualizing {len(true_positives_info)} true positive samples in side-by-side format...")
    
    # Generate visualizations for each true positive
    for i, tp in enumerate(true_positives_info):
        image_id = tp['image_id']
        image_path = self.get_image_path(image_id)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Could not find image for ID {image_id}")
            continue
        
        try:
            # Read the image
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
            
            # Get centroid of the bounding box (using average of GT and pred for better centering)
            gt_cx, gt_cy = self.get_bbox_centroid(tp['gt_bbox'])
            pred_cx, pred_cy = self.get_bbox_centroid(tp['pred_bbox'])
            cx = (gt_cx + pred_cx) / 2
            cy = (gt_cy + pred_cy) / 2
            
            # Create a 100x100 crop centered on the object
            height, width = image.shape[:2]
            
            # Calculate crop boundaries
            crop_size = 100
            x1 = max(0, int(cx - crop_size / 2))
            y1 = max(0, int(cy - crop_size / 2))
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if crop goes beyond image boundaries
            if x2 - x1 < crop_size:
                if x1 == 0:
                    x2 = min(width, crop_size)
                else:
                    x1 = max(0, x2 - crop_size)
            
            if y2 - y1 < crop_size:
                if y1 == 0:
                    y2 = min(height, crop_size)
                else:
                    y1 = max(0, y2 - crop_size)
            
            crop = image[y1:y2, x1:x2].copy()
            
            # Create a copy of the crop for each display
            gt_crop = crop.copy()
            pred_crop = crop.copy()
            
            # Convert ground truth bounding box to crop coordinates
            if self.bbox_format == 'xyxy':
                x1_gt, y1_gt, x2_gt, y2_gt = tp['gt_bbox']
                gt_crop_bbox = [
                    max(0, x1_gt - x1),
                    max(0, y1_gt - y1),
                    min(crop_size, x2_gt - x1),
                    min(crop_size, y2_gt - y1)
                ]
            else:  # xywh
                x_gt, y_gt, w_gt, h_gt = tp['gt_bbox']
                gt_crop_bbox = [
                    max(0, x_gt - x1),
                    max(0, y_gt - y1),
                    min(crop_size, w_gt),
                    min(crop_size, h_gt)
                ]
            
            # Convert prediction bounding box to crop coordinates
            if self.bbox_format == 'xyxy':
                x1_pred, y1_pred, x2_pred, y2_pred = tp['pred_bbox']
                pred_crop_bbox = [
                    max(0, x1_pred - x1),
                    max(0, y1_pred - y1),
                    min(crop_size, x2_pred - x1),
                    min(crop_size, y2_pred - y1)
                ]
            else:  # xywh
                x_pred, y_pred, w_pred, h_pred = tp['pred_bbox']
                pred_crop_bbox = [
                    max(0, x_pred - x1),
                    max(0, y_pred - y1),
                    min(crop_size, w_pred),
                    min(crop_size, h_pred)
                ]
            
            # Create a figure with two side-by-side subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot ground truth on the left
            ax1.imshow(gt_crop)
            if self.bbox_format == 'xyxy':
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]), 
                    gt_crop_bbox[2] - gt_crop_bbox[0], 
                    gt_crop_bbox[3] - gt_crop_bbox[1],
                    linewidth=2, 
                    edgecolor=GT_COLOR, 
                    facecolor='none'
                )
            else:  # xywh
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]), 
                    gt_crop_bbox[2], 
                    gt_crop_bbox[3],
                    linewidth=2, 
                    edgecolor=GT_COLOR, 
                    facecolor='none'
                )
            ax1.add_patch(rect_gt)
            ax1.set_title("Ground Truth", fontsize=12)
            ax1.axis('off')
            
            # Plot prediction on the right
            ax2.imshow(pred_crop)
            if self.bbox_format == 'xyxy':
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]), 
                    pred_crop_bbox[2] - pred_crop_bbox[0], 
                    pred_crop_bbox[3] - pred_crop_bbox[1],
                    linewidth=2, 
                    edgecolor=PRED_COLOR, 
                    facecolor='none'
                )
            else:  # xywh
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]), 
                    pred_crop_bbox[2], 
                    pred_crop_bbox[3],
                    linewidth=2, 
                    edgecolor=PRED_COLOR, 
                    facecolor='none'
                )
            ax2.add_patch(rect_pred)
            ax2.set_title(f"Prediction (Score: {tp['score']:.2f})", fontsize=12)
            ax2.axis('off')
            
            # Add a common title with details
            fig.suptitle(f"True Positive: {tp['category_name']}, IoU: {tp['iou']:.2f}", fontsize=14)
            
            # Save the figure
            img_name = os.path.basename(image_path)
            save_path = os.path.join(tp_dir, f"tp_side_by_side_{i}_{img_name}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ADDED: Save blank crop with no annotations
            save_blank_crop(crop, save_path)
            
        except Exception as e:
            print(f"Error processing true positive {i} from image {image_id}: {str(e)}")
    
    print(f"Side-by-side true positive visualization complete. Results saved to {tp_dir}")

def create_true_positives_grid_side_by_side(self, output_dir, max_samples=None, grid_size=(3, 4)):
    """
    Create a grid visualization showing multiple true positive examples with ground truth
    and prediction side by side. Also saves blank versions of each crop.
    
    Args:
        output_dir: Directory to save the visualization
        max_samples: Maximum number of samples to include (None for all)
        grid_size: Tuple of (rows, cols) for the grid layout
    """
    if not self.images_dir:
        print("Error: Cannot create true positives grid without images directory.")
        return
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import random
    
    # Create output directory
    tp_dir = os.path.join(output_dir, "true_positives")
    os.makedirs(tp_dir, exist_ok=True)
    
    # Get true positives info
    true_positives_info = []
    
    # Collect all true positive matches
    for image_id in self.gt_by_image.keys():
        gt_boxes = self.gt_by_image[image_id]
        pred_boxes = self.pred_by_image.get(image_id, [])
        
        if not gt_boxes or not pred_boxes:
            continue
        
        # Find matches
        for gt in gt_boxes:
            gt_bbox = gt['bbox']
            gt_cat_id = gt['category_id']
            
            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                pred_cat_id = pred['category_id']
                
                # Handle string category IDs
                if isinstance(pred_cat_id, str):
                    if hasattr(self, 'category_name_to_string'):
                        pred_cat_id = self.category_name_to_string.get(pred_cat_id, pred_cat_id)
                    else:
                        try:
                            from __main__ import category_name_to_string
                            pred_cat_id = category_name_to_string.get(pred_cat_id, pred_cat_id)
                        except (ImportError, AttributeError):
                            pass
                
                # Check for a match
                iou = self.calculate_iou(gt_bbox, pred_bbox)
                if iou >= self.iou_threshold and pred_cat_id == gt_cat_id:
                    true_positives_info.append({
                        'image_id': image_id,
                        'gt_bbox': gt_bbox,
                        'pred_bbox': pred_bbox,
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0),
                        'iou': iou
                    })
    
    # If no true positives found
    if not true_positives_info:
        print("No true positives found for grid visualization.")
        return
    
    # Group by category
    tp_by_category = {}
    for tp in true_positives_info:
        cat_name = tp['category_name']
        if cat_name not in tp_by_category:
            tp_by_category[cat_name] = []
        tp_by_category[cat_name].append(tp)
    
    # Sort categories by number of examples (most common first)
    sorted_categories = sorted(tp_by_category.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Determine how many to show
    rows, cols = grid_size
    grid_cells = rows * cols
    
    # Get samples for grid
    grid_samples = []
    
    # Strategy 1: Try to take examples from each category
    samples_per_category = grid_cells // len(sorted_categories)
    if samples_per_category > 0:
        for cat_name, samples in sorted_categories:
            if len(samples) > samples_per_category:
                selected = random.sample(samples, samples_per_category)
            else:
                selected = samples
            grid_samples.extend(selected)
            if len(grid_samples) >= grid_cells:
                break
    
    # If we still have space, fill with random samples
    if len(grid_samples) < grid_cells:
        remaining_slots = grid_cells - len(grid_samples)
        all_samples = [s for cat_samples in tp_by_category.values() for s in cat_samples 
                       if s not in grid_samples]
        if remaining_slots > len(all_samples):
            remaining_slots = len(all_samples)
        if remaining_slots > 0:
            grid_samples.extend(random.sample(all_samples, remaining_slots))
    
    # If we have too many, trim
    if len(grid_samples) > grid_cells:
        grid_samples = grid_samples[:grid_cells]
    
    # Create the grid figure - each sample requires 2 side-by-side images
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    
    # Collect original crops for blank saving
    original_crops = []
    
    # Function to process each sample
    def process_sample(ax_gt, ax_pred, tp):
        image_id = tp['image_id']
        image_path = self.get_image_path(image_id)
        
        if not image_path or not os.path.exists(image_path):
            ax_gt.text(0.5, 0.5, "Image not found", ha='center', va='center')
            ax_pred.text(0.5, 0.5, "Image not found", ha='center', va='center')
            ax_gt.axis('off')
            ax_pred.axis('off')
            return None
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                ax_gt.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                ax_pred.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                ax_gt.axis('off')
                ax_pred.axis('off')
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get centroid (average of GT and prediction)
            gt_cx, gt_cy = self.get_bbox_centroid(tp['gt_bbox'])
            pred_cx, pred_cy = self.get_bbox_centroid(tp['pred_bbox'])
            cx = (gt_cx + pred_cx) / 2
            cy = (gt_cy + pred_cy) / 2
            
            # Create crop
            height, width = image.shape[:2]
            crop_size = 100
            
            x1 = max(0, int(cx - crop_size / 2))
            y1 = max(0, int(cy - crop_size / 2))
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if crop goes beyond boundaries
            if x2 - x1 < crop_size:
                if x1 == 0:
                    x2 = min(width, crop_size)
                else:
                    x1 = max(0, x2 - crop_size)
            
            if y2 - y1 < crop_size:
                if y1 == 0:
                    y2 = min(height, crop_size)
                else:
                    y1 = max(0, y2 - crop_size)
            
            crop = image[y1:y2, x1:x2].copy()
            
            # Create separate copies for GT and prediction
            gt_crop = crop.copy()
            pred_crop = crop.copy()
            
            # Convert bounding boxes to crop coordinates
            def convert_bbox_to_crop(bbox):
                if self.bbox_format == 'xyxy':
                    x1_b, y1_b, x2_b, y2_b = bbox
                    return [
                        max(0, x1_b - x1),
                        max(0, y1_b - y1),
                        min(crop_size, x2_b - x1),
                        min(crop_size, y2_b - y1)
                    ]
                else:  # xywh
                    x_b, y_b, w_b, h_b = bbox
                    return [
                        max(0, x_b - x1),
                        max(0, y_b - y1),
                        min(crop_size, w_b),
                        min(crop_size, h_b)
                    ]
            
            gt_crop_bbox = convert_bbox_to_crop(tp['gt_bbox'])
            pred_crop_bbox = convert_bbox_to_crop(tp['pred_bbox'])
            
            # Display ground truth with green box
            ax_gt.imshow(gt_crop)
            if self.bbox_format == 'xyxy':
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]),
                    gt_crop_bbox[2] - gt_crop_bbox[0],
                    gt_crop_bbox[3] - gt_crop_bbox[1],
                    linewidth=1.5,
                    edgecolor=GT_COLOR,
                    facecolor='none'
                )
            else:
                rect_gt = patches.Rectangle(
                    (gt_crop_bbox[0], gt_crop_bbox[1]),
                    gt_crop_bbox[2],
                    gt_crop_bbox[3],
                    linewidth=1.5,
                    edgecolor=GT_COLOR,
                    facecolor='none'
                )
            ax_gt.add_patch(rect_gt)
            ax_gt.set_title("GT", fontsize=8)
            ax_gt.axis('off')
            
            # Display prediction with red box
            ax_pred.imshow(pred_crop)
            if self.bbox_format == 'xyxy':
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]),
                    pred_crop_bbox[2] - pred_crop_bbox[0],
                    pred_crop_bbox[3] - pred_crop_bbox[1],
                    linewidth=1.5,
                    edgecolor=PRED_COLOR,
                    facecolor='none'
                )
            else:
                rect_pred = patches.Rectangle(
                    (pred_crop_bbox[0], pred_crop_bbox[1]),
                    pred_crop_bbox[2],
                    pred_crop_bbox[3],
                    linewidth=1.5,
                    edgecolor=PRED_COLOR,
                    facecolor='none'
                )
            ax_pred.add_patch(rect_pred)
            ax_pred.set_title(f"Pred ({tp['score']:.2f})", fontsize=8)
            ax_pred.axis('off')
            
            # Add a tiny title at the top of the first column
            if ax_gt.get_subplotspec().is_first_col():
                ax_gt.annotate(
                    f"{tp['category_name']}\nIoU: {tp['iou']:.2f}",
                    xy=(-0.1, 0.5),
                    xycoords='axes fraction',
                    va='center',
                    ha='right',
                    fontsize=8,
                    rotation=90
                )
            
            # Return the original crop for blank saving
            return {
                'crop': crop,
                'image_id': image_id,
                'category': tp['category_name'],
                'score': tp['score'],
                'iou': tp['iou']
            }
            
        except Exception as e:
            ax_gt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=6)
            ax_pred.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=6)
            ax_gt.axis('off')
            ax_pred.axis('off')
            return None
    
    # Process each sample in the grid
    for i, tp in enumerate(grid_samples):
        if i < rows * cols:
            # Calculate row and column indices
            row = i // cols
            col = i % cols
            
            # Each sample uses 2 columns
            ax_gt = axes[row, col * 2]
            ax_pred = axes[row, col * 2 + 1]
            
            crop_info = process_sample(ax_gt, ax_pred, tp)
            if crop_info:
                original_crops.append({
                    'index': i,
                    **crop_info
                })
    
    # Hide any unused axes
    for i in range(len(grid_samples), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')
    
    # Set a main title
    fig.suptitle(f'True Positive Examples (IoU ≥ {self.iou_threshold})', fontsize=16)
    
    # Add legend at the bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GT_COLOR, lw=2, label='Ground Truth'),
        Line2D([0], [0], color=PRED_COLOR, lw=2, label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    
    # Adjust the spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the grid
    save_path = os.path.join(tp_dir, "true_positives_grid_side_by_side.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save individual blank crops
    for crop_info in original_crops:
        blank_filename = f"true_positive_{crop_info['category']}_{crop_info['index']}_blank.png"
        blank_save_path = os.path.join(tp_dir, blank_filename)
        save_blank_crop(crop_info['crop'], blank_save_path)
    
    print(f"Side-by-side true positives grid visualization saved to {save_path}")
    print(f"Also saved {len(original_crops)} blank crops to {tp_dir}")

def visualize_misclassifications_side_by_side(self, output_dir, max_samples=None):
    """
    Visualize misclassifications (correct box but wrong class) with ground truth and 
    prediction bounding boxes on separate images, shown side by side.
    Also saves a blank version of each crop.
    
    Args:
        output_dir: Directory to save the visualizations
        max_samples: Maximum number of samples to visualize (None for all)
    """
    if not self.images_dir:
        print("Error: Cannot visualize misclassifications without images directory.")
        return
    
    # Create base directory for misclassifications
    misclass_base_dir = os.path.join(output_dir, "misclassifications")
    os.makedirs(misclass_base_dir, exist_ok=True)
    
    # Create a specific directory for side-by-side visualizations
    side_by_side_dir = os.path.join(misclass_base_dir, "side_by_side")
    os.makedirs(side_by_side_dir, exist_ok=True)
    
    # Group by misclassification type (ground truth category -> predicted category)
    misclass_by_type = defaultdict(list)
    
    for mc in self.misclassifications_info:
        gt_cat_name = mc['gt_category_name']
        pred_cat_name = mc['category_name']
        # Create a clean directory name
        misclass_type = f"{gt_cat_name}_truth_{pred_cat_name}_predicted"
        # Replace any problematic characters for directory names
        misclass_type = misclass_type.replace(" ", "_").replace("/", "-").replace("\\", "-")
        misclass_by_type[misclass_type].append(mc)
    
    # Process each misclassification type
    for misclass_type, items in misclass_by_type.items():
        # Create directory for this specific misclassification type
        type_dir = os.path.join(side_by_side_dir, misclass_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # Limit samples if requested
        samples = items
        if max_samples is not None and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        print(f"Visualizing {len(samples)} samples of misclassification type '{misclass_type}' in side-by-side format...")
        
        # Generate individual visualizations for each sample
        for i, item in enumerate(samples):
            image_id = item['image_id']
            image_path = self.get_image_path(image_id)
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Could not find image for ID {image_id}")
                continue
            
            try:
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                
                # Get centroid of the bounding box (using average of GT and pred for better centering)
                gt_cx, gt_cy = self.get_bbox_centroid(item['gt_bbox'])
                pred_cx, pred_cy = self.get_bbox_centroid(item['bbox'])
                cx = (gt_cx + pred_cx) / 2
                cy = (gt_cy + pred_cy) / 2
                
                # Create a crop centered on the object
                height, width = image.shape[:2]
                
                # Calculate crop boundaries
                crop_size = 100
                x1 = max(0, int(cx - crop_size / 2))
                y1 = max(0, int(cy - crop_size / 2))
                x2 = min(width, x1 + crop_size)
                y2 = min(height, y1 + crop_size)
                
                # Adjust if crop goes beyond image boundaries
                if x2 - x1 < crop_size:
                    if x1 == 0:
                        x2 = min(width, crop_size)
                    else:
                        x1 = max(0, x2 - crop_size)
                
                if y2 - y1 < crop_size:
                    if y1 == 0:
                        y2 = min(height, crop_size)
                    else:
                        y1 = max(0, y2 - crop_size)
                
                crop = image[y1:y2, x1:x2].copy()
                
                # Create a copy of the crop for each display
                gt_crop = crop.copy()
                pred_crop = crop.copy()
                
                # Convert ground truth bounding box to crop coordinates
                if self.bbox_format == 'xyxy':
                    x1_gt, y1_gt, x2_gt, y2_gt = item['gt_bbox']
                    gt_crop_bbox = [
                        max(0, x1_gt - x1),
                        max(0, y1_gt - y1),
                        min(crop_size, x2_gt - x1),
                        min(crop_size, y2_gt - y1)
                    ]
                else:  # xywh
                    x_gt, y_gt, w_gt, h_gt = item['gt_bbox']
                    gt_crop_bbox = [
                        max(0, x_gt - x1),
                        max(0, y_gt - y1),
                        min(crop_size, w_gt),
                        min(crop_size, h_gt)
                    ]
                
                # Convert prediction bounding box to crop coordinates
                if self.bbox_format == 'xyxy':
                    x1_pred, y1_pred, x2_pred, y2_pred = item['bbox']
                    pred_crop_bbox = [
                        max(0, x1_pred - x1),
                        max(0, y1_pred - y1),
                        min(crop_size, x2_pred - x1),
                        min(crop_size, y2_pred - y1)
                    ]
                else:  # xywh
                    x_pred, y_pred, w_pred, h_pred = item['bbox']
                    pred_crop_bbox = [
                        max(0, x_pred - x1),
                        max(0, y_pred - y1),
                        min(crop_size, w_pred),
                        min(crop_size, h_pred)
                    ]
                
                # Create a figure with two side-by-side subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Plot ground truth on the left
                ax1.imshow(gt_crop)
                if self.bbox_format == 'xyxy':
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]), 
                        gt_crop_bbox[2] - gt_crop_bbox[0], 
                        gt_crop_bbox[3] - gt_crop_bbox[1],
                        linewidth=2, 
                        edgecolor=GT_COLOR, 
                        facecolor='none'
                    )
                else:  # xywh
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]), 
                        gt_crop_bbox[2], 
                        gt_crop_bbox[3],
                        linewidth=2, 
                        edgecolor=GT_COLOR, 
                        facecolor='none'
                    )
                ax1.add_patch(rect_gt)
                ax1.set_title(f"Ground Truth: {item['gt_category_name']}", fontsize=12)
                ax1.axis('off')
                
                # Plot prediction on the right
                ax2.imshow(pred_crop)
                if self.bbox_format == 'xyxy':
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]), 
                        pred_crop_bbox[2] - pred_crop_bbox[0], 
                        pred_crop_bbox[3] - pred_crop_bbox[1],
                        linewidth=2, 
                        edgecolor=PRED_COLOR, 
                        facecolor='none'
                    )
                else:  # xywh
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]), 
                        pred_crop_bbox[2], 
                        pred_crop_bbox[3],
                        linewidth=2, 
                        edgecolor=PRED_COLOR, 
                        facecolor='none'
                    )
                ax2.add_patch(rect_pred)
                ax2.set_title(f"Prediction: {item['category_name']} (Score: {item['score']:.2f})", fontsize=12)
                ax2.axis('off')
                
                # Add a common title with details
                fig.suptitle(f"Misclassification: {item['gt_category_name']} → {item['category_name']}, IoU: {item['iou']:.2f}", fontsize=14)
                
                # Save the figure
                img_name = os.path.basename(image_path)
                save_path = os.path.join(type_dir, f"misclass_side_by_side_{i}_{img_name}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # ADDED: Save blank crop with no annotations
                save_blank_crop(crop, save_path)
                
            except Exception as e:
                print(f"Error processing misclassification {i} from image {image_id}: {str(e)}")

def create_misclassifications_grid_side_by_side(self, output_dir, max_samples=None, grid_size=(3, 3)):
    """
    Create a grid visualization showing multiple misclassification examples with ground truth
    and prediction side by side. Also saves blank versions of each crop.
    
    Args:
        output_dir: Directory to save the visualization
        max_samples: Maximum number of samples to include per misclassification type (None for all)
        grid_size: Tuple of (rows, cols) for the grid layout
    """
    if not self.images_dir:
        print("Error: Cannot create misclassifications grid without images directory.")
        return
    
    # Create base directory for misclassifications
    misclass_base_dir = os.path.join(output_dir, "misclassifications")
    os.makedirs(misclass_base_dir, exist_ok=True)
    
    # Group by misclassification type (ground truth category -> predicted category)
    misclass_by_type = defaultdict(list)
    
    for mc in self.misclassifications_info:
        gt_cat_name = mc['gt_category_name']
        pred_cat_name = mc['category_name']
        misclass_type = f"{gt_cat_name}_truth_{pred_cat_name}_predicted"
        misclass_type = misclass_type.replace(" ", "_").replace("/", "-").replace("\\", "-")
        misclass_by_type[misclass_type].append(mc)
    
    # Sort by frequency (most common first)
    sorted_types = sorted(misclass_by_type.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Process top misclassification types
    for misclass_type, items in sorted_types:
        if len(items) < 4:  # Skip types with too few examples to make a grid worthwhile
            continue
            
        print(f"Creating side-by-side grid for misclassification type: {misclass_type}")
        
        # Limit samples if requested
        samples = items
        if max_samples is not None and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        # Further limit to fit grid size
        rows, cols = grid_size
        grid_cells = rows * cols
        if len(samples) > grid_cells:
            samples = samples[:grid_cells]
        
        # Create the grid figure - each sample requires 2 side-by-side images
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
        
        # Flatten axes if needed
        if rows > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = np.array(axes).flatten() if cols > 1 else np.array([axes]).flatten()
        
        # Collect original crops for blank saving
        original_crops = []
        
        # Function to process each sample
        def process_sample(ax_gt, ax_pred, item):
            image_id = item['image_id']
            image_path = self.get_image_path(image_id)
            
            if not image_path or not os.path.exists(image_path):
                ax_gt.text(0.5, 0.5, "Image not found", ha='center', va='center')
                ax_pred.text(0.5, 0.5, "Image not found", ha='center', va='center')
                ax_gt.axis('off')
                ax_pred.axis('off')
                return None
            
            try:
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    ax_gt.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                    ax_pred.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                    ax_gt.axis('off')
                    ax_pred.axis('off')
                    return None
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get centroid (average of GT and prediction)
                gt_cx, gt_cy = self.get_bbox_centroid(item['gt_bbox'])
                pred_cx, pred_cy = self.get_bbox_centroid(item['bbox'])
                cx = (gt_cx + pred_cx) / 2
                cy = (gt_cy + pred_cy) / 2
                
                # Create crop
                height, width = image.shape[:2]
                crop_size = 100
                
                x1 = max(0, int(cx - crop_size / 2))
                y1 = max(0, int(cy - crop_size / 2))
                x2 = min(width, x1 + crop_size)
                y2 = min(height, y1 + crop_size)
                
                # Adjust if crop goes beyond boundaries
                if x2 - x1 < crop_size:
                    if x1 == 0:
                        x2 = min(width, crop_size)
                    else:
                        x1 = max(0, x2 - crop_size)
                
                if y2 - y1 < crop_size:
                    if y1 == 0:
                        y2 = min(height, crop_size)
                    else:
                        y1 = max(0, y2 - crop_size)
                
                crop = image[y1:y2, x1:x2].copy()
                
                # Create separate copies for GT and prediction
                gt_crop = crop.copy()
                pred_crop = crop.copy()
                
                # Convert bounding boxes to crop coordinates
                def convert_bbox_to_crop(bbox):
                    if self.bbox_format == 'xyxy':
                        x1_b, y1_b, x2_b, y2_b = bbox
                        return [
                            max(0, x1_b - x1),
                            max(0, y1_b - y1),
                            min(crop_size, x2_b - x1),
                            min(crop_size, y2_b - y1)
                        ]
                    else:  # xywh
                        x_b, y_b, w_b, h_b = bbox
                        return [
                            max(0, x_b - x1),
                            max(0, y_b - y1),
                            min(crop_size, w_b),
                            min(crop_size, h_b)
                        ]
                
                gt_crop_bbox = convert_bbox_to_crop(item['gt_bbox'])
                pred_crop_bbox = convert_bbox_to_crop(item['bbox'])
                
                # Display ground truth with green box
                ax_gt.imshow(gt_crop)
                if self.bbox_format == 'xyxy':
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]),
                        gt_crop_bbox[2] - gt_crop_bbox[0],
                        gt_crop_bbox[3] - gt_crop_bbox[1],
                        linewidth=1.5,
                        edgecolor=GT_COLOR,
                        facecolor='none'
                    )
                else:
                    rect_gt = patches.Rectangle(
                        (gt_crop_bbox[0], gt_crop_bbox[1]),
                        gt_crop_bbox[2],
                        gt_crop_bbox[3],
                        linewidth=1.5,
                        edgecolor=GT_COLOR,
                        facecolor='none'
                    )
                ax_gt.add_patch(rect_gt)
                ax_gt.set_title(f"GT: {item['gt_category_name']}", fontsize=8)
                ax_gt.axis('off')
                
                # Display prediction with blue box
                ax_pred.imshow(pred_crop)
                if self.bbox_format == 'xyxy':
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]),
                        pred_crop_bbox[2] - pred_crop_bbox[0],
                        pred_crop_bbox[3] - pred_crop_bbox[1],
                        linewidth=1.5,
                        edgecolor=PRED_COLOR,
                        facecolor='none'
                    )
                else:
                    rect_pred = patches.Rectangle(
                        (pred_crop_bbox[0], pred_crop_bbox[1]),
                        pred_crop_bbox[2],
                        pred_crop_bbox[3],
                        linewidth=1.5,
                        edgecolor=PRED_COLOR,
                        facecolor='none'
                    )
                ax_pred.add_patch(rect_pred)
                ax_pred.set_title(f"Pred: {item['category_name']} ({item['score']:.2f})", fontsize=8)
                ax_pred.axis('off')
                
                # Return the original crop for blank saving
                return {
                    'crop': crop,
                    'image_id': image_id,
                    'gt_category': item['gt_category_name'],
                    'pred_category': item['category_name']
                }
                
            except Exception as e:
                ax_gt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=6)
                ax_pred.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=6)
                ax_gt.axis('off')
                ax_pred.axis('off')
                return None
        
        # Process each sample in the grid
        for i, item in enumerate(samples):
            if i < rows * cols:
                # Calculate row and column indices
                row = i // cols
                col = i % cols
                
                # Each sample uses 2 columns
                if rows > 1:
                    ax_gt = axes[row, col * 2]
                    ax_pred = axes[row, col * 2 + 1]
                else:
                    # Handle special case for single row
                    ax_gt = axes[col * 2] if cols > 1 else axes[0]
                    ax_pred = axes[col * 2 + 1] if cols > 1 else axes[1]
                
                crop_info = process_sample(ax_gt, ax_pred, item)
                if crop_info:
                    original_crops.append({
                        'index': i,
                        **crop_info
                    })
        
        # Hide any unused axes
        for i in range(len(samples) * 2, rows * cols * 2):
            axes_flat[i].axis('off')
        
        # Set a main title
        gt_name = items[0]['gt_category_name']
        pred_name = items[0]['category_name']
        fig.suptitle(f'Misclassification: {gt_name} → {pred_name} (n={len(samples)})', fontsize=16)
        
        # Add legend at the bottom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=GT_COLOR, lw=2, label='Ground Truth'),
            Line2D([0], [0], color=PRED_COLOR, lw=2, label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
        
        # Adjust the spacing
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the grid
        safe_type = misclass_type.replace(" ", "_").replace("/", "-").replace("\\", "-")
        save_path = os.path.join(misclass_base_dir, f"{safe_type}_grid_side_by_side.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save individual blank crops
        for crop_info in original_crops:
            blank_filename = f"{safe_type}_sample_{crop_info['index']}_blank.png"
            blank_save_path = os.path.join(misclass_base_dir, blank_filename)
            save_blank_crop(crop_info['crop'], blank_save_path)
        
        print(f"Side-by-side misclassification grid saved to {save_path}")
        print(f"Also saved {len(original_crops)} blank crops to {misclass_base_dir}")