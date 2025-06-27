import os
import json
import ast
import types
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import argparse
from typing import Dict, List, Tuple, Set
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm
import cv2
from PIL import Image
import matplotlib.patches as patches

from visualize_box import * 

# (
#     visualize_box, 
#     visualize_polygon, 
#     create_overlay_visualization, 
#     visualize_true_positives,
#     visualize_true_positives_side_by_side,
#     create_true_positives_grid,
#     create_true_positives_grid_side_by_side,
#     visualize_all_detections,
# )

GT_COLOR = 'green'        
PRED_COLOR = '#196DEB'

category_name_to_string = {
    "objects": 0,
    "access_aisle": 1,
    "curbside": 2,
    "dp_no_aisle": 3,
    "dp_one_aisle": 4,
    "dp_two_aisle": 5,
    "one_aisle": 6,
    "two_aisle": 7,
}

class DetectionEvaluator:
    """Class for evaluating object detection results against ground truth annotations."""

    def __init__(self, annotations_path: str, predictions_path: str, iou_threshold: float = 0.5, 
                 bbox_format: str = 'xyxy', use_filename_as_id: bool = False,
                 images_dir: str = None):
        """
        Initialize the evaluator with paths to annotations and predictions.
        
        Args:
            annotations_path: Path to the ground truth COCO format JSON file
            predictions_path: Path to the model predictions JSON file
            iou_threshold: IoU threshold for considering a detection as correct
            bbox_format: Bounding box format: 'xyxy' for [x1,y1,x2,y2] or 'xywh' for [x,y,width,height]
            use_filename_as_id: If True, use image filenames as IDs instead of numeric IDs
            images_dir: Directory containing the original images (required for visualization)
        """
        self.annotations_path = annotations_path
        self.predictions_path = predictions_path
        self.iou_threshold = iou_threshold
        self.bbox_format = bbox_format
        self.use_filename_as_id = use_filename_as_id
        self.images_dir = images_dir
        
        # Create mapping from image ID to image data
        self.image_info = {}
        self.filename_to_id = {}

        # Load data
        self.annotations = self._load_annotations(annotations_path)

        for img in self.annotations['images']:
            self.image_info[img['id']] = {
                'file_name': img['file_name'],
                'height': img['height'],
                'width': img['width']
            }
            # Create mapping from filename to ID for use with filename-based predictions
            self.filename_to_id[img['file_name']] = img['id']
            # Also map the name without path and extension
            base_name = os.path.basename(img['file_name'])
            name_without_ext = os.path.splitext(base_name)[0]
            self.filename_to_id[base_name] = img['id']
            self.filename_to_id[name_without_ext] = img['id']

        # Create mapping from category ID to category name
        self.categories = {}
        self.category_names = []
        for cat in self.annotations['categories']:
            self.categories[cat['id']] = cat['name']
            self.category_names.append(cat['name'])
        
        # Group annotations by image ID
        self.gt_by_image = defaultdict(list)
        for ann in self.annotations['annotations']:
            self.gt_by_image[ann['image_id']].append(ann)
        
        
        self.predictions = self._load_predictions(predictions_path)
        # Group predictions by image ID (if they're in a similar format)
        self.pred_by_image = defaultdict(list)
        for pred in self.predictions:
            self.pred_by_image[pred['image_id_num']].append(pred)
    
    def _load_annotations(self, path: str) -> Dict:
        """Load and parse the COCO annotations file."""
        with open(path, 'r') as f:
            annotations = json.load(f)

        # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2] if needed
        if self.bbox_format == 'xyxy':
            for ann in annotations['annotations']:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [x, y, x + w, y + h]
        
        return annotations
    
    def _load_predictions(self, path: str) -> List:
        """
        Load and parse the model predictions file.
        
        The format is expected to be similar to COCO's, i.e., a list of dictionaries with:
        - image_id: ID of the image or filename if use_filename_as_id is True
        - category_id: ID of the predicted category
        - bbox: [x1, y1, x2, y2] format or [x, y, width, height] depending on bbox_format
        - score: confidence score (optional)
        """
        with open(path, 'r') as f:
            predictions = json.load(f)
        # If predictions use filenames as IDs, convert to numeric IDs
        if self.use_filename_as_id:
            for pred in predictions:
                if isinstance(pred['image_id'], str):
                    # Try to map the filename to a numeric ID
                    filename = pred['image_id']
                    if filename in self.filename_to_id:
                        pred['image_id_num'] = self.filename_to_id[filename]
                    else:
                        # Try without path and extension
                        base_name = os.path.basename(filename)
                        if base_name in self.filename_to_id:
                            pred['image_id_num'] = self.filename_to_id[base_name]
                        else:
                            # Try without extension
                            name_without_ext = os.path.splitext(base_name)[0]
                            if name_without_ext in self.filename_to_id:
                                pred['image_id_num'] = self.filename_to_id[name_without_ext]
                            else:
                                print(f"Warning: Could not map filename '{filename}' to an image ID")
                                # Keep it as is or skip
        return predictions
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: Bounding boxes in format specified by self.bbox_format
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if self.bbox_format == 'xywh':
            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
            x1_1, y1_1 = box1[0], box1[1]
            x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
            
            x1_2, y1_2 = box2[0], box2[1]
            x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        else:
            # Already in [x1, y1, x2, y2] format
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _get_category_index(self, category_id: int) -> int:
        """Convert category ID to index in confusion matrix."""
        # Get all category IDs in sorted order
        cat_ids = sorted(self.categories.keys())
        
        # Find the index of this category ID
        try:
            return cat_ids.index(category_id)
        except ValueError:
            # If category ID is not found, return the last index (for "other")
            return len(self.categories)
    
    def get_image_path(self, image_id):
        """Get the full path to an image based on its ID."""
        if not self.images_dir:
            return None
            
        file_name = self.image_info[image_id]['file_name']
        # Handle both absolute and relative paths
        if os.path.isabs(file_name):
            return file_name
        else:
            return os.path.join(self.images_dir, file_name)

    def get_bbox_centroid(self, bbox):
        """Calculate the centroid of a bounding box."""
        if self.bbox_format == 'xyxy':
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        else:  # xywh format
            x, y, w, h = bbox
            return (x + w / 2, y + h / 2)

    def evaluate(self) -> Dict:
        """
        Evaluate the predictions against ground truth.
        
        Returns:
            Dictionary containing evaluation results including confusion matrix,
            precision, recall, F1-score, and overall metrics.
        """
        total_gt = sum(len(anns) for anns in self.gt_by_image.values())
        total_pred = sum(len(preds) for preds in self.pred_by_image.values())

        # Initialize confusion matrix
        num_categories = len(self.categories)
        conf_matrix = np.zeros((num_categories + 1, num_categories + 1), dtype=int)
        # Extra row/column for false positives/negatives
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Initialize per-class counters
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        
        # Store error information for visualization
        self.false_positives_info = []
        self.false_negatives_info = []
        self.misclassifications_info = []  # Track misclassifications separately
        
        # For each image, compare predictions to ground truth using Hungarian matching
        for image_id in self.gt_by_image.keys():
            gt_boxes = self.gt_by_image[image_id]
            pred_boxes = self.pred_by_image.get(image_id, [])
            
            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)
            
            # If no predictions or no ground truth, handle as special case
            if n_pred == 0:
                # All ground truth objects are false negatives
                for gt in gt_boxes:
                    false_negatives += 1
                    gt_cat_id = gt['category_id']
                    class_fn[gt_cat_id] += 1
                    
                    # Update confusion matrix - false negative
                    gt_cat_idx = self._get_category_index(gt_cat_id)
                    conf_matrix[gt_cat_idx, num_categories] += 1
                    
                    # Store false negative info for visualization
                    self.false_negatives_info.append({
                        'image_id': image_id,
                        'bbox': gt['bbox'],
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown')
                    })
                continue
                
            if n_gt == 0:
                # All predictions are false positives
                for pred in pred_boxes:
                    false_positives += 1
                    pred_cat_id = pred['category_id']
                    pred_cat_id = category_name_to_string[pred['category_id']]
                    class_fp[pred_cat_id] += 1
                    
                    # Update confusion matrix - false positive
                    pred_cat_idx = self._get_category_index(pred_cat_id)
                    conf_matrix[num_categories, pred_cat_idx] += 1
                    
                    # Store false positive info for visualization
                    self.false_positives_info.append({
                        'image_id': image_id,
                        'bbox': pred['bbox'],
                        'category_id': pred_cat_id,
                        'category_name': self.categories.get(pred_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0)
                    })
                continue
            
            # Build cost matrix for Hungarian algorithm
            # We use negative IoU as cost since we want to maximize IoU
            cost_matrix = np.zeros((n_gt, n_pred))
            
            # Also store the IoU values for potential greedy fallback
            iou_matrix = np.zeros((n_gt, n_pred))
            
            for i, gt in enumerate(gt_boxes):
                gt_bbox = gt['bbox']
                gt_cat_id = gt['category_id']
                
                for j, pred in enumerate(pred_boxes):
                    pred_bbox = pred['bbox']
                    pred_cat_id = pred['category_id']
                    pred_cat_id = category_name_to_string[pred['category_id']]
                    
                    iou = self.calculate_iou(gt_bbox, pred_bbox)
                    # Store IoU for potential greedy fallback
                    iou_matrix[i, j] = iou
                    
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
                # Try to apply Hungarian algorithm for optimal matching
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Process matched pairs from Hungarian algorithm
                for i, j in zip(row_indices, col_indices):
                    # Skip invalid matches (below threshold)
                    if cost_matrix[i, j] == float('inf'):
                        continue
                    
                    gt = gt_boxes[i]
                    pred = pred_boxes[j]
                    
                    gt_cat_id = gt['category_id']
                    pred_cat_id = pred['category_id']
                    pred_cat_id = category_name_to_string[pred_cat_id]
                    
                    # Update confusion matrix
                    pred_cat_idx = self._get_category_index(pred_cat_id)
                    gt_cat_idx = self._get_category_index(gt_cat_id)
                    
                    conf_matrix[gt_cat_idx, pred_cat_idx] += 1
                    
                    # Update counts
                    if pred_cat_id == gt_cat_id:
                        true_positives += 1
                        class_tp[gt_cat_id] += 1
                    else:
                        # This is a misclassification (correct box but wrong class)
                        false_positives += 1
                        class_fp[pred_cat_id] += 1
                        class_fn[gt_cat_id] += 1
                        
                        # Store misclassification info explicitly
                        iou = iou_matrix[i, j]  # Use the pre-calculated IoU
                        misclass_info = {
                            'image_id': image_id,
                            'bbox': pred['bbox'],
                            'gt_bbox': gt['bbox'],
                            'category_id': pred_cat_id,
                            'gt_category_id': gt_cat_id,
                            'category_name': self.categories.get(pred_cat_id, 'unknown'),
                            'gt_category_name': self.categories.get(gt_cat_id, 'unknown'),
                            'score': pred.get('score', 1.0),
                            'wrong_class': True,
                            'iou': iou
                        }
                        
                        # Add to both false positives (for consistency) and misclassifications
                        self.false_positives_info.append(misclass_info)
                        self.misclassifications_info.append(misclass_info)
                    
                    matched_gt_indices.add(i)
                    matched_pred_indices.add(j)
                    
            except ValueError as e:
                print('cost matrix:', cost_matrix)
                # exit()
                # Handle case where Hungarian algorithm fails
                print(f"Warning: Hungarian algorithm failed on image {image_id}: {e}")
                print("Falling back to greedy matching algorithm...")
                
                # Fallback: Use greedy matching when Hungarian algorithm fails
                # Create a list of (gt_index, pred_index, iou) tuples for all combinations
                all_pairs = []
                for i in range(n_gt):
                    for j in range(n_pred):
                        iou = iou_matrix[i, j]  # Use the pre-calculated IoU
                        if iou >= self.iou_threshold:
                            all_pairs.append((i, j, iou))
                
                # Sort by IoU (highest first)
                all_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Greedily match pairs
                for i, j, iou in all_pairs:
                    if i not in matched_gt_indices and j not in matched_pred_indices:
                        matched_gt_indices.add(i)
                        matched_pred_indices.add(j)
                        
                        # Process this match
                        gt = gt_boxes[i]
                        pred = pred_boxes[j]
                        
                        gt_cat_id = gt['category_id']
                        pred_cat_id = pred['category_id']
                        pred_cat_id = category_name_to_string[pred_cat_id]
                        
                        # Update confusion matrix
                        pred_cat_idx = self._get_category_index(pred_cat_id)
                        gt_cat_idx = self._get_category_index(gt_cat_id)
                        
                        conf_matrix[gt_cat_idx, pred_cat_idx] += 1
                        
                        # Update counts
                        if pred_cat_id == gt_cat_id:
                            true_positives += 1
                            class_tp[gt_cat_id] += 1
                        else:
                            # This is a misclassification (correct box but wrong class)
                            false_positives += 1
                            class_fp[pred_cat_id] += 1
                            class_fn[gt_cat_id] += 1
                            
                            # Store misclassification info explicitly
                            misclass_info = {
                                'image_id': image_id,
                                'bbox': pred['bbox'],
                                'gt_bbox': gt['bbox'],
                                'category_id': pred_cat_id,
                                'gt_category_id': gt_cat_id,
                                'category_name': self.categories.get(pred_cat_id, 'unknown'),
                                'gt_category_name': self.categories.get(gt_cat_id, 'unknown'),
                                'score': pred.get('score', 1.0),
                                'wrong_class': True,
                                'iou': iou
                            }
                            
                            # Add to both false positives (for consistency) and misclassifications
                            self.false_positives_info.append(misclass_info)
                            self.misclassifications_info.append(misclass_info)
            
            # Handle unmatched ground truth boxes (false negatives)
            for i in range(n_gt):
                if i not in matched_gt_indices:
                    false_negatives += 1
                    gt_cat_id = gt_boxes[i]['category_id']
                    class_fn[gt_cat_id] += 1
                    
                    # Update confusion matrix - false negative
                    gt_cat_idx = self._get_category_index(gt_cat_id)
                    conf_matrix[gt_cat_idx, num_categories] += 1
                    
                    # Store false negative info for visualization
                    self.false_negatives_info.append({
                        'image_id': image_id,
                        'bbox': gt_boxes[i]['bbox'],
                        'category_id': gt_cat_id,
                        'category_name': self.categories.get(gt_cat_id, 'unknown')
                    })
            
            # Handle unmatched predictions (false positives)
            for j in range(n_pred):
                if j not in matched_pred_indices:
                    false_positives += 1
                    pred_cat_id = pred_boxes[j]['category_id']
                    pred_cat_id = category_name_to_string[pred_cat_id]
                    class_fp[pred_cat_id] += 1
                    
                    # Update confusion matrix - false positive
                    pred_cat_idx = self._get_category_index(pred_cat_id)
                    conf_matrix[num_categories, pred_cat_idx] += 1
                    
                    # Store false positive info for visualization
                    self.false_positives_info.append({
                        'image_id': image_id,
                        'bbox': pred_boxes[j]['bbox'],
                        'category_id': pred_cat_id,
                        'category_name': self.categories.get(pred_cat_id, 'unknown'),
                        'score': pred_boxes[j].get('score', 1.0)
                    })
        
        # Check for any predictions on images that don't have ground truth
        for image_id in self.pred_by_image.keys():
            if image_id not in self.gt_by_image:
                # All predictions are false positives
                for pred in self.pred_by_image[image_id]:
                    false_positives += 1
                    pred_cat_id = pred['category_id']
                    pred_cat_id = category_name_to_string[pred_cat_id]
                    class_fp[pred_cat_id] += 1
                    
                    # Update confusion matrix - false positive
                    pred_cat_idx = self._get_category_index(pred_cat_id)
                    conf_matrix[num_categories, pred_cat_idx] += 1
                    
                    # Store false positive info for visualization
                    self.false_positives_info.append({
                        'image_id': image_id,
                        'bbox': pred['bbox'],
                        'category_id': pred_cat_id,
                        'category_name': self.categories.get(pred_cat_id, 'unknown'),
                        'score': pred.get('score', 1.0)
                    })
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class metrics
        class_metrics = {}
        for cat_id, cat_name in self.categories.items():
            tp = class_tp[cat_id]
            fp = class_fp[cat_id]
            fn = class_fn[cat_id]
            
            cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            
            class_metrics[cat_name] = {
                'precision': cat_precision,
                'recall': cat_recall,
                'f1_score': cat_f1,
                'support': tp + fn  # Total number of ground truth objects of this class
            }
        
        # Count misclassifications
        misclassification_count = len(self.misclassifications_info)
        
        # Validate counts
        print(f"Total predicted objects: {total_pred}")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
        print(f"Sum of TP+FP: {true_positives + false_positives}")
        
        return {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'misclassifications': misclassification_count,
            'class_metrics': class_metrics,
            'iou_threshold': self.iou_threshold,
            'total_gt': total_gt,
            'total_pred': total_pred
        }

    def save_blank_crop(self, crop, save_path):
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

    # def visualize_false_positives(self, output_dir: str, max_samples: int = None):
    #     """
    #     Visualize false positives and save them as 100x100 crops centered around the object.
    #     Images are organized in subdirectories according to their class.
        
    #     Args:
    #         output_dir: Directory to save the visualizations
    #         max_samples: Maximum number of samples to visualize (None for all)
    #     """
    #     if not self.images_dir:
    #         print("Error: Cannot visualize false positives without images directory.")
    #         return
        
    #     fp_dir = os.path.join(output_dir, "false_positives")
    #     os.makedirs(fp_dir, exist_ok=True)
        
    #     # Limit the number of samples if requested
    #     fp_samples = self.false_positives_info
    #     if max_samples is not None and len(fp_samples) > max_samples:
    #         fp_samples = np.random.choice(fp_samples, max_samples, replace=False).tolist()
        
    #     # Track classes for reporting
    #     class_counts = {}
        
    #     for i, fp in enumerate(fp_samples):
    #         image_id = fp['image_id']
    #         image_path = self.get_image_path(image_id)
    #         category_name = fp['category_name']
            
    #         # Create class subdirectory
    #         class_dir = os.path.join(fp_dir, category_name)
    #         os.makedirs(class_dir, exist_ok=True)
            
    #         # Update class counts
    #         if category_name not in class_counts:
    #             class_counts[category_name] = 0
    #         class_counts[category_name] += 1
            
    #         if not image_path or not os.path.exists(image_path):
    #             print(f"Warning: Could not find image for ID {image_id}")
    #             continue
            
    #         try:
    #             # Read the image
    #             image = cv2.imread(image_path)
    #             if image is None:
    #                 print(f"Warning: Could not read image {image_path}")
    #                 continue
                    
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                
    #             # Get centroid of the bounding box
    #             cx, cy = self.get_bbox_centroid(fp['bbox'])
                
    #             # Create a 100x100 crop centered on the object
    #             height, width = image.shape[:2]
                
    #             # Calculate crop boundaries
    #             crop_size = 100
    #             x1 = max(0, int(cx - crop_size / 2))
    #             y1 = max(0, int(cy - crop_size / 2))
    #             x2 = min(width, x1 + crop_size)
    #             y2 = min(height, y1 + crop_size)
                
    #             # Adjust if crop goes beyond image boundaries
    #             if x2 - x1 < crop_size:
    #                 if x1 == 0:
    #                     x2 = min(width, crop_size)
    #                 else:
    #                     x1 = max(0, x2 - crop_size)
                
    #             if y2 - y1 < crop_size:
    #                 if y1 == 0:
    #                     y2 = min(height, crop_size)
    #                 else:
    #                     y1 = max(0, y2 - crop_size)
                
    #             crop = image[y1:y2, x1:x2]
                
    #             # Convert bounding box to crop coordinates
    #             if self.bbox_format == 'xyxy':
    #                 x1_b, y1_b, x2_b, y2_b = fp['bbox']
    #                 crop_bbox = [
    #                     max(0, x1_b - x1),
    #                     max(0, y1_b - y1),
    #                     min(crop_size, x2_b - x1),
    #                     min(crop_size, y2_b - y1)
    #                 ]
    #             else:  # xywh
    #                 x_b, y_b, w_b, h_b = fp['bbox']
    #                 crop_bbox = [
    #                     max(0, x_b - x1),
    #                     max(0, y_b - y1),
    #                     min(crop_size, w_b),
    #                     min(crop_size, h_b)
    #                 ]
                
    #             # Create figure with clean visualization
    #             fig, ax = plt.subplots(figsize=(5, 5))
    #             ax.imshow(crop)
                
    #             # Draw the prediction bbox
    #             if self.bbox_format == 'xyxy':
    #                 rect = patches.Rectangle((crop_bbox[0], crop_bbox[1]), 
    #                                     crop_bbox[2] - crop_bbox[0], 
    #                                     crop_bbox[3] - crop_bbox[1],
    #                                     linewidth=2, 
    #                                     edgecolor=PRED_COLOR, 
    #                                     facecolor='none')
    #             else:  # xywh
    #                 rect = patches.Rectangle((crop_bbox[0], crop_bbox[1]), 
    #                                     crop_bbox[2], 
    #                                     crop_bbox[3],
    #                                     linewidth=2, 
    #                                     edgecolor=PRED_COLOR, 
    #                                     facecolor='none')
    #             ax.add_patch(rect)

    #             # If there's a ground truth box (wrong class case), show it too
    #             if 'gt_bbox' in fp:
    #                 if self.bbox_format == 'xyxy':
    #                     x1_gt, y1_gt, x2_gt, y2_gt = fp['gt_bbox']
    #                     gt_crop_bbox = [
    #                         max(0, x1_gt - x1),
    #                         max(0, y1_gt - y1),
    #                         min(crop_size, x2_gt - x1),
    #                         min(crop_size, y2_gt - y1)
    #                     ]
    #                     rect_gt = patches.Rectangle((gt_crop_bbox[0], gt_crop_bbox[1]), 
    #                                         gt_crop_bbox[2] - gt_crop_bbox[0], 
    #                                         gt_crop_bbox[3] - gt_crop_bbox[1],
    #                                         linewidth=2, 
    #                                         edgecolor=GT_COLOR, 
    #                                         facecolor='none',
    #                                         linestyle='--')
    #                 else:  # xywh
    #                     x_gt, y_gt, w_gt, h_gt = fp['gt_bbox']
    #                     gt_crop_bbox = [
    #                         max(0, x_gt - x1),
    #                         max(0, y_gt - y1),
    #                         min(crop_size, w_gt),
    #                         min(crop_size, h_gt)
    #                     ]
    #                     rect_gt = patches.Rectangle((gt_crop_bbox[0], gt_crop_bbox[1]), 
    #                                         gt_crop_bbox[2], 
    #                                         gt_crop_bbox[3],
    #                                         linewidth=2, 
    #                                         edgecolor=GT_COLOR, 
    #                                         facecolor='none',
    #                                         linestyle='--')
    #                 ax.add_patch(rect_gt)
                
    #             # Add title with details
    #             if 'wrong_class' in fp and fp['wrong_class']:
    #                 plt.close(fig)
    #                 continue
    #                 title = f"FP (Wrong Class)\nPred: {fp['category_name']} ({fp['score']:.2f})\nGT: {fp['gt_category_name']}"
    #             else:
    #                 title = f"False Positive\n{fp['category_name']} ({fp['score']:.2f})"
                
    #             ax.set_title(title, fontsize=10)
    #             ax.axis('off')  # Hide axes for cleaner visualization
                
    #             # Save the figure in the class subdirectory
    #             img_name = os.path.basename(image_path)
    #             save_path = os.path.join(class_dir, f"fp_{i}_{img_name}")
    #             plt.tight_layout()
    #             plt.savefig(save_path, dpi=100, bbox_inches='tight')
    #             plt.close(fig)
                
    #             # Save blank crop with no annotations
    #             self.save_blank_crop(crop, save_path)
                
    #         except Exception as e:
    #             print(f"Error processing false positive {i} from image {image_id}: {str(e)}")
        
    #     # Print summary of classes
    #     print(f"False positives saved by class:")
    #     for cls, count in class_counts.items():
    #         print(f"  - {cls}: {count} images")

    # def visualize_false_negatives(self, output_dir: str, max_samples: int = None):
    #     """
    #     Visualize false negatives and save them as 100x100 crops centered around the object.
    #     Images are organized in subdirectories according to their class.
        
    #     Args:
    #         output_dir: Directory to save the visualizations
    #         max_samples: Maximum number of samples to visualize (None for all)
    #     """
    #     if not self.images_dir:
    #         print("Error: Cannot visualize false negatives without images directory.")
    #         return
        
    #     fn_dir = os.path.join(output_dir, "false_negatives")
    #     os.makedirs(fn_dir, exist_ok=True)
        
    #     # Limit the number of samples if requested
    #     fn_samples = self.false_negatives_info
    #     if max_samples is not None and len(fn_samples) > max_samples:
    #         fn_samples = np.random.choice(fn_samples, max_samples, replace=False).tolist()
        
    #     # Track classes for reporting
    #     class_counts = {}
        
    #     for i, fn in enumerate(fn_samples):
    #         image_id = fn['image_id']
    #         image_path = self.get_image_path(image_id)
    #         category_name = fn['category_name']
            
    #         # Create class subdirectory
    #         class_dir = os.path.join(fn_dir, category_name)
    #         os.makedirs(class_dir, exist_ok=True)
            
    #         # Update class counts
    #         if category_name not in class_counts:
    #             class_counts[category_name] = 0
    #         class_counts[category_name] += 1
            
    #         if not image_path or not os.path.exists(image_path):
    #             print(f"Warning: Could not find image for ID {image_id}")
    #             continue
            
    #         try:
    #             # Read the image
    #             image = cv2.imread(image_path)
    #             if image is None:
    #                 print(f"Warning: Could not read image {image_path}")
    #                 continue
                    
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                
    #             # Get centroid of the bounding box
    #             cx, cy = self.get_bbox_centroid(fn['bbox'])
                
    #             # Create a 100x100 crop centered on the object
    #             height, width = image.shape[:2]
                
    #             # Calculate crop boundaries
    #             crop_size = 100
    #             x1 = max(0, int(cx - crop_size / 2))
    #             y1 = max(0, int(cy - crop_size / 2))
    #             x2 = min(width, x1 + crop_size)
    #             y2 = min(height, y1 + crop_size)
                
    #             # Adjust if crop goes beyond image boundaries
    #             if x2 - x1 < crop_size:
    #                 if x1 == 0:
    #                     x2 = min(width, crop_size)
    #                 else:
    #                     x1 = max(0, x2 - crop_size)
                
    #             if y2 - y1 < crop_size:
    #                 if y1 == 0:
    #                     y2 = min(height, crop_size)
    #                 else:
    #                     y1 = max(0, y2 - crop_size)
                
    #             crop = image[y1:y2, x1:x2]
                
    #             # Convert bounding box to crop coordinates
    #             if self.bbox_format == 'xyxy':
    #                 x1_b, y1_b, x2_b, y2_b = fn['bbox']
    #                 crop_bbox = [
    #                     max(0, x1_b - x1),
    #                     max(0, y1_b - y1),
    #                     min(crop_size, x2_b - x1),
    #                     min(crop_size, y2_b - y1)
    #                 ]
    #             else:  # xywh
    #                 x_b, y_b, w_b, h_b = fn['bbox']
    #                 crop_bbox = [
    #                     max(0, x_b - x1),
    #                     max(0, y_b - y1),
    #                     min(crop_size, w_b),
    #                     min(crop_size, h_b)
    #                 ]
                
    #             # Create figure with clean visualization
    #             fig, ax = plt.subplots(figsize=(5, 5))
    #             ax.imshow(crop)
                
    #             # Draw the ground truth bbox
    #             if self.bbox_format == 'xyxy':
    #                 rect = patches.Rectangle((crop_bbox[0], crop_bbox[1]), 
    #                                     crop_bbox[2] - crop_bbox[0], 
    #                                     crop_bbox[3] - crop_bbox[1],
    #                                     linewidth=2, 
    #                                     edgecolor=GT_COLOR, 
    #                                     facecolor='none')
    #             else:  # xywh
    #                 rect = patches.Rectangle((crop_bbox[0], crop_bbox[1]), 
    #                                     crop_bbox[2], 
    #                                     crop_bbox[3],
    #                                     linewidth=2, 
    #                                     edgecolor=GT_COLOR, 
    #                                     facecolor='none')
    #             ax.add_patch(rect)
                
    #             # Add title with details
    #             title = f"False Negative\n{fn['category_name']}"
    #             ax.set_title(title, fontsize=10)
    #             ax.axis('off')  # Hide axes for cleaner visualization
                
    #             # Save the figure in the class subdirectory
    #             img_name = os.path.basename(image_path)
    #             save_path = os.path.join(class_dir, f"fn_{i}_{img_name}")
    #             plt.tight_layout()
    #             plt.savefig(save_path, dpi=100, bbox_inches='tight')
    #             plt.close(fig)
                
    #             # Save blank crop with no annotations
    #             self.save_blank_crop(crop, save_path)
                
    #         except Exception as e:
    #             print(f"Error processing false negative {i} from image {image_id}: {str(e)}")
        
    #     # Print summary of classes
    #     print(f"False negatives saved by class:")
    #     for cls, count in class_counts.items():
    #         print(f"  - {cls}: {count} images")
            
    def visualize_errors(self, output_dir: str, max_samples: int = None):
        """
        Visualize all error types: false positives, false negatives, and misclassifications.
        
        Args:
            output_dir: Directory to save the visualizations
            max_samples: Maximum number of samples to visualize per category (None for all)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Visualizing false positives...")
        self.visualize_false_positives(output_dir, max_samples)
        
        print(f"Visualizing false negatives...")
        self.visualize_false_negatives(output_dir, max_samples)
        
        print(f"Visualizing misclassifications by type...")
        self.visualize_misclassifications(output_dir, max_samples)
            
        print(f"Visualizing misclassifications in side-by-side format...")
        # self.visualize_misclassifications_side_by_side(output_dir, max_samples)
        
        print(f"Creating misclassification side-by-side grid visualizations...")
        # self.create_misclassifications_grid_side_by_side(output_dir, max_samples)
            
        # Create an error distribution histogram
        self.plot_error_distribution(output_dir)
        
        print(f"Visualization complete. Results saved to {output_dir}")
        
    def plot_error_distribution(self, output_dir: str):
        """
        Create a visualization of the distribution of different error types.
        
        Args:
            output_dir: Directory to save the visualization
        """
        # Get counts of different error types
        fp_count = len([fp for fp in self.false_positives_info if 'wrong_class' not in fp or not fp['wrong_class']])
        fn_count = len(self.false_negatives_info)
        misclass_count = len(self.misclassifications_info)
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        categories = ['False Positives\n(No Match)', 'False Negatives\n(Missed Objects)', 'Misclassifications\n(Wrong Class)']
        counts = [fp_count, fn_count, misclass_count]
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        plt.bar(categories, counts, color=colors)
        
        # Add labels and title
        plt.ylabel('Count')
        plt.title('Distribution of Error Types')
        
        # Add count labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center')
        
        # Add a legend explaining the error types
        plt.figtext(0.5, 0.01, 
                    'False Positives: Detections with no matching ground truth\n'
                    'False Negatives: Ground truth objects that were not detected\n'
                    'Misclassifications: Objects detected with correct location but wrong class',
                    ha='center', fontsize=10, bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the text
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()
        
        # If we have misclassifications, create a heatmap of confusion patterns
        if misclass_count > 0:
            self.plot_misclassification_heatmap(output_dir)

    def plot_misclassification_heatmap(self, output_dir: str):
        """
        Create a heatmap showing the most common misclassification patterns.
        
        Args:
            output_dir: Directory to save the visualization
        """
        # Count occurrences of each misclassification pattern
        misclass_patterns = defaultdict(int)
        
        for mc in self.misclassifications_info:
            gt_name = mc['gt_category_name']
            pred_name = mc['category_name']
            misclass_patterns[(gt_name, pred_name)] += 1
        
        # Get unique category names for ground truth and predictions
        gt_categories = sorted(list(set(mc['gt_category_name'] for mc in self.misclassifications_info)))
        pred_categories = sorted(list(set(mc['category_name'] for mc in self.misclassifications_info)))
        
        # Create confusion matrix
        matrix = np.zeros((len(gt_categories), len(pred_categories)))
        
        for (gt_name, pred_name), count in misclass_patterns.items():
            gt_idx = gt_categories.index(gt_name)
            pred_idx = pred_categories.index(pred_name)
            matrix[gt_idx, pred_idx] = count
        
        # Only continue if we have data to visualize
        if np.sum(matrix) > 0:
            # Create the figure
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
                        xticklabels=pred_categories, yticklabels=gt_categories)
            
            plt.xlabel('Predicted Class')
            plt.ylabel('Ground Truth Class')
            plt.title('Misclassification Patterns')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'misclassification_heatmap.png'))
            plt.close()
                    
    def visualize_misclassifications(self, output_dir: str, max_samples: int = None):
        """
        Visualize misclassifications (correct box but wrong class) and save them 
        organized by misclassification type (e.g., "a_truth_b_predicted").
        
        Args:
            output_dir: Directory to save the visualizations
            max_samples: Maximum number of samples to visualize per misclassification type
        """
        if not self.images_dir:
            print("Error: Cannot visualize misclassifications without images directory.")
            return
        
        # Create base directory for misclassifications
        misclass_base_dir = os.path.join(output_dir, "misclassifications")
        os.makedirs(misclass_base_dir, exist_ok=True)
        
        # Create a summary file to track misclassification statistics
        summary_path = os.path.join(misclass_base_dir, "misclassification_summary.txt")
        with open(summary_path, 'w') as summary_file:
            summary_file.write("Misclassification Summary\n")
            summary_file.write("=======================\n\n")
            summary_file.write(f"Total misclassifications: {len(self.misclassifications_info)}\n\n")
        
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
        
        # Append statistics to summary file
        with open(summary_path, 'a') as summary_file:
            summary_file.write("Misclassification Types:\n")
            
            # Sort by frequency (most common first)
            sorted_types = sorted(misclass_by_type.items(), key=lambda x: len(x[1]), reverse=True)
            
            for misclass_type, items in sorted_types:
                summary_file.write(f"  {misclass_type}: {len(items)} instances\n")
            
            summary_file.write("\n")
        
        # Process each misclassification type
        for misclass_type, items in misclass_by_type.items():
            # Create directory for this specific misclassification type
            type_dir = os.path.join(misclass_base_dir, misclass_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # Create a mini-report for this misclassification type
            with open(os.path.join(type_dir, "info.txt"), 'w') as info_file:
                gt_cat_name = items[0]['gt_category_name']
                pred_cat_name = items[0]['category_name']
                
                info_file.write(f"Misclassification: {gt_cat_name} → {pred_cat_name}\n")
                info_file.write(f"Number of instances: {len(items)}\n")
                
                # Calculate average confidence score
                avg_score = sum(item['score'] for item in items) / len(items)
                info_file.write(f"Average confidence score: {avg_score:.4f}\n\n")
                
                info_file.write("Image IDs with this misclassification:\n")
                for item in items:
                    if 'file_name' in self.image_info.get(item['image_id'], {}):
                        file_name = self.image_info[item['image_id']]['file_name']
                        info_file.write(f"  {item['image_id']} ({file_name}): confidence {item['score']:.4f}\n")
                    else:
                        info_file.write(f"  {item['image_id']}: confidence {item['score']:.4f}\n")
            
            # Limit samples if requested
            samples = items
            if max_samples is not None and len(samples) > max_samples:
                samples = np.random.choice(samples, max_samples, replace=False).tolist()
            
            print(f"Visualizing {len(samples)} samples of misclassification type '{misclass_type}'...")
            
            # Create visualization grid if there are many samples
            if len(samples) > 9:
                # Create a grid visualization of examples (up to 3x3)
                grid_samples = samples[:9]  # Take first 9 for grid
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                axes = axes.flatten()
                
                for i, (ax, item) in enumerate(zip(axes, grid_samples)):
                    image_id = item['image_id']
                    image_path = self.get_image_path(image_id)
                    
                    if not image_path or not os.path.exists(image_path):
                        ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                        ax.axis('off')
                        continue
                    
                    try:
                        # Read the image
                        image = cv2.imread(image_path)
                        if image is None:
                            ax.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                            ax.axis('off')
                            continue
                            
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                        
                        # Get centroid of the bounding box
                        cx, cy = self.get_bbox_centroid(item['bbox'])
                        
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
                        
                        crop = image[y1:y2, x1:x2]
                        
                        # Display the crop
                        ax.imshow(crop)
                        
                        # Add a title with the scores
                        ax.set_title(f"GT: {item['gt_category_name']}\nPred: {item['category_name']} ({item['score']:.2f})", fontsize=8)
                        
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=8, wrap=True)
                        ax.axis('off')
                
                # Save the grid
                plt.tight_layout()
                plt.savefig(os.path.join(type_dir, "misclass_grid.png"), dpi=150)
                plt.close(fig)
            
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
                    
                    # Get centroid of the bounding box
                    cx, cy = self.get_bbox_centroid(item['bbox'])
                    
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
                    
                    crop = image[y1:y2, x1:x2]
                    
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
                    
                    # Create figure with clean visualization
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(crop)
                    
                    # Draw the prediction bbox in red
                    if self.bbox_format == 'xyxy':
                        rect_pred = patches.Rectangle(
                            (pred_crop_bbox[0], pred_crop_bbox[1]), 
                            pred_crop_bbox[2] - pred_crop_bbox[0], 
                            pred_crop_bbox[3] - pred_crop_bbox[1],
                            linewidth=2, 
                            edgecolor=PRED_COLOR, 
                            facecolor='none',
                            label=f"Pred: {item['category_name']}"
                        )
                    else:  # xywh
                        rect_pred = patches.Rectangle(
                            (pred_crop_bbox[0], pred_crop_bbox[1]), 
                            pred_crop_bbox[2], 
                            pred_crop_bbox[3],
                            linewidth=2, 
                            edgecolor=PRED_COLOR, 
                            facecolor='none',
                            label=f"Pred: {item['category_name']}"
                        )
                    ax.add_patch(rect_pred)
                    
                    # Draw the ground truth bbox in green (dashed)
                    if self.bbox_format == 'xyxy':
                        rect_gt = patches.Rectangle(
                            (gt_crop_bbox[0], gt_crop_bbox[1]), 
                            gt_crop_bbox[2] - gt_crop_bbox[0], 
                            gt_crop_bbox[3] - gt_crop_bbox[1],
                            linewidth=2, 
                            edgecolor=GT_COLOR, 
                            facecolor='none',
                            linestyle='--',
                            label=f"GT: {item['gt_category_name']}"
                        )
                    else:  # xywh
                        rect_gt = patches.Rectangle(
                            (gt_crop_bbox[0], gt_crop_bbox[1]), 
                            gt_crop_bbox[2], 
                            gt_crop_bbox[3],
                            linewidth=2, 
                            edgecolor=GT_COLOR, 
                            facecolor='none',
                            linestyle='--',
                            label=f"GT: {item['gt_category_name']}"
                        )
                    ax.add_patch(rect_gt)
                    
                    # Add title with details
                    title = f"Misclassification\nGT: {item['gt_category_name']}\nPred: {item['category_name']} ({item['score']:.2f})"
                    ax.set_title(title, fontsize=10)
                    
                    # Add legend
                    ax.legend(loc='lower right', fontsize=8)
                    
                    ax.axis('off')  # Hide axes for cleaner visualization
                    
                    # Save the figure
                    img_name = os.path.basename(image_path)
                    save_path = os.path.join(type_dir, f"misclass_{i}_{img_name}")
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error processing misclassification {i} from image {image_id}: {str(e)}")
    
    def plot_confusion_matrix(self, results: Dict, normalize: bool = True, save_path: str = None):
        """
        Plot the confusion matrix.
        
        Args:
            results: Results dictionary from evaluate()
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the figure (if None, the figure is displayed)
        """
        conf_matrix = results['confusion_matrix']
        
        # Create labels including "background" or "false positive"
        labels = self.category_names + ['background']
        
        # Handle normalization
        if normalize:
            # Normalize by row (ground truth) to show recall
            row_sums = conf_matrix.sum(axis=1)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            conf_matrix_norm = conf_matrix / row_sums[:, np.newaxis]
        else:
            conf_matrix_norm = conf_matrix
        
        print(conf_matrix_norm)
        conf_matrix_norm_transposed = np.transpose(conf_matrix_norm)

        # Create the figure
        plt.figure(figsize=(13, 11))
        sns.heatmap(conf_matrix_norm_transposed, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', xticklabels=labels[:], yticklabels=labels[:])
        
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_class_metrics(self, results: Dict, save_path: str = None):
        """
        Plot per-class metrics.
        
        Args:
            results: Results dictionary from evaluate()
            save_path: Path to save the figure (if None, the figure is displayed)
        """
        class_metrics = results['class_metrics']
        
        # Sort classes by support (number of ground truth instances)
        sorted_classes = sorted(class_metrics.items(),
                               key=lambda x: x[1]['support'], reverse=True)
        
        class_names = [cls[0] for cls in sorted_classes]
        precision = [cls[1]['precision'] for cls in sorted_classes]
        recall = [cls[1]['recall'] for cls in sorted_classes]
        f1_score = [cls[1]['f1_score'] for cls in sorted_classes]
        support = [cls[1]['support'] for cls in sorted_classes]
        
        # Create bar chart
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(class_names))
        width = 0.2
        
        # Plot bars
        ax1.bar(x - width, precision, width, label='Precision', color='blue', alpha=0.7)
        ax1.bar(x, recall, width, label='Recall', color='green', alpha=0.7)
        ax1.bar(x + width, f1_score, width, label='F1-Score', color='red', alpha=0.7)
        
        # Configure primary y-axis (metrics)
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.1)
        
        # Configure x-axis
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Create secondary y-axis for support
        ax2 = ax1.twinx()
        ax2.plot(x, support, 'o-', color='purple', label='Support')
        ax2.set_ylabel('Number of Instances')
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Class-wise Metrics')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_results(self, results: Dict, output_dir: str, visualize_errors: bool = True, max_samples: int = None):
        """
        Save evaluation results to files.
        
        Args:
            results: Results dictionary from evaluate()
            output_dir: Directory to save results
            visualize_errors: Whether to visualize false positives and false negatives
            max_samples: Maximum number of error samples to visualize per category
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(results, normalize=True,
                                  save_path=os.path.join(output_dir, 'confusion_matrix_norm.png'))
        self.plot_confusion_matrix(results, normalize=False,
                                  save_path=os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Save class metrics plot
        self.plot_class_metrics(results, save_path=os.path.join(output_dir, 'class_metrics.png'))
        
        # Save numeric results as JSON
        results_copy = results.copy()
        results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        # Save a summary text file
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(f"Object Detection Evaluation Summary\n")
            f.write(f"===================================\n\n")
            f.write(f"IoU Threshold: {results['iou_threshold']}\n")
            f.write(f"Total Ground Truth Objects: {results['total_gt']}\n")
            f.write(f"Total Predicted Objects: {results['total_pred']}\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"  True Positives: {results['true_positives']}\n")
            f.write(f"  False Positives: {results['false_positives']}\n")
            f.write(f"  False Negatives: {results['false_negatives']}\n\n")
            
            f.write(f"Per-Class Metrics:\n")
            for cls_name, metrics in results['class_metrics'].items():
                f.write(f"  {cls_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"    Support: {metrics['support']}\n")
                f.write(f"\n")
        
        # Visualize errors if requested and images directory is provided
        if visualize_errors and self.images_dir:
            viz_dir = os.path.join(output_dir, 'error_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            print(f"Visualizing false positives...")
            self.visualize_false_positives(viz_dir, max_samples)
            
            print(f"Visualizing false negatives...")
            self.visualize_false_negatives(viz_dir, max_samples)
            
            print(f"Visualizing misclassifications...")
            self.visualize_misclassifications(viz_dir, max_samples)
            
            print(f"Visualization complete. Results saved to {viz_dir}")


def convert_model_output(model_output_path: str, output_format_path: str, bbox_format: str = 'xyxy', use_filename_as_id: bool = False):
    """
    Convert model output to the format expected by the evaluator.
    This is a placeholder function - you'll need to customize this based on
    the actual format of your model's output.

    Expects model output to be
    {
        "image_id_1": [
            {"category_id": 1, "bbox": "[x1, y1, x2, y2]", "score": "0.95"},
            {"category_id": 2, "bbox": [x1, y1, x2, y2], "score": 0.87}
        ],
        "image_id_2": [
            {"category_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.91}
        ]
    }
    
    Args:
        model_output_path: Path to the raw model output
        output_format_path: Path to save the converted output
        bbox_format: Bounding box format: 'xyxy' for [x1,y1,x2,y2] or 'xywh' for [x,y,width,height]
        use_filename_as_id: If True, keep filenames as image IDs; otherwise convert to numeric IDs
    """
    # This is just an example. Modify according to your model's output format
    try:
        with open(model_output_path, 'r') as f:
            model_data = json.load(f)
        
        # Initialize list for converted predictions
        converted_predictions = []
        
        # Assuming model_data is a dictionary with image_id as keys
        # and predictions as values
        for image_id, preds in model_data.items():
            for pred in preds:
                # Handle bbox format based on bbox_format parameter
                pred['bbox'] = ast.literal_eval(pred['bbox'])
                pred['score'] = float(pred['score'])
                if len(pred['bbox']) == 4:
                    if bbox_format == 'xyxy':
                        # Already in the desired format or convert if needed
                        if 'width' in pred and 'height' in pred:  # Explicitly labeled width/height
                            x, y, w, h = pred['bbox']
                            bbox = [x, y, x + w, y + h]
                        elif pred['bbox'][2] > 0 and pred['bbox'][3] > 0 and pred['bbox'][2] < pred['bbox'][0]:
                            # Likely width/height format if width is smaller than x
                            x, y, w, h = pred['bbox']
                            bbox = [x, y, x + w, y + h]
                        else:  # Already in x1,y1,x2,y2 format
                            bbox = pred['bbox']
                    else:  # xywh format
                        if pred['bbox'][2] > pred['bbox'][0] and pred['bbox'][3] > pred['bbox'][1]:
                            # Currently in x1,y1,x2,y2 format, convert to x,y,w,h
                            x1, y1, x2, y2 = pred['bbox']
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                        else:
                            # Already in x,y,w,h format
                            bbox = pred['bbox']
                else:
                    # Unexpected format
                    print(f"Warning: Unexpected bbox format: {pred['bbox']}")
                    bbox = pred['bbox']
                
                # Create image ID field - can be filename or numeric ID
                if not use_filename_as_id and not isinstance(image_id, int):
                    try:
                        img_id = int(image_id)
                    except (ValueError, TypeError):
                        # Keep as filename/string if we can't convert and 
                        # we're using filenames as IDs
                        img_id = image_id
                else:
                    img_id = image_id
                        
                # Convert to the expected format
                converted_pred = {
                    'image_id': img_id,
                    'category_id': pred['category_id'],
                    'bbox': bbox,
                    'score': pred.get('score', 1.0)  # Default to 1.0 if score not provided
                }
                converted_predictions.append(converted_pred)
        
        # Save converted predictions
        with open(output_format_path, 'w') as f:
            json.dump(converted_predictions, f)
            
        return True
    except Exception as e:
        print(f"Error converting model output: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection results with enhanced visualizations')
    parser.add_argument('--gt', required=True, help='Path to ground truth annotations (COCO format)')
    parser.add_argument('--pred', required=True, help='Path to model predictions')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    parser.add_argument('--convert', action='store_true', help='Convert model output format')
    parser.add_argument('--bbox-format', choices=['xyxy', 'xywh'], default='xyxy', 
                       help='Bounding box format: xyxy=[x1,y1,x2,y2], xywh=[x,y,width,height]')
    parser.add_argument('--use-filename-as-id', action='store_true',
                       help='Use filenames as image IDs instead of numeric IDs')
    parser.add_argument('--images-dir', required=True,
                       help='Directory containing the original images (required for visualization)')
    
    # Visualization options
    parser.add_argument('--visualize-errors', action='store_true', 
                       help='Visualize false positives and false negatives')
    parser.add_argument('--visualize-correct', action='store_true',
                       help='Visualize correct detections (true positives)')
    parser.add_argument('--side-by-side', action='store_true',
                       help='For true positives, show ground truth and prediction side-by-side')
    parser.add_argument('--visualize-all', action='store_true',
                       help='Create full-image visualizations with all detections')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of samples to visualize per category')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of full images to visualize')
    parser.add_argument('--grid-size', nargs=2, type=int, default=[4, 5],
                       help='Grid size for visualization grids (rows cols)')
    
    args = parser.parse_args()
    
    # If conversion is needed
    if args.convert:
        converted_path = args.pred + '.converted.json'
        if convert_model_output(args.pred, converted_path, args.bbox_format, args.use_filename_as_id):
            args.pred = converted_path

    # Create evaluator
    evaluator = DetectionEvaluator(
        args.gt, 
        args.pred, 
        args.iou, 
        args.bbox_format,
        args.use_filename_as_id,
        args.images_dir
    )

    # Run evaluation
    results = evaluator.evaluate()
    
    # Attach visualization methods to the evaluator instance
    evaluator.visualize_true_positives = types.MethodType(visualize_true_positives, evaluator)
    evaluator.visualize_true_positives_side_by_side = types.MethodType(visualize_true_positives_side_by_side, evaluator)
    evaluator.create_true_positives_grid = types.MethodType(create_true_positives_grid, evaluator)
    evaluator.create_true_positives_grid_side_by_side = types.MethodType(create_true_positives_grid_side_by_side, evaluator)
    evaluator.visualize_all_detections = types.MethodType(visualize_all_detections, evaluator)
    evaluator.visualize_misclassifications_side_by_side = types.MethodType(visualize_misclassifications_side_by_side, evaluator)
    evaluator.create_misclassifications_grid_side_by_side = types.MethodType(create_misclassifications_grid_side_by_side, evaluator)
    evaluator.visualize_false_positives = types.MethodType(visualize_false_positives, evaluator)
    evaluator.visualize_false_negatives = types.MethodType(visualize_false_negatives, evaluator)


    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    viz_dir = os.path.join(args.output, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate standard evaluation results
    evaluator.plot_confusion_matrix(results, normalize=True,
                                   save_path=os.path.join(args.output, 'confusion_matrix_norm.png'))
    evaluator.plot_confusion_matrix(results, normalize=False,
                                   save_path=os.path.join(args.output, 'confusion_matrix.png'))
    evaluator.plot_class_metrics(results, save_path=os.path.join(args.output, 'class_metrics.png'))
    
    # Visualize errors if requested
    if args.visualize_errors:
        print("Visualizing false positives...")
        evaluator.visualize_false_positives(viz_dir, args.max_samples)
        
        print("Visualizing false negatives...")
        evaluator.visualize_false_negatives(viz_dir, args.max_samples)
        
        print("Visualizing misclassifications...")
        evaluator.visualize_misclassifications(viz_dir, args.max_samples)
    
        print("Visualizing misclassifications in side-by-side format...")
        evaluator.visualize_misclassifications_side_by_side(viz_dir, args.max_samples)
        
        print("Creating misclassification side-by-side grid visualizations...")
        # evaluator.create_misclassifications_grid_side_by_side(viz_dir, args.max_samples, tuple(args.grid_size))

    # Visualize correct detections if requested
    if args.visualize_correct:
        if args.side_by_side:
            print("Visualizing correct detections (true positives) side-by-side...")
            evaluator.visualize_true_positives_side_by_side(viz_dir, args.max_samples)
            
            print("Creating true positives grid visualization side-by-side...")
            # evaluator.create_true_positives_grid_side_by_side(viz_dir, args.max_samples, tuple(args.grid_size))
        else:
            print("Visualizing correct detections (true positives)...")
            # evaluator.visualize_true_positives(viz_dir, args.max_samples)
            
            print("Creating true positives grid visualization...")
            # evaluator.create_true_positives_grid(viz_dir, args.max_samples, tuple(args.grid_size))
    
    # Visualize all detections on full images if requested
    if args.visualize_all:
        print("Creating full-image visualizations...")
        evaluator.visualize_all_detections(viz_dir, None, args.max_images)
    
    # Save a summary text file
    with open(os.path.join(args.output, 'summary.txt'), 'w') as f:
        f.write(f"Object Detection Evaluation Summary\n")
        f.write(f"===================================\n\n")
        f.write(f"IoU Threshold: {results['iou_threshold']}\n")
        f.write(f"Total Ground Truth Objects: {results['total_gt']}\n")
        f.write(f"Total Predicted Objects: {results['total_pred']}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall: {results['recall']:.4f}\n")
        f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
        f.write(f"  True Positives: {results['true_positives']}\n")
        f.write(f"  False Positives: {results['false_positives']}\n")
        f.write(f"  False Negatives: {results['false_negatives']}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        for cls_name, metrics in results['class_metrics'].items():
            f.write(f"  {cls_name}:\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"    Support: {metrics['support']}\n")
            f.write(f"\n")
    
    # Save numeric results as JSON
    results_copy = results.copy()
    results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
    with open(os.path.join(args.output, 'evaluation_results.json'), 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    # Print a brief summary
    print(f"\nEvaluation Complete. Results saved to {args.output}")
    print(f"Overall Precision: {results['precision']:.4f}")
    print(f"Overall Recall: {results['recall']:.4f}")
    print(f"Overall F1-Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()