import os
import json
import argparse
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from datetime import datetime


def find_annotation_files(coco_dir):
    """Find all COCO annotation JSON files in the directory."""
    ann_files = []
    for root, _, files in os.walk(coco_dir):
        for file in files:
            if file.endswith('.json'):
                ann_files.append(os.path.join(root, file))
    
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {coco_dir}")
    
    return ann_files


def detect_split(ann_file):
    """Detect dataset split (train/val/test) from filename or parent directory."""
    filename = os.path.basename(ann_file)
    dirname = os.path.basename(os.path.dirname(ann_file))
    
    # Try to identify split from filename or parent directory
    for possible_split in ['train', 'val', 'test', 'validation']:
        if possible_split in filename.lower() or possible_split in dirname.lower():
            # Normalize 'validation' to 'val'
            return 'val' if possible_split == 'validation' else possible_split
    
    # If no split is identified, use the basename without extension
    return os.path.splitext(filename)[0]


def create_output_directories(output_dir, splits):
    """Create the necessary output directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main directory for images
    main_images_dir = os.path.join(output_dir, "images")
    os.makedirs(main_images_dir, exist_ok=True)
    
    # Create output directories for each split
    for split in splits:
        split_images_dir = os.path.join(output_dir, 'images', split)
        os.makedirs(split_images_dir, exist_ok=True)
        
        if split != "combined":  # Don't create visualization dir for combined split
            visualize_dir = os.path.join(output_dir, "visualized_crops", split)
            os.makedirs(visualize_dir, exist_ok=True)
    
    # Create annotations directory
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    return annotations_dir


def initialize_datasets(splits):
    """Initialize dataset structures for each split."""
    split_datasets = {}
    
    for split in splits:
        split_datasets[split] = {
            "info": {
                "description": f"Cropped COCO dataset ({split} split) with polygon annotations",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "COCO to Cropped COCO script",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }
    
    return split_datasets


def find_image(img_filename, coco_dir, img_dir):
    """Find the image file - first in the img_dir, then search recursively if needed."""
    img_path = os.path.join(img_dir, img_filename)
    
    if os.path.exists(img_path):
        return img_path
    
    # Try to find the image in other directories
    for root, _, files in os.walk(coco_dir):
        potential_path = os.path.join(root, img_filename)
        if os.path.exists(potential_path):
            return potential_path
    
    return None


def calculate_centroid(ann, bbox):
    """Calculate the centroid of an annotation (from segmentation or bbox)."""
    if 'segmentation' in ann and ann['segmentation']:
        # Handle different segmentation formats
        if isinstance(ann['segmentation'], list):
            if isinstance(ann['segmentation'][0], list):  # Polygon format
                # Flatten the list of polygons and get all x, y coordinates
                all_x = []
                all_y = []
                
                for seg in ann['segmentation']:
                    # Each segmentation is [x1, y1, x2, y2, ...]
                    points = np.array(seg).reshape(-1, 2)
                    all_x.extend(points[:, 0])
                    all_y.extend(points[:, 1])
                
                # Calculate centroid as the mean of all polygon points
                center_x = int(np.mean(all_x))
                center_y = int(np.mean(all_y))
                return center_x, center_y
    
    # Fall back to bbox center if no valid segmentation or other format
    center_x = int(bbox[0] + bbox[2] / 2)
    center_y = int(bbox[1] + bbox[3] / 2)
    return center_x, center_y


def calculate_crop_coordinates(center_x, center_y, crop_size, img_width, img_height):
    """Calculate the crop coordinates centered around a point."""
    # Calculate the ideal crop coordinates (centroid at center)
    ideal_x1 = center_x - crop_size // 2
    ideal_y1 = center_y - crop_size // 2
    ideal_x2 = ideal_x1 + crop_size
    ideal_y2 = ideal_y1 + crop_size
    
    # Check if this crop would require padding
    needs_padding = (ideal_x1 < 0 or ideal_y1 < 0 or 
                   ideal_x2 > img_width or ideal_y2 > img_height)
    
    # Calculate the valid region within the image
    valid_x1 = max(0, ideal_x1)
    valid_y1 = max(0, ideal_y1)
    valid_x2 = min(img_width, ideal_x2)
    valid_y2 = min(img_height, ideal_y2)
    
    return {
        'ideal_x1': ideal_x1, 
        'ideal_y1': ideal_y1,
        'ideal_x2': ideal_x2,
        'ideal_y2': ideal_y2,
        'valid_x1': valid_x1,
        'valid_y1': valid_y1,
        'valid_x2': valid_x2,
        'valid_y2': valid_y2,
        'needs_padding': needs_padding
    }


def create_crop_image(img_np, crop_coords, crop_size):
    """Create a cropped image with padding if necessary."""
    # Skip if no valid region
    if crop_coords['valid_x2'] <= crop_coords['valid_x1'] or crop_coords['valid_y2'] <= crop_coords['valid_y1']:
        return None
    
    # Create a black canvas for the final crop
    final_crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    
    # Extract the valid portion from the image
    valid_crop = img_np[
        crop_coords['valid_y1']:crop_coords['valid_y2'], 
        crop_coords['valid_x1']:crop_coords['valid_x2']
    ]
    
    # Calculate where to place this valid portion in the final crop
    dest_x1 = max(0, -crop_coords['ideal_x1'])  # How many pixels padding needed on left
    dest_y1 = max(0, -crop_coords['ideal_y1'])  # How many pixels padding needed on top
    
    # Place the valid portion into the black canvas
    final_crop[
        dest_y1:dest_y1 + valid_crop.shape[0],
        dest_x1:dest_x1 + valid_crop.shape[1]
    ] = valid_crop
    
    return final_crop


def transform_polygon_segmentation(ann, crop_coords, crop_size):
    """Transform polygon segmentation to match crop coordinates."""
    new_segmentation = []
    
    if 'segmentation' in ann and ann['segmentation']:
        # Handle polygon format segmentation
        if isinstance(ann['segmentation'], list) and isinstance(ann['segmentation'][0], list):
            for seg in ann['segmentation']:
                # Convert polygon to numpy array of (x,y) pairs
                poly = np.array(seg).reshape(-1, 2)
                
                # Shift polygon coordinates to the crop coordinate system
                poly[:, 0] = poly[:, 0] - crop_coords['ideal_x1']
                poly[:, 1] = poly[:, 1] - crop_coords['ideal_y1']
                
                # Filter out points outside the crop
                valid_poly = (poly[:, 0] >= 0) & (poly[:, 0] < crop_size) & \
                           (poly[:, 1] >= 0) & (poly[:, 1] < crop_size)
                
                # Only keep polygons with at least 3 points
                if np.sum(valid_poly) >= 3:
                    # Convert numpy values to Python float to avoid serialization issues
                    valid_points = [float(x) for x in poly[valid_poly].flatten()]
                    new_segmentation.append(valid_points)
    
    return new_segmentation


def create_bbox_segmentation(bbox, crop_coords, crop_size):
    """Create a rectangular segmentation from a bounding box."""
    x, y, w, h = bbox
    new_x = float(max(0, x - crop_coords['ideal_x1']))
    new_y = float(max(0, y - crop_coords['ideal_y1']))
    new_w = float(min(crop_size - new_x, w))
    new_h = float(min(crop_size - new_y, h))
    
    # Create a simple rectangular polygon (using Python float)
    rect_poly = [
        float(new_x), float(new_y),
        float(new_x + new_w), float(new_y),
        float(new_x + new_w), float(new_y + new_h),
        float(new_x), float(new_y + new_h)
    ]
    
    return [rect_poly]


def calculate_bbox_from_segmentation(segmentation, crop_size):
    """Calculate bounding box and area from segmentation."""
    all_x = []
    all_y = []
    
    for seg in segmentation:
        points = np.array(seg).reshape(-1, 2)
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
    
    if all_x and all_y:
        min_x = float(max(0, min(all_x)))
        min_y = float(max(0, min(all_y)))
        max_x = float(min(crop_size, max(all_x)))
        max_y = float(min(crop_size, max(all_y)))
        
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        area = float((max_x - min_x) * (max_y - min_y))
    else:
        # Fallback if no valid points
        bbox = [0.0, 0.0, float(crop_size), float(crop_size)]
        area = float(crop_size * crop_size)
    
    return bbox, area


def visualize_crop(img_np, final_crop, class_name, crop_coords, new_segmentation, 
                  new_bbox, ann, bbox, obj_index, output_dir, split, img_filename):
    """Create visualizations of the original image and the crop with annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image with centroid
    ax1.imshow(img_np)
    center_x = crop_coords['ideal_x1'] + (crop_coords['ideal_x2'] - crop_coords['ideal_x1']) // 2
    center_y = crop_coords['ideal_y1'] + (crop_coords['ideal_y2'] - crop_coords['ideal_y1']) // 2
    ax1.plot(center_x, center_y, 'ro', markersize=10)
    
    # Draw the crop boundaries with color based on padding status
    rect_color = 'r' if crop_coords['needs_padding'] else 'g'
    rect = plt.Rectangle((crop_coords['ideal_x1'], crop_coords['ideal_y1']), 
                         crop_coords['ideal_x2'] - crop_coords['ideal_x1'], 
                         crop_coords['ideal_y2'] - crop_coords['ideal_y1'], 
                         linewidth=2, edgecolor=rect_color, facecolor='none')
    ax1.add_patch(rect)
    
    # Draw original segmentation polygon if available
    if 'segmentation' in ann and isinstance(ann['segmentation'], list) and isinstance(ann['segmentation'][0], list):
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape(-1, 2)
            ax1.plot(poly[:, 0], poly[:, 1], 'g-', linewidth=2)
    else:
        # Draw bbox if no polygon
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax1.add_patch(rect)
    
    # Set title 
    title = f"{class_name} centroid"
    if crop_coords['needs_padding']:
        title += " (would be skipped with --skip_padded)"
    ax1.set_title(title)
    
    # Show crop with transformed segmentation
    ax2.imshow(final_crop)
    
    # Draw transformed segmentation polygons
    for seg in new_segmentation:
        poly = np.array(seg).reshape(-1, 2)
        ax2.plot(poly[:, 0], poly[:, 1], 'g-', linewidth=2)
    
    # Draw transformed bbox
    x, y, w, h = new_bbox
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
    ax2.add_patch(rect)
    
    ax2.set_title(f"Cropped Image with Annotation")
    
    visualize_dir = os.path.join(output_dir, "visualized_crops", split)
    os.makedirs(visualize_dir, exist_ok=True)
    plt.savefig(os.path.join(visualize_dir, f"vis_{os.path.splitext(img_filename)[0]}_obj{obj_index}.jpg"))
    plt.close(fig)


def process_image_annotations(img_id, img_info, coco, cat_ids, coco_dir, output_dir, img_dir, split, 
                            split_images_dir, split_datasets, image_id_counter, annotation_id_counter,
                            category_mapping, category_counts, crop_size, visualize, skip_padded):
    """Process all annotations for a single image."""
    # Find image file
    img_path = find_image(img_info['file_name'], coco_dir, img_dir)
    if not img_path:
        print(f"Warning: Image {img_info['file_name']} not found. Skipping.")
        return image_id_counter, annotation_id_counter
    
    # Load image
    try:
        img = Image.open(img_path)
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}. Skipping.")
        return image_id_counter, annotation_id_counter
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    
    # Process each annotation (object)
    for obj_index, ann in enumerate(anns):
        bbox = ann['bbox']  # [x, y, width, height]
        cat_id = ann['category_id']
        
        if cat_id not in cat_ids:
            continue
        
        # Get class name
        class_name = coco.loadCats([cat_id])[0]['name']
        category_counts[class_name] = category_counts.get(class_name, 0) + 1
        
        # Calculate centroid 
        center_x, center_y = calculate_centroid(ann, bbox)
        
        # Calculate crop coordinates
        crop_coords = calculate_crop_coordinates(
            center_x, center_y, crop_size, img_np.shape[1], img_np.shape[0]
        )
        
        # Skip if crop would require padding and skip_padded is True
        if skip_padded and crop_coords['needs_padding']:
            print(f"Skipping object in {img_info['file_name']} that would require padding")
            continue
        
        # Create the cropped image
        final_crop = create_crop_image(img_np, crop_coords, crop_size)
        if final_crop is None:
            print(f"Warning: No valid region for object in {img_info['file_name']}. Skipping.")
            continue
        
        # Increment image ID counter for our new dataset
        image_id_counter += 1
        
        # Create a unique filename for the crop
        crop_filename = f"{class_name}_{os.path.splitext(img_info['file_name'])[0]}_obj{obj_index}.jpg"
        crop_path = os.path.join(split_images_dir, crop_filename)
        
        # Save crop
        Image.fromarray(final_crop).save(crop_path)
        print(f"Saved crop: {crop_path} (split: {split})")
        
        # Add this image to the appropriate split dataset
        split_datasets[split]["images"].append({
            "id": image_id_counter,
            "file_name": crop_filename,
            "width": crop_size,
            "height": crop_size,
            "license": 1,
            "original_img_id": img_id,
            "original_object_id": ann.get("id", obj_index),
            "split": split
        })
        
        # Transform the segmentation for the cropped image
        new_segmentation = transform_polygon_segmentation(ann, crop_coords, crop_size)
        
        # If no valid segmentation after transformation, create one from the bounding box
        if not new_segmentation:
            new_segmentation = create_bbox_segmentation(bbox, crop_coords, crop_size)
        
        # Calculate new bbox and area from the segmentation
        new_bbox, new_area = calculate_bbox_from_segmentation(new_segmentation, crop_size)
        
        # Increment annotation ID counter
        annotation_id_counter += 1
        
        # Add annotation for this object in the crop
        split_datasets[split]["annotations"].append({
            "id": annotation_id_counter,
            "image_id": image_id_counter,
            "category_id": cat_id,  # Keep the original category ID
            "segmentation": new_segmentation,
            "area": new_area,
            "bbox": new_bbox,
            "iscrowd": ann.get("iscrowd", 0),
            "original_annotation_id": ann.get("id", None)
        })
        
        # Visualize if requested
        if visualize and obj_index < 5:  # Only show first 5 crops per image to avoid too many plots
            visualize_crop(
                img_np, final_crop, class_name, crop_coords, new_segmentation, 
                new_bbox, ann, bbox, obj_index, output_dir, split, img_info['file_name']
            )
    
    return image_id_counter, annotation_id_counter


def process_annotation_file(ann_file, split, coco_dir, output_dir, target_classes, crop_size, 
                          visualize, skip_padded, split_datasets, image_id_counter, 
                          annotation_id_counter, category_mapping, category_counts):
    """Process a single COCO annotation file."""
    print(f"Processing annotation file: {ann_file} (split: {split})")
    
    # Load COCO annotations
    coco = COCO(ann_file)
    
    # Get image directory (usually in the same directory as the annotation file or parent)
    img_dir = os.path.dirname(ann_file)
    
    # Path for the output images for this split
    split_images_dir = os.path.join(output_dir, 'images', split)
    
    # Get category IDs for target classes
    if target_classes:
        cat_ids = []
        for target_class in target_classes:
            cats = coco.loadCats(coco.getCatIds(catNms=[target_class]))
            if cats:
                cat_ids.extend([cat['id'] for cat in cats])
            else:
                print(f"Warning: Class '{target_class}' not found in dataset")
    else:
        # Use all categories if no target classes specified
        cat_ids = coco.getCatIds()
    
    if not cat_ids:
        print("No valid category IDs found. Skipping this annotation file.")
        return image_id_counter, annotation_id_counter
    
    # Add categories to our new dataset, preserving original IDs
    for cat_id in cat_ids:
        cat_info = coco.loadCats([cat_id])[0]
        if cat_id not in category_mapping:
            category_mapping[cat_id] = cat_id  # Keep same ID
            
            new_category = {
                "id": cat_id,
                "name": cat_info["name"],
                "supercategory": cat_info.get("supercategory", "none")
            }
            
            # Add to all split datasets if not already there
            for s in split_datasets:
                # Check if this category is already in the split dataset
                if not any(c["id"] == cat_id for c in split_datasets[s]["categories"]):
                    split_datasets[s]["categories"].append(new_category)
            
            category_counts[cat_info["name"]] = 0
    
    # Get image IDs containing the target categories
    img_ids = set()
    for cat_id in cat_ids:
        img_ids.update(coco.getImgIds(catIds=[cat_id]))
    
    # Process each image
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        
        # Process all annotations for this image
        image_id_counter, annotation_id_counter = process_image_annotations(
            img_id, img_info, coco, cat_ids, coco_dir, output_dir, img_dir, split, split_images_dir,
            split_datasets, image_id_counter, annotation_id_counter,
            category_mapping, category_counts, crop_size, visualize, skip_padded
        )
    
    return image_id_counter, annotation_id_counter


def create_combined_dataset(split_datasets):
    """Create a combined dataset from all splits."""
    combined_dataset = {
        "info": {
            "description": "Combined cropped COCO dataset with polygon annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "COCO to Cropped COCO script",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add all categories to combined dataset (ensuring no duplicates)
    category_ids = set()
    for dataset in split_datasets.values():
        for category in dataset["categories"]:
            if category["id"] not in category_ids:
                combined_dataset["categories"].append(category)
                category_ids.add(category["id"])
    
    # Add all images and annotations to combined dataset
    for dataset in split_datasets.values():
        combined_dataset["images"].extend(dataset["images"])
        combined_dataset["annotations"].extend(dataset["annotations"])
    
    return combined_dataset


def save_datasets(split_datasets, combined_dataset, annotations_dir):
    """Save all datasets to JSON files."""
    total_images = 0
    total_annotations = 0
    
    # Save each split dataset
    for split, dataset in split_datasets.items():
        if split != "combined":  # Skip the combined dataset, we'll save it separately
            split_annotations_path = os.path.join(annotations_dir, f"{split}_annotations.json")
            with open(split_annotations_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"Split: {split}")
            print(f"  Images: {len(dataset['images'])}")
            print(f"  Annotations: {len(dataset['annotations'])}")
            print(f"  Categories: {len(dataset['categories'])}")
            
            total_images += len(dataset['images'])
            total_annotations += len(dataset['annotations'])
    
    # Save the combined dataset
    combined_annotations_path = os.path.join(annotations_dir, "combined_annotations.json")
    with open(combined_annotations_path, 'w') as f:
        json.dump(combined_dataset, f, indent=2)
    
    print(f"\nTotal statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Total categories: {len(combined_dataset['categories'])}")


def create_cropped_coco_dataset(coco_dir, output_dir, target_classes=None, crop_size=100, visualize=False, skip_padded=False):
    """
    Create a new COCO dataset with crops of objects from the original COCO dataset,
    maintaining polygon annotations within the crops.
    
    This function has been refactored for better readability and maintainability.
    All numeric values are converted to standard Python types before JSON serialization.
    """
    # Find annotation files
    ann_files = find_annotation_files(coco_dir)
    
    # Detect dataset splits (train/val/test)
    splits = [detect_split(ann_file) for ann_file in ann_files]
    print(f"Detected dataset splits: {splits}")
    
    # Create output directories
    annotations_dir = create_output_directories(output_dir, splits + ["combined"])
    
    # Initialize dataset structures
    split_datasets = initialize_datasets(splits)
    
    # Tracking variables
    image_id_counter = 0
    annotation_id_counter = 0
    category_mapping = {}  # Maps original category IDs to new ones (but we'll keep them the same)
    category_counts = {}
    
    # Process each annotation file
    for i, ann_file in enumerate(ann_files):
        split = splits[i]
        
        # Process this annotation file
        image_id_counter, annotation_id_counter = process_annotation_file(
            ann_file, split, coco_dir, output_dir, target_classes, crop_size, 
            visualize, skip_padded, split_datasets, image_id_counter, 
            annotation_id_counter, category_mapping, category_counts
        )
    
    # Report category statistics
    print("\nCategory statistics:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} instances")
    
    # Create a combined dataset
    combined_dataset = create_combined_dataset(split_datasets)
    
    # Save all datasets
    save_datasets(split_datasets, combined_dataset, annotations_dir)
    
    print(f"Dataset saved to: {output_dir}")
    print(f"Annotations saved to: {annotations_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a cropped COCO dataset with polygon annotations")
    parser.add_argument("--coco_dir", required=True, help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", required=True, help="Path to save the cropped COCO dataset")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names to include")
    parser.add_argument("--crop_size", type=int, default=100, help="Size of crop (default: 100)")
    parser.add_argument("--visualize", action="store_true", help="Visualize centroids and crops for debugging")
    parser.add_argument("--skip_padded", action="store_true", help="Skip crops that would require padding")
    
    args = parser.parse_args()
    
    # Create the cropped COCO dataset
    create_cropped_coco_dataset(
        args.coco_dir,
        args.output_dir,
        args.classes,
        args.crop_size,
        args.visualize,
        args.skip_padded
    )