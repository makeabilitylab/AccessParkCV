import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from shapely.geometry import Polygon
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Polygon as mplPolygon

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create cropped COCO dataset around object centroids')
    parser.add_argument('--coco_json', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing COCO images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for cropped images and annotations')
    parser.add_argument('--classes', nargs='+', type=str, help='List of classes to crop around (if not specified, all classes will be used)')
    parser.add_argument('--crop_size', type=int, default=100, help='Size of the crop (default: 100)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of crops with annotations')
    parser.add_argument('--vis_samples', type=int, default=10, help='Number of sample visualizations to generate (default: 10)')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save visualizations (defaults to output_dir/visualizations)')
    parser.add_argument('--yolo', action='store_true', help='Also export dataset in YOLO polygon format')
    parser.add_argument('--yolo_dir', type=str, default=None, help='Directory to save YOLO format data (defaults to output_dir/yolo)')
    parser.add_argument('--no_padding', action='store_true', help='Only save crops that fit completely within the original image (no padding required)')
    return parser.parse_args()

def setup_directories(args):
    """Create necessary output directories."""
    images_output_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    vis_dir = None
    if args.visualize:
        vis_dir = args.vis_dir if args.vis_dir else os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    yolo_dir = None
    if args.yolo:
        yolo_dir = args.yolo_dir if args.yolo_dir else os.path.join(args.output_dir, 'yolo')
        os.makedirs(yolo_dir, exist_ok=True)
        # Create images and labels directories for YOLO format
        os.makedirs(os.path.join(yolo_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels'), exist_ok=True)
    
    return images_output_dir, vis_dir, yolo_dir

def get_target_categories(coco, class_names=None):
    """Get category IDs for target classes."""
    if not class_names:
        return coco.getCatIds()
    
    target_cat_ids = []
    for class_name in class_names:
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            target_cat_ids.extend(cat_ids)
        else:
            print(f"Warning: Class '{class_name}' not found in dataset")
    
    return target_cat_ids

def calculate_centroid(polygon):
    """Calculate the centroid of a polygon."""
    if len(polygon) < 6:  # Need at least 3 points (x1,y1,x2,y2,x3,y3)
        return None
    
    # Reshape into [[x1,y1], [x2,y2], ...]
    points = np.array(polygon).reshape(-1, 2)
    
    try:
        poly = Polygon(points)
        if poly.is_valid:
            centroid = poly.centroid
            return [centroid.x, centroid.y]
    except Exception as e:
        print(f"Error calculating centroid: {e}")
    
    # Fallback: simple average if shapely fails
    return [np.mean(points[:, 0]), np.mean(points[:, 1])]

def clip_polygon(polygon, bounds):
    """Clip polygon to fit within the crop bounds."""
    x_min, y_min, x_max, y_max = bounds
    points = np.array(polygon).reshape(-1, 2)
    
    # Clip x-coordinates
    points[:, 0] = np.clip(points[:, 0], x_min, x_max)
    
    # Clip y-coordinates
    points[:, 1] = np.clip(points[:, 1], y_min, y_max)
    
    # Return clipped points without subtracting origin (will adjust later)
    return points.flatten().tolist()

def create_centered_crop(img, centroid, crop_size):
    """Create a crop centered exactly on the given centroid, with black padding if needed."""
    cx, cy = int(centroid[0]), int(centroid[1])
    half_size = crop_size // 2
    
    # Calculate ideal crop boundaries (exact centering)
    ideal_x_min = cx - half_size
    ideal_y_min = cy - half_size
    ideal_x_max = ideal_x_min + crop_size
    ideal_y_max = ideal_y_min + crop_size
    
    # Check if padding is required
    padding_required = (ideal_x_min < 0 or 
                        ideal_y_min < 0 or 
                        ideal_x_max > img.shape[1] or 
                        ideal_y_max > img.shape[0])
    
    # Create a black background image for padding
    padded_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    
    # Calculate source and destination coordinates for copying
    # Source coordinates (from original image)
    src_x_min = max(0, ideal_x_min)
    src_y_min = max(0, ideal_y_min)
    src_x_max = min(img.shape[1], ideal_x_max)
    src_y_max = min(img.shape[0], ideal_y_max)
    
    # Destination coordinates (in padded image)
    dst_x_min = max(0, -ideal_x_min)
    dst_y_min = max(0, -ideal_y_min)
    dst_x_max = dst_x_min + (src_x_max - src_x_min)
    dst_y_max = dst_y_min + (src_y_max - src_y_min)
    
    # Copy the valid portion of the original image to the padded image
    if src_x_max > src_x_min and src_y_max > src_y_min:
        padded_img[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img[src_y_min:src_y_max, src_x_min:src_x_max]
    
    crop_info = {
        'ideal_bounds': [ideal_x_min, ideal_y_min, ideal_x_max, ideal_y_max],
        'actual_bounds': [src_x_min, src_y_min, src_x_max, src_y_max],
        'dst_offset': [dst_x_min, dst_y_min],
        'padding_required': padding_required
    }
    
    return padded_img, crop_info

def process_annotation_for_crop(ann, crop_info, crop_size):
    """Process an annotation to fit within the crop boundaries."""
    if not ann.get('segmentation') or len(ann['segmentation']) == 0:
        return None
    
    x_min, y_min, x_max, y_max = crop_info['actual_bounds']
    ideal_x_min, ideal_y_min = crop_info['ideal_bounds'][0], crop_info['ideal_bounds'][1]
    dst_x_min, dst_y_min = crop_info['dst_offset']
    
    # Process each segmentation polygon
    new_segments = []
    is_visible = False
    
    for segment in ann['segmentation']:
        if not segment:
            continue
            
        # Reshape polygon to check intersection
        poly_points = np.array(segment).reshape(-1, 2)
        
        # Check if any point is within the crop
        x_in_crop = (poly_points[:, 0] >= x_min) & (poly_points[:, 0] <= x_max)
        y_in_crop = (poly_points[:, 1] >= y_min) & (poly_points[:, 1] <= y_max)
        
        if np.any(x_in_crop & y_in_crop):
            is_visible = True
            
            # Clip polygon to crop bounds
            clipped_poly = clip_polygon(segment, [x_min, y_min, x_max, y_max])
            
            # Adjust polygon coordinates to be relative to the centered crop
            points = np.array(clipped_poly).reshape(-1, 2)
            points[:, 0] = points[:, 0] - x_min + dst_x_min
            points[:, 1] = points[:, 1] - y_min + dst_y_min
            clipped_poly = points.flatten().tolist()
            
            if len(clipped_poly) >= 6:  # At least 3 points
                new_segments.append(clipped_poly)
    
    if not is_visible or not new_segments:
        return None
    
    # Update bbox to match the cropped coordinates
    x1, y1, w, h = ann['bbox']
    x2, y2 = x1 + w, y1 + h
    
    # Adjust coordinates to the centered crop (relative to padded image)
    x1 = max(x_min, x1) - ideal_x_min
    y1 = max(y_min, y1) - ideal_y_min
    x2 = min(x_max, x2) - ideal_x_min
    y2 = min(y_max, y2) - ideal_y_min
    
    new_w = max(0, x2 - x1)
    new_h = max(0, y2 - y1)
    
    return {
        'segmentation': new_segments,
        'area': ann['area'] * (len(new_segments) / len(ann['segmentation'])),  # Approximate
        'bbox': [x1, y1, new_w, new_h],
        'category_id': ann['category_id'],
        'iscrowd': ann.get('iscrowd', 0),
        'original_ann_id': ann['id']
    }

def visualize_crop_with_annotations(img, annotations, categories, title=None, save_path=None):
    """Visualize a cropped image with its annotations."""
    fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
    
    # Display the image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Create a color map for categories
    cat_colors = {}
    for cat in categories:
        cat_colors[cat['id']] = [random.random() for _ in range(3)]
    
    # Add annotations
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = next((cat['name'] for cat in categories if cat['id'] == cat_id), 'Unknown')
        color = cat_colors.get(cat_id, [0.5, 0.5, 0.5])
        
        # Draw segmentation polygons
        for seg in ann['segmentation']:
            if len(seg) >= 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
                polygon = np.array(seg).reshape(-1, 2)
                p = mplPolygon(polygon, fill=True, alpha=0.4, color=color)
                ax.add_patch(p)
        
        # Draw bounding box
        x, y, w, h = ann['bbox']
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
        
        # Add label
        ax.text(x, y, cat_name, fontsize=10, 
                bbox=dict(facecolor=color, alpha=0.7, pad=1))
    
    if title:
        ax.set_title(title)
    
    ax.set_axis_off()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_centroid(img_info, img, categories, vis_dir, crop_size):
    """Create a visualization showing the centroid position."""
    if 'centroid' not in img_info:
        return
        
    cx, cy = img_info['centroid']
    ideal_x_min, ideal_y_min = img_info['ideal_bounds'][0], img_info['ideal_bounds'][1]
    
    # Create a figure showing the centroid position
    fig, ax = plt.figure(figsize=(8, 8)), plt.gca()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Mark the center of the crop with a crosshair
    ax.axhline(crop_size // 2, color='r', linestyle='--', alpha=0.5, label='Center of crop')
    ax.axvline(crop_size // 2, color='r', linestyle='--', alpha=0.5)
    
    # Mark the centroid (should be at the center)
    centroid_x = cx - ideal_x_min
    centroid_y = cy - ideal_y_min
    ax.scatter(centroid_x, centroid_y, color='lime', s=100, marker='+', label='Object centroid')
    
    # Add a title
    cat_id = img_info['centroid_category_id']
    cat_name = next((cat['name'] for cat in categories if cat['id'] == cat_id), 'Unknown')
    ax.set_title(f"Crop {img_info['id']}: Centered on {cat_name} object")
    
    ax.legend(loc='upper right')
    ax.set_axis_off()
    
    # Save visualization
    center_vis_path = os.path.join(vis_dir, f"centroid_{img_info['id']:06d}.png")
    plt.savefig(center_vis_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

def process_crop(img, img_info, ann, coco, args, images_output_dir, new_image_id):
    """Process a single crop around an object's centroid."""
    # Get the centroid of the object
    segmentation = ann['segmentation']
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return None, None, None
    
    centroid = calculate_centroid(segmentation[0])
    if not centroid:
        return None, None, None
    
    # Create the centered crop
    cropped_img, crop_info = create_centered_crop(img, centroid, args.crop_size)
    
    # Skip crops that require padding if no_padding option is set
    if args.no_padding and crop_info['padding_required']:
        return None, None, None
    
    # Save the cropped image
    crop_filename = f"crop_{new_image_id:06d}.jpg"
    crop_path = os.path.join(images_output_dir, crop_filename)
    cv2.imwrite(crop_path, cropped_img)
    
    # Create new image info
    new_img_info = {
        'id': new_image_id,
        'file_name': crop_filename,
        'width': args.crop_size,
        'height': args.crop_size,
        'original_image_id': img_info['id'],
        'original_file_name': img_info['file_name'],
        'crop_bounds': crop_info['actual_bounds'],
        'ideal_bounds': crop_info['ideal_bounds'],
        'centroid': [int(centroid[0]), int(centroid[1])],
        'centroid_category_id': ann['category_id']
    }
    
    # Get all annotations for this image
    img_ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    img_anns = coco.loadAnns(img_ann_ids)
    
    # Process annotations within the crop
    new_anns = []
    for other_ann in img_anns:
        processed_ann = process_annotation_for_crop(other_ann, crop_info, args.crop_size)
        if processed_ann:
            processed_ann['image_id'] = new_image_id
            new_anns.append(processed_ann)
    
    return new_img_info, new_anns, crop_info

def generate_visualizations(new_dataset, images_output_dir, vis_dir, sample_size, crop_size):
    """Generate visualizations of the cropped images."""
    print(f"Generating {sample_size} visualizations...")
    
    # Create a dictionary to group annotations by image_id
    anns_by_image = {}
    for ann in new_dataset['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    # Sample images to visualize
    total_images = len(new_dataset['images'])
    sample_size = min(sample_size, total_images)
    sample_image_ids = random.sample([img['id'] for img in new_dataset['images']], sample_size)
    
    # Create visualizations
    for img_id in tqdm(sample_image_ids):
        # Find the image info
        img_info = next((img for img in new_dataset['images'] if img['id'] == img_id), None)
        if not img_info:
            continue
            
        # Load the cropped image
        img_path = os.path.join(images_output_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Create centroid visualization
        visualize_centroid(img_info, img, new_dataset['categories'], vis_dir, crop_size)
        
        # Create annotation visualization
        annotations = anns_by_image.get(img_id, [])
        
        # Create title with class information
        class_names = []
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = next((cat['name'] for cat in new_dataset['categories'] if cat['id'] == cat_id), 'Unknown')
            if cat_name not in class_names:
                class_names.append(cat_name)
        
        title = f"Crop {img_id}: {', '.join(class_names[:5])}"
        if len(class_names) > 5:
            title += f" and {len(class_names) - 5} more"
            
        # Save visualization
        vis_path = os.path.join(vis_dir, f"vis_{img_id:06d}.png")
        visualize_crop_with_annotations(
            img, 
            annotations, 
            new_dataset['categories'],
            title=title,
            save_path=vis_path
        )
    
    print(f"Visualizations saved to {vis_dir}")

def convert_to_yolo_format(new_dataset, images_output_dir, yolo_dir):
    """Convert the dataset to YOLO polygon format."""
    print("Converting to YOLO format...")
    
    # Create category mapping from COCO ID to YOLO index
    categories = new_dataset['categories']
    cat_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    # Create class names file
    class_names = [cat['name'] for cat in categories]
    with open(os.path.join(yolo_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))
    
    # Process each image and its annotations
    for img_info in tqdm(new_dataset['images']):
        img_id = img_info['id']
        
        # Copy the image to YOLO images directory
        src_img_path = os.path.join(images_output_dir, img_info['file_name'])
        dst_img_path = os.path.join(yolo_dir, 'images', img_info['file_name'])
        if os.path.exists(src_img_path):
            # Copy the image to the YOLO directory
            import shutil
            shutil.copy2(src_img_path, dst_img_path)
        
        # Get annotations for this image
        anns = [ann for ann in new_dataset['annotations'] if ann['image_id'] == img_id]
        
        # Create YOLO label file (uses the same basename as the image file)
        label_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        label_path = os.path.join(yolo_dir, 'labels', label_filename)
        
        with open(label_path, 'w') as f:
            for ann in anns:
                # Get class index
                class_idx = cat_mapping.get(ann['category_id'], 0)
                
                # Process each segmentation polygon
                for segment in ann['segmentation']:
                    if len(segment) < 6:  # Need at least 3 points
                        continue
                    
                    # Convert to normalized coordinates
                    points = np.array(segment).reshape(-1, 2)
                    
                    # Normalize coordinates (YOLO format uses values from 0 to 1)
                    points[:, 0] = points[:, 0] / img_info['width']
                    points[:, 1] = points[:, 1] / img_info['height']
                    
                    # Clip to ensure values are between 0 and 1
                    points = np.clip(points, 0, 1)
                    
                    # Format as YOLO polygon: class_idx x1 y1 x2 y2 ... xn yn
                    yolo_line = f"{class_idx} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                    f.write(yolo_line + '\n')
    
    # Create data.yaml file
    data_yaml = {
        'path': os.path.abspath(yolo_dir),
        'train': 'images',
        'val': '',  # No validation set by default
        'test': '',  # No test set by default
        'nc': len(categories),
        'names': class_names
    }
    
    # Save as YAML
    try:
        import yaml
        with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    except ImportError:
        # Fallback to simple text format if PyYAML not available
        with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write(f"train: images\n")
            f.write(f"val: \n")
            f.write(f"test: \n")
            f.write(f"nc: {len(categories)}\n")
            f.write(f"names: {class_names}\n")
    
    print(f"YOLO format data saved to {yolo_dir}")
    print(f"Found {len(new_dataset['categories'])} classes and processed {len(new_dataset['images'])} images")

def main():
    args = parse_arguments()
    
    # Setup directories
    images_output_dir, vis_dir, yolo_dir = setup_directories(args)
    
    # Initialize COCO API
    coco = COCO(args.coco_json)
    
    # Get target categories
    target_cat_ids = get_target_categories(coco, args.classes)
    
    # Get annotation IDs for target categories
    target_ann_ids = coco.getAnnIds(catIds=target_cat_ids)
    target_anns = coco.loadAnns(target_ann_ids)
    
    # Prepare new COCO format dataset
    new_dataset = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'categories': coco.dataset.get('categories', []),
        'images': [],
        'annotations': []
    }
    
    # Track new image and annotation IDs
    new_image_id = 1
    new_ann_id = 1
    
    # Process each annotation
    print(f"Processing {len(target_anns)} annotations...")
    for ann in tqdm(target_anns):
        # Skip annotations without segmentation
        if not ann.get('segmentation') or len(ann['segmentation']) == 0:
            continue
        
        # Get the image
        img_id = ann['image_id']
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        
        # Process the crop
        new_img_info, new_anns, crop_info = process_crop(
            img, img_info, ann, coco, args, images_output_dir, new_image_id)
            
        if not new_img_info or not new_anns:
            continue
        
        # Add the image to the dataset
        new_dataset['images'].append(new_img_info)
        
        # Add the annotations to the dataset
        for new_ann in new_anns:
            new_ann['id'] = new_ann_id
            new_dataset['annotations'].append(new_ann)
            new_ann_id += 1
        
        new_image_id += 1
    
    # Save the new annotations
    output_json_path = os.path.join(args.output_dir, 'instances_cropped.json')
    with open(output_json_path, 'w') as f:
        json.dump(new_dataset, f)
    
    print(f"Created new dataset with {new_image_id-1} images and {new_ann_id-1} annotations")
    print(f"Output directory: {args.output_dir}")
    
    # Generate visualizations if requested
    if args.visualize and new_image_id > 1:
        generate_visualizations(
            new_dataset, 
            images_output_dir, 
            vis_dir, 
            min(args.vis_samples, new_image_id-1),
            args.crop_size
        )
    
    # Convert to YOLO format if requested
    if args.yolo and new_image_id > 1:
        convert_to_yolo_format(new_dataset, images_output_dir, yolo_dir)

if __name__ == '__main__':
    main()