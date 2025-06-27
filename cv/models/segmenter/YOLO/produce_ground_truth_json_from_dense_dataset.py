import os
import json
import glob
import numpy as np
import math
from pathlib import Path
from shapely.geometry import Polygon, Point, LineString

import cv2
import numpy as np
import ast
from shapely.wkt import loads
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import random


"""
Produces ground truth json from the dataset that has ALL objects labeled, not just the center parking space and neighboring aisles

YOLO Polygon to Custom JSON Converter

This script converts YOLO format datasets with polygon annotations to a custom JSON format.
It processes both images and their corresponding label files to generate a structured JSON
output that maps image filenames to their polygon annotations with category information.

The expected YOLO dataset structure:
- dataset_root/
  - train/
    - images/
      - *.jpg, *.jpeg, *.png, *.bmp
    - labels/
      - *.txt
  - test/
    - images/
      - *.jpg, *.jpeg, *.png, *.bmp
    - labels/
      - *.txt
  - data.yaml (for class names)

Each YOLO label file should contain annotations in the format:
  class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
Where:
  - class_id: Integer ID of the object class
  - x1, y1, x2, y2, ...: Coordinates of polygon vertices (normalized 0-1 values)

The output JSON format:
{
  "image_filename.jpg": [
    {
      "xyxyxyxy": [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
      ],
      "ground_truth_polygon": [
        [x1, y1],
        [x2, y2],
        ...
      ],
      "category_id": "class_name"
    },
    ...
  ],
  ...
}

The coordinates in both "xyxyxyxy" and "ground_truth_polygon" are converted from 
normalized YOLO format (0-1) to actual pixel positions based on the image size.

Usage:
  python convert_yolo_to_json.py --dataset /path/to/your/yolo/dataset --output output.json --subset test

Requirements:
  - Python 3.6+
  - NumPy
  - PyYAML (optional, for parsing YAML files)
"""

class_idx_to_name = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']
parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])

def normalize_to_pixel_coords(coords, img_width, img_height):
    """
    Convert normalized coordinates (0-1) to pixel coordinates
    
    Args:
        coords: List of [x, y] coordinates in normalized format
        img_width: Width of the image in pixels
        img_height: Height of the image in pixels
    
    Returns:
        List of [x, y] coordinates in pixel format
    """
    pixel_coords = []
    for x, y in coords:
        px = x * img_width
        py = y * img_height
        pixel_coords.append([px, py])
    return pixel_coords

def compute_oriented_bbox(polygon):
    """
    Compute an oriented bounding box from a polygon
    
    Args:
        polygon: List of [x, y] coordinates of polygon vertices
    
    Returns:
        List of [x, y] coordinates of the oriented bounding box (4 points)
    """
    # Convert to numpy array
    points = np.array(polygon)
    
    # Get the minimal area bounding rectangle
    # First, try to use OpenCV if available
    try:
        import cv2
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rect)
        # Return in the same format as input
        return box.tolist()
    except ImportError:
        # Fallback method if OpenCV is not available
        # This is a simpler approach that will not always give the minimal area box
        
        # Compute the convex hull
        hull = points[_convex_hull(points)]
        
        # Find the minimum and maximum x, y coordinates
        min_x = np.min(hull[:, 0])
        max_x = np.max(hull[:, 0])
        min_y = np.min(hull[:, 1])
        max_y = np.max(hull[:, 1])
        
        # Create an axis-aligned bounding box
        bbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        
        return bbox


def _convex_hull(points):
    """
    Compute the convex hull of a set of points using Graham scan
    (Simple implementation for the fallback case)
    
    Args:
        points: NumPy array of [x, y] coordinates
    
    Returns:
        Indices of the convex hull vertices
    """
    # Find the point with the lowest y-coordinate
    start = np.argmin(points[:, 1])
    
    # Define a function to compute the angle
    def angle(p1, p2):
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    
    # Sort points by polar angle with respect to the start point
    start_point = points[start]
    angles = [angle(start_point, p) for p in points]
    sorted_indices = np.argsort(angles)
    
    # Initialize the hull with the start point and the first two sorted points
    hull = [start, sorted_indices[0]]
    
    # Process the remaining points
    for i in range(1, len(sorted_indices)):
        point = sorted_indices[i]
        
        while len(hull) > 1:
            # Check if the current point makes a right turn
            p1 = points[hull[-2]]
            p2 = points[hull[-1]]
            p3 = points[point]
            
            # Cross product to determine turn direction
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            
            if cross >= 0:  # Left turn or collinear
                break
            
            hull.pop()  # Remove the last point if it makes a right turn
            
        hull.append(point)
    
    return hull

def extract_category(filename, roboflow_category):
    if roboflow_category in ['left_aisle', 'right_aisle', 'access_aisle']:
        return "access_aisle"

    categories = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']
    for category in categories:
        if filename.startswith(category + "_"):
            return category

    return roboflow_category

    # return None

def filter_overlapping_detections(detections, iou_threshold=0.5):
    """
    Filter overlapping polygon detections, keeping only the one with higher confidence
    when IoU exceeds the threshold.
    
    Args:
        detections: List of detection dictionaries with 'conf' and 'xyxyxyxy' keys
        iou_threshold: IoU threshold above which to consider detections as overlapping
        
    Returns:
        List of filtered detections
    """
    if not detections:
        return []
    
    # Sort detections by confidence score (descending)
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    # Convert polygon coordinates to Shapely Polygon objects
    polygons = []
    for detection in sorted_detections:
        coords = detection['xyxyxyxy']
        poly = Polygon(coords)
        polygons.append(poly)
    
    # Initialize list to keep track of indices to keep
    indices_to_keep = []
    
    for i in range(len(sorted_detections)):
        should_keep = True
        
        # Check if current detection overlaps with any higher confidence detection
        for j in indices_to_keep:
            # Calculate IoU
            intersection_area = polygons[i].intersection(polygons[j]).area
            union_area = polygons[i].area + polygons[j].area - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                should_keep = False
                break
                
        if should_keep:
            indices_to_keep.append(i)
    
    # Return filtered detections
    return [sorted_detections[i] for i in indices_to_keep]

def get_polygon_edges(coords):
    # Calculate edges and their properties
    edges = []
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])
        
        # Calculate vector, length, and midpoint
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        midpoint = (p1 + p2) / 2

        # Unit vector and normal vector (perpendicular)
        unit_vector = edge_vector / edge_length
        normal_vector = np.array([-unit_vector[1], unit_vector[0]])
        
        edges.append({
            'index': i,
            'length': edge_length,
            'vector': edge_vector,
            'unit_vector': unit_vector,
            'normal_vector': normal_vector,
            'start': p1,
            'end': p2,
            'midpoint': midpoint,
            'edge_line': LineString([p1, p2])
        })

    return edges

def find_side_edges(edges):
    """
    Find the opposing edges of a parking space, that are the "side" of the parking space (i.e. where you get out from)
    """
    # Find the two pairs of opposing edges (in a quadrilateral)
    # Pair 1: edges 0 and 2
    # Pair 2: edges 1 and 3
    pair1_lengths = [edges[0]['length'], edges[2]['length']]
    pair2_lengths = [edges[1]['length'], edges[3 % len(edges)]['length']]

    # First check: If all edges in one pair are longer than all edges in the other pair
    if min(pair1_lengths) > max(pair2_lengths):
        # All edges in pair 1 are longer than all edges in pair 2
        selected_pair = [edges[0], edges[2]]
    elif min(pair2_lengths) > max(pair1_lengths):
        # All edges in pair 2 are longer than all edges in pair 1
        selected_pair = [edges[1], edges[3 % len(edges)]]
    else:
        # No clear dominance, so check by length similarity
        pair1_diff = abs(edges[0]['length'] - edges[2]['length'])
        pair2_diff = abs(edges[1]['length'] - edges[3 % len(edges)]['length'])
        
        # Select the pair with smaller length difference
        if pair1_diff <= pair2_diff:
            selected_pair = [edges[0], edges[2]]
        else:
            selected_pair = [edges[1], edges[3 % len(edges)]]
    
    return selected_pair

def enhanced_filter_detections(detections, imgsz=100, iou_threshold=0.5, share_edge_proportion=0.4, proximity_threshold=10, overlap_proximity_threshold=20):
    """
    Enhanced filter for overlapping polygon detections with special rules:
    1. Keep object containing the center of the image
    2. Keep "access_aisle" objects that cover sufficient portion of the center object's two longest edges
    
    Args:
        detections: List of detection dictionaries with 'conf', 'xyxyxyxy', and 'category_id' keys
        imgsz: Image size (assuming square image)
        iou_threshold: IoU threshold for basic overlap filtering
        share_edge_proportion: Minimum proportion of center object's edge that must be covered by the access_aisle
        proximity_threshold: Maximum distance (in pixels) to consider a point "near" the access_aisle
        overlap_proximity_threshold: Relaxed distance threshold for points that are inside the access_aisle
        
    Returns:
        List of filtered detections
    """
    if not detections:
        return []
    
    # First, run the basic IoU filtering
    # filtered_by_iou = filter_overlapping_detections(detections, iou_threshold)
    filtered_by_iou = detections

    # Get image dimensions to find the center
    height, width = imgsz, imgsz
    image_center = Point(width / 2, height / 2)
    
    # Convert all detections to Shapely Polygons
    all_polygons = []
    for detection in filtered_by_iou:
        coords = detection['xyxyxyxy']
        poly = Polygon(coords)
        detection['polygon'] = poly
        all_polygons.append(detection)

    # Find the center parking space (object containing the image center)
    # If there are multiple, take the one with the highest confidence
    center_object = None
    center_object_candidates = []
    for detection in all_polygons:
        if detection['category_id'] in parking_categories and detection['polygon'].contains(image_center):
            center_object_candidates.append(detection)
    # center_object_candidates.sort(key=lambda x: x['conf'], reverse=True)
    center_object = center_object_candidates[0] if len(center_object_candidates) > 0 else None
    
    # If no object contains the center, return empty
    if center_object is None:
        return []
    
    # Initialize the result list with the center object
    result = [center_object]

    center_edges = get_polygon_edges(center_object['xyxyxyxy'])
    # Find the edges that we care about
    long_edges = find_side_edges(center_edges)
    center_object['side_edges'] = long_edges
    center_object['access_edges'] = {'edge1': edge_to_simple_representation(long_edges[0]),
                                     'edge2': edge_to_simple_representation(long_edges[1])}
    
    # Find "access_aisle" objects near the two longest edges of the center object
    for detection in all_polygons:
        # Skip if it's the center object or not an access_aisle
        if detection == center_object or detection['category_id'] != 'access_aisle':
            continue
        
        # Get edges of the access_aisle object
        aisle_edges = get_polygon_edges(detection['xyxyxyxy'])

        # Find the best edge-to-edge alignment with the two longest edges
        is_neighboring = False
        for center_edge in long_edges:
            center_edge_line = center_edge['edge_line']
            for aisle_edge in aisle_edges:
                aisle_edge_line = aisle_edge['edge_line']
                # Check minimum distance between the edges
                min_distance = center_edge_line.distance(aisle_edge_line)

                if min_distance <= overlap_proximity_threshold: # increase distance to be conservative
                        # Check if aisle covers sufficient portion of center edge
                        num_samples = 20
                        
                        # Sample points along the center edge
                        center_points = [center_edge_line.interpolate(i/num_samples, normalized=True) 
                                        for i in range(num_samples + 1)]
                        
                        # Count how many center edge points are close to the aisle edge
                        # For each point, use overlap_proximity_threshold if the point is inside the aisle polygon
                        close_points = 0
                        for p in center_points:
                            # Check if point is inside the aisle polygon
                            point_inside_aisle = detection['polygon'].contains(p)
                            # Use appropriate threshold based on point location
                            point_threshold = overlap_proximity_threshold if point_inside_aisle else proximity_threshold
                            # Check if point is close enough to the aisle edge
                            if p.distance(aisle_edge_line) <= point_threshold:
                                close_points += 1
                        # If sufficient proportion of center edge is covered, consider them aligned
                        if close_points >= share_edge_proportion * (num_samples + 1):
                            is_neighboring = True
                            break
                            
            if is_neighboring:
                break
        
        if is_neighboring:
            result.append(detection)
    
    # if there's no parking space
    detected_categories = [item['category_id'] for item in result]
    for obj_cat in detected_categories:
        if obj_cat in parking_categories:
            return result
    else:
         return []

def edge_to_simple_representation(edge):
    return sorted((tuple(map(int, edge['start'])), tuple(map(int, edge['end']))))

def calculate_parking_space_width(center_object):
    """
    Calculate the width of the parking space in the center of the image.
    Width is defined as the perpendicular distance between the two longest edges.
    
    Args:
        center_object: center_object dictionary with 'xyxyxyxy' keys.
    Returns:
        width: Float representing the width of the parking space in pixels.
               Returns None if no detections exist.
    """
    if center_object is None:
        return None, None
        
    # Check if it has the required keys
    if 'xyxyxyxy' not in center_object:
        return None, None
        
    # Get polygon coordinates
    coords = center_object['xyxyxyxy']
    
    edges = get_polygon_edges(coords)
    selected_pair = find_side_edges(edges)
    center_object['side_edges'] = selected_pair
    
    # Calculate the distance between midpoints of the selected edges
    midpoint1 = selected_pair[0]['midpoint']
    midpoint2 = selected_pair[1]['midpoint']
    
    width = np.linalg.norm(midpoint2 - midpoint1)

    # Store the width in the center_object
    center_object['width'] = width
    center_object['access_edges'] = {'edge1': edge_to_simple_representation(selected_pair[0]),
                                     'edge2': edge_to_simple_representation(selected_pair[1])}
    return width, center_object

def calculate_object_width(center_object, neighboring_object):
    """
    Calculate the width of a neighboring object relative to the center object.
    
    Width is defined as:
    1. From the midpoint of the long center edge closest to the object, draw a perpendicular line
    2. Find intersection points with the neighboring object
    3. Width is the distance to the farthest intersection (or to the single intersection)
    
    Args:
        center_object: Dictionary containing the center parking space with 'xyxyxyxy' coordinates
        neighboring_object: Dictionary containing the neighboring object with 'xyxyxyxy' coordinates
        
    Returns:
        width: Float representing the width of the neighboring object in pixels.
               Returns 0 if no intersection is found.
    """
    import numpy as np
    
    # Check if both objects have the required coordinates
    if 'xyxyxyxy' not in center_object or 'ground_truth_polygon' not in neighboring_object:
        return 0
    
    # Get polygon coordinates
    center_coords = center_object['xyxyxyxy']
    neighbor_coords = neighboring_object['ground_truth_polygon']
    
    # center_edges = find_side_edges(get_polygon_edges(center_coords))

    # Get the edges we care about, that we calculated earlier
    side_edges = center_object['side_edges']
    edge1, edge2 = side_edges

    # We need to find which of the two longest edges is closest to the neighboring object
    # For this, calculate the centroid of the neighboring object
    neighbor_centroid = np.mean(neighbor_coords, axis=0)
    
    # Find distances from the midpoints of the two longest edges to the neighboring centroid    
    dist1 = np.linalg.norm(edge1['midpoint'] - neighbor_centroid)
    dist2 = np.linalg.norm(edge2['midpoint'] - neighbor_centroid)
    
    # Select the edge closest to the neighboring object
    closest_edge = edge1 if dist1 < dist2 else edge2
    
    # Calculate ray from midpoint in the direction of the normal vector
    midpoint = closest_edge['midpoint']
    normal = closest_edge['normal_vector']
    
    # Calculate the centroid of the center object
    center_centroid = np.mean(center_coords, axis=0)

    # Check if normal is pointing away from the center object's centroid
    # If dot product of (midpoint - center_centroid) and normal is negative, flip normal
    if np.dot(midpoint - center_centroid, normal) < 0:
        normal = -normal
    
    # Now we need to find intersections of this ray with the edges of the neighboring object
    intersections = []

    # Check intersection with each edge of the neighboring object
    for i in range(len(neighbor_coords)):
        n_p1 = np.array(neighbor_coords[i])
        n_p2 = np.array(neighbor_coords[(i + 1) % len(neighbor_coords)])

        # Ray equation: midpoint + t * normal, t >= 0
        # Line segment equation: n_p1 + s * (n_p2 - n_p1), 0 <= s <= 1
        
        # Set up parametric equations and solve
        # We need to solve for s and t in the equation:
        # midpoint + t * normal = n_p1 + s * (n_p2 - n_p1)
        
        edge_vector = n_p2 - n_p1
        
        # Check if lines are parallel (cross product is zero)
        cross_prod = normal[0] * edge_vector[1] - normal[1] * edge_vector[0]
        if abs(cross_prod) < 1e-10:  # Practically parallel
            continue
        
        # Vector from line point to ray point
        d = midpoint - n_p1

        # Calculate parameter s for the line segment
        # Using the formula for line-line intersection
        s = (d[1] * normal[0] - d[0] * normal[1]) / cross_prod
        
        if 0 <= s <= 1:
            # Calculate the intersection point on the edge
            intersection_on_edge = n_p1 + s * edge_vector
            
            # Calculate t parameter for the ray
            # Use the component with larger normal value for numerical stability
            if abs(normal[0]) > abs(normal[1]):
                t = (intersection_on_edge[0] - midpoint[0]) / normal[0]
            else:
                t = (intersection_on_edge[1] - midpoint[1]) / normal[1]
            
            # If t >= 0, the intersection is in the direction of the ray
            if t >= 0:
                intersection_point = midpoint + t * normal
                distance = t  # This is the distance along the ray
                intersections.append((intersection_point, distance))
    
    # If no intersections, width is 0
    if not intersections:
        calculated_width = 0 
    elif len(intersections) == 1: # If one intersection, return that distance
        calculated_width = intersections[0][1]
    else:
        calculated_width = max(intersections, key=lambda x: x[1])[1]     # If multiple intersections, return the distance to the farthest one

    neighboring_object['width'] = calculated_width
    closest_edge_identifier = (tuple([int(dim) for dim in closest_edge['start']]), tuple([int(dim) for dim in closest_edge['end']]))
    neighboring_object['closest_parking_edge'] = closest_edge_identifier

    return neighboring_object['width']

def visualize_objects_on_image(image_path, objects):
    """
    Visualize polygon objects on an image.
    
    Args:
        image_path (str): Path to the image file
        objects (list): List of polygon objects with various attributes
        
    Returns:
        None: Displays the visualization
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert from BGR to RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    
    # Define colors for different object types
    color_map = {
        'dp_one_aisle': 'blue',
        'access_aisle': 'green',
        # Add more categories as needed
    }
    
    # Process each object
    for obj in objects:
        category = obj['category_id']
        color = color_map.get(category, 'red')  # Default to red if category not in map
        
        # Draw the main polygon (xyxyxyxy)
        if 'xyxyxyxy' in obj and obj['xyxyxyxy']:
            points = np.array(obj['xyxyxyxy'])
            polygon_patch = MplPolygon(points, closed=True, fill=False, 
                                      edgecolor=color, linewidth=2, label=f"{category} (Detected)")
            ax.add_patch(polygon_patch)
        
        # Draw the ground truth polygon if available
        if 'ground_truth_polygon' in obj and obj['ground_truth_polygon']:
            gt_points = np.array(obj['ground_truth_polygon'])
            gt_polygon_patch = MplPolygon(gt_points, closed=True, fill=False, 
                                         edgecolor=color, linestyle='--', linewidth=2, label=f"{category} (Ground Truth)")
            ax.add_patch(gt_polygon_patch)
        
        # Draw access edges if available
        if 'access_edges' in obj and obj['access_edges']:
            for edge_key, edge_points in obj['access_edges'].items():
                if len(edge_points) == 2:  # Make sure we have start and end points
                    start, end = edge_points
                    ax.plot([start[0], end[0]], [start[1], end[1]], color='purple', 
                           linewidth=3, marker='o', label=f"{edge_key}")
        
        # Add text label for the category near the first point
        if 'xyxyxyxy' in obj and obj['xyxyxyxy']:
            first_point = obj['xyxyxyxy'][0]
            ax.text(first_point[0], first_point[1], category, 
                   color='white', backgroundcolor=color, fontsize=8)
    
    # Handle Shapely Polygon objects
    for obj in objects:
        if 'polygon' in obj and str(obj['polygon']).startswith('<POLYGON'):
            # Convert the string representation to a shapely Polygon
            polygon_str = str(obj['polygon'])
            try:
                # If it's a WKT string, we can parse it directly
                if polygon_str.startswith('<POLYGON'):
                    # Extract the WKT part
                    wkt_part = polygon_str.split('((')[1].split('))')[0]
                    wkt_str = f"POLYGON (({wkt_part}))"
                    shapely_polygon = loads(wkt_str)
                    
                    # Get the exterior coordinates
                    exterior_coords = list(shapely_polygon.exterior.coords)
                    
                    # Create a matplotlib polygon patch
                    polygon_patch = MplPolygon(exterior_coords, closed=True, 
                                             fill=False, edgecolor='orange', linestyle='-.',
                                             linewidth=1.5, label="Shapely Polygon")
                    ax.add_patch(polygon_patch)
            except Exception as e:
                print(f"Error processing Shapely polygon: {e}")
    
    # Handle side edges
    for obj in objects:
        if 'side_edges' in obj and isinstance(obj['side_edges'], list):
            for edge in obj['side_edges']:
                if 'start' in edge and 'end' in edge:
                    try:
                        # Convert numpy arrays to lists if needed
                        start = edge['start'].tolist() if hasattr(edge['start'], 'tolist') else edge['start']
                        end = edge['end'].tolist() if hasattr(edge['end'], 'tolist') else edge['end']
                        
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                               color='yellow', linewidth=2, linestyle='-.', 
                               marker='x', label=f"Side Edge {edge.get('index', '')}")
                    except Exception as e:
                        print(f"Error processing side edge: {e}")
    
    # Add a title
    ax.set_title("Object Visualization on Image")
    
    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
    
    # Display the image
    plt.tight_layout()
    plt.savefig('debug.png')

def yolo_to_custom_json(dataset_path, output_json_path, subset=None, img_width=100, img_height=100):
    """
    Convert YOLO polygon annotations to custom JSON format
    
    Args:
        dataset_path: Path to YOLO dataset containing images and labels
        output_json_path: Path where the output JSON file will be saved
        subset: Optional subset to process (e.g., 'train', 'test'). If None, process all images.
        img_width: Width of the images in pixels
        img_height: Height of the images in pixels
    """
    # Dictionary to store the result
    result = {}
    
    # Get all image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    if subset:
        # Only look in the specified subset directory
        subset_images_dir = os.path.join(dataset_path, subset, 'images')
        if os.path.exists(subset_images_dir):
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(subset_images_dir, '**', ext), recursive=True))
        else:
            print(f"Warning: Subset directory '{subset_images_dir}' not found.")
    else:
        # Look in all subset directories
        for potential_subset in ['train', 'test', 'val']:
            subset_images_dir = os.path.join(dataset_path, potential_subset, 'images')
            if os.path.exists(subset_images_dir):
                for ext in image_extensions:
                    image_paths.extend(glob.glob(os.path.join(subset_images_dir, '**', ext), recursive=True))
    
    # Read class names from data.yaml
    classes = read_classes(dataset_path)
    
    # Process each image
    for image_path in image_paths:
        # Get image filename
        image_filename = os.path.basename(image_path)

        # Determine which subset this image belongs to by examining its path
        path_parts = Path(image_path).parts
        current_subset = None
        for part in path_parts:
            if part in ['train', 'test', 'val', 'test_only_full_objs']:
                current_subset = part
                break
        
        if not current_subset:
            print(f"Warning: Could not determine subset for image {image_path}. Skipping.")
            continue
        
        # Find corresponding label file (same name, .txt extension, in labels directory)
        label_dir = os.path.join(dataset_path, current_subset, 'labels')
        label_path = os.path.join(label_dir, image_filename.rsplit('.', 1)[0] + '.txt')
        
        # If label file exists, process it
        if os.path.exists(label_path):
            result[image_filename] = []

            # Read and parse label file
            with open(label_path, 'r') as f:
                center_annotation = None
                for line in f:
                    # print(line)
                    parts = line.strip().split()
                    
                    class_id = int(parts[0])
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    
                    # Extract polygon coordinates (YOLO format uses normalized coordinates)
                    polygon = []
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i + 1])
                            polygon.append([x, y])
                    
                    # Convert normalized coordinates to pixel coordinates
                    pixel_polygon = normalize_to_pixel_coords(polygon, img_width, img_height)
                    

                    if len(parts) < 9:  # At least class_id + 4 points (8 coordinates)
                        obb = pixel_polygon
                    else:
                        # Compute the oriented bounding box from the pixel polygon
                        obb = compute_oriented_bbox(pixel_polygon)
                    
                    real_category = extract_category(image_filename, class_name)
                    # Add annotation to result
                    annotation = {
                        "xyxyxyxy": obb,
                        "ground_truth_polygon": pixel_polygon,
                        "category_id": real_category,
                        "roboflow_category_id": class_name
                    }

                    result[image_filename].append(annotation)

                # Filter out for only the parking space in the center and the aisles next to it
                filtered_detections = enhanced_filter_detections(result[image_filename], iou_threshold=0.5, share_edge_proportion=0.4, proximity_threshold=5, overlap_proximity_threshold=15)
                
                # Find widths
                for detection in filtered_detections:
                    if detection['category_id'] in parking_categories:
                        calculate_parking_space_width(detection)
                        center_annotation = detection
                
                for detection in filtered_detections:
                    if detection['category_id'] not in parking_categories:
                        calculate_object_width(center_annotation, detection)

                # visualize_objects_on_image(image_path, filtered_detections)

                for detection in filtered_detections:
                    if detection['category_id'] in parking_categories:
                        del detection['side_edges']
                    del detection['polygon']
                result[image_filename] = filtered_detections

    
    # Write output to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Conversion completed. JSON saved to {output_json_path}")
    print(f"Processed {len(result)} images with annotations.")
    print(f"Coordinates converted to pixel values (based on {img_width}x{img_height} image size).")


def read_classes(dataset_path):
    """
    Read class names from data.yaml
    
    Args:
        dataset_path: Path to YOLO dataset
    
    Returns:
        List of class names
    """
    # Try to read from data.yaml
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(data_yaml_path):
        try:
            import yaml
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        except (ImportError, Exception) as e:
            print(f"Error reading data.yaml: {e}")
    
    # Try alternative locations
    for potential_subset in ['train', 'test', 'val']:
        alt_yaml_path = os.path.join(dataset_path, potential_subset, 'data.yaml')
        if os.path.exists(alt_yaml_path):
            try:
                import yaml
                with open(alt_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        return data['names']
            except (ImportError, Exception):
                pass
    
    # Try to read from classes.txt
    classes_path = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f]
    
    # If no class file found, return an empty list
    print("Warning: No class file found. Using numeric class IDs.")
    return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO polygon annotations to custom JSON format')
    parser.add_argument('--dataset', type=str, required=True, help='Path to YOLO dataset')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file path')
    parser.add_argument('--subset', type=str, default=None, 
                        help='Specific subset to process (e.g., train, test). If not specified, process all subsets.')
    parser.add_argument('--width', type=int, default=100, help='Width of the images in pixels (default: 100)')
    parser.add_argument('--height', type=int, default=100, help='Height of the images in pixels (default: 100)')
    
    args = parser.parse_args()
    
    yolo_to_custom_json(args.dataset, args.output, args.subset, args.width, args.height)

