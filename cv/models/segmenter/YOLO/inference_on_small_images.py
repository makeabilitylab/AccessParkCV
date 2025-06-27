import torch
import cv2
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from shapely.geometry import Polygon, Point, LineString
from matplotlib.path import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = True

# model = YOLO('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runsobb/train/weights/best.pt')
model = YOLO('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train2/weights/best.pt')
model.info()

class_idx_to_name = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']
parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])

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
    filtered_by_iou = filter_overlapping_detections(detections, iou_threshold)
    
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
    center_object_candidates.sort(key=lambda x: x['conf'], reverse=True)
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

def calculate_parking_space_width(detections):
    """
    Calculate the width of the parking space in the center of the image.
    Width is defined as the perpendicular distance between the two longest edges.
    
    Args:
        detections: List of filtered detection dictionaries with 'xyxyxyxy' keys.
                   The first object should be the center parking space.
    Returns:
        width: Float representing the width of the parking space in pixels.
               Returns None if no detections exist.
        center_object: The center parking space object (for reference)
    """
    # If no detections, return None
    if not detections:
        return None, None
        
    # The first object in the filtered detections is the center parking space
    center_object = None
    for detection in detections:
        if detection['category_id'] in parking_categories:
            center_object = detection
            break
            
    if center_object is None:
        return None, None
        
    # Check if it has the required keys
    if 'xyxyxyxy' not in center_object:
        return None, None
        
    # Get side edges
    selected_pair = center_object['side_edges']
    
    # Calculate the distance between midpoints of the selected edges
    midpoint1 = selected_pair[0]['midpoint']
    midpoint2 = selected_pair[1]['midpoint']
    
    width = np.linalg.norm(midpoint2 - midpoint1)

    # Store the width in the center_object
    center_object['width'] = width
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
    
    # Check if both objects have the required coordinates
    if 'xyxyxyxy' not in center_object or 'xyxyxyxy' not in neighboring_object:
        return 0
    
    # Get polygon coordinates
    center_coords = center_object['xyxyxyxy']
    neighbor_coords = neighboring_object['xyxyxyxy']
    
    # Get the edges we care about, that we calculated earlier
    side_edges = center_object['side_edges']
    edge1, edge2 = side_edges

    # We need to find which of the two longest edges is closest to the neighboring object
    # For this, calculate the centroid of the neighboring object
    neighbor_centroid = np.mean(neighbor_coords, axis=0)
    
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

# Modify the visualization function to highlight the center object
def visualize_detections(image_path, detections, output_path=None, show=True, figsize=(10, 10)):
    """
    Visualize polygon detections on an image and optionally save the result.
    
    Args:
        image_path (str): Path to the input image
        detections (list): List of detection dictionaries with 'xyxyxyxy', 'conf', and 'category_id' keys
        output_path (str, optional): Path to save the output image. If None, the image won't be saved.
        show (bool): Whether to display the image
        figsize (tuple): Figure size for the output visualization
        
    Returns:
        None
    """
    # Read the image
    img = np.array(Image.open(image_path))
    height, width = img.shape[0], img.shape[1]
    image_center = (width / 2, height / 2)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    
    # Draw center point
    ax.plot(image_center[0], image_center[1], 'yo', markersize=8)
    
    # Define colors for different categories (can be extended)
    category_colors = {
        'access_aisle': 'red',
        'dp_one_aisle': 'blue',
    }
    
    # Add detections
    for detection in detections:
        # Get polygon coordinates
        coords = detection['xyxyxyxy']
        category = detection['category_id']
        conf = detection['conf']
        
        # Check if this polygon contains the center
        poly = Polygon(coords)
        is_center_object = poly.contains(Point(image_center))
        
        # Get color for category (default to green if not in dictionary)
        color = category_colors.get(category, 'green')
        
        # Create polygon patch
        polygon = patches.Polygon(
            coords, 
            closed=True, 
            fill=is_center_object,  # Fill the center object
            alpha=0.3 if is_center_object else 1.0,
            edgecolor=color, 
            facecolor=color if is_center_object else 'none',
            linewidth=3 if is_center_object else 2
        )
        ax.add_patch(polygon)
        
        # Add text with category and confidence
        centroid_x = sum(point[0] for point in coords) / len(coords)
        centroid_y = sum(point[1] for point in coords) / len(coords)
        
        label = f"{category}: {conf:.2f}"
        if is_center_object:
            label += " (CENTER)"
            
        ax.text(
            centroid_x, 
            centroid_y,
            label,
            color='white',
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
        )
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Tight layout
    plt.tight_layout()
    
    # Save the image if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    # Show the image if needed
    if show:
        plt.show()
    else:
        plt.close()

def predict_with_model(file_path, visualize=None):
    results = model(file_path)
    total_detected_objs = []
    for result in results:
        for i in range(len(result.obb.cls.int())):
            single_detection = {
                "conf": float(result.obb.conf[i]),
                "xyxyxyxy": result.obb.xyxyxyxy[i].tolist(),
                "category_id": result.names[result.obb.cls.int()[i].item()],
            }
            total_detected_objs.append(single_detection)
        filtered_detections = enhanced_filter_detections(total_detected_objs, iou_threshold=0.5, share_edge_proportion=0.4, proximity_threshold=5, overlap_proximity_threshold=15)
        
        calculate_parking_space_width(filtered_detections) # Calculate width of parking space
        # Calculate width of access aisles
        for detection in filtered_detections:
            if detection['category_id'] not in parking_categories:
                calculate_object_width(filtered_detections[0], detection)

        if visualize:
            # Visualize if num parking > 1 or access aisle > 2, or if no objects
            # Extract all category_id values
            category_ids = [item["category_id"] for item in filtered_detections]
            # Count the occurrences of each category
            category_counts = Counter(category_ids)
            parking_sum = category_counts['dp_no_aisle'] + category_counts['dp_one_aisle'] \
                          + category_counts['dp_two_aisle'] + category_counts['one_aisle'] \
                          + category_counts['two_aisle'] + category_counts['curbside']
            if category_counts['access_aisle'] > 2 or parking_sum > 1 or parking_sum == 0:
                basename = os.path.basename(file_path)
                output_path = os.path.join(visualize, f"{basename}_visualize_filtered.png")
                visualize_detections(file_path, filtered_detections, output_path=output_path, show=False)
    
    return filtered_detections

def make_access_edges_serializable(data):
    return data
    # return [list(s) for s in data]

def from_access_edges_serializable(serialized_data):
    return [set(tuple(item) for item in s) for s in serialized_data]

def save_results_as_json(results_dict, output_json_path):
    """
    Save detection results as a JSON file.
    
    Args:
        results_dict (dict): Dictionary with filenames as keys and lists of detection objects as values
        output_json_path (str): Path to save the JSON file
        
    Returns:
        None
    """

    # Convert Shapely objects which aren't JSON serializable
    serializable_results = {}
    
    for filename, detections in results_dict.items():
        serializable_detections = []
        for detection in detections:
            # Create a copy without the Shapely polygon object if it exists
            serializable_detection = {k: v for k, v in detection.items() if k != 'polygon'}
            if 'access_edges' in serializable_detection:
                serializable_detection['access_edges'] = make_access_edges_serializable(serializable_detection['access_edges'])
                del serializable_detection['side_edges']

            serializable_detections.append(serializable_detection)
        serializable_results[filename] = serializable_detections
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_json_path}")

def process_images_in_folder(folder_path, output_path):
    """
    Applies a specified function to every image in a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing the images.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path '{folder_path}' is not valid.")

    visualize_dir = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train/visualizations/"
    os.makedirs(visualize_dir, exist_ok=True)
    visualize_dir = None

    filename_to_detections = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # if filename != "dp_one_aisle_142_180_26734_png_obj0_jpg.rf.5cef4d46a2c99e85d7fab580e03bdf9b.jpg":
        #     continue
        # else:
        #     print("Found the image!")
        #     detected_objs = predict_with_model(file_path, visualize=visualize_dir)
        #     exit()

        print(f"Processing {filename}")
        detected_objs = predict_with_model(file_path, visualize=visualize_dir)
        basename = os.path.basename(filename)
        filename_to_detections[basename] = detected_objs

    save_results_as_json(filename_to_detections, output_path)

process_images_in_folder('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/dataset_made_from_crops/YOLO_larger/test_only_full_objs/images',
                         '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train2/testset_results_larger.json')

# process_images_in_folder('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/test_image/images/',
#                          '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/test_image/testset_results.json')