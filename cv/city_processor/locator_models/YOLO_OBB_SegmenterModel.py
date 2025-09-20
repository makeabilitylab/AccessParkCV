import torch
import cv2
import os
import glob
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from shapely.geometry import Polygon, Point, LineString

import matplotlib.colors as mcolors

class DisabilityParkingSpaceSegmenterYOLO:

    def __init__(self, model_path, confidence_threshold=0.3, imgsz=100, verbose=False):

        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None

        self.class_idx_to_name = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']
        self.parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])
        
        self.open_model()

    def open_model(self):
        self.model = model = YOLO(self.model_path)

    def filter_overlapping_detections(self, detections, iou_threshold=0.5):
        """
        Filter overlapping polygon detections, keeping only the one with higher confidence
        when IoU exceeds the threshold.
        
        Args:
            detections: List of detection dictionaries with 'conf' and 'polygon' keys
            iou_threshold: IoU threshold above which to consider detections as overlapping
            
        Returns:
            List of filtered detections
        """
        if not detections:
            return []
        
        # Sort detections by confidence score (descending)
        sorted_detections = sorted(detections, key=lambda x: x['segmenter_conf'], reverse=True)
        
        # Convert polygon coordinates to Shapely Polygon objects
        polygons = []
        for detection in sorted_detections:
            coords = detection['polygon']
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

    def get_polygon_edges(self, coords):
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

    def find_side_edges(self, edges):
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

    def edge_to_simple_representation(self, edge):
        return sorted((tuple(map(int, edge['start'])), tuple(map(int, edge['end']))))

    def enhanced_filter_detections(self, detections, imgsz=100, iou_threshold=0.5, share_edge_proportion=0.4, proximity_threshold=10, overlap_proximity_threshold=20):
        """
        Enhanced filter for overlapping polygon detections with special rules:
        1. Keep object containing the center of the image
        2. Keep "access_aisle" objects that cover sufficient portion of the center object's two longest edges
        
        Args:
            detections: List of detection dictionaries with 'conf', 'polygon', and 'category_id' keys
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
        filtered_by_iou = self.filter_overlapping_detections(detections, iou_threshold)
        
        # Get image dimensions to find the center
        height, width = imgsz, imgsz
        image_center = Point(width / 2, height / 2)
        
        # Convert all detections to Shapely Polygons
        all_polygons = []
        for detection in filtered_by_iou:
            coords = detection['polygon']
            poly = Polygon(coords)
            detection['polygon_obj'] = poly
            all_polygons.append(detection)

        # Find the center parking space (object containing the image center)
        # If there are multiple, take the one with the highest confidence
        center_object = None
        center_object_candidates = []
        for detection in all_polygons:
            if detection['segmenter_category_id'] in self.parking_categories and detection['polygon_obj'].contains(image_center):
                center_object_candidates.append(detection)
        center_object_candidates.sort(key=lambda x: x['segmenter_conf'], reverse=True)
        center_object = center_object_candidates[0] if len(center_object_candidates) > 0 else None
        
        # If no object contains the center, return empty
        if center_object is None:
            return []
        
        # Initialize the result list with the center object
        result = [center_object]

        center_edges = self.get_polygon_edges(center_object['polygon'])
        # Find the edges that we care about
        long_edges = self.find_side_edges(center_edges)
        center_object['side_edges'] = long_edges
        center_object['access_edges'] = {'edge1': self.edge_to_simple_representation(long_edges[0]),
                                        'edge2': self.edge_to_simple_representation(long_edges[1])}
        center_object['access_aisles'] = {str(sorted(self.edge_to_simple_representation(long_edges[0]))): [],
                                          str(sorted(self.edge_to_simple_representation(long_edges[1]))): []}
        
        # Find "access_aisle" objects near the two longest edges of the center object
        for detection in all_polygons:
            # Skip if it's the center object or not an access_aisle
            if detection == center_object or detection['segmenter_category_id'] != 'access_aisle':
                continue
            
            # Get edges of the access_aisle object
            aisle_edges = self.get_polygon_edges(detection['polygon'])

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
                                point_inside_aisle = detection['polygon_obj'].contains(p)
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
                    center_object['access_aisles'][str(sorted(self.edge_to_simple_representation(center_edge)))].append(detection)
                    break
            
            if is_neighboring:
                result.append(detection)
        
        # if there's no parking space
        detected_categories = [item['segmenter_category_id'] for item in result]
        for obj_cat in detected_categories:
            if obj_cat in self.parking_categories:
                return result
        else:
            return []


    def calculate_parking_space_width(self, detections):
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
            if detection['segmenter_category_id'] in self.parking_categories:
                center_object = detection
                break
                
        if center_object is None:
            return None, None
            
        # Check if it has the required keys
        if 'polygon' not in center_object:
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

    def calculate_object_width(self, center_object, neighboring_object):
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
        if 'polygon' not in center_object or 'polygon' not in neighboring_object:
            return 0
        
        # Get polygon coordinates
        center_coords = center_object['polygon']
        neighbor_coords = neighboring_object['polygon']

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
        neighboring_object['closest_parking_edge'] = sorted(self.edge_to_simple_representation(closest_edge))
        return neighboring_object['width']

    def total_width_including_aisles(self, detections):
        # find parking space
        parking_space = None
        access_aisle_widths = {}
        for detection in detections:
            if detection['segmenter_category_id'] in self.parking_categories:
                parking_space = detection
            else: # detection['category_id'] == 'access_aisle
                # take the max parking width on that side of parking space
                access_aisle_widths[str(detection['closest_parking_edge'])] = max(detection['width'], access_aisle_widths.get(str(detection['closest_parking_edge']), 0))

        if parking_space is None: return 0
        
        total_width = parking_space['width'] + sum(access_aisle_widths.values())
        return total_width

    def characterize(self, image, tile_x=None, tile_y=None):
        """
        Runs the locator model on an image. Returns a list of dictionaries containing detected objects:

        [
            {
                'class_name': dp_one_aisle,
                'bbox': (x1, y1, x2, y2),
                'polygon': (x1, y1, x2, y2, x3, y3) # if available
            }
            {
            ...
            }
        ]
        """
        results = self.model(image, verbose=False)
        if len(results) == 0: # no detections
            return []

        total_detected_objs = []
        for result in results:
            for i in range(len(result.obb.cls.int())):
                single_detection = {
                    "segmenter_conf": float(result.obb.conf[i]),
                    "polygon": result.obb.xyxyxyxy[i].tolist(),
                    "segmenter_category_id": result.names[result.obb.cls.int()[i].item()],
                }
                total_detected_objs.append(single_detection)
            filtered_detections = self.enhanced_filter_detections(total_detected_objs, iou_threshold=0.5, share_edge_proportion=0.4, proximity_threshold=5, overlap_proximity_threshold=15)

        if len(filtered_detections) == 0:
            return []
            
        parking_space = filtered_detections[0] # Parking space is always the first object
        # remove uneccessary fields
        for detection in filtered_detections:
            if 'side_edges' in detection:
                del detection['side_edges']
                del detection['access_edges']

        return filtered_detections
        
    def transform_local_coords_to_tile_coords(self, x, y, topleft_pix, pixel_to_tile_coords_func):
        tile_x, tile_y = topleft_pix[0]
        real_x = x + topleft_pix[1]
        real_y = y + topleft_pix[2]
        return pixel_to_tile_coords_func((tile_x, tile_y), 512, real_x, real_y)

    # def save_visualization(self, image, cropped_result, detections, id, output_dir, scale_factor=8, line_width=12, darkness_percent=70, dotted_lines=True, round_corners=False):
    def save_visualization(self, image, cropped_result, detections, id, output_dir, scale_factor=8, line_width=2, darkness_percent=0, dotted_lines=True, round_corners=False):
        """
        Save a visualization of the image with detections using a simple color scheme:
        - Access aisles are always orange
        - Parking spaces are green
        - No text labels on boxes
        - Polygon outlines on the outside of regions
        - Areas outside polygons are darkened
        - Image is resized for smoother lines (default 4x larger)
        
        Args:
            image: PIL Image object containing the image
            cropped_result: The cropped detection result
            detections: List of detection objects with polygon coordinates
            id: Identifier for the image
            output_dir: Directory to save visualization results
            scale_factor: Factor to scale the image by (default=4)
            line_width: Width of polygon lines (default=8)
            darkness_percent: Percentage to darken areas outside polygons (default=50)
            dotted_lines: Whether to use dotted lines for polygons (default=False)
            round_corners: Whether to round the corners of polygons (default=False)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.path as mpath
        from PIL import Image
        import os
        from scipy.ndimage import binary_dilation
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        
        # Save the original blank image
        image.save(os.path.join(output_dir, f'{id}_blank.jpg'))
        
        # Resize the image to be larger by scale_factor (for smoother polygon lines)
        orig_width, orig_height = image.size
        new_width, new_height = orig_width * scale_factor, orig_height * scale_factor
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # resized_image.save(os.path.join(output_dir, f'{id}_blank1.jpg'))

        # Convert image to numpy array for matplotlib
        img = np.array(resized_image)
        
        # Create figure and axis with no padding
        fig_width, fig_height = new_width/100, new_height/100  # Adjust figure size to match resized image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Remove figure padding/margins completely
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Display the image without any borders
        ax.imshow(img)

        # Define simple color scheme
        AISLE_COLOR = '#FFCF03'  # Access aisles
        SPACE_COLOR = '#00DDFF'  # Parking spaces
        
        # Create a mask for all polygons combined (for darkening outside areas)
        mask = np.zeros((new_height, new_width), dtype=np.bool_)
        
        # Apply scale factor to polygon coordinates (already passed as a parameter)
        
        # Add detections - process in two passes to ensure parking spaces are on top
        # First pass: Process all non-parking spaces (aisles)
        # Second pass: Process all parking spaces
        
        # Create lists to store parking spaces and aisles separately
        parking_polygons = []
        aisle_polygons = []
        
        # Process all detections and categorize them
        for detection in detections:
            # Get polygon coordinates and scale them by scale_factor
            coords = np.array(detection['polygon']) * scale_factor
            
            # Create a mask for this polygon
            poly_path = mpath.Path(coords)
            x, y = np.meshgrid(np.arange(new_width), np.arange(new_height))
            points = np.column_stack((x.flatten(), y.flatten()))
            poly_mask = poly_path.contains_points(points).reshape(new_height, new_width)
            
            # Add this polygon to the combined mask
            mask = mask | poly_mask
            
            # Determine the category and store for later drawing
            if detection['segmenter_category_id'] in self.parking_categories:
                color = SPACE_COLOR  # Parking space
                parking_polygons.append((coords, color))
            else:
                color = AISLE_COLOR  # Access aisle
                aisle_polygons.append((coords, color))
        
        # Define linestyle based on dotted_lines parameter
        linestyle = 'dashed' if dotted_lines else 'solid'
        
        # Function to create rounded polygon if requested
        def create_polygon(coords, color):
            if round_corners:
                # Create a path with rounded corners
                vertices = []
                codes = []
                
                # Start with the first point
                vertices.append(coords[0])
                codes.append(Path.MOVETO)
                
                # Add curves for each corner
                for i in range(1, len(coords)):
                    # Add a point before the corner
                    prev_point = coords[i-1]
                    current_point = coords[i]
                    
                    # Calculate distance for control points (30% of segment length is a good starting point)
                    segment_length = np.sqrt(np.sum((current_point - prev_point)**2))
                    control_dist = min(segment_length * 0.3, 20 * scale_factor)  # Limit the control distance
                    
                    # Direction vector for the segment
                    direction = (current_point - prev_point) / segment_length
                    
                    # Add a point just before the corner
                    curve_start = current_point - direction * control_dist
                    vertices.append(curve_start)
                    codes.append(Path.LINETO)
                    
                    # Add the corner point
                    vertices.append(current_point)
                    codes.append(Path.CURVE3)  # Use a quadratic Bezier curve
                
                # Close the path smoothly
                vertices.append(coords[0])
                codes.append(Path.CURVE3)
                
                # Create the path and patch
                path = Path(vertices, codes)
                return PathPatch(path, edgecolor=color, facecolor='none', linewidth=line_width, linestyle=linestyle)
            else:
                # Create standard polygon with straight edges
                return plt.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=line_width,
                    linestyle=linestyle
                )
        
        # Draw aisles first (they'll be underneath)
        for coords, color in aisle_polygons:
            polygon = create_polygon(coords, color)
            ax.add_patch(polygon)
        
        # Draw parking spaces on top
        for coords, color in parking_polygons:
            polygon = create_polygon(coords, color)
            ax.add_patch(polygon)
        
        # Darken areas outside the polygons
        # Create a version of the image with darkened areas outside the mask
        darkened_img = img.copy()
        darkness_factor = (100 - darkness_percent) / 100  # Convert percentage to factor (e.g., 50% -> 0.5)
        darkened_img[~mask] = (darkened_img[~mask] * darkness_factor).astype(np.uint8)
        
        # Display the darkened image
        ax.imshow(darkened_img)
        
        # Remove axis ticks and borders
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        save_path = os.path.join(output_dir, f'{id}.jpg')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        return
        
        # # Set axis boundaries to match image dimensions
        # ax.set_xlim(0, new_width)
        # ax.set_ylim(new_height, 0)  # Note: y-axis is inverted in matplotlib
        
        # # Fix for white line at bottom of image - ensure margins are zero
        # # ax.margins(0, 0)    
        # fig.tight_layout(pad=0)

        # # Save the visualization with no borders or padding
        # save_path = os.path.join(output_dir, f'{id}.jpg')
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        # plt.close()

    def characterize_results(self, cropped_results, topleft_x, topleft_y, pixel_to_tile_coords_func, visualize_dir=None):
        something_detected = []
        if len(cropped_results) == 0: # if there were no detections
            return cropped_results, something_detected
        
        for cropped_result in cropped_results:
            detections = self.characterize(cropped_result['cropped_image'])

            # If something was detected, add True to list. False if not
            something_detected.append(len(detections) > 0)

            if visualize_dir:
                self.save_visualization(cropped_result['cropped_image'], cropped_result, detections, cropped_result['id'], visualize_dir)
 
            if len(detections) == 0: # if nothing was detected
                cropped_result['polygon'] = []
                cropped_result['width'] = 'Unknown'
                cropped_result['access_aisles'] = {}
            else:
                for detection in detections:       
                    if detection['segmenter_category_id'] in self.parking_categories:

                        # if visualize_dir:
                        #     self.save_visualization(cropped_result['cropped_image'], detections, cropped_result['id'], visualize_dir)

                        cropped_result['polygon'] = [self.transform_local_coords_to_tile_coords(x, y, cropped_result['cropped_topleft_pix'], pixel_to_tile_coords_func)
                                                     for x, y in detection['polygon']]
                        # access aisle polygons
                        cropped_result['access_aisles'] = {key: [] for key in detection['access_aisles'].keys()}
                        for edge_string in cropped_result['access_aisles'].keys():
                            for aisle_object in detection['access_aisles'][edge_string]:
                                aisle_obj_dict = {}
                                aisle_obj_dict['polygon'] = [self.transform_local_coords_to_tile_coords(x, y, cropped_result['cropped_topleft_pix'], pixel_to_tile_coords_func)
                                                            for x, y in aisle_object['polygon']]
                                aisle_obj_dict['segmenter_conf'] = aisle_object['segmenter_conf']
                                cropped_result['access_aisles'][edge_string].append(aisle_obj_dict)

        # print(cropped_results, something_detected)
        # if sum(something_detected) > 0:
        #     exit()
        return cropped_results, something_detected