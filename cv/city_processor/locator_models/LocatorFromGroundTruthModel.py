import os
import json
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from utils import deg2num, num2deg, unnormalize_pixel_coord
from lrucache import LRUCache
from classids_to_names import ids_to_names
from shapely.geometry import Polygon, Point, LineString


parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])

class DisabilityParkingGroundTruthLocator:

    def __init__(self, label_path, imgsz=512):

        self.label_path = label_path
        self.acceptable_tile_formats = (".txt")

        self.imgsz = imgsz

        self.tiles_to_files = self.find_downloaded_labels()

        self.parking_categories = set(['curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle'])

    def filename_to_tileid(self, filename: str) -> tuple:
        """
        Converts a filename from the tile2net Slippy naming convention to a tuple of (x, y).
        Parameters:
            Filename in the format "{x}_{y}.{png/jpg/etc}" (e.g. 299878_401132.jpeg)
        Returns:
            Tuple representing x, y of tile. (e.g. (299878, 401132))
        """
        name, extension = os.path.splitext(filename)
        x, y = name.split("_")
        return (int(x), int(y))

    def find_downloaded_labels(self) -> dict:
        tiles_to_files = {}

        for filename in os.listdir(self.label_path):
            if filename.endswith(self.acceptable_tile_formats):
                try:
                    filepath = os.path.join(self.label_path, filename)
                    x, y = self.filename_to_tileid(filename)
                    tiles_to_files[(x,y)] = filepath
                except ValueError:
                    continue
        return tiles_to_files

    def convert_polygon_to_bbox(self, coords):
        # Initialize min/max values using the first point
        x_min = x_max = coords[0][0]
        y_min = y_max = coords[0][1]
        
        # Find min/max x and y coordinates
        for x, y in coords:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Create bbox in [x_min, y_min, x_max, y_max] format
        bbox = ((x_min, y_min), (x_max, y_max))
        
        return bbox


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

    def enhanced_filter_detections(self, center_object, detections, share_edge_proportion=0.4, proximity_threshold=10, overlap_proximity_threshold=20):
        """
        Enhanced filter for overlapping polygon detections with special rules:
        1. Keep object containing the center of the image
        2. Keep "access_aisle" objects that cover sufficient portion of the center object's two longest edges
        
        Args:
            detections: List of detection dictionaries with 'conf', 'xyxyxyxy', and 'category_id' keys
            imgsz: Image size (assuming square image)
            share_edge_proportion: Minimum proportion of center object's edge that must be covered by the access_aisle
            proximity_threshold: Maximum distance (in pixels) to consider a point "near" the access_aisle
            overlap_proximity_threshold: Relaxed distance threshold for points that are inside the access_aisle
            
        Returns:
            List of the center object (parking space) and neighboring access aisles. Center object is always in the first position.
            New center object dictionary containing the neighboring access aisle
        """
        if not detections:
            return [center_object]
        
        # Convert all detections to Shapely Polygons
        all_polygons = []
        for detection in detections:
            coords = detection['polygon']
            poly = Polygon(coords)
            detection['polygon_obj'] = poly
            all_polygons.append(detection)

        # If no object contains the center, return empty
        if center_object is None:
            return [None]
        
        # Initialize the result list with the center object
        result = [center_object]
            
        center_edges = self.get_polygon_edges(center_object['polygon'])
        # Find the edges that we care about
        long_edges = self.find_side_edges(center_edges)
        center_object['side_edges'] = long_edges
        center_object['access_edges'] = {'edge1': self.edge_to_simple_representation(long_edges[0]),
                                        'edge2': self.edge_to_simple_representation(long_edges[1])}
        
        # Find "access_aisle" objects near the two longest edges of the center object
        for detection in all_polygons:
            # Skip if it's the center object or not an access_aisle
            if detection == center_object or detection['class_name'] != 'access_aisle':
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
                    break
            
            if is_neighboring:
                result.append(detection)
        
        for detection in detections:
            del detection['polygon_obj']

        # if there's no parking space
        detected_categories = [item['class_name'] for item in result]
        for obj_cat in detected_categories:
            if obj_cat in parking_categories:
                return result
        else:
            return [None]


    def edge_to_simple_representation(self, edge):
        return (tuple(map(int, edge['start'])), tuple(map(int, edge['end'])))

    def find_closest_edge(self, center_object, neighboring_object):
        side_edges = center_object['side_edges']
        edge1, edge2 = side_edges

        neighbor_coords = neighboring_object['polygon']

        # We need to find which of the two longest edges is closest to the neighboring object
        # For this, calculate the centroid of the neighboring object
        neighbor_centroid = np.mean(neighbor_coords, axis=0)
        
        dist1 = np.linalg.norm(edge1['midpoint'] - neighbor_centroid)
        dist2 = np.linalg.norm(edge2['midpoint'] - neighbor_centroid)
        
        # Select the edge closest to the neighboring object
        closest_edge = edge1 if dist1 < dist2 else edge2

        return self.edge_to_simple_representation(closest_edge)

    def find_neighboring_access_aisles(self, detections):
        access_aisle_objects = []
        parking_space_objects = []
        for detection in detections:
            if detection['class_name'] == 'access_aisle':
                access_aisle_objects.append(detection)
            else:
                detection['access_aisles'] = {} # key: closest parking edge, val: access aisle dict
                parking_space_objects.append(detection)
        
        for detection in parking_space_objects:
            neighboring_objs = self.enhanced_filter_detections(detection, access_aisle_objects)
            center_obj = neighboring_objs[0]
            for i in range(1, len(neighboring_objs)):
                closest_edge = self.find_closest_edge(center_obj, neighboring_objs[i])
                closest_edge_str = str(sorted(closest_edge))
                if closest_edge_str not in center_obj['access_aisles']:
                    center_obj['access_aisles'][closest_edge_str] = []
                center_obj['access_aisles'][closest_edge_str].append(copy.deepcopy(neighboring_objs[i])) # deepcopy as an aisle can be shared between two parking spaces, so we treat each as unique

    def read_objects_from_label_text(self, label_path):
        objects = []
        with open(label_path, 'r') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                # Split the line into values
                values = line.strip().split()
                # The first value is the class ID
                class_id = int(values[0])
                class_name = ids_to_names[class_id]

                # The rest are x,y coordinates in pairs
                coordinates = [float(coord) for coord in values[1:]]
                # Check if we have an even number of coordinates (x,y pairs)
                if len(coordinates) % 2 != 0:
                    print(f"Warning: Skipping malformed line, odd number of coordinates: {line}")
                    continue
                # Create polygon as list of (x,y) tuples
                polygon = []
                for i in range(0, len(coordinates), 2):
                    x, y = coordinates[i], coordinates[i+1]
                    polygon.append((unnormalize_pixel_coord(x), unnormalize_pixel_coord(y)))
                
                # Add to objects list
                objects.append({
                    'source_image': os.path.basename(label_path),
                    'class_name': class_name,
                    'polygon': polygon,
                    'bbox': self.convert_polygon_to_bbox(polygon)
                })

        return objects

    def detect(self, image, tile_x=None, tile_y=None):
        """
        Runs the locator model on an image. Returns a list of dictionaries containing detected objects:

        [{
            'class_name': dp_one_aisle,
            'bbox': (x1, y1, x2, y2),
            'polygon': (x1, y1, x2, y2, x3, y3) # if available
        }]
        
        
        Returns a list containing detected objects:
        [((bbox coords), class_name), ...]
        """
        if (tile_x, tile_y) not in self.tiles_to_files:
            return []

        detected_objects = self.read_objects_from_label_text(self.tiles_to_files[(tile_x, tile_y)])
        self.find_neighboring_access_aisles(detected_objects) # Find and assign the polygons of neighboring access_aisles
        return detected_objects

    def save_visualization(self, image, detections, id, output_dir, topleft_coords):
        """
        Save a visualization of the image with detections using a simple color scheme:
        - Access aisles are always orange
        - Parking spaces are green
        - No text labels on boxes
        
        Args:
            image: PIL Image object containing the image
            detections: List of detection objects with polygon coordinates
            id: Identifier for the image
            output_dir: Directory to save visualization results
            topleft_coords: Top-left coordinates for coordinate transformation
        """
        # Save the original blank image
        image.save(os.path.join(output_dir, f'{id}_blank.jpg'))
        
        # Convert image to numpy array for matplotlib
        img = np.array(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        
        # Define simple color scheme
        AISLE_COLOR = '#FFCF03'  # Access aisles
        SPACE_COLOR = 'green'   # Parking spaces
        
        # Add detections
        for detection in detections:
            # Get polygon coordinates
            coords = [self.transform_uncropped_coords_to_crop_coords(x, y, topleft_coords) 
                    for x, y in detection['polygon']]
            
            # Determine the color based on category
            if 'class_name' in detection and detection['class_name'] == 'access_aisle':
                color = AISLE_COLOR
            elif 'predicted_class' in detection and detection['predicted_class'] in self.parking_categories:
                color = SPACE_COLOR
            else:
                # Default fallback
                color = SPACE_COLOR
            
            # Create polygon patch without fill and no label
            polygon = plt.Polygon(
                coords,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=4
            )
            
            # Add polygon to the plot
            ax.add_patch(polygon)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a small legend
        # legend_elements = [
        #     plt.Line2D([0], [0], color=SPACE_COLOR, lw=2, label='Parking Space'),
        #     plt.Line2D([0], [0], color=AISLE_COLOR, lw=2, label='Access Aisle')
        # ]
        # ax.legend(handles=legend_elements, loc='upper right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join(output_dir, f'{id}.jpg')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()

    def old_save_visualization(self, image, detections, id, output_dir, topleft_coords):

        image.save(os.path.join(output_dir, f'{id}_blank.jpg'))

        img = np.array(image)
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Define colors for different categories (can be extended)
        category_colors = {
            'access_aisle': '#FFCF03',
            'parking_space': 'green',
        }

        # Add detections
        for detection in detections:
            # Get polygon coordinates
            # print(detection)
            coords = [self.transform_uncropped_coords_to_crop_coords(x, y, topleft_coords) for x, y in detection['polygon']]

            if 'predicted_class' in detection and detection['predicted_class'] in self.parking_categories:
                category = 'parking_space' 
            else:
                assert detection['class_name'] == 'access_aisle'
                category = 'access_aisle'

            height, width = img.shape[0], img.shape[1]
            image_center = (width / 2, height / 2)
            image_center = (width / 2, height / 2)
            
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

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{id}.jpg')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def transform_local_coords_to_tile_coords(self, x, y, topleft_pix, pixel_to_tile_coords_func):
        tile_x, tile_y = topleft_pix[0]
        real_x = x + topleft_pix[1]
        real_y = y + topleft_pix[2]
        return pixel_to_tile_coords_func((tile_x, tile_y), 512, real_x, real_y)

    def transform_uncropped_coords_to_crop_coords(self, x, y, topleft_pix):
        real_x = x - topleft_pix[1]
        real_y = y - topleft_pix[2]
        return real_x, real_y

    def characterize_results(self, cropped_results, topleft_x, topleft_y, pixel_to_tile_coords_func, visualize_dir=None):
        something_detected = []
        if len(cropped_results) == 0: # if there were no detections
            return cropped_results, something_detected

        for cropped_result in cropped_results:
            # If something was detected, add True to list. False if not
            something_detected.append(len(cropped_result) > 0)

            if len(cropped_result) == 0: # if nothing was detected
                cropped_result['polygon'] = []
                cropped_result['width'] = 'Unknown'
            else:
                detections = []
                detections.append(cropped_result)
                for key, value in cropped_result['access_aisles'].items():
                    detections.extend(value)

                for detection in detections:  
                    if 'predicted_class' in detection and detection['predicted_class'] in self.parking_categories:
                        if visualize_dir:
                            self.save_visualization(cropped_result['cropped_image'], detections, cropped_result['id'], visualize_dir, detection['cropped_topleft_pix'])

                      
        # convert polygons to tilecoords
        for cropped_result in cropped_results:
            tile_x, tile_y = cropped_result['polygon_tilex_tiley']
            cropped_result['polygon'] = [pixel_to_tile_coords_func((tile_x, tile_y), 512, coord[0], coord[1]) for coord in cropped_result['polygon']]

            access_aisles_polygon_list = {}
            for edgestring, aislelist in cropped_result['access_aisles'].items():
                access_aisles_polygon_list[edgestring] = []
                for aisledict in aislelist:
                    aisle_obj = {}
                    aisle_obj['polygon'] = [pixel_to_tile_coords_func((tile_x, tile_y), 512, coord[0], coord[1]) for coord in aisledict['polygon']]
                    aisle_obj['segmenter_conf'] = 1
                    access_aisles_polygon_list[edgestring].append(aisle_obj)
            cropped_result['access_aisles'] = access_aisles_polygon_list

        return cropped_results, something_detected