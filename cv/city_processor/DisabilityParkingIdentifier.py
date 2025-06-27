import os
import json
import numpy as np
import torch
import copy
from typing import Tuple, List
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from shapely.geometry import Polygon, Point, LineString
from ultralytics import YOLO
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
from lrucache import LRUCache
from utils import deg2num, num2deg, meters_to_feet

class DisabilityParkingIdentifier:

    def __init__(self, locator_model, characterizer_models=[]):

        # Tool 
        self._project_id = None
        self._input_tile_dir = None
        self._output_dir = None
        self._progress_file = None
        self._bbox_latlong_topleft = None
        self._bbox_latlong_botright = None
        self.topleft_coords = None
        self.botright_coords = None

        self._zoom_level = 20  # Slippy zoom level https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        self.imgsz = 256
        self.estimated_parking_size = 50 # estimated size of a parking spot, to exclude from if its within 
                                    # that far from the border
        self.from_crs = CRS.from_epsg(4326)
        self.to_crs = CRS.from_epsg(3857)

        # Models
        self.locator_model = locator_model 
        self.filter_models = [] # Class objects that have a "filter" function that takes an image, and returns True to keep
        self.characterizer_models = characterizer_models # Class objects that have a "characterize" function that takes an image, and returns key value pairs for characterization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Tile processing
        self.acceptable_tile_formats = (".png", ".jpg", ".jpeg", ".tif")
        self.available_tile_paths = {} # Dictionary of available tiles, as {(x,y): filepath}
        self._open_tile_cache = LRUCache()
        self.finished_tiles = {}  # For a 2x2 tile to be finished means
                                  # it, it's south, east, and southeast
                                  # half-overlapping tiles are also complete.

    def initialize(self, progress_file: str, 
                         bbox_latlong_topleft: Tuple[str, str],
                         bbox_latlong_botright: Tuple[str, str]):
        """
        Initialize from a progress file.
        """
        print("## #     Initializing     # ##")
        if not os.path.isfile(progress_file):
            raise Exception(f'Progress file "{progress_file}" does not exist.')
        
        progress_file = self.read_progress_file(progress_file)
        self._project_id = progress_file['project_id']
        self._input_tile_dir = progress_file['input_tile_dir']
        self._output_dir = progress_file['output_dir']
        self._zoom_level = progress_file['zoom_level']

        self._bbox_latlong_topleft = (float(bbox_latlong_topleft[0]), float(bbox_latlong_topleft[1]))
        self._bbox_latlong_botright = (float(bbox_latlong_botright[0]), float(bbox_latlong_botright[1]))

        self.finished_tiles = progress_dict['finished_tiles']

        print(f'Resuming project with project id: {self._project_id}.')
        print(f'                output directory: {self._output_dir}.')

        # Find tiles and load models
        self.startup()

    def initialize(self, project_id: str, 
                         input_tile_dir: str,
                         output_dir: str,
                         bbox_latlong_topleft: Tuple[str, str],
                         bbox_latlong_botright: Tuple[str, str],
                         zoom_level=20):
        """
        Initialize from scratch
        """
        print("## #     Initializing     # ##")

        self._project_id = project_id

        if not os.path.isdir(input_tile_dir):
            raise Exception(f'Input directory "{input_tile_dir}" does not exist.')

        if not os.path.isdir(output_dir):           
            print(f"Output directory {output_dir} does not exist. \n  Creating it.")
            os.makedirs(output_dir, exist_ok=True)

        self._input_tile_dir = input_tile_dir
        self._output_dir = output_dir
        self._bbox_latlong_topleft = (float(bbox_latlong_topleft[0]), float(bbox_latlong_topleft[1]))
        self._bbox_latlong_botright = (float(bbox_latlong_botright[0]), float(bbox_latlong_botright[1]))

        self._zoom_level = zoom_level

        # Find tiles and load models
        self.startup()

    def startup(self):
        self.available_tile_paths = self.find_downloaded_tiles() # Find paths of source tiles

    def read_progress_file(progress_file: str) -> dict:
        with open(progress_file, 'r') as file:
            data = json.load(file)
        return data
    
    def write_progress_file(self):
        progress_dict = {}
        progress_dict['project_id'] = self._project_id
        progress_dict['input_tile_dir'] = self._input_tile_dir
        progress_dict['output_dir'] = self._output_dir         
        progress_dict['zoom_level'] = self._zoom_level

        progress_dict['finished_tiles'] = self.finished_tiles
        
        project_dir_path = os.path.join(self._output_dir, self._project_id)
        os.makedirs(project_dir_path, exist_ok=True)

        save_path = os.path.join(project_dir_path, 'progress_file.json')
        with open(save_path, 'w') as file:
            json.dump(progress_dict, file, indent=4)
        
        print(f'Progress file written to {save_path}.')

    def write_clean_results(self):
        result_dict = {}
        result_dict['project_id'] = self._project_id
        result_dict['input_tile_dir'] = self._input_tile_dir
        result_dict['output_dir'] = self._output_dir         
        result_dict['zoom_level'] = self._zoom_level

        result_dict['parking_spaces'] = []
        for tile_pos, parking_list in self.finished_tiles.items():
            for spot in parking_list:
                parking_dict = {}
                parking_dict['id'] = spot['id']
                parking_dict['class'] = spot['predicted_class']
                parking_dict['centroid_latlong'] = spot['centroid']
                parking_dict['total_width'] = spot['total_width']
                parking_dict['bbox'] = spot['bbox']
                parking_dict['bbox_conf'] = spot['bbox_conf'] if 'bbox_conf' in spot else None
                parking_dict['polygon'] = spot['polygon'] if 'polygon' in spot else None
                result_dict['parking_spaces'].append(parking_dict)
        
        project_dir_path = os.path.join(self._output_dir, self._project_id)
        os.makedirs(project_dir_path, exist_ok=True)

        save_path = os.path.join(project_dir_path, 'total_spaces.json')
        with open(save_path, 'w') as file:
            json.dump(result_dict, file, indent=4)
        
        print(f'Results written to {save_path}.')

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

    def find_downloaded_tiles(self) -> dict:
        tiles_to_files = {}

        for filename in os.listdir(self._input_tile_dir):
            if filename.endswith(self.acceptable_tile_formats):
                try:
                    filepath = os.path.join(self._input_tile_dir, filename)
                    x, y = self.filename_to_tileid(filename)
                    tiles_to_files[(x,y)] = filepath
                except ValueError:
                    continue
        print(f"Found {len(tiles_to_files)} total source tiles.")
        return tiles_to_files

    def find_nearest_downloaded_tile(self, tile_x: int, tile_y: int) -> tuple:
        closest_tile = None
        min_distance = float('inf')
        
        for coord in self.available_tile_paths.keys():
            distance = ((coord[0] - tile_x) ** 2 + (coord[1] - tile_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_tile = coord
                
        return closest_tile

    def open_tile_img_toPIL(self, tile_x: int, tile_y: int, imgsz=256):
        """
        Given a tile x and y, returns the PIL image of that file.
        If the file doesn't exist, returns a black image.
        """
        # If the image was already opened and in the cache
        if self._open_tile_cache.contains((tile_x, tile_y)):
            return self._open_tile_cache.get((tile_x, tile_y))

        try:
            file_path = self.available_tile_paths[(tile_x, tile_y)]
            # Attempt to open the image file
            image = Image.open(file_path)
            # resize to desired size
            image = image.resize((imgsz, imgsz))
        except:        
            # Create a 256x256 black image if the file is not found
            image = Image.new('RGB', (imgsz, imgsz), (0, 0, 0))

        self._open_tile_cache.put((tile_x, tile_y), image) # add opened image to cache
        return image

    def requested_bbox_to_downloaded_tiles_bbox(self):
        # Find the top left and bottom right bounding tiles that we have downloaded,
        # to fit requested bbox. 
        desired_bbox_topleft = deg2num(self._bbox_latlong_topleft[0], self._bbox_latlong_topleft[1], self._zoom_level)
        desired_bbox_botright = deg2num(self._bbox_latlong_botright[0], self._bbox_latlong_botright[1], self._zoom_level)

        topleft_coords = list(self.find_nearest_downloaded_tile(desired_bbox_topleft[0], desired_bbox_topleft[1]))
        print(topleft_coords)
        topleft_coords[0] -= 1  # Shift window up so that parking spaces on north border are gotten in full
        topleft_coords[1] -= 1  # Shift window left so that parking spaces on west border are gotten in full
                                # (because a tile is considered complete if its east, south, and southeast tiles are computed)
        if bool(topleft_coords[0] & 1): # if odd
            topleft_coords[0] -= 1
        if bool(topleft_coords[1] & 1): # if odd
            topleft_coords[1] -= 1
        topleft_coords = tuple(topleft_coords)

        botright_coords = list(self.find_nearest_downloaded_tile(desired_bbox_botright[0], desired_bbox_botright[1]))
        if bool(botright_coords[0] & 1): # if odd
            botright_coords[0] += 1
        if bool(botright_coords[1] & 1): # if odd
            botright_coords[1] += 1
        botright_coords = tuple(botright_coords)

        # print(topleft_coords)
        # exit()
        return topleft_coords, botright_coords

    def stitch_tile_from_topleft_coord(self, tile_x: int, tile_y: int):
        topleft = self.open_tile_img_toPIL(tile_x, tile_y, imgsz=self.imgsz)
        topright = self.open_tile_img_toPIL(tile_x+1, tile_y, imgsz=self.imgsz)
        botleft = self.open_tile_img_toPIL(tile_x, tile_y+1, imgsz=self.imgsz)
        botright = self.open_tile_img_toPIL(tile_x+1, tile_y+1, imgsz=self.imgsz)
        
        # Create a new image with size 512x512
        new_image = Image.new('RGB', (512, 512))

        # Paste the four images into the new image
        new_image.paste(topleft, (0, 0))
        new_image.paste(topright, (256, 0))
        new_image.paste(botleft, (0, 256))
        new_image.paste(botright, (256, 256))

        return new_image

    def pixel_to_tile_coords(self, topleft_tile_coords: Tuple[int, int], imgsz: int, x: int, y: int):
        """
        Given a pixel position and a tile it's in, returns the tile and pixel position as a tuple
        """
        base_tile_size = imgsz // 2
        tile_x = int(topleft_tile_coords[0] + (x // base_tile_size))
        tile_y = int(topleft_tile_coords[1] + (y // base_tile_size))
        ret_x = x % base_tile_size
        ret_y = y % base_tile_size
        pre_tup = tile_x, tile_y, ret_x, ret_y
        return ((tile_x, tile_y), ret_x, ret_y)

    def convert_point_list_to_tile_coords(self, tile_x, tile_y, imgsz, point_list):
        ret_list = [self.pixel_to_tile_coords((tile_x, tile_y), imgsz, coord[0], coord[1]) for coord in point_list]
        return ret_list

    def tile_coord_to_latlong(self, tile_coords):
        tile_x, tile_y = tile_coords[0]
        xpos, ypos = tile_coords[1], tile_coords[2]
        xpos = xpos / self.imgsz
        ypos = ypos / self.imgsz
        lat, long = num2deg(tile_x + xpos, tile_y + ypos, self._zoom_level)
        return lat, long

    def tile_coord_to_3857(self, tile_coords):
        lat, long = self.tile_coord_to_latlong(tile_coords)
        x3857, y3857 = transform(self.from_crs, self.to_crs, xs=[long], ys=[lat])
        x3857, y3857 = x3857[0], y3857[0]
        return x3857, y3857
    
    def convert_polygon_tilecoord_list_to_3857(self, polygon_list):
        coords_in_3857 = [self.tile_coord_to_3857(tile_coord) for tile_coord in polygon_list]
        return coords_in_3857

    def edge_to_simple_representation(self, edge):
        return (tuple(map(int, edge['start'])), tuple(map(int, edge['end'])))

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

    def find_width_of_parking_space(self, parking_space):
        """
        Given a parking space dict, returns the width in FEET. Space is assumed to be triangular or a quadrilateral.

        Width is define as the distance between the midpoints of the two longest opposing edges.

        If no pair of opposing edges is definitively longer, we take the pair with the smallest variance.
        """
        if parking_space is None:
            return None
            
        # Check if it has the required keys
        if 'polygon' not in parking_space or parking_space['polygon'] == []:
            return None
            
        # Get polygon coordinates
        coords = parking_space['polygon']
        
        center_edges = self.get_polygon_edges(coords)
        long_edges = self.find_side_edges(center_edges)
        parking_space['side_edges'] = long_edges

        selected_pair = parking_space['side_edges']

        # Calculate the distance between midpoints of the selected edges
        midpoint1 = selected_pair[0]['midpoint']
        midpoint2 = selected_pair[1]['midpoint']
        
        width = np.linalg.norm(midpoint2 - midpoint1)

        # Store the width in the center_object
        # print("line340:", width, meters_to_feet(width))
        parking_space['space_width'] = meters_to_feet(width)
        parking_space['access_edges'] = {'edge1': self.edge_to_simple_representation(selected_pair[0]),
                                        'edge2': self.edge_to_simple_representation(selected_pair[1])}
        return parking_space['space_width']

    def find_access_aisle_width(self, center_object, neighboring_object):
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
            width: Float representing the width of the neighboring object in FEET.
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

        neighboring_object['aisle_width'] = meters_to_feet(calculated_width)
        closest_edge_identifier = (tuple([int(dim) for dim in closest_edge['start']]), tuple([int(dim) for dim in closest_edge['end']]))
        neighboring_object['closest_parking_edge'] = closest_edge_identifier
        return neighboring_object['aisle_width']


    def find_combined_width_of_space_and_aisles(self, parking_obj):
        width_of_space = self.find_width_of_parking_space(parking_obj) # stores it in item['space_width']
        if width_of_space is None:
            parking_obj['total_width'] = 'Unknown'
            return 'Unknown'
        parking_obj['total_width'] = width_of_space
        for side, aisle_list in parking_obj['access_aisles'].items(): # for list of access aisle in neighboring aisles, to each side
            max_aisle_on_this_side_width = 0
            for i, aisle in enumerate(aisle_list): # find the maximum width of an aisle on this side
                aisle_list[i]['polygon'] = self.convert_polygon_tilecoord_list_to_3857(aisle_list[i]['polygon'])
                # aisle['bbox'] = self.convert_polygon_tilecoord_list_to_3857(aisle['bbox'])
                max_aisle_on_this_side_width = max(self.find_access_aisle_width(parking_obj, aisle_list[i]), max_aisle_on_this_side_width)
            parking_obj['total_width'] += max_aisle_on_this_side_width
        return parking_obj['total_width']


    def locate_objects(self, image, 
                             parking_crop_box_size: int,
                             tile_x,
                             tile_y,
                             location_check_func=None):
        """
        Detects objects in the image, crops them out (with box size parameter)
        and returns a list of detected objects as a tuple:
        {'cropped_image': cropped out space,
         'predicted_class': predicted class as string,
         'centroid': centroid of space,
         'bbox': bounding box as (x1, y1, x2, y2)
        }

        Takes a function that the centroid location is passed into, and returns True if it is within a bad region
        False if not.
        """
        results = self.locator_model.detect(image, tile_x=tile_x, tile_y=tile_y)
        detected_objects = []

        for i, result in enumerate(results):
            predicted_class = result['class_name']
            (x1, y1), (x2, y2) = result['bbox']

            # Calculate the centroid of the bounding box
            centroid_x = int((x1 + x2) // 2)
            centroid_y = int((y1 + y2) // 2)

            location_check_result = (location_check_func is not None) and (location_check_func(image.size[0], centroid_x, centroid_y))
            if location_check_result:
                continue
            
            if predicted_class == 'access_aisle':
                continue

            # Calculate the top-left and bottom-right coordinates for the crop
            half_N = parking_crop_box_size // 2
            left = max(centroid_x - half_N, 0)
            top = max(centroid_y - half_N, 0)
            right = min(centroid_x + half_N, image.width)
            bottom = min(centroid_y + half_N, image.height)
            
            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))


            # Append the cropped object and its original position
            tilecoords_bbox_topleft = self.pixel_to_tile_coords((tile_x, tile_y), image.size[0], x1, y1)
            tilecoords_bbox_botright = self.pixel_to_tile_coords((tile_x, tile_y), image.size[0], x2, y2)

            detected_object = {
                'id': f'{tile_x}_{tile_y}_obj{i}',
                'cropped_image': cropped_image,
                'predicted_class': predicted_class,
                'centroid': self.pixel_to_tile_coords((tile_x, tile_y), image.size[0], centroid_x, centroid_y),
                'bbox': (tilecoords_bbox_topleft, tilecoords_bbox_botright),
                'cropped_topleft_pix': ((tile_x, tile_y), left, top),
            }
            if 'polygon' in result:
                detected_object['polygon'] = result['polygon']
                detected_object['polygon_tilex_tiley'] = (tile_x, tile_y)
                # polygon_coords = [self.pixel_to_tile_coords((tile_x, tile_y), image.size[0], coord[0], coord[1]) for coord in result['polygon']]
                # detected_object['polygon'] = polygon_coords
            if 'bbox_conf' in result:
                detected_object['bbox_conf'] = result['bbox_conf']
            if 'access_aisles' in result:
                detected_object['access_aisles'] = result['access_aisles']
                detected_object['access_aisles_tilex_tiley'] = (tile_x, tile_y)
                # for key, val in detected_object['access_aisles'].items():
                #     for aisle in val:
                #         aisle['polygon'] = self.convert_point_list_to_tile_coords(tile_x, tile_y, image.size[0], aisle['polygon'])
                #         aisle['bbox'] = self.convert_point_list_to_tile_coords(tile_x, tile_y, image.size[0], aisle['bbox'])

            detected_objects.append(detected_object)

        return detected_objects

    def complete_tile(self, tile_x: int, tile_y: int) -> list:
        
        def top_left_tile_check(imgsz, centroid_x, centroid_y):
            near_top_border = centroid_x <= self.estimated_parking_size
            near_left_border = centroid_y <= self.estimated_parking_size
            near_right_border = (imgsz - centroid_x) <= self.estimated_parking_size
            near_bottom_border = (imgsz - centroid_y) <= self.estimated_parking_size
            return near_right_border or near_bottom_border or near_top_border or near_left_border

        def east_tile_check(imgsz, centroid_x, centroid_y):
            vertical_middle = imgsz // 2
            within_vertical_middle = abs(centroid_x - vertical_middle) <= self.estimated_parking_size
            outside_top_and_bottom = centroid_y > self.estimated_parking_size and centroid_y < (imgsz - self.estimated_parking_size)
            return not (within_vertical_middle and outside_top_and_bottom)

        def south_tile_check(imgsz, centroid_x, centroid_y):
            horizontal_middle = imgsz // 2
            within_horizontal_middle = abs(centroid_y - horizontal_middle) <= self.estimated_parking_size
            outside_left_and_right = centroid_x > self.estimated_parking_size and (imgsz - centroid_x) > self.estimated_parking_size
            return not (within_horizontal_middle and outside_left_and_right)
        
        def southeast_tile_check(imgsz, centroid_x, centroid_y):
            center_x = imgsz // 2
            center_y = center_x
            center_of_img = self.estimated_parking_size // 2
            # Define the boundaries of the M pixel square centered at the center
            left_bound = center_x - center_of_img
            right_bound = center_x + center_of_img
            top_bound = center_y - center_of_img
            bottom_bound = center_y + center_of_img
            # Check if the centroid is within the defined square
            within_center_square = (left_bound <= centroid_x <= right_bound) and (top_bound <= centroid_y <= bottom_bound)
            return not within_center_square

        # Run on stitched tile tile_x tile_y
        # only take stuff where centroid is outside of N pix border 
        img = self.stitch_tile_from_topleft_coord(tile_x, tile_y)
        # img.save(f"{tile_x}_{tile_y}.png")
        result = self.locate_objects(img, 100, 
                                     tile_x, tile_y,
                                     location_check_func=top_left_tile_check)

        # Run on stitched tile tile_x + 1, tile_y (rightwards)
        # only take stuff within N pixels of vertical middle, outside of N pix of south & north border
        img = self.stitch_tile_from_topleft_coord(tile_x+1, tile_y)
        result.extend(self.locate_objects(img, 100, 
                                          tile_x+1, tile_y,
                                          location_check_func=east_tile_check))
        # img.save('teststitch2.png')

        # Run on stitched tile tile_x, tile_y + 1 (downwards)
        # only take stuff within N pixels of horiz middle, outside of N pix of east & west border
        img = self.stitch_tile_from_topleft_coord(tile_x, tile_y+1)
        result.extend(self.locate_objects(img, 100, 
                                          tile_x, tile_y+1,
                                          location_check_func=south_tile_check))
        # img.save(f'teststitch3_{tile_x}_{tile_y+1}.png')

        # Run on stitched tile tile_x + 1, tile_y + 1 (right down)
        # only take stuff within N pix of center of tile
        img = self.stitch_tile_from_topleft_coord(tile_x+1, tile_y+1)
        result.extend(self.locate_objects(img, 100, 
                                          tile_x+1, tile_y+1,
                                          location_check_func=southeast_tile_check))
        # img.save(f'teststitch4_{tile_x+1}_{tile_y+1}.png')

        return result

    def run_on_tile(self, tile_x: int, tile_y: int, visualize_dir=None):
        ##  Save aggregated centroids and crops
        cropped_results = self.complete_tile(tile_x, tile_y)

        # Pass crops through filter

        # Pass crops through characterizers
        for characterizer in self.characterizer_models:
            cropped_results, something_detected = characterizer.characterize_results(cropped_results, tile_x, tile_y, self.pixel_to_tile_coords, visualize_dir=visualize_dir)

        # # DEBUG visualize
        # crop_img_save_dir = os.path.join(self._output_dir, 'cropped_imgs')
        # for i, item in enumerate(cropped_results):
        #     centroid_x = item['centroid'][1]
        #     centroid_y = item['centroid'][2]
        #     item['cropped_image'].save(os.path.join(self.crop_img_save_dir, f'crop_{centroid_x}_{centroid_y}.png'))

        # Add results to self.finished_tiles
        tile_objects = []
        for item in cropped_results:
            # Convert all tilecoords points (e.g. ([167990, 365888], 32.64, 252.32) ) to 3857 x & y
            
            item['bbox'] = self.convert_polygon_tilecoord_list_to_3857(item['bbox'])

            if 'polygon' in item:
                item['polygon'] = self.convert_polygon_tilecoord_list_to_3857(item['polygon'])

                # Derive total width in feet, only if we have a successful mask
                self.find_combined_width_of_space_and_aisles(item)
            else:
                item['total_width'] = "Unknown"

            object_dict = {
                'id': item['id'],
                'predicted_class': item['predicted_class'],
                'centroid': self.tile_coord_to_latlong(item['centroid']),
                'bbox': item['bbox'],
                'total_width': item['total_width']
            }
            if 'polygon' in item:
                object_dict['polygon'] = item['polygon']
            if 'bbox_conf' in item:
                object_dict['bbox_conf'] = item['bbox_conf'] 

            # access_aisle_polygon_list = []
            # for edge_string, access_aisle_list in item['access_aisles'].items():
            #     for access_aisle in access_aisle_list:
            #         access_aisle_polygon_list.append(access_aisle['polygon'])
            # object_dict['access_aisles'] = access_aisle_polygon_list
            # print(item)
            object_dict['access_aisles'] = item['access_aisles']
            
            tile_objects.append(object_dict)

        self.finished_tiles[str((tile_x, tile_y))] = tile_objects

    def run(self, visualize_dir=None, topleft_coords_override=None):
        """
        Locates the disability parking spaces, characterizes them, and saves 
        output as json in the output directory
        """
        topleft_coords, botright_coords = self.requested_bbox_to_downloaded_tiles_bbox()

        if topleft_coords_override:
            topleft_coords = topleft_coords_override
        
        self.topleft_coords = topleft_coords
        self.botright_coords = botright_coords

        # Makedirs
        self.crop_img_save_dir = os.path.join(self._output_dir, 'cropped_imgs')
        os.makedirs(self.crop_img_save_dir, exist_ok=True)
        if visualize_dir:
            os.makedirs(visualize_dir, exist_ok=True)

        # call run_on_tile on every even numbered tile

        # Calculate total iterations
        total_iterations = ((botright_coords[1] - topleft_coords[1]) // 2) * ((botright_coords[0] - topleft_coords[0]) // 2)
        print(f"Total tiles to scan: {total_iterations}")

        # Create progress bar
        save_it = 0
        with tqdm(total=total_iterations, desc="Processing tiles") as pbar:
            for tile_y in range(topleft_coords[1] + 2, botright_coords[1], 2): # +2 to skip the first column and first row. This is because the top half/left of the images will be black causing potentially occluded detections.
                for tile_x in range(topleft_coords[0] + 2, botright_coords[0], 2):
                    # print(self.topleft_coords)
                    # print(tile_x, tile_y)
                    # exit()
                    
                    if str((tile_x, tile_y)) not in self.finished_tiles:
                        self.run_on_tile(tile_x, tile_y, visualize_dir=visualize_dir)
                    
                    pbar.update(1)

                    if save_it == 200:
                        self.write_progress_file()
                        save_it = 0
                    save_it += 1

        self.write_progress_file()
        self.write_clean_results()

        # self.run_on_tile(topleft_coords[0], topleft_coords[1])
        # print(topleft_coords)
        # print(botright_coords)