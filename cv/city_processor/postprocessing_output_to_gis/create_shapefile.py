import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import shutil
from shapely.geometry import Polygon, box

# project_name = 'dc_audifield'
# path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted/dc_audifield_predicted/total_spaces.json'
# output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/shapefiles/dc_audifield'

project_name = 'seattle_choropleth'
path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_choropleth/predicted/seattle_choropleth/total_spaces.json'
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/shapefiles/seattle_choropleth'

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def create_shapefile_from_json(json_path, output_shapefile_path):
    """
    Create a GIS shapefile from a JSON file containing objects with bounding boxes.
    
    Parameters:
    json_file_path (str): Path to the JSON file
    output_shapefile_path (str): Path where shapefile will be saved
    
    Returns:
    GeoDataFrame: The created GeoDataFrame with bounding boxes
    # """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create lists to store geometries and attributes
    geometries = []
    attributes = []

    # Process each object
    for obj in data['parking_spaces']:
        # Extract bounding box coordinates
        # Assuming format is [min_x, min_y, max_x, max_y]
        if 'polygon' in obj:
            polygon = obj['polygon']
            geom = Polygon(obj['polygon'])
        else:
            bbox = obj['bbox']
            # print(bbox)

            # Create a shapely box from the coordinates
            geom = box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
       
        geometries.append(geom)
        
        # Store all attributes for this object
        obj_attrs = {k: v for k, v in obj.items() if (k != 'bbox' or k != 'polygon')}
        attributes.append(obj_attrs)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs="EPSG:3857")
    
    # Save to shapefile
    gdf.to_file(output_shapefile_path)
    
    shutil.make_archive(os.path.join(output_shapefile_path, f'{project_name}'), 'zip', output_shapefile_path)

    return gdf

create_shapefile_from_json(path_to_json, output_path)