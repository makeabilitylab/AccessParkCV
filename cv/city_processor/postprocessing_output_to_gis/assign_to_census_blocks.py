import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import os

# project_name = 'seattle_northgate'
# path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted/seattle_northgate_predicted/total_spaces.json'
# output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_blocks/seattle_northgate'

# project_name = 'seattle_roosevelt'
# path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted/seattle_roosevelt/total_spaces.json'
# output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_blocks/seattle_roosevelt'

project_name = 'seattle_choropleth'
path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_choropleth/predicted/seattle_choropleth/total_spaces.json'
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_blocks/seattle_choropleth'


def get_census_blocks_for_points(points, shapefile_path):
    """
    Find the Census block for each lat/long point using GeoPandas and local Census TIGER/Line shapefiles.
    
    Parameters:
    -----------
    points : list of [lat, lon] coordinates
    shapefile_path : str
        Path to the Census block shapefile (.shp)
        
    Returns:
    --------
    pandas.DataFrame with original points and their Census block information
    """
    # Convert points to a DataFrame
    points_df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    
    # Create a GeoDataFrame from the points
    geometry = [Point(lon, lat) for lat, lon in points]
    points_gdf = gpd.GeoDataFrame(
        points_df, 
        geometry=geometry,
        crs="EPSG:4326"  # WGS84 coordinate system
    )
    
    # Load Census block shapefiles
    print(f"Loading shapefile from: {shapefile_path}")
    blocks_gdf = gpd.read_file(shapefile_path)
    
    # Ensure the Census blocks are in the same CRS as the points
    blocks_gdf = blocks_gdf.to_crs(points_gdf.crs)
    
    # Perform spatial join to find which block contains each point
    joined = gpd.sjoin(points_gdf, blocks_gdf, how="left", predicate="within")
    
    # Select relevant columns for the result
    # Adjusting to include common Census block fields, but this can be customized
    # based on what's available in your specific shapefile
    possible_columns = ['latitude', 'longitude', 'GEOID', 'GEOID20', 'BLOCKCE', 'BLOCKCE20', 
                       'NAME', 'MTFCC', 'ALAND', 'AWATER', 'STATEFP', 'COUNTYFP', 'TRACTCE']
    result_columns = ['latitude', 'longitude']
    
    # Add any of the possible columns that exist in the joined DataFrame
    for col in possible_columns:
        if col in joined.columns:
            result_columns.append(col)
    
    # Create result DataFrame
    result_df = joined[result_columns].copy()
    
    return result_df

def process_points_in_batches(points, shapefile_path, batch_size=1000):
    """
    Process points in batches to handle large datasets efficiently
    """
    results = []
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
        batch_results = get_census_blocks_for_points(batch, shapefile_path)
        results.append(batch_results)
    
    return pd.concat(results, ignore_index=True)

def count_points_per_census_block(points, shapefile_path, output_csv_path, batch_size=1000):
    """
    Count the number of points in each census block and save to CSV
    
    Parameters:
    -----------
    points : list of [lat, lon] coordinates
    shapefile_path : str
        Path to the Census block shapefile (.shp)
    output_csv_path : str
        Path where the output CSV will be saved
    batch_size : int
        Number of points to process in each batch
    
    Returns:
    --------
    pandas.DataFrame with census block information and point counts
    """
    # Process all points to get block assignments
    results = process_points_in_batches(points, shapefile_path, batch_size)
    
    # Determine which column contains the block identifier
    block_id_column = None
    for col in ['GEOID', 'GEOID20', 'BLOCKCE', 'BLOCKCE20']:
        if col in results.columns:
            block_id_column = col
            break
    
    if block_id_column is None:
        raise ValueError("Could not find a suitable block ID column in the results")
    
    # Count points per census block
    counts = results[block_id_column].value_counts().reset_index()
    counts.columns = ['block_id', 'point_count']
    
    # Get additional block information if available
    info_columns = [col for col in results.columns if col not in ['latitude', 'longitude']]
    
    if len(info_columns) > 1:  # If we have more info than just the block ID
        # Get unique block information (take first occurrence of each block)
        block_info = results.drop_duplicates(subset=block_id_column)[info_columns]
        
        # Merge with counts
        counts = pd.merge(
            counts, 
            block_info, 
            left_on='block_id', 
            right_on=block_id_column,
            how='left'
        )
        
        # Drop duplicate ID column
        if 'block_id' != block_id_column:
            counts = counts.drop(columns=block_id_column)
    
    # Save to CSV
    print(f"Saving counts to {output_csv_path}")
    counts.to_csv(output_csv_path, index=False)
    
    return counts

# Example usage
if __name__ == "__main__":

    # Read the JSON file
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    points = [item['centroid_latlong'] for item in data['parking_spaces']]

    # Path to your local TIGER/Line shapefile
    shapefile_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_blocks/tl_2024_53_tabblock20.zip"
    
    # Get Census blocks
    # For small number of points:
    results = get_census_blocks_for_points(points, shapefile_path)
    
    # For large number of points, use batching:
    # results = process_points_in_batches(points, shapefile_path, batch_size=1000)
    
    print(results)
    
    os.makedirs(output_path, exist_ok=True)
    # Count points per census block and save to CSV
    counts = count_points_per_census_block(points, shapefile_path, os.path.join(output_path, "census_blocks_results.csv"))
    
    print("Top blocks by point count:")
    print(counts.sort_values('point_count', ascending=False).head(10))
