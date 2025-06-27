import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import os

# Configure your project paths
project_name = 'seattle_choropleth'
path_to_json = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_choropleth/predicted/seattle_choropleth/total_spaces.json'
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_tracts/seattle_choropleth'

def get_census_tracts_for_points(points, tract_shapefile_path):
    """
    Find the Census tract for each lat/long point using GeoPandas and 
    Census TIGER/Line shapefiles.
    
    Parameters:
    -----------
    points : list of [lat, lon] coordinates
    tract_shapefile_path : str
        Path to the Census tract shapefile (.shp or .zip)
        
    Returns:
    --------
    pandas.DataFrame with original points and their Census tract information
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
    
    # Load Census tract shapefiles
    print(f"Loading tract shapefile from: {tract_shapefile_path}")
    tracts_gdf = gpd.read_file(tract_shapefile_path)
    
    # Ensure the Census tracts are in the same CRS as the points
    tracts_gdf = tracts_gdf.to_crs(points_gdf.crs)
    
    # Perform spatial join to find which tract contains each point
    joined = gpd.sjoin(points_gdf, tracts_gdf, how="left", predicate="within")
    
    # Determine which column contains the tract identifier
    tract_id_column = None
    for col in ['GEOID', 'GEOID20', 'TRACTCE', 'TRACTCE20']:
        if col in joined.columns:
            tract_id_column = col
            break
    
    if tract_id_column is None:
        raise ValueError("Could not find a suitable tract ID column in the results")
    
    # Ensure tract ID is a string to avoid groupby issues
    joined[tract_id_column] = joined[tract_id_column].astype(str)
    
    # Select relevant columns for the result
    possible_columns = ['latitude', 'longitude', 'GEOID', 'TRACTCE', 'NAME', 
                       'ALAND', 'AWATER', 'STATEFP', 'COUNTYFP']
    
    result_columns = ['latitude', 'longitude']
    
    # Add any of the possible columns that exist in the joined DataFrame
    for col in possible_columns:
        if col in joined.columns:
            result_columns.append(col)
    
    # Create result DataFrame
    result_df = joined[result_columns].copy()
    
    return result_df, tract_id_column

def process_points_in_batches(points, tract_shapefile_path, batch_size=1000):
    """
    Process points in batches to handle large datasets efficiently
    """
    results = []
    tract_id_column = None
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
        batch_results, batch_tract_column = get_census_tracts_for_points(batch, tract_shapefile_path)
        results.append(batch_results)
        
        # Use the tract ID column from the first batch
        if tract_id_column is None:
            tract_id_column = batch_tract_column
    
    return pd.concat(results, ignore_index=True), tract_id_column

def count_points_per_census_tract(points, tract_shapefile_path, output_csv_path, batch_size=1000):
    """
    Count the number of points in each census tract and save to CSV
    
    Parameters:
    -----------
    points : list of [lat, lon] coordinates
    tract_shapefile_path : str
        Path to the Census tract shapefile (.shp or .zip)
    output_csv_path : str
        Path where the output CSV will be saved
    batch_size : int
        Number of points to process in each batch
    
    Returns:
    --------
    pandas.DataFrame with census tract information and point counts
    """
    # Process all points to get tract assignments
    results, tract_id_column = process_points_in_batches(points, tract_shapefile_path, batch_size)
    
    # Count points per census tract
    tract_counts = results[tract_id_column].value_counts().reset_index()
    tract_counts.columns = ['tract_id', 'point_count']
    
    # Get additional tract information
    tract_info_columns = [col for col in results.columns if col not in ['latitude', 'longitude']]
    
    if len(tract_info_columns) > 1:  # If we have more info than just the tract ID
        # Get unique tract information (take first occurrence of each tract)
        tract_info = results.drop_duplicates(subset=[tract_id_column])[tract_info_columns]
        
        # Merge with counts
        tract_counts = pd.merge(
            tract_counts, 
            tract_info, 
            left_on='tract_id', 
            right_on=tract_id_column,
            how='left'
        )
        
        # Drop duplicate ID column
        if 'tract_id' != tract_id_column:
            tract_counts = tract_counts.drop(columns=tract_id_column)
    
    # Save to CSV
    print(f"Saving counts to {output_csv_path}")
    tract_counts.to_csv(output_csv_path, index=False)
    
    return tract_counts

# Example usage
if __name__ == "__main__":
    # Read the JSON file
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    points = [item['centroid_latlong'] for item in data['parking_spaces']]

    # Path to your local TIGER/Line shapefiles for census tracts
    tract_shapefile_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/postprocessing/census_blocks/tl_2024_53_tract.zip"
    
    # Check if tract shapefile exists
    if not os.path.exists(tract_shapefile_path):
        raise FileNotFoundError(f"Tract shapefile not found at {tract_shapefile_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Count points per census tract and save to CSV
    counts = count_points_per_census_tract(
        points, 
        tract_shapefile_path, 
        os.path.join(output_path, "census_tracts_results.csv")
    )
    
    print(f"Found {len(counts)} census tracts with parking spaces")
    print("\nTop tracts by point count:")
    print(counts.sort_values('point_count', ascending=False).head(10))