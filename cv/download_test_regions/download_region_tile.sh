#!/bin/bash


# DC
region_name='dc'
# project_name='dc_usps'
# location="38.921611, -76.995726, 38.91534589480659, -76.98772430419922"
# project_name='dc_unionstation'
# location="38.900216, -77.012636, 38.89397221118196, -77.00454711914062"
project_name='dc_audifield'
location="38.8720477, -77.0170474, 38.86564216992963, -77.0090103149414"


# Seattle
# region_name="king"
# project_name="seattle_roosevelt"
# location="47.681744, -122.325996, 47.67550974029736, -122.31689651540462"
# project_name="seattle_northgate"
# location="47.710431, -122.328963, 47.704166826506054, -122.31964309743587"

# LA
# region_name="la"
# project_name="la_downtown_chinatown"
# location="34.0631990, -118.2416627, 34.056641640355934, -118.23348999023438"

# project_name="la_altadena"
# location="34.191647, -118.133806, 34.18510984477344, -118.12568664550781"

# project_name="la_ktown"
# location="34.064376, -118.305435, 34.057779382846725, -118.29734802246094"

# project_name="la_torrance"
# location="33.835593,-118.356391, 33.828786509874305, -118.34815979003906"


# Masachusetts
# region_name="ma"
# project_name="mass_acton_stroad"
# location="42.489280, -71.419658, 42.48323834594139, -71.4114761352539"

# project_name="mass_waltham"
# location="42.372182, -71.222412, 42.36615433532088, -71.21440887451172" 


# # NJ
# region_name="nj"
# project_name="nj_test"
# location="40.8036030, -74.4844324, 40.797437319357236, -74.476318359375"


output_dir="/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions"
python -m tile2net generate -l "$location" -o "$output_dir" -n "$project_name" -z 20 --stitch_step 2

overlap_input_dir="${output_dir}/${project_name}/tiles/static/${region_name}/512_20"
overlap_output_dir="${output_dir}/${project_name}/tiles/overlapping_stitched"
python create_overlapped_stitches.py -i "$overlap_input_dir" -o "$overlap_output_dir" -p "*.jpeg" --tile_size 256 256