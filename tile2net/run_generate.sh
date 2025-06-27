#!/bin/bash

# Tests
# location="47.664828, -122.317529, 47.661953, -122.313538" # UW small
# location="38.899525, -77.051475, 38.896202, -77.045145" # DC small

# Spring Hill
# location="35.79430, -87.00349, 35.64893, -86.84475" # Spring hill all

# DC
# location="38.99790, -77.11791, 38.91502, -76.90172" # DC North half
# location="38.90855, -77.12214, 38.79624,-76.90082" # DC South half
# location="38.92105, -77.12388, 38.90376, -76.90082" # DC middle


# Seattle
# location="47.95720383, -122.44897267, 47.75111338, -122.14324443" # Seattle north third X
# location="47.95099549, -122.41157158, 47.84355653, -122.14093344" # Seattle north third north X
# location="47.85383368, -122.44735166, 47.75293806, -122.14384431" # Seattle north third south X


# location="47.76808322, -122.44476540, 47.59624429, -122.14745170" # Seattle middle third X
# location="47.76675377, -122.44173661, 47.68086660, -122.14581843" # Seattle middle north X
# location="47.69371854, -122.44591288, 47.59684943, -122.14581843" # Seattle middle south X


# location="47.61137370, -122.44616783, 47.40770646, -122.15446381" # Seattle south third X 
# location="47.60960690, -122.44362084, 47.49905493, -122.15399054" # Seattle south third north X
# location="47.51990139, -122.44836887, 47.40916010, -122.15517755" # Seattle south third south X

# location="47.95720383, -122.44897267, 47.40916010, -122.15517755" # Seattle all 

location="34.0516083,-118.2539977, 34.0484493, -118.2503215" # LA small

output_dir="/gscratch/scrubbed/jaredhwa/DisabilityParking/data/tile2net"
# output_dir="/gscratch/makelab/jaredhwa/DisabilityParking/tile2net_tiles"
python -m tile2net generate -l "$location" -o "$output_dir" -n LA -z 20 --stitch_step 2
