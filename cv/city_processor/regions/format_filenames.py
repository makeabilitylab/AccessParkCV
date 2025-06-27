import os
import re

def rename_files_in_directory(directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)
    
    # Regular expression to match the pattern and extract the first two numbers
    pattern = r'^(\d+_\d+).*(\.[^.]+)$'
    
    for filename in files:
        # Skip directories
        if os.path.isdir(os.path.join(directory_path, filename)):
            continue
            
        # Apply the regex pattern to the filename
        match = re.match(pattern, filename)
        if match:
            # Extract the first two numbers and the extension
            new_filename = match.group(1) + match.group(2)
            
            # Create full file paths
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)
            
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {filename} -> {new_filename}')
        else:
            print(f'Skipped: {filename} (did not match expected pattern)')

# Paths to your directories
image_directory = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/images'
text_directory = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/labels'

# Process both directories
print("Processing image directory...")
rename_files_in_directory(image_directory)

print("\nProcessing text directory...")
rename_files_in_directory(text_directory)

print("\nRenaming complete!")