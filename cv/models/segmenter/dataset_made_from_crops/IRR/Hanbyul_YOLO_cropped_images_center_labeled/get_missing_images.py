import os
import shutil

def copy_unique_images(dir_a, dir_b, dir_c):
    """
    Copy images from dir_a to dir_c if they don't exist in dir_b,
    comparing filenames after removing '.rf.' and everything that follows.
    """
    # Check if directories exist
    if not os.path.isdir(dir_a):
        print(f"Error: Source directory A '{dir_a}' does not exist.")
        return
    
    if not os.path.isdir(dir_b):
        print(f"Error: Comparison directory B '{dir_b}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dir_c):
        try:
            os.makedirs(dir_c)
            print(f"Created destination directory C: '{dir_c}'")
        except Exception as e:
            print(f"Error creating destination directory: {e}")
            return
    
    # Get all files from directory A and B
    files_a = [f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f))]
    files_b = [f for f in os.listdir(dir_b) if os.path.isfile(os.path.join(dir_b, f))]
    
    # Transform filenames from directory B for comparison
    transformed_names_b = set()
    for filename in files_b:
        # Get the part before '.rf.' if it exists
        if '.rf.' in filename:
            transformed_name = filename.split('.rf.')[0]
        else:
            transformed_name = filename
        transformed_names_b.add(transformed_name)
    
    # Track statistics
    copied_count = 0
    skipped_count = 0
    
    # Process files from directory A
    for filename in files_a:
        # Get the transformed name for comparison
        if '.rf.' in filename:
            transformed_name = filename.split('.rf.')[0]
        else:
            transformed_name = filename
        
        # Check if this file exists in directory B (by transformed name)
        if transformed_name not in transformed_names_b:
            # This file is unique to directory A
            source_path = os.path.join(dir_a, filename)
            dest_path = os.path.join(dir_c, filename)
            
            # Copy the file
            if os.path.exists(dest_path):
                print(f"Skipped copying: '{filename}' - already exists in destination.")
                skipped_count += 1
            else:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                print(f"Copied unique file: '{filename}'")
        else:
            skipped_count += 1
    
    print(f"\nSummary: {copied_count} unique files copied to directory C, {skipped_count} files skipped.")

# Example usage
if __name__ == "__main__":
    dir_a = input("Enter directory A path (source): ")
    dir_b = input("Enter directory B path (comparison): ")
    dir_c = input("Enter directory C path (destination for unique files): ")
    copy_unique_images(dir_a, dir_b, dir_c)