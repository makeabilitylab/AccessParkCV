import os
import shutil

def copy_files_with_transformed_names(source_directory, destination_directory):
    # Check if the source directory exists
    if not os.path.isdir(source_directory):
        print(f"Error: The source directory '{source_directory}' does not exist.")
        return
    
    # Check if the destination directory exists, create it if not
    if not os.path.exists(destination_directory):
        try:
            os.makedirs(destination_directory)
            print(f"Created destination directory: '{destination_directory}'")
        except Exception as e:
            print(f"Error creating destination directory: {e}")
            return
    
    # Get all files in the source directory
    files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    
    # Track statistics
    copied_count = 0
    skipped_count = 0
    
    for filename in files:
        # Check if the filename contains '.rf.'
        if '.rf.' in filename:
            # Split the filename at '.rf.' and take the first part
            new_filename = filename.split('.rf.')[0]
            new_filename += '.jpg'
            # new_filename += '.txt'

            # Get the file extension from the original filename if needed
            if '.' in new_filename:
                # If the original part already has an extension, use it as is
                pass
            else:
                # If the original part doesn't have an extension, preserve the original extension
                original_ext = os.path.splitext(filename)[1]
                new_filename += original_ext
            
            # Create the full paths
            old_path = os.path.join(source_directory, filename)
            new_path = os.path.join(destination_directory, new_filename)
            
            # Copy the file (checking if destination already exists)
            if os.path.exists(new_path):
                print(f"Skipped: '{new_filename}' already exists in destination.")
                skipped_count += 1
            else:
                shutil.copy2(old_path, new_path)  # Use copy2 to preserve metadata
                copied_count += 1
                print(f"Copied: '{filename}' -> '{new_filename}'")
        else:
            print(f"Skipped: '{filename}' - doesn't contain '.rf.'")
            skipped_count += 1
    
    print(f"\nSummary: {copied_count} files copied with transformed names, {skipped_count} files skipped.")

# Example usage
if __name__ == "__main__":
    source_dir = input("Enter the source directory path: ")
    dest_dir = input("Enter the destination directory path: ")
    copy_files_with_transformed_names(source_dir, dest_dir)