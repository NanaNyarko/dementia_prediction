import os
import shutil

def move_folders(source_dir):
    # Get a list of subdirectories in the source directory
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    # Iterate over each subdirectory in the source directory
    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        # Get a list of subdirectories in the current subdirectory
        subsubdirs = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
        # Check if any of the subdirectories match the patterns like 'no_dementia', 'mild_dementia', etc.
        for subsubdir in subsubdirs:
            if subsubdir.endswith("_dementia"):
                source_path = os.path.join(subdir_path, subsubdir)
                dest_path = os.path.join(source_dir, subsubdir)
                # Move the contents of the subdirectory one level up into the source directory
                for item in os.listdir(source_path):
                    item_path = os.path.join(source_path, item)
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                    # Check if the item already exists in the destination directory
                    dest_item_path = os.path.join(dest_path, item)
                    if os.path.exists(dest_item_path):
                        # If the item already exists, rename it before moving
                        dest_item_path += "_moved"
                    # Move the item to the destination directory
                    shutil.move(item_path, dest_item_path)
                    print(f"Moved '{item}' to '{dest_item_path}'")
                # Remove the now empty subdirectory
                os.rmdir(source_path)
                print(f"Removed empty directory '{source_path}'")

if __name__ == "__main__":
    source_dir = r"C:\Users\akosu\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Dissertation\Datasets\data\train"
    move_folders(source_dir)
