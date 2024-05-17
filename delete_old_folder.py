import os
import shutil

def delete_old_folder(root_dir):
    # Get list of directories in the root directory
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Iterate over each directory
    for dir_name in dirs:
        current_dir = os.path.join(root_dir, dir_name)
        old_folder = os.path.join(current_dir, 'OLD')
        
        # Check if the 'OLD' folder exists
        if os.path.exists(old_folder):
            # Delete the 'OLD' folder and its contents
            try:
                shutil.rmtree(old_folder)
                print(f"Deleted 'OLD' folder and its contents in '{current_dir}'")
            except OSError as e:
                print(f"Error deleting 'OLD' folder in '{current_dir}': {e}")

if __name__ == "__main__":
    root_dir = r"C:\Users\akosu\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Dissertation\Datasets\data\raw_train"
    
    delete_old_folder(root_dir)
