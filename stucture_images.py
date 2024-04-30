import os
import shutil

def move_files_up_and_suffix(source_dir):
    # Get list of directories in the source directory
    dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Move files from each 'RAW' directory to its parent directory
    for dir_name in dirs:
        raw_dir = os.path.join(source_dir, dir_name, 'RAW')
        
        # Check if the 'RAW' directory exists
        if os.path.exists(raw_dir):
            files = os.listdir(raw_dir)
            
            # Move each file from 'RAW' directory to its parent directory with suffix
            for file in files:
                source_path = os.path.join(raw_dir, file)
                dest_filename = f"{dir_name}_{file}"
                dest_path = os.path.join(source_dir, dir_name, dest_filename)
                shutil.move(source_path, dest_path)
                print(f"Moved '{file}' from '{raw_dir}' to '{os.path.join(source_dir, dir_name)}' with suffix '{dir_name}_'")
            
            # Remove the now empty 'RAW' directory
            os.rmdir(raw_dir)
            print(f"Removed empty directory '{raw_dir}'")

if __name__ == "__main__":
    source_dir = r"C:\Users\akosu\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Dissertation\Datasets\data\raw_train"
    
    move_files_up_and_suffix(source_dir)
