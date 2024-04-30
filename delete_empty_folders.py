import os

def delete_empty_folders(directory):
    # Iterate over all items in the directory
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check if the current directory is empty
        if not dirs and not files:
            # Delete the empty directory
            os.rmdir(root)
            print(f"Deleted empty directory: {root}")

if __name__ == "__main__":
    directory = r"C:\Users\akosu\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Dissertation\Datasets\data\train"
    delete_empty_folders(directory)
