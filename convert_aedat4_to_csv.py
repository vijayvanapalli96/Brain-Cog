import tonic
import numpy as np
import os
import glob

# --- USER CONFIGURATION ---
# Set the path to the folder containing your organized class sub-directories.
# This script will find .aedat4 files here and create .csv files in the same location.
organized_data_path = "data/my_grouped_recordings"
# --------------------------

def convert_aedat4_to_csv(filepath):
    """Reads an .aedat4 file and saves its event data as a .csv file."""
    try:
        # Define the new filepath with a .csv extension
        csv_filepath = os.path.splitext(filepath)[0] + ".csv"
        
        # Avoid re-converting files that already exist
        if os.path.exists(csv_filepath):
            print(f"Skipping {filepath}, CSV already exists.")
            return True, False # Success, but no new file was created

        # Read the aedat4 file
        events = tonic.io.read_aedat4(filepath)
        
        # Create a structured numpy array (t, x, y, p)
        events_np = np.stack([events['t'], events['x'], events['y'], events['p']], axis=1)
        
        # Save to CSV with a header
        np.savetxt(
            csv_filepath, 
            events_np, 
            delimiter=',', 
            header='t,x,y,p', 
            fmt='%d', 
            comments=''
        )
        print(f"Successfully converted {filepath} -> {csv_filepath}")
        return True, True # Success, and a new file was created
    except Exception as e:
        print(f"Could not convert {filepath}. Error: {e}")
        return False, False

if __name__ == '__main__':
    print(f"Starting conversion process in '{organized_data_path}'...")
    
    # Find all .aedat4 files in the subdirectories
    filepaths = glob.glob(os.path.join(organized_data_path, "**/*.aedat4"), recursive=True)
    
    if not filepaths:
        print(f"No .aedat4 files found in '{organized_data_path}'. Make sure you have run the organization script first.")
    else:
        print(f"Found {len(filepaths)} total .aedat4 files to process.")
        converted_count = 0
        failed_count = 0
        
        for path in filepaths:
            success, was_converted = convert_aedat4_to_csv(path)
            if success and was_converted:
                converted_count += 1
            elif not success:
                failed_count += 1

        print("\n--- Conversion Summary ---")
        print(f"Newly converted files: {converted_count}")
        if failed_count > 0:
             print(f"Failed conversions:    {failed_count}")
        print("Conversion process complete.") 