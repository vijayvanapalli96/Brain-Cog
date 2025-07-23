import os
import glob
import shutil

# --- USER CONFIGURATION ---
# 1. Set the path to the folder containing your class sub-directories with the raw .aedat4 files.
source_dir = "Recordings"

# 2. Set the path where the organized, training-ready dataset will be created.
#    This is the path you will use in your training script.
organized_dir = "data/my_grouped_recordings"

# 3. Define the logic for grouping the signs into broader categories.
#    The script will check if any keyword is present in the original folder name.
GROUP_MAPPING = {
    # New Class Name: [Keywords to identify old class names]
    "Mandatory": [
        "wear", "use_", "keep", "close_door"
    ],
    "Prohibition": [
        "no_", "not_"
    ],
    "Warning": [
        "warning", "hazard", "zone", "slippery", "unstable", "unprotected",
        "toxic", "loud_noise", "strong_winds", "radioactive", "quicksand",
        "oxidising", "overhead", "optical_radiation", "non_ionising", "magnetic",
        "laser", "landslide", "hot_steam", "high_surf", "high_sound", "guard_dog",
        "forklift", "flood", "flammable", "falling_ice", "explosive", "electricity",
        "drop", "crushing", "corrosive", "bull", "biological", "battery_charging",
        "automatic_start", "asphyxiating", "arc_flash", "shark"
    ],
    "Safe_Condition": [
        "stretcher", "defibrillator", "exit"
    ],
    "Fire_Safety": [
        "fire"
    ],
    # A special case for first aid, as it can sometimes be confused with general warnings
    "First_Aid": [
        "first_aid"
    ]
}
# --------------------------

def get_new_class_name(original_class_name):
    """Finds the new group for a class based on the GROUP_MAPPING."""
    for group, keywords in GROUP_MAPPING.items():
        for keyword in keywords:
            if keyword in original_class_name.lower():
                return group
    return None

def organize_event_files(source, destination):
    """
    Crawls the source directory, finds all .aedat4 files, and copies them
    into a new, clean directory structure organized by the new groups.
    """
    print(f"Starting to organize and group files from '{source}' into '{destination}'...")
    
    # Ensure the destination directory exists and is empty
    if os.path.exists(destination):
        print(f"Destination directory '{destination}' already exists. Clearing it out.")
        shutil.rmtree(destination)
    os.makedirs(destination)

    # Find all .aedat4 files in the source directory
    search_pattern = os.path.join(source, "**", "*.aedat4")
    filepaths = glob.glob(search_pattern, recursive=True)

    if not filepaths:
        print(f"No .aedat4 files found in '{source}'. Nothing to do.")
        return

    print(f"Found {len(filepaths)} event files to organize.")
    organized_count = 0
    uncategorized_count = 0
    
    for filepath in filepaths:
        try:
            # The class name is the parent directory of the file's parent directory
            parent_dir = os.path.dirname(filepath)
            original_class_name = os.path.basename(os.path.dirname(parent_dir))

            # Get the new, grouped class name
            new_class_name = get_new_class_name(original_class_name)

            if not new_class_name:
                print(f"Warning: Could not categorize '{original_class_name}'. Skipping file '{filepath}'.")
                uncategorized_count += 1
                continue

            # Create the new class directory in the destination
            class_dest_dir = os.path.join(destination, new_class_name)
            if not os.path.exists(class_dest_dir):
                os.makedirs(class_dest_dir)

            # Copy the file
            shutil.copy(filepath, class_dest_dir)
            organized_count += 1
        
        except Exception as e:
            print(f"Could not process file {filepath}. Error: {e}")

    print("\n--- Organization Summary ---")
    print(f"Successfully organized {organized_count}/{len(filepaths)} files into {len(GROUP_MAPPING)} groups.")
    if uncategorized_count > 0:
        print(f"Skipped {uncategorized_count} files that could not be categorized.")
    print(f"Your training-ready dataset is now located at: '{destination}'")

if __name__ == '__main__':
    organize_event_files(source_dir, organized_dir) 