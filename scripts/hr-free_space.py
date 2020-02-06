import glob
import os

from gurih.utils import validate_nonavailability

if __name__ == "__main__":
    mp3s = glob.glob("*.mp3")

    file_nonavailability_dict = validate_nonavailability(mp3s, "npz")

    available_files = [f"{k.replace('npz', 'mp3')}" for k, v in file_nonavailability_dict.items() if not v]

    confirmation = str(input(f"You're going to delete {len(available_files)} files, this change is permanent, are you sure (Y/n)? "))

    if confirmation == "Y":
        for available_file in available_files:
            os.remove(available_file)
        
        print(f"Removed {len(available_files)} files")
    else:
        print(f"Mission aborted")

    