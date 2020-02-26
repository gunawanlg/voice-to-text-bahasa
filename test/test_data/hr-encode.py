import glob
import os
import json
from datetime import datetime

if __name__ == "__main__":
    # Get all of the npz files
    npzs = glob.glob("*.npz")

    # Get all of the available texts to compare it with the ones required    
    available_txts = set(glob.glob("*.txt"))
    required_txts = set([f"{npz.replace('npz', 'txt')}" for npz in npzs])

    assert len(available_txts.intersection(required_txts)) == len(required_txts)

    encoded_dict = {}
    date = datetime.today().strftime("%Y%m%d")

    for i, npz in enumerate(npzs):
        encoded_dict[i] = npz.replace(".npz", "")
        os.rename(npz, f"{i}.npz")
        os.rename(f"{npz.replace('npz', 'txt')}", f"{i}.txt")

    with open(f"{date}_audio_encoding.json", "w") as f:
        json.dump(encoded_dict, f)