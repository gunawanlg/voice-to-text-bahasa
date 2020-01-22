import os

def generate_filenames(dir):
    return [dir + filename for filename in os.listdir(dir) if filename[-3:] in ["mp3", "ogg", "wav"]]