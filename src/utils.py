import os

def generate_filenames(dir):
    """
    Generate filenames of audio files in the
    directory. The generated filenames are the full
    path of the filenames.

    Parameters
    ----------
    dir : string
        The directory where the audio files located.

    Returns
    -------
    [list] : 1-d array
        Array of the filenames and the full path to
        the full name so it'd be easy to load em.

    Example
    -------
    >>> dir = "/data/raw"
    >>> generate_filenames(dir)
    >>> ["OSR_us_000_0010_8k.wav", "OSR_us_000_001_8k.wav"]
    """
    
    return [dir + filename for filename in os.listdir(dir) if filename[-3:] in ["mp3", "ogg", "wav"]]