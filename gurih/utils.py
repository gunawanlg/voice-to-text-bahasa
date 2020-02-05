import os

def batch(iterable, b=1):
    """
    Create batch from iterable.

    Parameters
    ----------
    iterable : iterable
        iterable to create batches from
    b : int, optional, [default = 1]
        batch size
    
    Returns
    -------
    batches : iterable
        generator of batch

    Example
    -------
    >>> l = list(range(10))
    >>> batches = batch(l, 3)
    >>> batches
    <generator object batch at 0x005A0370>
    >>> for b in batches:
    ...    print(b)
    ...
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]
    """
    l = len(iterable)
    for ndx in range(0, l, b):
        yield iterable[ndx:min(ndx + b, l)]

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