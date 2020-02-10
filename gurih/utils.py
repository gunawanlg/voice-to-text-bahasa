import os

def batch(l, b=1, n=None):
    """
    Create batch from iterable.

    Parameters
    ----------
    l : list
        list to create batches from
    b : int, optional, [default = 1]
        batch size
    n : int, optional, [default = None]
        if None: len(batches[-1]) < b if len(iterable) % b != 0
        else: len(batches[-1]) == b if len(iterable) % b == 0
        this will override b param
    
    Returns
    -------
    batches : iterable
        generator of batch

    Example
    -------
    If n is None, or not inputted
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

    if n is not None:
    >>> l = list(range(10))
    >>> batches = batch(l, n=3)
    >>> batches
    <generator object batch at 0x006C0F30>
    >>> for b in batches:
    ...    print(b)
    ...
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8, 9]
    """    
    if n is not None:
        assert n > 0
        b = int(len(l) / n)
        for ndx in range(0, n-1):
            yield l[ndx*b:ndx*b+b]
        yield l[n*b-b:]
    else:
        assert b > 0
        m = len(l)
        for ndx in range(0, m, b):
            yield l[ndx:min(ndx + b, m)]

def validate_nonavailability(X, file_type):
    """
    Validate if files with the input file_type exist or not

    Parameters
    ----------
    X : list
        list of data to be validated

    file_type : str
        type of the files to be compared with X


    Return
    ------
    availability_dict : dict
        dictionary of file availability

    Example
    -------
    Let's say we have a file named "a.npz" in the check_dir
    >>> X = ["a.mp3", "b.mp3"]
    >>> file_type = "npz"
    >>> file_nonavailability_dict = validate_nonavailability(X, "npz")
    >>> file_nonavailability_dict
    {"a.npz": False, "b.npz": True}
    """

    # Get a list of existing file files
    finished_files = glob.glob(f"*.{file_type}")

    # Get a list of all possible generated files
    possible_file_path_absolutes =  [f"{x.split('.')[0]}.{file_type}" for x in X]

    # Create a dictionary for faster querying
    file_nonavailability_dict = {possible_file_path_absolute: True for possible_file_path_absolute in possible_file_path_absolutes}

    for finished_file in finished_files:
        file_nonavailability_dict[finished_file] = False

    return file_nonavailability_dict

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