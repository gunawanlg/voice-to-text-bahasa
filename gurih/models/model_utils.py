import tensorflow.keras.backend as K

def ctc_decode(ctc_matrix, idx_to_char_map, **kwargs):
    """
    Decode ctc matrix output into human readable text using
    tensorflow.keras.backend.ctc_decode

    Parameters
    ----------
    ctc_matrix : np.array(shape=[m, max_sequence_length, vocab_len])
        output from ASR model where m denotes number of samples
    idx_to_char_map : dict
        map index output to character, including blank token
    mode : str, optional
        default greedy by looking for max probabilty in each timestep
        other options:
            - keras
    """
    output = K.ctc_decode(ctc_matrix,
                          ctc_matrix[0].shape[0],
                          **kwargs)
    
    for i, example in enumerate(output):
        output_text = ""
        for timestep in example.numpy():
            output_text += idx_to_char_map[timestep]
            output[i] = output_text

    return output