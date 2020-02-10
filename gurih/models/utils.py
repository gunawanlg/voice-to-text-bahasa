import numpy
import tensorflow.keras.backend as K


def wer(y_true, y_pred, write_html=False):
    """
    Given two list of strings how many word error rate(insert, delete or 
    substitution).

    Parameters
    ----------
    y_true : list of str, str
        references, ground truth label string
    y_pred : list of str, str
        hypothesis, prediction from model
    write_html : bool, optional
        write output html or not. Only valid if given str input.
    
    Returns
    -------
    wer : float
        Word error rate number of (substitution + insertion + deletion) divided 
        by number of words in references.

    Examples
    --------
    >>> y_true = ['aku dan dia', 'dia dan kamu', 'kamu dan aku']
    >>> y_pred = ['aky dan dia', 'diaa dan kamu', 'kamu aku']
    >>> wer(y_true, y_pred)
    33.33333333333333
    
    >>> y_true = 'aku, kamu, dan dia'
    >>> y_pred = 'ak, dia, dan'
    >>> wer(y_true, y_pred)
    50.0
    """
    if (type(y_true) == list) and (type(y_pred) == list):
        result = 0
        for r, h in zip(y_true, y_pred):
            result += __single_wer(r.split(' '), h.split(' '), write_html=False)
        result /= len(y_pred)
    else:
        result = __single_wer(y_true, y_pred, write_html=write_html)

    return result

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

    Returns
    -------
    y_preds : list of str
        string prediction from the model
    """
    ctc_matrix_decoded = K.ctc_decode(ctc_matrix,
                          [ctc_matrix.shape[1]]*ctc_matrix.shape[0],
                          **kwargs)
    
    ctc_decoded, _ = ctc_matrix_decoded
    ctc_decoded = ctc_decoded[0].numpy()

    y_preds = []
    for y_pred in ctc_decoded:
        output_text = ""
        for idx in y_pred:
            if idx in idx_to_char_map:
                output_text += idx_to_char_map[idx]    
        output_text = output_text.strip()
        y_preds.append(output_text)

    return y_preds

class CharMap:
    """
    Defines character map used in ASR model
    """
    CHAR_TO_IDX_MAP = {
        " ": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "j": 10,
        "k": 11,
        "l": 12,
        "m": 13,
        "n": 14,
        "o": 15,
        "p": 16,
        "q": 17,
        "r": 18,
        "s": 19,
        "t": 20,
        "u": 21,
        "v": 22,
        "w": 23,
        "x": 24,
        "y": 25,
        "z": 26,
        ".": 27,
        ",": 28,
        "%": 29,
    }

    IDX_TO_CHAR_MAP = {v: k for k, v in CHAR_TO_IDX_MAP.items()}

    def __len__(self):
        return len(self.CHAR_TO_IDX_MAP) - 1


def __single_wer(r, h, write_html):
        d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        result = float(d[len(r)][len(h)]) / len(r) * 100

        if write_html:
            x = len(r)
            y = len(h)

            html = '<html><body><head><meta charset="utf-8"></head>' \
                '<style>.g{background-color:#0080004d}.r{background-color:#ff00004d}.y{background-color:#ffa50099}</style>'

            while True:
                if x == 0 or y == 0:
                    break

                if r[x - 1] == h[y - 1]:
                    x = x - 1
                    y = y - 1
                    html = '%s ' % h[y] + html
                elif d[x][y] == d[x - 1][y - 1] + 1:    # substitution
                    x = x - 1
                    y = y - 1
                    html = '<span class="y">%s(%s)</span> ' % (h[y], r[x]) + html
                elif d[x][y] == d[x - 1][y] + 1:        # deletion
                    x = x - 1
                    html = '<span class="r">%s</span> ' % r[x] + html
                elif d[x][y] == d[x][y - 1] + 1:        # insertion
                    y = y - 1
                    html = '<span class="g">%s</span> ' % h[y] + html
                else:
                    raise ValueError('\nWe got an error.')
                    break

            html = html + '</body></html>'

            with open('diff.html', 'w', encoding='utf8') as f:
                f.write(html)

        return result