import numpy as np


def wer_and_cer(y_true, y_pred, html_filename=None, return_stats=False):
    """
    Calculate both word error rate and character error rate for given strings
    or list of strings.

    Parameters
    ----------
    y_true : list of str, str
        references, ground truth label string
    y_pred : list of str, str
        hypothesis, prediction from model
    html_filename : str, optional
        write output html or not. Only valid if given str input

   Returns
    -------
    out : dict
        dictionary contain wer and cer values with keys ['wer', 'cer']
    """
    if (isinstance(y_true, str) & isinstance(y_pred, str)):
        results_wer = __single_wer(y_true.split(' '), y_pred.split(' '),
                                   html_filename=html_filename,
                                   return_stats=return_stats)
        result_cer = __single_wer(list(y_true), list(y_pred),
                                  html_filename=html_filename,
                                  return_stats=return_stats)

        return {"wer": results_wer, "cer": result_cer}
    elif (isinstance(y_true, list) & isinstance(y_pred, list)):
        wer = []
        cer = []
        stats_wer = []
        stats_cer = []

        for r, h in zip(y_true, y_pred):
            if return_stats is True:
                wer_, stats_wer_ = __single_wer(r.split(' '), h.split(' '),
                                                html_filename=None,
                                                return_stats=return_stats)
                cer_, stats_cer_ = __single_wer(list(r), list(h),
                                                html_filename=None,
                                                return_stats=return_stats)
                stats_wer.append(stats_wer_)
                stats_cer.append(stats_cer_)
            else:
                wer_ = __single_wer(r.split(' '), h.split(' '),
                                    html_filename=None,
                                    return_stats=return_stats)
                cer_ = __single_wer(list(r), list(h),
                                    html_filename=None,
                                    return_stats=return_stats)
            wer.append(wer_)
            cer.append(cer_)

        if return_stats is True:
            return {
                "wer": (np.array(wer), np.array(stats_wer)),
                "cer": (np.array(cer), np.array(stats_cer))
            }
        else:
            return {
                "wer": np.array(wer),
                "cer": np.array(cer)
            }
    else:
        raise TypeError(f"Type mismatch. Got {type(y_true)} != {type(y_pred)}")


def cer(y_true, y_pred, html_filename=None, return_stats=False):
    """
    Given to list of strings how many character error rate (insertion, deletion
    , and substitution)

    Parameters
    ----------
    y_true : list of str, str
        references, ground truth label string
    y_pred : list of str, str
        hypothesis, prediction from model
    html_filename : str, optional
        write output html or not. Only valid if given str input

    Returns
    -------
    cer : float
        Word error rate number of (substitution + insertion + deletion) divided
        by number of words in references.

    Examples
    --------
    >>> y_true = ['aku dan dia', 'dia dan kamu', 'kamu dan aku']
    >>> y_pred = ['aky dan dia', 'diaa dan kamu', 'kamu aku']
    >>> cer(y_true, y_pred)
    16.919191919191917

    >>> y_true = 'aku, kamu, dan dia'
    >>> y_pred = 'ak, dia, dan'
    >>> cer(y_true, y_pred)
    75.0
    """
    if (isinstance(y_true, str) & isinstance(y_pred, str)):
        result = __single_wer(list(y_true), list(y_pred),
                              html_filename=html_filename,
                              return_stats=return_stats)
        return result
    elif (isinstance(y_true, list) & isinstance(y_pred, list)):
        cer = []
        stats = []

        for r, h in zip(y_true, y_pred):
            if return_stats is True:
                cer_, stats_ = __single_wer(list(r), list(h),
                                            html_filename=None,
                                            return_stats=return_stats)
                stats.append(stats_)
            else:
                cer_ = __single_wer(list(r), list(h),
                                    html_filename=None,
                                    return_stats=return_stats)
            cer.append(cer_)

        if return_stats is True:
            return np.array(cer), np.array(stats)
        else:
            return np.array(cer)
    else:
        raise TypeError(f"Type mismatch. Got {type(y_true)} != {type(y_pred)}")


def wer(y_true, y_pred, html_filename=None, return_stats=False):
    """
    Given two list of strings how many word error rate (insertion, deletion,
    and substitution)

    Parameters
    ----------
    y_true : list of str, str
        references, ground truth label string
    y_pred : list of str, str
        hypothesis, prediction from model
    html_filename : str, optional
        write output html or not. Only valid if given str input.
    return_stats : str, [default=False]
        if True, will return number of subsitution, insertion, and deletion in
        dictionary of stats

    Returns
    -------
    wer : float, numpy.ndarray
        Word error rate number of (substitution + insertion + deletion) divided
        by number of words in references.
    stats : numpy.ndarray, optional
        this only returned if return_stats is true. Keys are 'ok', 'sub', 'del'
        and 'ins'

    Examples
    --------
    >>> y_true = ['aku dan dia', 'dia dan kamu', 'kamu dan aku']
    >>> y_pred = ['aky dan dia', 'diaa dan kamu', 'kamu aku']
    >>> wer(y_true, y_pred).mean()
    33.33333333333333

    >>> y_true = 'aku, kamu, dan dia'
    >>> y_pred = 'ak, dia, dan'
    >>> wer(y_true, y_pred)
    75.0
    """
    if (isinstance(y_true, str) & isinstance(y_pred, str)):
        result = __single_wer(y_true.split(' '), y_pred.split(' '),
                              html_filename=html_filename,
                              return_stats=return_stats)
        return result
    elif (isinstance(y_true, list) & isinstance(y_pred, list)):
        wer = []
        stats = []

        for r, h in zip(y_true, y_pred):
            if return_stats is True:
                wer_, stats_ = __single_wer(r.split(' '), h.split(' '),
                                            html_filename=None,
                                            return_stats=return_stats)
                stats.append(stats_)
            else:
                wer_ = __single_wer(r.split(' '), h.split(' '),
                                    html_filename=None,
                                    return_stats=return_stats)
            wer.append(wer_)

        if return_stats is True:
            return np.array(wer), np.array(stats)
        else:
            return np.array(wer)
    else:
        raise TypeError(f"Type mismatch. Got {type(y_true)} != {type(y_pred)}")


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


def __single_wer(r, h, html_filename=None, return_stats=False):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
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

    # Backtrace
    x = len(r)
    y = len(h)

    # Keys in order, 'ok', 'sub', 'ins', 'del'
    if return_stats is True:
        numOk = 0
        numSub = 0
        numDel = 0
        numIns = 0

    html = '<html><body><head><meta charset="utf-8"></head>' \
        '<style>.g{background-color:#0080004d}.r{background-color:#ff00004d}.' \
        'y{background-color:#ffa50099}</style>'

    while True:
        if x == 0 or y == 0:
            break

        if r[x - 1] == h[y - 1]:
            if return_stats is True:
                numOk += 1
            x = x - 1
            y = y - 1
            html = '%s ' % h[y] + html
        elif d[x][y] == d[x - 1][y - 1] + 1:    # substitution
            if return_stats is True:
                numSub += 1
            x = x - 1
            y = y - 1
            html = '<span class="y">%s(%s)</span> ' % (h[y], r[x]) + html
        elif d[x][y] == d[x - 1][y] + 1:        # deletion
            if return_stats is True:
                numDel += 1
            x = x - 1
            html = '<span class="r">%s</span> ' % r[x] + html
        elif d[x][y] == d[x][y - 1] + 1:        # insertion
            if return_stats is True:
                numIns += 1
            y = y - 1
            html = '<span class="g">%s</span> ' % h[y] + html
        else:
            raise ValueError('\nWe got an error.')
            break

    html = html + '</body></html>'

    if html_filename is not None:
        with open(html_filename, 'w', encoding='utf8') as f:
            f.write(html)

    if return_stats is True:
        stats = []
        stats.extend([numOk, numSub, numDel, numIns])
        return result, stats
    else:
        return result
