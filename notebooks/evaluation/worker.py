import os
import re
import numpy as np
from string import ascii_lowercase
from collections import defaultdict, Counter
from multiprocessing import Pool

import sentencepiece as spm


def prefix_beam_search(ctc, lm=None, k=25, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.

    Parameters
    ----------
    ctc : np.ndarray
        The CTC output. Should be a 2D array (timesteps x alphabet_size)
    lm : function, [default=None]
        Should take as input a string and output a probability
    k : int, [default=25]
        The beam width. Will keep the 'k' most likely candidates at each timestep
    alpha : float, [default=0.30]
        The language model weight. Should usually be between 0 and 1.
    beta : float, [default=0.5]
        The language model compensation term. The higher the 'alpha', the higher the 'beta'.
    prune : float, [default=0.001]
        Only extend prefixes with chars with an emission probability higher than 'prune'.

    Returns
    -------
    string: The decoded CTC output.
    """

    lm = (lambda l: 1) if lm is None else lm  # if no LM is provided, just set to returning 1
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    alphabet = list(ascii_lowercase) + [' ', '>', '%']
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc))  # just add an imaginative zero'th step
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:

            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == '%':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):
                        lm_prob = lm(l_plus.strip(' >')) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    return A_prev[0].strip('>')


class SPLM:
    def __init__(self, spp, log=True, regularize=True):
        self.spp = spp
        self.log = log
        self.regularize = regularize

    def __call__(self, sentence, **kwargs):
        return self.sp_score(sentence, **kwargs)

    def sp_score(self, sentence, l=-1, alpha=0.2):
        """Score sentence using unigram model of sentencepiece"""
        if self.regularize:
            encoded = self.spp.sample_encode_as_ids(sentence, l, alpha)
        else:
            encoded = self.spp.encode_as_ids(sentence)

        score = 0
        for idx in encoded:
            # return emission log probabilities, so just add them by chain-rule
            score += self.spp.GetScore(idx)

        if not self.log:
            score = 10 ** score

        return score


# Load unigram model
sp = spm.SentencePieceProcessor()
sp.load('m_botchan.model')
sp = SPLM(sp, log=False)


def worker(b):
    res = prefix_beam_search(b,
                             lm=sp,
                             k=100,
                             alpha=0.30,
                             beta=5,
                             prune=0.001)
    return res


def ctc_beam_search_sp_mp(examples):
    # create the threadpool
    with Pool(os.cpu_count() - 1) as p:
        # schedule one map/worker for each row in the original data
        q = p.map(worker, [b for b in examples])

    return q
