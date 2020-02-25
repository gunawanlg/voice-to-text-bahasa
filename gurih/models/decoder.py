import inspect

import numpy as np
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class CTCDecoder(BaseEstimator):
    """
    Generate transcript for MFCC input using ASRModel.

    Parameters
    ----------
    idx_to_char_map : dict
        dictionary mapping index to character
    low_memory : bool, [default=False]
        if True, will only store wer_ attribute.

    Returns
    -------
    wer_ : float
        Word error rate of given sequence input.
    """
    def __init__(self, idx_to_char_map):
        self.idx_to_char_map = idx_to_char_map

    def fit(self, X, y=None):
        """DO nothing"""
        self.X_ = X
        return self

    def predict(self, X):
        """
        Predict input mfcc sequence.

        Parameters
        ----------
        X : np.array[shape=(m, ctc_output_length, features)]
            m examples of ctc_matrix

        Returns
        -------
        y_pred : list of str
            output transcription/
        """
        check_is_fitted(self, 'X_')
        if inspect.isgenerator(X):
            y_pred = []
            for x in X:
                x_tmp = np.expand_dims(x, axis=0)
                y = self._ctc_decode(x_tmp,
                                     self.idx_to_char_map,
                                     greedy=False)
                y_pred.extend(y)
        else:
            y_pred = self._ctc_decode(X,
                                      self.idx_to_char_map,
                                      greedy=False)

        return y_pred

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def _ctc_decode(self, ctc_matrix, idx_to_char_map, **kwargs):
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
