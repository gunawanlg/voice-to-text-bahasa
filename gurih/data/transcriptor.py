import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from gurih.models.model_utils import ctc_decode, wer


class ASRTranscriptor(BaseEstimator):
    """
    Generate transcript for MFCC input using ASRModel.

    Parameters
    ----------
    ASRModel : _BaseModel
        Pretrained model with predict() method.
    idx_to_char_map : dict
        dictionary mapping index to character
    low_memory : bool, [default=False]
        if True, will only store wer_ attribute.

    Attributes
    ----------
    ctc_matrix_ : numpy.ndarray
        ctc matrix of given sequence input
    wer_ : float
        Word error rate of given sequence input.

    Examples
    --------
    >>> from gurih.models.model import BaselineASRmodel
    >>> from gurih.models.utils import CharMap
    >>> X = np.random.rand(1, 1000, 39) # 39 MFCC features,
                                        # 1000 sequence length
                                        # 1 input example
    >>> idx_to_char_map = CharMap.IDX_TO_CHAR_MAP
    >>> transcriptor = ASRTranscriptor(BaselineASRModel(), idx_to_char_map)
    >>> transcriptor.predict(X)
    """
    def __init__(self, ASRModel, idx_to_char_map, low_memory=False):
        self.model = ASRModel
        self.idx_to_char_map = idx_to_char_map
        self.low_memory = low_memory

    def fit(self, X, y=None):
        """
        If y is provided, generates evaluation metrics.

        Parameters
        ----------
        X : np.array[shape=(m, max_seq_length, features)]
            m examples mfcc input of audio data
        y : list of str
            transcription of given input sequences X
        """
        if self.low_memory is True:
            return self.__fit_generator(X, y)
        else:
            ctc_matrix = self.model.predict(X)
            self.ctc_matrix_ = ctc_matrix

            if y is not None:
                y_pred = ctc_decode(ctc_matrix,
                                    self.idx_to_char_map,
                                    greedy=False)
                self.wer_ = wer(y, y_pred, write_html=False)

        return self

    def predict(self, X):
        """
        Predict input mfcc sequence.

        Parameters
        ----------
        X : np.array[shape=(m, max_seq_length, features)]
            m examples mfcc input of audio data

        Returns
        -------
        y_pred : list of str
            output transcription/
        """
        check_is_fitted(self, 'ctc_matrix_')
        if self.low_memory is False:
            y_pred = ctc_decode(self.ctc_matrix_,
                                self.idx_to_char_map,
                                greedy=False)
        else:
            y_pred = []
            for ctc in self.ctc_matrix_:
                y = ctc_decode(ctc,
                               self.idx_to_char_map,
                               greedy=False)
                y_pred.append(y)

        return y_pred

    def __fit_generator(self, X, y):
        ctc_matrix = self.__ctc_matrix_gen(X)
        self.ctc_matrix_ = self.__ctc_matrix_gen(X)  # generator function

        if y is not None:
            y_pred = []
            for ctc in ctc_matrix:
                y_pred_cache = ctc_decode(ctc,
                                          self.idx_to_char_map,
                                          greedy=False)
                y = y_pred_cache[0]
                y_pred.append(y)
            self.wer_ = wer(y, y_pred, write_html=False)

    def __ctc_matrix_gen(self, X):
        for x in X:
            ctc = self.model.predict(np.expand_dims(x, axis=0))
            yield ctc
