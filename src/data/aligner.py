from sklearn.base import TransformerMixin

class Aligner(TransformerMixin):
    """Align transcription to mp3"""
    def __init__(self, type='aeneas'):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        return None