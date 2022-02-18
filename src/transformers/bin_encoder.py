from typing import List

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np

class BinEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, bins: List[int]) -> None:
        """
        params:
            bins (List[int]): bins by right border
        """
        self.bins = bins
    
    def fit(self, X: np.array, y=None):
        return self
    
    def transform(self, X: np.array) -> np.array:
        """
        Divide array on bins

        params:
            X (np.array): processing data

        returns:
            (np.array) one dimensional discretized data
        """
        return np.digitize(X, bins=self.bins, right=True)


if __name__ == "__main__":
    bin_encoder = BinEncoder(bins=[2, 5, 7, 10])
    X = np.arange(0, 15, dtype=int)
    print(X, bin_encoder.fit_transform(X))

"""
input: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14] 
output: [0 0 0 1 1 1 2 2 3 3 3 4 4 4 4]
"""