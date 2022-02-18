from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np

class OutliersIQRHandler(BaseEstimator, TransformerMixin):
    strategy = "median"

    def __init__(self, strategy: str = "median") -> None:
        """
        params:
            replace_value (str): 'median' or 'mean'
        """
        self.strategy = strategy
        self.replacing_strategies = {
            "median": np.median,
            "mean": np.mean
        }
        self.replacing_value = None
    
    def fit(self, X: np.array, y=None):
        self.replacing_value = self.replacing_strategies[self.strategy](X)
        return self
    
    def transform(self, X: np.array) -> np.array:
        """
        Detects ourliers using IQR
        IQR = Q3 - Q1
        lower_bound = Q3 + IQR * 1.5
        upper_bound = Q1 - IQR * 1.5

        params:
            X (np.array): processing data

        returns:
            (np.array) processed data with the specified startegy
        """

        Q1, Q3 =  np.quantile(X, 0.25), np.quantile(X, 0.75)
        IQR = Q3 - Q1        
        lower_bound, upper_bound = Q1 - IQR * 1.5, Q3 + IQR * 1.5

        return np.where((X > upper_bound) | (X < lower_bound), self.replacing_value, X)


if __name__ == "__main__":
    outliers_handler = OutliersIQRHandler()
    X = np.array(list(range(100)) + list(range(10**3, 10**3 + 10)))
    print(X, outliers_handler.fit_transform(X))