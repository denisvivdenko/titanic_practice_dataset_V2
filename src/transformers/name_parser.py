from typing import Tuple
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

@FunctionTransformer
def parse_names(full_names: np.array) -> np.array:
    @np.vectorize
    def _parse_names(full_name: str) -> Tuple[str, str]:
        """
            Extracts surname and title.

        Example:
            'Braund, Mr. Owen Harris' -> ["Braund", "Mr"]
        """
        surname, name = full_name.split(", ")
        title = name.split(". ")[0]
        return (surname, title)
    return _parse_names(full_names)


if __name__ == "__main__":
    data = pd.read_csv("src\\datasets\\train.csv")["Name"]
    parse_names.fit_transform(data.values[:2])
    
    # (array(['Braund', 'Cumings'], dtype='<U7'), array(['Mr', 'Mrs'], dtype='<U3'))