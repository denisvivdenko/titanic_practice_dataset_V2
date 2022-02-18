from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import numpy as np
import pandas as pd

@FunctionTransformer
def parse_surnames(full_names: np.array) -> np.array:
    _parse_surnames = np.vectorize(lambda full_name: full_name.split(", ")[0])
    return _parse_surnames(full_names)

@FunctionTransformer
def parse_titles(full_name: np.array) -> np.array:
    _parse_titles = np.vectorize(lambda full_name: full_name.split(", ")[1].split(" ")[0])
    return _parse_titles(full_name)

if __name__ == "__main__":
    data = pd.read_csv("src\\datasets\\train.csv")["Name"]
    print(data.values[:2])
    print(parse_surnames.fit_transform(data.values[:2]))
    print(parse_titles.fit_transform(data.values[:2]))
"""
['Braund, Mr. Owen Harris' 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)']
['Braund' 'Cumings']
['Mr.' 'Mrs.']
"""