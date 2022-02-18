import numpy as np
from sklearn.preprocessing import FunctionTransformer

@FunctionTransformer
def extract_alphabetic_code_by_split(X: np.array, split_value: str = " ") -> np.array:
    """
        Splits string and removes numerical part from it. If string is empty gives it unknown.
    """
    extract_code = np.vectorize(lambda x: " ".join([code for code in x.split(split_value) if not code.isnumeric()]))
    extracted_codes = extract_code(X)
    return np.where(extracted_codes == "", "unknown", extracted_codes)

@FunctionTransformer
def extract_alphabetic_code(X: np.array) -> np.array:
    """
        Removes numbers part from string. If string is empty gives it unknown.
    """
    extract_alphabetic = np.vectorize(lambda x: ''.join(filter(str.isalpha, str(x))))
    extracted_codes = extract_alphabetic(X)
    return np.where(extracted_codes == "", "unknown", extracted_codes)

if __name__ == "__main__":
    print(extract_alphabetic_code_by_split.fit_transform(np.array(["B1C 12", "AB 3", "K 1", "12"])))
    print(extract_alphabetic_code.fit_transform(np.array(["B1C 12", "AB 3B", "K 121C"])))

"""
input: ["B1C 12", "AB 3", "K 1", "12"]
output: ['B1C' 'AB' 'K', 'unknown']

input: ["B1C 12", "AB 3B", "K 121C"]
output: ['BC' 'ABB' 'KC']
"""
