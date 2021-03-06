import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.transformers.outliers_handler import OutliersIQRHandler
from src.transformers.bin_encoder import BinEncoder

data = pd.read_csv("src\\datasets\\train.csv")

age_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("outliers_handler", OutliersIQRHandler(strategy="median")),
    ("discretizer", BinEncoder(bins=[5, 12, 18, 25, 45, 60, 80])),
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

fare_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("outliers_handler", OutliersIQRHandler(strategy="median"))
])

categorical_features_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

numerical_discrete_features_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median"))
])

pipeline = ColumnTransformer([
    ("age_feature", age_feature_pipeline, ["Age"]),
    ("fare_feature", fare_feature_pipeline, ["Fare"]),
    ("categorical_features", categorical_features_pipeline, ["Sex", "Embarked"]),
    ("numerical_discrete_features", numerical_discrete_features_pipeline, ["SibSp", "Pclass", "Parch"])
], sparse_threshold=0)

if __name__ == "__main__":
    print(pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId).head())