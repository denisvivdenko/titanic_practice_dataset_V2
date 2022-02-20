import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.transformers.outliers_handler import OutliersIQRHandler
from src.transformers.name_parser import parse_surnames
from src.transformers.name_parser import parse_titles
from src.transformers.alphabetic_code_extractor import extract_alphabetic_code
from src.transformers.alphabetic_code_extractor import extract_alphabetic_code_by_split

data = pd.read_csv("src\\datasets\\train.csv")

surname_feature_pipeline = Pipeline([
    ("surname_parser", parse_surnames),
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

name_title_feature_pipeline = Pipeline([
    ("name_title_parser", parse_titles),
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

age_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("outliers_handler", OutliersIQRHandler(strategy="median"))
])

ticket_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("alphabetic_code_extractor", extract_alphabetic_code_by_split),
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

cabin_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("alphabetic_code_extractor", extract_alphabetic_code),
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
    ("name_title_feature", name_title_feature_pipeline, ["Name"]),
    ("surname_feature", surname_feature_pipeline, ["Name"]),
    ("age_feature", age_feature_pipeline, ["Age"]),
    ("ticket_feature", ticket_feature_pipeline, ["Ticket"]),
    ("cabin_feature", cabin_feature_pipeline, ["Cabin"]),
    ("fare_feature", fare_feature_pipeline, ["Fare"]),
    ("categorical_features", categorical_features_pipeline, ["Sex", "Embarked"]),
    ("numerical_discrete_features", numerical_discrete_features_pipeline, ["SibSp", "Pclass", "Parch"])
], sparse_threshold=0)

if __name__ == "__main__":
    print(pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId).head())