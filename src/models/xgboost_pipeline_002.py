import pandas as pd
import xgboost as xgb 

from src.pipelines.pipeline_002 import pipeline
from src.models.model_builder import ModelBuilder

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("src\\datasets\\train.csv")
test_data = pd.read_csv("src\\datasets\\test.csv")

parameters = { 'max_depth': [3, 6, 10, 15],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [10, 30, 50],
           'colsample_bytree': [0.3, 0.7]}

xgb_classifier = xgb.XGBClassifier(seed = 20)

model = ModelBuilder(pipeline=pipeline, estimator=xgb_classifier, tuning_parameters=parameters)
model.tune_parameters(data=data, metric="roc_auc", verbose=10)
model.predict(test_data).to_csv(f"src\\predictions\\{model.get_model_score():.2f}_xgboost_pipeline_02_01.csv")

print(model.get_model_score())
print(model.get_model_params())

