import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.pipelines.pipeline_004 import pipeline
from src.models.model_builder import ModelBuilder


data = pd.read_csv("src\\datasets\\train.csv")
test_data = pd.read_csv("src\\datasets\\test.csv")

parameters = {'base_estimator__max_depth': range(2, 11, 2),
              'base_estimator__min_samples_leaf':[5, 10, 15],
              'n_estimators':[10, 50, 100],
              'learning_rate':[0.01, 0.1]}

adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

model = ModelBuilder(pipeline=pipeline, estimator=adaboost_classifier, tuning_parameters=parameters)
model.tune_parameters(data=data, metric="roc_auc", verbose=10)
model.predict(test_data).to_csv(f"src\\predictions\\{model.get_model_score():.2f}_adaboost_pipeline_02_01.csv")

print(model.get_model_params())


