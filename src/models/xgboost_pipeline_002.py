import pandas as pd
import xgboost as xgb 
from sklearn.model_selection import GridSearchCV

from src.pipelines.pipeline_002 import pipeline
from src.scripts.gridsearch_cv_info import print_gridsearch_cv_results

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("src\\datasets\\train.csv")
X_train, y_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId), data.Survived

params = { 'max_depth': [3, 6, 10, 15],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [10, 30, 50],
           'colsample_bytree': [0.3, 0.7]}

xgbr = xgb.XGBClassifier(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring="roc_auc", 
                   verbose=10)

clf.fit(X_train, y_train)
print_gridsearch_cv_results(clf)

test_data = pd.read_csv("src\\datasets\\test.csv")
model = clf.best_estimator_
prediction = pd.DataFrame(model.predict(pipeline.transform(test_data)), columns=["Survived"], index=test_data.PassengerId)
prediction.to_csv("src\\predictions\\xgboost_pipeline_002.csv")