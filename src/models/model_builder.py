import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class ModelBuilder:
    def __init__(self, pipeline: Pipeline, estimator, tuning_parameters: dict) -> None:
        """
            Wrapper that transforms data and tune model parameters
        """
        self.pipeline = pipeline
        self.estimator = estimator
        self.tuning_parameters = tuning_parameters
        self.classifier: GridSearchCV = None

    def tune_parameters(self, data: pd.DataFrame, metric: str, verbose: int = 10) -> None:
        X, y = pd.DataFrame(self.pipeline.fit_transform(data), index=data.PassengerId), data.Survived
        classifier = GridSearchCV(
            estimator=self.estimator, 
            param_grid=self.tuning_parameters,
            scoring=metric, 
            verbose=verbose
        )
        classifier.fit(X, y)
        self.classifier = classifier

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        model = self.classifier.best_estimator_
        predictions = model.predict(self.pipeline.transform(X))
        return pd.DataFrame(predictions, columns=["Survived"], index=X.PassengerId)

    def get_model_params(self) -> dict:
        return self.classifier.best_params_

    def get_model_score(self) -> float:
        return self.classifier.best_score_
 