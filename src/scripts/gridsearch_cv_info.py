from sklearn.model_selection import GridSearchCV

def print_gridsearch_cv_results(classifier: GridSearchCV) -> None:
    info = f"""
    params: {classifier.param_grid}
    score: {classifier.best_score_}
    best params: {classifier.best_params_}
    """
    print(info)