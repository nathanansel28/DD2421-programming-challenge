from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import SequentialFeatureSelector

import numpy as np
import pandas as pd
import xgboost as xgb

param_dist = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 6, 9, 12],
    "n_estimators": [100, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "min_child_weight": [1, 3, 5]
}

def hyperparam_tuning(X_train, y_train, k=5, n_iter=20) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y_train)), 
                                  eval_metric="mlogloss", random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, n_iter=n_iter,
        scoring='accuracy', cv=cv, verbose=2, n_jobs=-1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("Best parameters found:", random_search.best_params_)
    print("Best accuracy score:", random_search.best_score_)
    
    return random_search


def select_features_xgb(X_train, y_train, n_features: int = 10, cv_folds: int = 5):
    """
    Performs feature selection using XGBoost and Sequential Feature Selector.
    
    Parameters:
        X_train (pd.DataFrame): Training feature set
        y_train (pd.Series or np.array): Target labels
        n_features (int): Number of top features to select
        cv_folds (int): Number of cross-validation folds
    
    Returns:
        List[str]: Selected feature names
    """
    
    # Initialize XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Use Sequential Feature Selector (SFS) to select the best features
    sfs = SequentialFeatureSelector(
        xgb_clf, n_features_to_select=n_features, direction='forward', cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    sfs.fit(X_train, y_train)
    
    # Get the selected feature names
    selected_features = X_train.columns[sfs.get_support()].tolist()
    
    print(f"Selected Features: {selected_features}")
    return selected_features
