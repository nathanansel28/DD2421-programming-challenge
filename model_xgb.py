from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
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
