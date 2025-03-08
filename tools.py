from data import *
from model_nn import *
from base_models import Model

import xgboost as xgb
from typing import List, Union, Optional, Literal
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss



def train_kcv(
    model_type: Literal['gnb', 'nn', 'xgb', 'rf'],
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
    xgb_params: dict = {
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_train)),
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }, 
    rf_params: dict = {}
):

    cv_accuracies = []
    cv_log_losses = []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    if model_type == 'nn':
        X_train = scaler.fit_transform(X_train)

    for train_idx, val_idx in kf.split(X_train, y_train):
        if type(X_train) == np.ndarray:
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        elif type(X_train) == pd.DataFrame: 
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = Model(
            model_type=model_type, 
            xgb_params=xgb_params, 
            nn_params=X_train_fold.shape[1],
            rf_params=rf_params
        )
        model.fit(X_train_fold, y_train_fold, X_val=X_val_fold, y_val=y_val_fold)
        y_val_pred, y_val_proba = model.predict(X_val_fold) 

        cv_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
        cv_log_losses.append(log_loss(y_val_fold, y_val_proba))


    print(f"Cross-validation accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    print(f"Cross-validation log loss: {np.mean(cv_log_losses):.4f} ± {np.std(cv_log_losses):.4f}")

