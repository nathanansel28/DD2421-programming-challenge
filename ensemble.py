from data import *
from model_nn import *
from base_models import Model

from typing import Literal, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss


def soft_voting(
    list_X_preds: List[np.ndarray],
    weights: List[float] = None,
    mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
) -> Tuple[np.ndarray, np.ndarray]:
    list_X_preds = np.array(list_X_preds)

    if weights is None:
        weights = np.ones(len(list_X_preds))  # Default to equal weights
    else:
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize weights to sum to 1

    if mean_type == 'arithmetic':
        weighted_proba = np.average(list_X_preds, axis=0, weights=weights)
    elif mean_type == 'geometric':
        weighted_proba = np.exp(np.average(np.log(list_X_preds + 1e-9), axis=0, weights=weights))    
    else:
        raise ValueError("mean_type must be either 'arithmetic' or 'geometric'")

    weighted_proba /= np.sum(weighted_proba, axis=1, keepdims=True)
    weighted_pred = np.argmax(weighted_proba, axis=1)

    return weighted_pred, weighted_proba

        
def train_ensemble(
    X, 
    y,
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
    mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
    k: int = 5, 
    verbose: int = 0
):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

    xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
    gnb_model = Model(model_type='gnb')
    nn_model = Model(model_type='nn', nn_params=X_train_scaled.shape[1])

    xgb_model.fit(X_train, y_train)
    gnb_model.fit(X_train, y_train)
    nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)

    xgb_y_val_pred, xgb_y_val_proba = xgb_model.predict(X_val) 
    gnb_y_val_pred, gnb_y_val_proba = gnb_model.predict(X_val) 
    nn_y_val_pred, nn_y_val_proba = nn_model.predict(X_val_scaled) 


    results = []
    weight_options = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
    for w_xgb in weight_options:
        for w_gnb in weight_options:
            if w_xgb + w_gnb <= 1:
                w_nn = 1 - (w_xgb + w_gnb)


            weighted_y_val_pred, weighted_y_val_proba = soft_voting(
                list_X_preds=[xgb_y_val_proba, gnb_y_val_proba, nn_y_val_proba],
                weights=[w_xgb, w_gnb, w_nn],
                mean_type=mean_type
            )

            results.append({
                'weights': (w_xgb, w_gnb, w_nn),
                'accuracy': accuracy_score(y_val, weighted_y_val_pred), 
                'log_loss': log_loss(y_val, weighted_y_val_proba)
            })


    best_result = min(results, key=lambda x: x['log_loss'])

    return results, best_result









    # xgb_model.fit(X_train_fold, y_train_fold)
    # gnb_model.fit(X_train_fold, y_train_fold)
    # nn_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)


    # cv_accuracies = []
    # cv_log_losses = []
    # kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # for train_idx, val_idx in kf.split(X_train, y_train):
    #     if type(X_train) == np.ndarray:
    #         X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    #     elif type(X_train) == pd.DataFrame: 
    #         X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    #     y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]


    #     xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
    #     nn_model = Model(model_type='nn', nn_params=X_train_fold.shape[1])
    #     gnb_model = Model(model_type='gnb')

    #     xgb_model.fit(X_train_fold, y_train_fold)
    #     gnb_model.fit(X_train_fold, y_train_fold)
    #     nn_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)







