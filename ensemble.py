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

        
# def train_ensemble(
#     X, 
#     y,
#     xgb_params: dict,
#     mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
#     k: int = 5,  # k-fold cross-validation
#     verbose: int = 0
# ):

#     # Split the data using StratifiedKFold
#     kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
#     results = []
#     weight_options = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
#     for train_idx, val_idx in kf.split(X, y):
#         y_train, y_val = y[train_idx], y[val_idx]
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]


#         X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

#         # Create and fit the models
#         xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
#         gnb_model = Model(model_type='gnb')
#         nn_model = Model(model_type='nn', nn_params=X_train_scaled.shape[1])

#         xgb_model.fit(X_train, y_train)
#         gnb_model.fit(X_train, y_train)
#         nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)

#         # Predict on validation set
#         xgb_y_val_pred, xgb_y_val_proba = xgb_model.predict(X_val)
#         gnb_y_val_pred, gnb_y_val_proba = gnb_model.predict(X_val)
#         nn_y_val_pred, nn_y_val_proba = nn_model.predict(X_val_scaled)

#         # Loop through weight options to find the best weights
#         for w_xgb in weight_options:
#             for w_gnb in weight_options:
#                 if w_xgb + w_gnb <= 1:
#                     w_nn = 1 - (w_xgb + w_gnb)

#                     weighted_y_val_pred, weighted_y_val_proba = soft_voting(
#                         list_X_preds=[xgb_y_val_proba, gnb_y_val_proba, nn_y_val_proba],
#                         weights=[w_xgb, w_gnb, w_nn],
#                         mean_type=mean_type
#                     )

#                     results.append({
#                         'fold': len(results) // (len(weight_options) * len(weight_options)),
#                         'weights': (w_xgb, w_gnb, w_nn),
#                         'accuracy': accuracy_score(y_val, weighted_y_val_pred), 
#                         'log_loss': log_loss(y_val, weighted_y_val_proba)
#                     })

#     # Calculate mean and std deviation for accuracy and log-loss across all folds
#     accuracies = [result['accuracy'] for result in results]
#     log_losses = [result['log_loss'] for result in results]

#     mean_accuracy = np.mean(accuracies)
#     std_accuracy = np.std(accuracies)
#     mean_log_loss = np.mean(log_losses)
#     std_log_loss = np.std(log_losses)

#     # Store these mean and std values
#     final_results = {
#         'mean_accuracy': mean_accuracy,
#         'std_accuracy': std_accuracy,
#         'mean_log_loss': mean_log_loss,
#         'std_log_loss': std_log_loss,
#         'results': results  # This includes individual fold results for reference
#     }

#     # Get the best result based on log_loss
#     best_result = min(results, key=lambda x: x['log_loss'])

#     return final_results, best_result



def train_ensemble(
    X, 
    y,
    xgb_params: dict,
    mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
    k: int = 5,  # k-fold cross-validation
    verbose: int = 0
):

    # Split the data using StratifiedKFold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    weight_results = {}  # This will hold the accuracy and log_loss for each weight combination
    
    weight_options = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
    
    for train_idx, val_idx in kf.split(X, y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        # Scale the data
        X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

        # Create and fit the models
        xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
        gnb_model = Model(model_type='gnb')
        nn_model = Model(model_type='nn', nn_params=X_train_scaled.shape[1])

        xgb_model.fit(X_train, y_train)
        gnb_model.fit(X_train, y_train)
        nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)

        # Predict on validation set
        xgb_y_val_pred, xgb_y_val_proba = xgb_model.predict(X_val)
        gnb_y_val_pred, gnb_y_val_proba = gnb_model.predict(X_val)
        nn_y_val_pred, nn_y_val_proba = nn_model.predict(X_val_scaled)

        # Loop through weight options to find the best weights
        for w_xgb in weight_options:
            for w_gnb in weight_options:
                if w_xgb + w_gnb <= 1:
                    w_nn = 1 - (w_xgb + w_gnb)

                    weighted_y_val_pred, weighted_y_val_proba = soft_voting(
                        list_X_preds=[xgb_y_val_proba, gnb_y_val_proba, nn_y_val_proba],
                        weights=[w_xgb, w_gnb, w_nn],
                        mean_type=mean_type
                    )

                    # Store accuracy and log_loss for this weight combination
                    weight_key = (round(w_xgb, 2), round(w_gnb, 2), round(w_nn, 2))
                    accuracy = accuracy_score(y_val, weighted_y_val_pred)
                    logloss = log_loss(y_val, weighted_y_val_proba)

                    if weight_key not in weight_results:
                        weight_results[weight_key] = {'accuracies': [], 'log_losses': []}
                    
                    weight_results[weight_key]['accuracies'].append(accuracy)
                    weight_results[weight_key]['log_losses'].append(logloss)
    
    # Now, calculate the mean and std for each weight combination
    final_results = []
    for weight_key, metrics in weight_results.items():
        accuracies = metrics['accuracies']
        log_losses = metrics['log_losses']

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_logloss = np.mean(log_losses)
        std_logloss = np.std(log_losses)

        final_results.append({
            'weights': weight_key,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_log_loss': mean_logloss,
            'std_log_loss': std_logloss
        })

    # Get the best result based on mean log_loss
    best_result = min(final_results, key=lambda x: x['mean_log_loss'])

    return final_results, best_result

