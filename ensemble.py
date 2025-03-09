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

    # If weights are provided, normalize them
    if weights is None:
        weights = np.ones(len(list_X_preds))  # Default to equal weights
    else:
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize weights to sum to 1

    # Handle NaN values by replacing them with a small value
    list_X_preds = np.nan_to_num(list_X_preds, nan=1e-9)

    if mean_type == 'arithmetic':
        weighted_proba = np.average(list_X_preds, axis=0, weights=weights)
    elif mean_type == 'geometric':
        weighted_proba = np.exp(np.average(np.log(list_X_preds + 1e-9), axis=0, weights=weights))    
    else:
        raise ValueError("mean_type must be either 'arithmetic' or 'geometric'")

    # Normalize the weighted probabilities to sum to 1
    weighted_proba /= np.sum(weighted_proba, axis=1, keepdims=True)
    
    # Get the predicted class
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



# def train_ensemble(
#     X, 
#     y,
#     xgb_params: dict,
#     rf_params: dict,
#     mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
#     k: int = 5,  # k-fold cross-validation
#     verbose: int = 0
# ):

#     # Split the data using StratifiedKFold
#     kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
#     weight_results = {}  # This will hold the accuracy and log_loss for each weight combination
    
#     weight_options = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
    
#     for train_idx, val_idx in kf.split(X, y):
#         y_train, y_val = y[train_idx], y[val_idx]
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

#         # Scale the data
#         X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

#         # Create and fit the models
#         xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
#         gnb_model = Model(model_type='gnb', selected_features=['x2', 'x3', 'x4', 'x6', 'x8', 'x9', 'x10', 'x11'])
#         nn_model = Model(model_type='nn', nn_params=X_train_scaled.shape[1])
#         rf_model = Model(model_type='rf', rf_params=rf_params)

#         xgb_model.fit(X_train, y_train)
#         gnb_model.fit(X_train, y_train)
#         nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)
#         rf_model.fit(X_train, y_train)

#         # Predict on validation set
#         xgb_y_val_pred, xgb_y_val_proba = xgb_model.predict(X_val)
#         gnb_y_val_pred, gnb_y_val_proba = gnb_model.predict(X_val)
#         nn_y_val_pred, nn_y_val_proba = nn_model.predict(X_val_scaled)
#         rf_y_val_pred, rf_y_val_proba = rf_model.predict(X_val)

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

#                     # Store accuracy and log_loss for this weight combination
#                     weight_key = (round(w_xgb, 2), round(w_gnb, 2), round(w_nn, 2))
#                     accuracy = accuracy_score(y_val, weighted_y_val_pred)
#                     logloss = log_loss(y_val, weighted_y_val_proba)

#                     if weight_key not in weight_results:
#                         weight_results[weight_key] = {'accuracies': [], 'log_losses': []}
                    
#                     weight_results[weight_key]['accuracies'].append(accuracy)
#                     weight_results[weight_key]['log_losses'].append(logloss)
    
#     # Now, calculate the mean and std for each weight combination
#     final_results = []
#     for weight_key, metrics in weight_results.items():
#         accuracies = metrics['accuracies']
#         log_losses = metrics['log_losses']

#         mean_accuracy = np.mean(accuracies)
#         std_accuracy = np.std(accuracies)
#         mean_logloss = np.mean(log_losses)
#         std_logloss = np.std(log_losses)

#         final_results.append({
#             'weights': weight_key,
#             'mean_accuracy': mean_accuracy,
#             'std_accuracy': std_accuracy,
#             'mean_log_loss': mean_logloss,
#             'std_log_loss': std_logloss
#         })

#     # Get the best result based on mean log_loss
#     best_result = min(final_results, key=lambda x: x['mean_log_loss'])

#     return final_results, best_result




def train_ensemble(
    X, 
    y,
    xgb_params: dict,
    rf_params: dict,
    mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
    k: int = 5,  # k-fold cross-validation
    verbose: int = 0
):

    # Split the data using StratifiedKFold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    weight_results = {}  # This will hold the accuracy and log_loss for each weight combination
    
    # weight_options = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
    weight_options = np.linspace(0, 1, 31)  # 0 to 1 in steps of 0.05
    
    for train_idx, val_idx in kf.split(X, y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        # Scale the data
        X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

        # Create and fit the models
        xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
        gnb_model = Model(model_type='gnb', selected_features=['x2', 'x3', 'x4', 'x6', 'x8', 'x9', 'x10', 'x11'])
        # gnb_model = Model(model_type='gnb')
        nn_model = Model(model_type='nn', nn_params=X_train_scaled.shape[1])
        rf_model = Model(model_type='rf', rf_params=rf_params)

        xgb_model.fit(X_train, y_train)
        gnb_model.fit(X_train, y_train)
        nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)
        rf_model.fit(X_train, y_train)

        # Prepare models and their predictions in lists for easier manipulation
        models_predictions = [
            xgb_model.predict(X_val)[1], 
            gnb_model.predict(X_val)[1], 
            nn_model.predict(X_val_scaled)[1], 
            rf_model.predict(X_val)[1]
        ]
        
        # Loop through weight options to find the best weights
        weight_combinations = generate_weight_combinations(len(models_predictions), weight_options)
        
        for weight_comb in weight_combinations:
            weighted_y_val_pred, weighted_y_val_proba = soft_voting(
                list_X_preds=models_predictions,
                weights=weight_comb,
                mean_type=mean_type
            )

            if sum(np.isnan(y_val)) > 0:
                raise ValueError(f"This is the error: y_val \n\n {y_val}")
            if sum(sum(np.isnan(weighted_y_val_proba))):
                raise ValueError(
                    f"This is the error:\n\n {weighted_y_val_proba}\n"
                    f"{weight_comb}"
                )
            
            # Store accuracy and log_loss for this weight combination
            weight_key = tuple(round(w, 2) for w in weight_comb)
            accuracy = accuracy_score(y_val, weighted_y_val_pred)
            logloss = log_loss(y_val, weighted_y_val_proba)

            if weight_key not in weight_results:
                weight_results[weight_key] = {'accuracies': [], 'log_losses': []}
            
            weight_results[weight_key]['accuracies'].append(accuracy)
            weight_results[weight_key]['log_losses'].append(logloss)
    
    print(f"Model trainings successful, proceeding to weight combination search...")
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
    print(f"Weight combination search done.")

    return final_results, best_result


import numpy as np
from itertools import product

def generate_weight_combinations(num_models: int, weight_options: np.ndarray):
    """Generate all possible weight combinations for a given number of models where the weights sum to 1"""
    
    weight_combinations = []
    
    # Generate combinations of weights for all models such that they sum to 1
    for comb in product(weight_options, repeat=num_models):
        if np.isclose(sum(comb), 1.0):  # Check if the sum is very close to 1
            # Sort the combination to avoid duplicates (e.g., [0.2, 0.5, 0.3] and [0.3, 0.2, 0.5] are equivalent)
            comb_sorted = tuple(sorted(comb))
            if comb_sorted not in weight_combinations:
                weight_combinations.append(comb_sorted)
    
    return weight_combinations
