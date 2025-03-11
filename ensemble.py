from data import *
from model_nn import *
from base_models import Model

from typing import Literal, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from itertools import product
import json



class Ensemble: 
    def __init__(
        self, 
        models: List[Model] = None, 
        model_weights: np.ndarray = None
    ):
        if models is None: 
            self.manual = True
            self.xgb_model = Model(model_type='xgb', xgb_params=xgb_params)
            self.gnb_model = Model(model_type='gnb', selected_features=['x2', 'x3', 'x4', 'x6', 'x8', 'x9', 'x10', 'x11'])
            self.nn_model = Model(model_type='nn', nn_params=X.shape[1]) # TODO: replace with correct parameter later
            self.rf_model = Model(model_type='rf', rf_params=rf_params)
        else: 
            self.manual = False
            self.models = models
        if model_weights is None: 
            self.model_weights = np.array([0.25, 0.25, 0.25, 0.25])
        else: 
            self.model_weights = model_weights


    def fit(
        self, X_train, y_train, X_val, y_val
    ) -> None:
        X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

        if self.manual:
            self.xgb_model.fit(X_train, y_train)
            self.gnb_model.fit(X_train, y_train)
            self.nn_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=0)
            self.rf_model.fit(X_train, y_train)
        else: 
            for model in self.models: 
                if model.model_type == 'nn':
                    model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=0)
                else: 
                    model.fit(X_train, y_train)
    

    def predict(
        self, X_pred, weights=None, mean_type='arithmetic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, X_pred_scaled = scale_data(X_pred, X_pred)

        if self.manual:
            models_predictions = [
                self.xgb_model.predict(X_pred)[1], 
                self.gnb_model.predict(X_pred)[1], 
                self.nn_model.predict(X_pred_scaled)[1], 
                self.rf_model.predict(X_pred)[1]
            ]
        else: 
            models_predictions = []
            for model in self.models: 
                if model.model_type == 'nn':
                    models_predictions.append(model.predict(X_pred_scaled)[1])
                else: 
                    models_predictions.append(model.predict(X_pred)[1])

        weights = self.model_weights if weights is None else weights
        return soft_voting(
            list_X_preds=models_predictions, weights=weights, mean_type=mean_type
        )


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



possible_models = ['xgb', 'gnb', 'nn', 'rf']
with open('params_xgb.json', 'r') as file:
    xgb_params = json.load(file)

with open('params_rf.json', 'r') as file: 
    rf_params = json.load(file)


def train_ensemble(
    X, 
    y,
    models: List[Model] = [
        Model(model_type='xgb', xgb_params=xgb_params), 
        Model(model_type='gnb'), 
        Model(model_type='nn', nn_params=X_train.shape[1]),
        Model(model_type='rf', rf_params=rf_params)
    ],
    k_fold_type: Literal['kcv', 'shuffle_split'] = 'kcv',
    mean_type: Literal['geometric', 'arithmetic'] = 'arithmetic',
    k: int = 5,  # k-fold cross-validation
    verbose: int = 0,
    weight_iterations: int = 31 
):

    # Split the data using StratifiedKFold
    if k_fold_type == 'kcv':
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    elif k_fold_type == 'shuffle_split':
        kf = StratifiedShuffleSplit(n_splits=k, test_size=2/3, random_state=42)

    weight_results = {}  # To hold the accuracy and log_loss for each weight combination
    weight_options = np.linspace(0, 1, weight_iterations)
    
    for train_idx, val_idx in kf.split(X, y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        X_train_scaled, X_val_scaled = scale_data(X_train, X_val) # scale data for NN

        models_predictions = []
        for model in models:
            if model.model_type == 'nn':
                model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, verbose=verbose)
                models_predictions.append(model.predict(X_val_scaled)[1])
            else: 
                model.fit(X_train, y_train)
                models_predictions.append(model.predict(X_val)[1])
        # print(models_predictions)
        # for model_pred in models_predictions:
        #     print(model_pred.shape)

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


def show_top_weights(final_results, n_top=5):
    # Sort by mean_accuracy (top 5 accuracies)
    top_5_accuracies = sorted(final_results, key=lambda x: x['mean_accuracy'], reverse=True)[:n_top]

    # Sort by mean_log_loss (bottom 5 log-losses)
    bottom_5_loglosses = sorted(final_results, key=lambda x: x['mean_log_loss'])[:n_top]

    print(f"Top {n_top} Accuracies:")
    for i, result in enumerate(top_5_accuracies, 1):
        print(f"{i}. Weights: {[round(float(w), 2) for w in result['weights']]} | "
              f"Mean Accuracy: {result['mean_accuracy']:.3f} | "
              f"Std Accuracy: {result['std_accuracy']:.3f} | "
              f"Mean Log Loss: {result['mean_log_loss']:.3f} | "
              f"Std Log Loss: {result['std_log_loss']:.3f}")

    print(f"\nBottom {n_top} Log Losses:")
    for i, result in enumerate(bottom_5_loglosses, 1):
        print(f"{i}. Weights: {[round(float(w), 2) for w in result['weights']]} | "
              f"Mean Accuracy: {result['mean_accuracy']:.3f} | "
              f"Std Accuracy: {result['std_accuracy']:.3f} | "
              f"Mean Log Loss: {result['mean_log_loss']:.3f} | "
              f"Std Log Loss: {result['std_log_loss']:.3f}\n")
