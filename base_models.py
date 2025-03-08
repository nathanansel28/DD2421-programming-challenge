from model_nn import * 

from typing import Literal, Tuple, Optional, Dict
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

Features = Literal['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

class Model:
    def __init__(
        self, 
        model_type: Literal['gnb', 'nn', 'xgb'],
        selected_features: List[Features] = None,
        xgb_params: Optional[Dict] = {}, 
        nn_params: Optional[int] = None, 
    ):
        self.model_type = model_type
        if self.model_type == 'gnb':
            self.model = GaussianNB()
        elif self.model_type == 'nn' and nn_params:
            self.model = create_nn_model(nn_params)
        elif self.model_type == 'xgb': 
            self.model = xgb.XGBClassifier(**xgb_params)
        if selected_features is None: 
            self.selected_features = [f'x{i}' for i in range(1, 14)]
        else: 
            self.selected_features = selected_features


    def fit(
        self, X_train, y_train, X_val=None, y_val=None, verbose=1
    ) -> None:
        X_train = self._select_features(X_train)

        if self.model_type in ['gnb', 'xgb']: 
            self.model.fit(X_train, y_train)
        elif self.model_type == 'nn':
            self.model.fit(
                X_train, y_train, 
                epochs=20, batch_size=16, verbose=verbose, 
                validation_data=(X_val, y_val)
            )


    def predict(
        self, X_pred
    ) -> Tuple[np.ndarray, np.ndarray]: 
        X_pred = self._select_features(X_pred)
        if self.model_type != 'nn':
            X_pred = X_pred[self.selected_features]
        if self.model_type in ['gnb', 'xgb']:     
            y_proba = self.model.predict_proba(X_pred)
            y_pred = self.model.predict(X_pred)
        elif self.model_type == 'nn': 
            y_proba = self.model.predict(X_pred)
            y_pred = np.argmax(y_proba, axis=1)

        return y_pred, y_proba

    def _select_features(self, X_input): 
        return X_input[self.selected_features] if self.model_type != 'nn' else X_input
