from model_nn import * 

from typing import Literal, Tuple
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB


class Model:
    def __init__(
        self, 
        model_type: Literal['gnb', 'nn', 'xgb'],
        xgb_params = None, 
        nn_params = None, 
    ):
        self.model_type = model_type
        if self.model_type == 'gnb':
            self.model = GaussianNB()
        elif self.model_type == 'nn':
            self.model = create_nn_model(nn_params)
        elif self.model_type == 'xgb': 
            self.model = xgb.XGBClassifier(**xgb_params)


    def fit(
        self, X_train, y_train, X_val=None, y_val=None
    ) -> None:
        if self.model_type in ['gnb', 'xgb']: 
            self.model.fit(X_train, y_train)

        elif self.model_type == 'nn':
            self.model.fit(
                X_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(X_val, y_val)
            )


    def predict(
        self, X_pred
    ) -> Tuple[np.ndarray, np.ndarray]: 
        if self.model_type in ['gnb', 'xgb']:     
            y_proba = self.model.predict_proba(X_pred)
            y_pred = self.model.predict(X_pred)
        elif self.model_type == 'nn': 
            y_proba = self.model.predict(X_pred)
            y_pred = np.argmax(y_proba, axis=1)

        return y_pred, y_proba
