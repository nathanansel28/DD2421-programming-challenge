from typing import List, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

features = [f"x{i}" for i in range(1, 14)]
train_raw = pd.read_csv('data//TrainOnMe.csv')[['y']+features]
test_raw = pd.read_csv('data//EvaluateOnMe.csv')[features]

"""
===================
DATA PRE-PROCESSING
===================
"""

# Select features and target
X_train = train_raw[features]
y_train = train_raw['y']
X_test = test_raw[features]

# Encode CATEGORICAL target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Converts ['Anthropic', 'OpenAI', 'Mistral'] to [0, 1, 2]

# Convert BOOLEAN columns to integers
bool_cols = X_train.select_dtypes(include=['bool']).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Encode CATEGORICAL features
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])  # Apply the same encoding to test


