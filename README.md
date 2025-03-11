# DD2421-programming-challenge
Submission by Nathan Ansel

# Model Explanation
Soft voting ensemble classifier of XGBoost, Gaussian Naive Bayes, Random Forest, and Deep Neural Network.

# How to Run Main File
1. Create a virtual environment and activate it.
2. `pip install -r requirements.txt`.
3. Head to `submission.ipynb` to view final submission.
4. Final submission is saved as `submission.txt`.

# Files
### Data & Prediction Files
1. `submission.txt`: Submission file.
2. `TrainOnMe.csv`: Training dataset.
3. `EvaluateOnMe.csv`: Test dataset.

### Python Files
Source code for all implementations.
1. `base_models.py`: To contain the `Model` class, a wrapper for l individual models used, providing common methods such as `fit()` and `predict()`. 
2. `data.py`: To process data.
3. `ensemble.py`: `Ensemble` class and ensemble training functions.
4. `model_nn.py`: Code for Deep Neural Network model.
5. `model_rf.py`: Code for Random Forests model.
6. `model_xgb.py`: Code for XGBoost model.
7. `tools.py`: Contain common training functions such as `train_kcv()` and `create_splits()`.

### Jupyter Notebooks
Here is where all the training and experimentation was done.
1. `validation.ipynb`: Validation of ensemble model.
2. `submission.ipynb`: Implementation to generate final predictions.
3. `playground.ipynb`: Exploratory data analysis.
4. `model_ensemble.ipynb`
5. `model_gnb.ipynb`
6. `model_nn.ipynb` 
7. `model_rf.ipynb` 
8. `model_xgb.ipynb` 
