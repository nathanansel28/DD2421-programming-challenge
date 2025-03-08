from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

# Define hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

def hyperparam_tuning(X_train, y_train, k=5, n_iter=20) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=n_iter,
        scoring='accuracy', cv=cv, verbose=2, n_jobs=-1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("Best parameters found:", random_search.best_params_)
    print("Best accuracy score:", random_search.best_score_)
    
    return random_search

