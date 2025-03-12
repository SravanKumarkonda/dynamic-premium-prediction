import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

class ModelTrainer:
    def __init__(self):
        self.model = None
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
    def train_model(self, X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', return_train_score=True)
        
        with mlflow.start_run() as run:
            # Train model
            grid_search.fit(X_train, y_train)
            
            # Make predictions
            y_pred = grid_search.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log basic metrics
            mlflow.log_metric("mse", float(mse))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            
            # Log GridSearchCV results
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log all CV results with valid metric names
            for i, (mean_score, std_score) in enumerate(zip(
                grid_search.cv_results_['mean_test_score'],
                grid_search.cv_results_['std_test_score']
            )):
                params = grid_search.cv_results_['params'][i]
                param_key = f"cv_score_n{params['n_estimators']}_d{params['max_depth']}_s{params['min_samples_split']}"
                mlflow.log_metric(param_key, mean_score)
                mlflow.log_metric(f"cv_std_{param_key}", std_score)
            
            # Log feature importances
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': grid_search.best_estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log top 10 feature importances
            for idx, row in feature_importance.head(10).iterrows():
                # Clean feature name for MLflow
                clean_feature_name = row['feature'].replace('=', '_').replace('[', '_').replace(']', '_')
                mlflow.log_metric(f"importance_{clean_feature_name}", row['importance'])
            
            # Save all metrics for GitHub Actions
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "best_cv_score": float(grid_search.best_score_),
                "best_params": grid_search.best_params_,
                "feature_importance": feature_importance.set_index('feature')['importance'].to_dict()
            }
            
            # Save metrics for GitHub Actions
            with open("/tmp/metrics.json", "w") as f:
                json.dump(metrics, f)
            
            # Log model
            mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        
        self.model = grid_search.best_estimator_
        return self.model