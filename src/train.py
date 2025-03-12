import mlflow
import pandas as pd
from data.data_ingestion import DataIngestion
from models.model_training import ModelTrainer
import os

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Set the experiment (this will create it if it doesn't exist)
    experiment_name = "dynamic_premium_prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Set the experiment as active
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"Error setting experiment: {str(e)}")
        return
    
    # Initialize data ingestion
    data_ingestion = DataIngestion("data/dynamic_pricing_dataset.csv")
    
    # Load and split data
    print("Loading data...")
    df = data_ingestion.read_data()
    X_train, X_test, y_train, y_test = data_ingestion.split_data(df)
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train, X_test, y_test)
    
    print("Training completed! Check MLflow UI for metrics.")

if __name__ == "__main__":
    main()