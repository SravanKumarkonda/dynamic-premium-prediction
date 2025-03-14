# Dynamic Premium Prediction

A machine learning system that predicts insurance premiums based on various driving and vehicle-related features.

## Project Overview

This project implements a dynamic premium prediction system using:
- Random Forest Regressor
- MLflow for experiment tracking
- Flask for web interface
- Docker for containerization


## Dataset

The dataset includes various features that influence insurance premiums:
- Speed and acceleration patterns
- Vehicle details (Make, Model, Year)
- Safety features
- Driving history (Accidents, Claims)
- Environmental conditions (Weather, Traffic, Road conditions)
- Current premium and policy information

## Technical Implementation

### Data Preprocessing
- Handled categorical variables using one-hot encoding
- Split data into training (80%) and testing (20%) sets
- Standardized numerical features

### Model Training
- Algorithm: Random Forest Regressor
- Hyperparameter Tuning using GridSearchCV:
  ```python
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [10, 20, 30],
      'min_samples_split': [2, 5, 10]
  }
  ```
- Metrics tracked: MSE, RMSE, R²

### MLflow Integration
- Experiment tracking
- Model versioning
- Parameter logging
- Metric monitoring

## Challenges and Solutions

1. **Feature Engineering**
   - Challenge: Complex categorical variables
   - Solution: Implemented comprehensive one-hot encoding

2. **Model Deployment**
   - Challenge: Consistent environment across development and production
   - Solution: Containerized application using Docker

3. **MLflow Integration**
   - Challenge: Model persistence across container restarts
   - Solution: Implemented volume mounting for MLflow artifacts

## Getting Started

### Prerequisites
- Docker Desktop
- Git
- Python 3.8 or higher

### Installation

1. Clone the repository:
git clone https://github.com/SravanKumarkonda/dynamic-premium-prediction.git

                                or

unzip dynamic-premium-prediction.zip file 

# Go to the project directory using terminal by running the following command:
cd dynamic-premium-prediction

2. Start the application:
docker-compose up --build


3. In a new terminal, train the model:
docker exec -it dynamic-premium-prediction-flask-app-1 python src/train.py

4.(Optional) In a new terminal, start the Notebook for EDA:
docker exec -it dynamic-premium-prediction-flask-app-1 jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

### Accessing the Applications

- Flask Web Interface: http://localhost:8000
- MLflow UI: http://localhost:5000


## Usage

1. Start the application using Docker Compose
2. Train the model using the provided command
3. Access the web interface to make predictions
4. Monitor experiments in MLflow UI

## Model Performance

Current model metrics:
- R² Score: [Your R² score]
- RMSE: [Your RMSE value]
- MSE: [Your MSE value]

## 

- Builds the Docker containers
- Runs model training
- Performs model evaluation
- Stores metrics and artifacts


