from flask import Flask, request, render_template, jsonify
import mlflow
import pandas as pd
import os

app = Flask(__name__)

def load_latest_model():
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
        # Get the dynamic_premium_prediction experiment
        experiment = mlflow.get_experiment_by_name("dynamic_premium_prediction")
        if experiment is None:
            raise Exception("Experiment 'dynamic_premium_prediction' not found")
            
        # Search in this specific experiment
        latest_run = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['start_time DESC'],
            max_results=1
        )
        
        if len(latest_run) == 0:
            raise Exception("No model runs found in 'dynamic_premium_prediction' experiment")
            
        print(f"Loading model from experiment: dynamic_premium_prediction, run_id: {latest_run.iloc[0].run_id}")
        return mlflow.sklearn.load_model(f"runs:/{latest_run.iloc[0].run_id}/model")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data
            features = {
                'Speed_kmh': float(request.form['speed']),
                'Acceleration_ms2': float(request.form['acceleration']),
                'Braking_Pattern': request.form['braking_pattern'],
                'Distance_Driven_km': float(request.form['distance']),
                'Location_History': request.form['location_history'],
                'Vehicle_Make': request.form['vehicle_make'],
                'Vehicle_Model': request.form['vehicle_model'],
                'Vehicle_Year': int(request.form['vehicle_year']),
                'Safety_Features': request.form['safety_features'],
                'Past_Claims': int(request.form['past_claims']),
                'Previous_Policy_Lapses': int(request.form['policy_lapses']),
                'Accident_History': request.form['accident_history'],
                'Road_Conditions': request.form['road_conditions'],
                'Weather_Data': request.form['weather'],
                'Traffic_Congestion': request.form['traffic'],
                'Current_Premium': float(request.form['current_premium']),
                'Deductibles': float(request.form['deductibles']),
                'Policy_Type': request.form['policy_type']
            }
            
            # Create DataFrame with a single row
            df = pd.DataFrame([features])
            
            # Define categorical columns (same as in training)
            categorical_cols = [
                'Braking_Pattern',
                'Location_History',
                'Vehicle_Make',
                'Vehicle_Model',
                'Safety_Features',
                'Accident_History',
                'Road_Conditions',
                'Weather_Data',
                'Traffic_Congestion',
                'Policy_Type'
            ]
            
            # One-hot encode categorical variables
            df_encoded = pd.get_dummies(df, columns=categorical_cols)
            
            # Load model
            model = load_latest_model()
            
            # Get the expected feature names from the model
            expected_features = model.feature_names_in_
            
            # Add missing columns with 0 values
            for feature in expected_features:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0
                    
            # Reorder columns to match training data
            df_encoded = df_encoded[expected_features]
            
            # Make prediction
            prediction = model.predict(df_encoded)[0]
            
        except Exception as e:
            return jsonify({"error": str(e)}), 400
            
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)