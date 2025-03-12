import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def read_data(self):
        df = pd.read_csv(self.data_path)
        return df
    
    def split_data(self, df):
        X = df.drop('Dynamic_Premium', axis=1)
        y = df['Dynamic_Premium']
        
        # Handle categorical variables
        categorical_cols = [
            'Safety_Features', 
            'Policy_Type',
            'Braking_Pattern',
            'Location_History',
            'Vehicle_Make',
            'Vehicle_Model',
            'Accident_History',
            'Road_Conditions',
            'Weather_Data',
            'Traffic_Congestion'
        ]
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)