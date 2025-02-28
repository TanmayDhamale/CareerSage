
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
ann_model = load_model('ann_model.h5')
scaler = joblib.load('feature_scaler.pkl')

# Load feature columns
with open('feature_columns.txt', 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

# Example function to prepare input data
def prepare_user_data(user_data_dict):
    # Create a DataFrame with the same columns as the training data
    df = pd.DataFrame(columns=feature_columns)
    
    # Fill the DataFrame with the user data
    user_row = pd.DataFrame([user_data_dict])
    
    # Convert to one-hot encoding if needed
    user_row = pd.get_dummies(user_row)
    
    # Make sure all columns from training exist
    for col in feature_columns:
        if col not in user_row.columns:
            user_row[col] = 0
    
    # Select only the columns used during training
    user_row = user_row[feature_columns]
    
    return user_row

# Example prediction function
def predict_career(user_data):
    # Prepare data
    prepared_data = prepare_user_data(user_data)
    
    # Scale the data
    scaled_data = scaler.transform(prepared_data)
    
    # Get predictions from each model
    ann_pred = ann_model.predict(scaled_data)
    rf_pred = rf_model.predict_proba(scaled_data)
    xgb_pred = xgb_model.predict_proba(scaled_data)
    gb_pred = gb_model.predict_proba(scaled_data)
    
    # Weighted ensemble prediction
    final_pred = (0.25 * ann_pred + 
                  0.25 * rf_pred + 
                  0.25 * xgb_pred + 
                  0.25 * gb_pred)
    
    # Return the class with highest probability
    return np.argmax(final_pred, axis=1)[0]

# Example usage
example_user = {
    'Age': 25,
    'Gender': 1,  # Encoded gender
    'Education': 1,  # Encoded education level
    'Preferred Industry': 3,  # Encoded industry preference
    'Writing': 1,
    'Finance': 0,
    'Programming': 1,
    'Engineering': 0,
    'Healthcare': 0,
    'Design': 1,
    'Marketing': 0,
    'Technology': 1,
    'Entertainment': 0,
    'Sports': 0,
    'Arts': 1
}

predicted_career = predict_career(example_user)
print(f"Predicted career label: {predicted_career}")
