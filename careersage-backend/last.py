import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

# Load Dataset
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_clean.csv")

# Ensure Column Names Are Clean
df.columns = df.columns.str.strip()

# Check if 'career_label' exists
if "career_label" not in df.columns:
    raise KeyError("Error: 'career_label' column not found in dataset.")

# 🔹 Explicit User Input Handling
def process_user_input(user_json):
    user_data = json.loads(user_json)
    df_new = pd.DataFrame([user_data])
    return df_new

# 🔹 Ensure Required Columns Exist
expected_columns = [
    "Age", "Gender", "Education_Level", "Programming_Skill_Level", "AI_ML_Skill_Level",
    "Data_Analysis_Skill_Level", "Cybersecurity_Skill_Level", "Web_Development_Skill_Level",
    "Interest_in_Management", "Interest_in_Research", "Job_Search_Count", "Career_Switches"
]

for col in expected_columns:
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in dataset. Adding default values (0).")
        df[col] = 0  # Default value

# 🔹 Feature Scaling (Normalization of Numerical Features)
numerical_features = [
    "Age", "Programming_Skill_Level", "AI_ML_Skill_Level", "Data_Analysis_Skill_Level",
    "Cybersecurity_Skill_Level", "Web_Development_Skill_Level", "Interest_in_Management",
    "Interest_in_Research", "Job_Search_Count", "Career_Switches"
]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 🔹 Convert Categorical Features
categorical_features = ["Education_Level", "Gender"]
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col])

# 🔹 Balance Dataset Using SMOTE
X = df.drop("career_label", axis=1)
y = df["career_label"]
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 🔹 Convert All Features to Float
X_resampled = X_resampled.astype(float)

# 🔹 Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔹 ANN Model with Early Stopping & Embedding Layer
def build_ann():
    model = Sequential([
        Embedding(input_dim=len(X_train.columns), output_dim=16, input_length=X_train.shape[1]),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

ann_model = build_ann()
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 🔹 Random Forest with Feature Importance
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, 50]}
rf_tuned = RandomizedSearchCV(rf, rf_params, cv=3, n_iter=5, scoring='accuracy')
rf_tuned.fit(X_train, y_train)

# 🔹 Feature Importance Analysis for RF
rf_feature_importance = pd.Series(rf_tuned.best_estimator_.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n🔹 Top Features Impacting Career Recommendation (RF Model):\n", rf_feature_importance.head(10))

# 🔹 XGBoost with Bayesian Optimization
xgb = XGBClassifier()
xgb_params = {'learning_rate': (0.01, 0.3), 'max_depth': (3, 10), 'n_estimators': (100, 500)}
xgb_tuned = BayesSearchCV(xgb, xgb_params, n_iter=10, cv=3, scoring='accuracy')
xgb_tuned.fit(X_train, y_train)

# 🔹 Feature Importance Analysis for XGBoost
xgb_feature_importance = pd.Series(xgb_tuned.best_estimator_.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n🔹 Top Features Impacting Career Recommendation (XGBoost Model):\n", xgb_feature_importance.head(10))

# 🔹 Accuracy Evaluation
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
y_pred_rf = rf_tuned.best_estimator_.predict(X_test)
y_pred_xgb = xgb_tuned.best_estimator_.predict(X_test)

ann_acc = accuracy_score(y_test, y_pred_ann)
rf_acc = accuracy_score(y_test, y_pred_rf)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"\n🔹 ANN Accuracy: {ann_acc * 100:.2f}%")
print(f"🔹 Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"🔹 XGBoost Accuracy: {xgb_acc * 100:.2f}%")

# 🔹 Save Models
joblib.dump(rf_tuned.best_estimator_, "random_forest.pkl")
joblib.dump(xgb_tuned.best_estimator_, "xgboost.pkl")
ann_model.save("ann_model.h5")