
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_processed.csv")

# Ensure Column Names Are Clean
df.columns = df.columns.str.strip()

# Check if 'career_label' exists
if "career_label" not in df.columns:
    raise KeyError("Error: 'career_label' column not found in dataset.")

# Add User Behavior Features
df["job_search_count"] = np.random.randint(1, 10, df.shape[0])
df["career_switches"] = np.random.randint(0, 5, df.shape[0])

# Handle Missing Values
df.fillna("Unknown", inplace=True)

# Convert Categorical Features
df = pd.get_dummies(df, drop_first=True)

# Balance Dataset Using SMOTE
X = df.drop("career_label", axis=1)
y = df["career_label"]
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert All Features to Float
X_resampled = X_resampled.astype(float)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔹 ANN Model
def build_ann():
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

ann_model = build_ann()
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 🔹 Random Forest with Hyperparameter Tuning
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, 50]}
rf_tuned = RandomizedSearchCV(rf, rf_params, cv=3, n_iter=5, scoring='accuracy')
rf_tuned.fit(X_train, y_train)

# 🔹 XGBoost with Bayesian Optimization
xgb = XGBClassifier()
xgb_params = {'learning_rate': (0.01, 0.3), 'max_depth': (3, 10), 'n_estimators': (100, 500)}
xgb_tuned = BayesSearchCV(xgb, xgb_params, n_iter=10, cv=3, scoring='accuracy')
xgb_tuned.fit(X_train, y_train)

# 🔹 Transformer-based Model (BERT for Job Recommendations)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

# Ensure 'job_history' exists before BERT encoding
if "job_history" in X_train.columns:
    X_train_bert = np.array([encode_text(text) for text in X_train["job_history"]])
    X_test_bert = np.array([encode_text(text) for text in X_test["job_history"]])
else:
    raise KeyError("Error: 'job_history' column not found in dataset.")

# Define and Train BERT Classifier
bert_classifier = Sequential([
    Input(shape=(X_train_bert.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')
])
bert_classifier.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
bert_classifier.fit(X_train_bert, y_train, epochs=30, batch_size=32, validation_split=0.2)

# 🔹 Reinforcement Learning (Feedback-Based Updates)
feedback_scores = np.random.uniform(0, 1, size=len(y_test))  # Simulating feedback
rewarded_samples = X_test[feedback_scores > 0.7]
rewarded_labels = y_test[feedback_scores > 0.7]
ann_model.fit(rewarded_samples, rewarded_labels, epochs=10, batch_size=16)

# 🔹 Accuracy Evaluation
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
y_pred_rf = rf_tuned.best_estimator_.predict(X_test)
y_pred_xgb = xgb_tuned.best_estimator_.predict(X_test)
y_pred_bert = np.argmax(bert_classifier.predict(X_test_bert), axis=1)

ann_acc = accuracy_score(y_test, y_pred_ann)
rf_acc = accuracy_score(y_test, y_pred_rf)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
bert_acc = accuracy_score(y_test, y_pred_bert)

print(f"ANN Accuracy: {ann_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print(f"BERT Accuracy: {bert_acc * 100:.2f}%")

# 🔹 Save Models
joblib.dump(rf_tuned.best_estimator_, "random_forest.pkl")
joblib.dump(xgb_tuned.best_estimator_, "xgboost.pkl")
bert_classifier.save("bert_classifier.h5")
ann_model.save("ann_model.h5")
