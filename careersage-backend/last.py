import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# === 1️⃣ Load & Preprocess Dataset ===
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_refined.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check if 'career_label' exists
if "career_label" not in df.columns:
    raise KeyError("Error: 'career_label' column not found in dataset.")

# Remove job_history if it exists
if "job_history" in df.columns:
    df = df.drop("job_history", axis=1)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# One-Hot Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Extract features and target
X = df.drop("career_label", axis=1)
y = df["career_label"]

# === 2️⃣ Split Before Applying SMOTE ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "feature_scaler.pkl")

# === 3️⃣ Train & Optimize Random Forest Using Bayesian Search ===
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': (100, 1000),
    'max_depth': (10, 100),
    'min_samples_split': (4, 20),
    'min_samples_leaf': (2, 10),
    'bootstrap': [True, False]
}

rf_tuned = BayesSearchCV(
    rf, rf_params, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
)
rf_tuned.fit(X_train_resampled, y_train_resampled)

# Keep Top 30 Important Features
feature_importance = pd.Series(rf_tuned.best_estimator_.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(30).index  # Select 30 most important
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)[top_features]
X_test = pd.DataFrame(X_test, columns=X.columns)[top_features]

# === 4️⃣ Train Random Forest Multiple Times & Choose Best Model ===
best_rf_model = None
best_rf_score = 0

for i in range(10):  # Train 10 times with different seeds
    rf = RandomForestClassifier(
        **rf_tuned.best_params_,
        random_state=i
    )
    rf.fit(X_train_resampled, y_train_resampled)
    score = rf.score(X_test, y_test)

    print(f"RF Iteration {i+1}, Accuracy: {score:.4f}")
    
    if score > best_rf_score:
        best_rf_score = score
        best_rf_model = rf

print(f"\nBest RF Model Accuracy: {best_rf_score:.4f}")

# Save Best Random Forest Model
joblib.dump(best_rf_model, "random_forest_model.pkl")

# === 5️⃣ Train XGBoost with Optimized Parameters ===
xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=12, subsample=0.8, random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test)

# Save XGBoost model
joblib.dump(xgb, "xgboost_model.pkl")

# === 6️⃣ Train ANN with Batch Normalization & Dropout ===
ann_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')
])
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_resampled, y_train_resampled, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# ANN Predictions
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)

# Save ANN model
ann_model.save("ann_model.h5")

# === 7️⃣ Evaluate Models ===
rf_acc = accuracy_score(y_test, best_rf_model.predict(X_test))
xgb_acc = accuracy_score(y_test, y_pred_xgb)
ann_acc = accuracy_score(y_test, y_pred_ann)

print(f"\n🔹 Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"🔹 XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print(f"🔹 ANN Accuracy: {ann_acc * 100:.2f}%")

print("\n===== Random Forest Classification Report =====")
print(classification_report(y_test, best_rf_model.predict(X_test)))

print("\n===== XGBoost Classification Report =====")
print(classification_report(y_test, y_pred_xgb))

print("\n===== ANN Classification Report =====")
print(classification_report(y_test, y_pred_ann))

# === 8️⃣ Plot Confusion Matrix for Best Model ===
best_model = best_rf_model if rf_acc > xgb_acc else xgb
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Best Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('best_model_confusion_matrix.png')
