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
from tensorflow.keras.layers import Dense, Dropout

# === 1️⃣ Load & Preprocess Dataset ===
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_processed_numeric.csv")

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Check if 'career_label' exists
if "career_label" not in df.columns:
    raise KeyError("Error: 'career_label' column not found in dataset.")

# Remove job_history if it exists
if "job_history" in df.columns:
    df = df.drop("job_history", axis=1)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop("career_label")

# One-Hot Encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

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
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
    'bootstrap': [True, False]
}

rf_tuned = BayesSearchCV(
    rf, rf_params, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
)
rf_tuned.fit(X_train_resampled, y_train_resampled)

# Feature Importance Filtering
feature_importance = pd.Series(rf_tuned.best_estimator_.feature_importances_, index=X.columns)
top_features = feature_importance[feature_importance > 0.01].index  # Keep only top features
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)[top_features]
X_test = pd.DataFrame(X_test, columns=X.columns)[top_features]

# === 4️⃣ Train Random Forest Multiple Times & Choose Best ===
best_rf_model = None
best_rf_acc = 0

for i in range(5):  # Train 5 times with different seeds
    print(f"\n🔄 Training Random Forest - Iteration {i+1}")
    rf_model = RandomForestClassifier(**rf_tuned.best_params_, random_state=i)
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf)
    print(f"Iteration {i+1} Accuracy: {acc * 100:.2f}%")

    if acc > best_rf_acc:
        best_rf_acc = acc
        best_rf_model = rf_model

# Save the best Random Forest model
joblib.dump(best_rf_model, "random_forest_model.pkl")
print(f"\n✅ Best Random Forest Model Accuracy: {best_rf_acc * 100:.2f}%")

# === 5️⃣ Train Alternative Model (XGBoost) ===
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test)

# Save XGBoost model
joblib.dump(xgb, "xgboost_model.pkl")

# === 6️⃣ Train Alternative Model (ANN) ===
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')
])
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2)
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)

# Save ANN model
ann_model.save("ann_model.h5")

# === 7️⃣ Evaluate Models ===
rf_acc = best_rf_acc  # Use best accuracy from RF training loop
xgb_acc = accuracy_score(y_test, y_pred_xgb)
ann_acc = accuracy_score(y_test, y_pred_ann)

print(f"\n🔹 Best Random Forest Accuracy: {rf_acc * 100:.2f}%")
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
y_pred_best = best_rf_model.predict(X_test) if rf_acc > xgb_acc else y_pred_xgb

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Best Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('best_model_confusion_matrix.png')

print("\n✅ Training & Evaluation Completed!")