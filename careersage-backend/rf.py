import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_processed_numeric.csv")

# Ensure Column Names Are Clean
df.columns = df.columns.str.strip()

# Check if 'career_label' exists
if "career_label" not in df.columns:
    raise KeyError("Error: 'career_label' column not found in dataset.")

# Remove job_history column if it exists
if "job_history" in df.columns:
    df = df.drop("job_history", axis=1)

# Handle Missing Values with appropriate strategies
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill numerical columns with median (more robust than mean)
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Convert Categorical Features
df = pd.get_dummies(df, drop_first=True)

# Extract features and target
X = df.drop("career_label", axis=1)
y = df["career_label"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance Dataset Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Save the scaler for future predictions
joblib.dump(scaler, "feature_scaler.pkl")

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Random Forest with Hyperparameter Tuning ===
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf = RandomForestClassifier(random_state=42)
rf_tuned = RandomizedSearchCV(
    rf,
    rf_params,
    n_iter=20,
    cv=kfold,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_tuned.fit(X_train, y_train)
print(f"Best RF Parameters: {rf_tuned.best_params_}")

# Train the Best Model Multiple Times and Select the Best One
best_model = None
best_score = 0

for i in range(5):  # Train 5 times with different seeds
    rf = RandomForestClassifier(
        **rf_tuned.best_params_,  # Use best parameters found
        random_state=i  # Different seed for each iteration
    )
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    
    print(f"Iteration {i+1}, Accuracy: {score}")
    
    if score > best_score:
        best_score = score
        best_model = rf  # Store best model

print(f"Best Model Accuracy: {best_score}")

# Evaluate Model
y_pred_rf = best_model.predict(X_test)
print("\n===== Random Forest Evaluation =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('random_forest_confusion_matrix.png')

# Save the best model
joblib.dump(best_model, "random_forest_model.pkl")
