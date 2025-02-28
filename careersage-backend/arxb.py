
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
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

# Feature scaling (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance Dataset Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split Dataset with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Save the scaler for future predictions
joblib.dump(scaler, "feature_scaler.pkl")

# Save feature columns for future reference
with open('feature_columns.txt', 'w') as f:
    for col in X.columns:
        f.write(f"{col}\n")

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Improved ANN Model ===
def build_improved_ann(input_dim, output_dim, dropout_rate=0.3):
    model = Sequential([
        Input(shape=(input_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Train improved ANN
ann_model = build_improved_ann(X_train.shape[1], len(np.unique(y_train)))
ann_history = ann_model.fit(
    X_train,
    y_train,
    epochs=100,  # Train longer with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

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

# === XGBoost with Enhanced Parameters ===
xgb_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_tuned = RandomizedSearchCV(
    xgb,
    xgb_params,
    n_iter=20,
    cv=kfold,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_tuned.fit(X_train, y_train)
print(f"Best XGB Parameters: {xgb_tuned.best_params_}")

# === Gradient Boosting Classifier ===
gb = GradientBoostingClassifier(random_state=42)
gb_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

gb_tuned = RandomizedSearchCV(
    gb,
    gb_params,
    n_iter=20,
    cv=kfold,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
gb_tuned.fit(X_train, y_train)
print(f"Best GB Parameters: {gb_tuned.best_params_}")

# === Accuracy Evaluation ===
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
y_pred_rf = rf_tuned.best_estimator_.predict(X_test)
y_pred_xgb = xgb_tuned.best_estimator_.predict(X_test)
y_pred_gb = gb_tuned.best_estimator_.predict(X_test)

# Create a function to evaluate and display detailed metrics
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    
    return acc

# Evaluate all models
ann_acc = evaluate_model(y_test, y_pred_ann, "ANN Model")
rf_acc = evaluate_model(y_test, y_pred_rf, "Random Forest")
xgb_acc = evaluate_model(y_test, y_pred_xgb, "XGBoost")
gb_acc = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Plot training history for ANN
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(ann_history.history['accuracy'])
plt.plot(ann_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(ann_history.history['loss'])
plt.plot(ann_history.history['val_loss'])
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('ann_training_history.png')

# Plot model comparison
model_names = ['ANN', 'Random Forest', 'XGBoost', 'Gradient Boosting']
accuracies = [ann_acc, rf_acc, xgb_acc, gb_acc]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.savefig('model_comparison.png')

# === Save Models ===
print("\n=== Saving Models ===")
# Save the best performing model based on accuracy
best_model_name = model_names[np.argmax(accuracies)]
print(f"Best performing model: {best_model_name}")

# Save all models
joblib.dump(rf_tuned.best_estimator_, "random_forest_model.pkl")
joblib.dump(xgb_tuned.best_estimator_, "xgboost_model.pkl")
joblib.dump(gb_tuned.best_estimator_, "gradient_boosting_model.pkl")
ann_model.save("ann_model.h5")

# Create a model selection function for prediction
def predict_career(features, models, scaler):
    """
    Ensemble prediction function that takes the weighted average of all models
    """
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Get predictions from each model
    ann_pred = models['ann'].predict(scaled_features)
    rf_pred = models['rf'].predict_proba(scaled_features)
    xgb_pred = models['xgb'].predict_proba(scaled_features)
    gb_pred = models['gb'].predict_proba(scaled_features)
    
    # Weighted average (can be adjusted based on individual model performance)
    weights = {
        'ann': 0.25,
        'rf': 0.25,
        'xgb': 0.25,
        'gb': 0.25
    }
    
    # Adjust weights based on which model performed best
    best_model = model_names[np.argmax(accuracies)]
    if best_model == 'ANN':
        weights['ann'] = 0.4
        weights['rf'] = 0.2
        weights['xgb'] = 0.2
        weights['gb'] = 0.2
    elif best_model == 'Random Forest':
        weights['ann'] = 0.2
        weights['rf'] = 0.4
        weights['xgb'] = 0.2
        weights['gb'] = 0.2
    elif best_model == 'XGBoost':
        weights['ann'] = 0.2
        weights['rf'] = 0.2
        weights['xgb'] = 0.4
        weights['gb'] = 0.2
    elif best_model == 'Gradient Boosting':
        weights['ann'] = 0.2
        weights['rf'] = 0.2
        weights['xgb'] = 0.2
        weights['gb'] = 0.4
    
    # Weighted ensemble prediction
    final_pred = (weights['ann'] * ann_pred + 
                  weights['rf'] * rf_pred + 
                  weights['xgb'] * xgb_pred + 
                  weights['gb'] * gb_pred)
    
    # Return the class with highest probability
    return np.argmax(final_pred, axis=1)

# Save a small example of how to use the model in the future
with open('prediction_example.py', 'w') as f:
    f.write("""
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
""")

print("\n=== Training Complete ===")
print("All models have been trained and saved.")
print(f"ANN Accuracy: {ann_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
print(f"Best model: {model_names[np.argmax(accuracies)]}")