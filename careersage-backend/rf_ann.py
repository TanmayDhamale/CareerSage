# Import required libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import TensorFlow/Keras for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load the processed dataset
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_processed.csv")

# Define Features (X) and Target (y)
X = df.drop(columns=['Career Label'])  # Use all columns except the target
y = df['Career Label']

# Label Encode the Target Variable (Career Label)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the encoder for future use
joblib.dump(label_encoder, "career_label_encoder.pkl")

# Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------- RANDOM FOREST MODEL -------------------------- #
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Evaluation
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred, target_names=label_encoder.classes_))

# Save the trained model
joblib.dump(rf_model, "career_recommendation_rf.pkl")
print("Random Forest Model saved successfully!")

# -------------------------- ANN MODEL -------------------------- #
# Normalize the data for ANN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Convert y_train and y_test to int32
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

print("\nTraining ANN Model...")
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])

# Compile ANN
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train ANN Model
history = ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predictions & Evaluation
ann_y_pred_probs = ann_model.predict(X_test)
ann_y_pred = np.argmax(ann_y_pred_probs, axis=1)  # Convert softmax probabilities to class labels
ann_accuracy = accuracy_score(y_test, ann_y_pred)

print(f"ANN Accuracy: {ann_accuracy:.2f}")
print("ANN Classification Report:\n", classification_report(y_test, ann_y_pred, target_names=label_encoder.classes_))

# Save the trained ANN model
ann_model.save("career_recommendation_ann.h5")
print("ANN Model saved successfully!")

# -------------------------- COMPARE RESULTS -------------------------- #
print("\n\U0001F4A1 Model Performance Comparison \U0001F4A1")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"ANN Accuracy: {ann_accuracy:.2f}")

if ann_accuracy > rf_accuracy:
    print("✅ ANN performed better!")
else:
    print("✅ Random Forest performed better!")
