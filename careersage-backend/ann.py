# Import required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load the dataset
#df = pd.read_csv("career_recommendation_dataset.csv")
df = pd.read_csv("/Users/tanmay/Developer/CareerSage/careersage-backend/career_recommendation_dataset.csv")

# 2️⃣ Display first few rows of the dataset
print(df.head())

# 3️⃣ Define Features (X) and Target (y)
# Selecting relevant columns (Modify these based on your dataset)
X = df[['Age', 'Gender', 'Education', 'Skills', 'Interests']]
y = df['Career Label']

# Convert categorical variables into numerical values (one-hot encoding)
X = pd.get_dummies(X)

# 4️⃣ Split dataset into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Make predictions
y_pred = model.predict(X_test)

# 7️⃣ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# 8️⃣ Save the trained model for future use
joblib.dump(model, "career_recommendation_model.pkl")
print("Model saved successfully!")



# import joblib

# # Load the trained model
# model = joblib.load("career_recommendation_model.pkl")

# # Example user input (modify this based on your dataset features)
# new_user = pd.DataFrame([{
#     'Age': 25,
#     'Gender': 'Male',
#     'Education_Level': "Bachelor's",
#     'Skill1': 'Programming',
#     'Skill2': 'Data Science',
#     'Interest1': 'Technology',
#     'Interest2': 'AI'
# }])

# # Convert categorical variables
# new_user = pd.get_dummies(new_user)

# # Ensure same feature columns as training
# missing_cols = set(X.columns) - set(new_user.columns)
# for col in missing_cols:
#     new_user[col] = 0  # Add missing columns with 0 values

# # Predict career for the new user
# predicted_career = model.predict(new_user)
# print(f"Recommended Career: {predicted_career[0]}")