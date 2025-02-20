import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class CareerRecommendationSystem:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        
    def preprocess_data(self, df):
        """Enhanced data preprocessing with feature engineering"""
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Handle missing values more sophisticatedly
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
            
        # Fill categorical missing values with mode
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Feature engineering
        if 'years_experience' in df.columns:
            df['experience_squared'] = df['years_experience'] ** 2
            
        if 'education_level' in df.columns:
            education_weights = {
                'high_school': 1,
                'bachelor': 2,
                'master': 3,
                'phd': 4
            }
            df['education_weight'] = df['education_level'].map(education_weights)
        
        # Create interaction features
        if 'skills' in df.columns and 'years_experience' in df.columns:
            df['skill_experience_ratio'] = df['skills'].str.count(',') / (df['years_experience'] + 1)
        
        # Convert categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        return df_encoded
    
    def prepare_data(self, df, target_column='career_label'):
        """Prepare data for training with advanced preprocessing"""
        # Preprocess features
        X = self.preprocess_data(df.drop(target_column, axis=1))
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df[target_column])
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Apply SMOTE for balance
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def build_ann(self, input_dim, num_classes):
        """Build an improved ANN with regularization and batch normalization"""
        model = Sequential([
            Dense(256, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X_train, y_train, input_dim, num_classes):
        """Train multiple models with optimized parameters"""
        # Callbacks for ANN
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5)
        ]
        
        # Train ANN
        ann = self.build_ann(input_dim, num_classes)
        self.models['ann'] = ann
        ann.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['rf'] = rf.fit(X_train, y_train)
        
        # Train XGBoost
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['xgb'] = xgb.fit(X_train, y_train)
        
        # Train Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            random_state=self.random_state
        )
        self.models['gb'] = gb.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and print detailed metrics"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'ann':
                y_pred = np.argmax(model.predict(X_test), axis=1)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'report': report
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print("Classification Report:")
            print(report)
        
        return results
    
    def save_models(self, path_prefix="models/"):
        """Save all trained models"""
        for name, model in self.models.items():
            if name == 'ann':
                model.save(f"{path_prefix}ann_model.h5")
            else:
                joblib.dump(model, f"{path_prefix}{name}_model.pkl")

# Usage example
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("career_recommendation_processed_numeric.csv")
    
    # Initialize and train the system
    career_system = CareerRecommendationSystem()
    X_train, X_test, y_train, y_test = career_system.prepare_data(df)
    
    # Get the number of features and classes
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Train all models
    career_system.train_models(X_train, y_train, input_dim, num_classes)
    
    # Evaluate models
    results = career_system.evaluate_models(X_test, y_test)
    
    # Save models
    career_system.save_models()