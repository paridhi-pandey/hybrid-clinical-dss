import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import glob

def train_model():
    print("Finding data files...")
    
    # Locate data files dynamically
    base_dir = os.path.dirname(__file__)
    training_dir = os.path.join(base_dir, '../data/training')
    testing_dir = os.path.join(base_dir, '../data/testing')
    
    train_patterns = glob.glob(os.path.join(training_dir, '*.csv'))
    test_patterns = glob.glob(os.path.join(testing_dir, '*.csv'))
    
    if not train_patterns or not test_patterns:
        raise FileNotFoundError("Could not find CSV files in both data/training/ and data/testing/ directories.")
        
    train_file = train_patterns[0]
    test_file = test_patterns[0]
    
    print(f"Loading Training data from: {train_file}")
    df_train = pd.read_csv(train_file)
    print(f"Loading Testing data from: {test_file}")
    df_test = pd.read_csv(test_file)
    
    # Remove unnamed columns which often appear due to trailing commas
    df_train = df_train.loc[:, ~df_train.columns.str.contains('Unnamed')]
    df_test = df_test.loc[:, ~df_test.columns.str.contains('Unnamed')]
    
    # Make sure we only use columns that exist in both train and test to avoid KeyErrors
    common_cols = [c for c in df_train.columns if c in df_test.columns]
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]
    
    print("\nDataset common columns between splits before processing:")
    print(df_train.columns.tolist())
    
    # Assume the last column is the target variable
    target_col = df_train.columns[-1]
    feature_cols = df_train.columns[:-1]
    
    print(f"\nTarget column automatically identified as: '{target_col}'")
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' ensures we don't crash on unseen categories during testing/prediction
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # Target encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Define model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create final training pipeline (preprocessor uses pandas DataFrames)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
                               
    print("\nTraining Random Forest model on training dataset...")
    pipeline.fit(X_train, y_train_encoded)
    
    print("\nEvaluating model on testing dataset...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    # Output metrics
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    
    # Get unique classes present in the test set to avoid warnings
    unique_test_classes = np.unique(y_test_encoded)
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.inverse_transform(unique_test_classes)))
    
    # Save the models and artifacts
    models_dir = os.path.join(base_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(models_dir, 'pipeline_model.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    joblib.dump(feature_cols.tolist(), os.path.join(models_dir, 'feature_names.pkl'))
    joblib.dump(list(numeric_features), os.path.join(models_dir, 'numeric_features.pkl'))
    joblib.dump(list(categorical_features), os.path.join(models_dir, 'categorical_features.pkl'))
    
    print("Model and preprocessing artifacts saved successfully in models/")

if __name__ == "__main__":
    train_model()
