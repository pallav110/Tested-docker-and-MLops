import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set MLflow tracking URI - can be local or remote
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Clean feature names to ensure they're compatible with MLflow metrics
# Replace parentheses and other invalid characters with underscores
clean_feature_names = [name.replace('(', '_').replace(')', '_').replace(' ', '_') for name in feature_names]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
params = {
    "n_estimators": 100,
    "max_depth": 7,
    "min_samples_split": 2
}

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log feature importance as artifact
    feature_importance = model.feature_importances_
    for i, importance in enumerate(feature_importance):
        mlflow.log_metric(f"importance_{clean_feature_names[i]}", importance)
    
    # Log model with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.sklearn.log_model(
        model, 
        "model",
        signature=signature,
        registered_model_name="iris_classifier"
    )
    
    print(f"Model trained and logged to MLflow (Run ID: {run.info.run_id})")
    print(f"Accuracy: {accuracy:.4f}")