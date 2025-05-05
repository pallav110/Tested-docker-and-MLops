import mlflow
import numpy as np
from typing import List, Dict, Union, Any
import os
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_name: str = None, stage: str = None, run_id: str = None):
        """
        Initialize inference service with a model from MLflow
        """
        # Set MLflow tracking URI to the local database
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        try:
            # If run_id is provided, try multiple approaches to load the model
            if run_id:
                self._try_load_model(run_id)
            else:
                # Try registry or fallback to latest run
                self._try_registry_or_fallback(model_name, stage)
                
            self.feature_names = ["sepal length (cm)", "sepal width (cm)", 
                                "petal length (cm)", "petal width (cm)"]
            self.target_names = ["setosa", "versicolor", "virginica"]
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._print_debug_info(run_id)
            raise e
    
    def _try_load_model(self, run_id):
        """Try multiple approaches to load the model"""
        # Print debug information first
        self._print_debug_info(run_id)
        
        # Approach 1: Look for model in artifacts/model subdirectory
        try:
            artifacts_model_path = os.path.join("/app/mlruns/1", run_id, "artifacts", "model")
            logger.info(f"Trying to load model from artifacts/model path: {artifacts_model_path}")
            self.model = mlflow.pyfunc.load_model(artifacts_model_path)
            logger.info(f"Successfully loaded model from artifacts/model path")
            return
        except Exception as e:
            logger.warning(f"artifacts/model path failed: {e}")
        
        # Approach 2: Standard MLflow path with runs URI
        try:
            logger.info(f"Trying to load model using standard MLflow path: runs:/{run_id}/model")
            self.model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            logger.info(f"Successfully loaded model using standard MLflow path")
            return
        except Exception as e:
            logger.warning(f"Standard MLflow path failed: {e}")
        
        # Approach 3: Direct run directory path
        try:
            run_path = os.path.join("/app/mlruns/1", run_id)
            logger.info(f"Trying to load model from run directory: {run_path}")
            self.model = mlflow.pyfunc.load_model(run_path)
            logger.info(f"Successfully loaded model from run directory")
            return
        except Exception as e:
            logger.warning(f"Run directory path failed: {e}")
        
        # Approach 4: Search for model files recursively
        try:
            import glob
            model_files = glob.glob(f"/app/mlruns/1/{run_id}/**/MLmodel", recursive=True)
            if model_files:
                model_dir = os.path.dirname(model_files[0])
                logger.info(f"Found MLmodel file at: {model_files[0]}")
                logger.info(f"Trying to load model from found directory: {model_dir}")
                self.model = mlflow.pyfunc.load_model(model_dir)
                logger.info(f"Successfully loaded model from found directory")
                return
        except Exception as e:
            logger.warning(f"Found directory path failed: {e}")
        
        # Approach 5: Load pickle file directly if it exists
        try:
            for potential_path in [
                os.path.join("/app/mlruns/1", run_id, "artifacts", "model", "model.pkl"),
                os.path.join("/app/mlruns/1", run_id, "artifacts", "model.pkl"),
                os.path.join("/app/mlruns/1", run_id, "model.pkl")
            ]:
                if os.path.exists(potential_path):
                    logger.info(f"Found pickle file at: {potential_path}")
                    with open(potential_path, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"Successfully loaded model from pickle file")
                    return
        except Exception as e:
            logger.warning(f"Loading pickle file failed: {e}")
        
        raise ValueError(f"Failed to load model using any method for run_id {run_id}")
    
    def _try_registry_or_fallback(self, model_name, stage):
        """Try loading from registry or fall back to latest run"""
        try:
            # Try loading from registry
            self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
            logger.info(f"Loaded model {model_name} from registry (stage: {stage})")
        except Exception as e:
            logger.warning(f"Error loading from registry: {e}")
            # Find the latest run as fallback
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            experiments = client.search_experiments(filter_string="name = 'iris-classification'")
            if not experiments:
                raise ValueError("No experiments found")
                
            latest_run = client.search_runs(
                experiment_ids=[experiments[0].experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            if not latest_run:
                raise ValueError("No runs found")
                
            latest_run_id = latest_run[0].info.run_id
            logger.info(f"Falling back to latest run: {latest_run_id}")
            self._try_load_model(latest_run_id)
    
    def _print_debug_info(self, run_id=None):
        """Print debugging information about the run directory"""
        if run_id:
            # Check main run directory
            run_dir = f"/app/mlruns/1/{run_id}"
            logger.info(f"Debugging - checking run directory: {run_dir}")
            if os.path.exists(run_dir):
                logger.info(f"Run directory exists. Contents:")
                for item in os.listdir(run_dir):
                    path = os.path.join(run_dir, item)
                    if os.path.isfile(path):
                        logger.info(f"  - {item} (file, {os.path.getsize(path)} bytes)")
                    else:
                        logger.info(f"  - {item} (directory)")
                
                # Check artifacts directory if it exists
                artifacts_dir = os.path.join(run_dir, "artifacts")
                if os.path.exists(artifacts_dir):
                    logger.info(f"Artifacts directory exists. Contents:")
                    for item in os.listdir(artifacts_dir):
                        path = os.path.join(artifacts_dir, item)
                        if os.path.isfile(path):
                            logger.info(f"  - {item} (file, {os.path.getsize(path)} bytes)")
                        else:
                            logger.info(f"  - {item} (directory)")
                    
                    # Check model directory if it exists
                    model_dir = os.path.join(artifacts_dir, "model")
                    if os.path.exists(model_dir):
                        logger.info(f"Model directory exists. Contents:")
                        for item in os.listdir(model_dir):
                            path = os.path.join(model_dir, item)
                            if os.path.isfile(path):
                                logger.info(f"  - {item} (file, {os.path.getsize(path)} bytes)")
                            else:
                                logger.info(f"  - {item} (directory)")
            else:
                logger.warning(f"Run directory does not exist")
    
    def predict(self, features):
        """
        Make prediction with preprocessed features
        
        Args:
            features: List of feature values or DataFrame
            
        Returns:
            List of predicted class names
        """
        # Handle different model types (mlflow.pyfunc vs direct scikit-learn model)
        if hasattr(self.model, 'predict'):
            # Direct scikit-learn model
            predictions = self.model.predict(features)
        else:
            # MLflow pyfunc model
            predictions = self.model.predict(features)
        
        # Convert numeric predictions to class names if they're numeric
        if hasattr(predictions, 'dtype') and np.issubdtype(predictions.dtype, np.integer):
            return [self.target_names[int(pred)] for pred in predictions]
        
        return predictions
    
    def predict_with_validation(self, data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Validate input data and make prediction
        
        Args:
            data: List of dictionaries with feature names and values
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Validate and extract features
        features = []
        for item in data:
            try:
                # Ensure all features are present
                item_features = [float(item.get(feat, 0.0)) for feat in self.feature_names]
                features.append(item_features)
            except (ValueError, TypeError):
                return {"error": "Invalid input format"}
        
        # Make predictions
        predictions = self.predict(features)
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }