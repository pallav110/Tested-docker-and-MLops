from flask import Flask, request, jsonify, render_template
import traceback
from inference import ModelInference
import logging
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
model = None

# Use this approach instead of before_first_request
def load_model():
    global model
    try:
        # Get this run_id from your train.py output
        run_id = "3374cd7d316f412f9d9d95614e39d005"  # Replace with your actual run ID
        
        # Print debug info about mounted directories
        logger.info("Checking mlruns directory...")
        if os.path.exists("/app/mlruns"):
            logger.info("mlruns directory exists")
            if os.path.exists(f"/app/mlruns/1/{run_id}"):
                logger.info(f"Run directory exists: /app/mlruns/1/{run_id}")
                logger.info(f"Contents: {os.listdir(f'/app/mlruns/1/{run_id}')}")
            else:
                logger.warning(f"Run directory does not exist: /app/mlruns/1/{run_id}")
        else:
            logger.warning("mlruns directory does not exist!")
        
        # Only use the ModelInference class
        model = ModelInference(run_id=run_id)
        logger.info(f"Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        
# Load model at startup
load_model()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint - display API information"""
    return jsonify({
        "status": "online",
        "api_version": "1.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/": "API information (this message)",
            "/health": "Health check endpoint",
            "/predict": "Prediction endpoint (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Get input data
        data = request.json
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid input format. Expected a list of objects"}), 400

        # Make prediction
        prediction = model.predict(data)
        
        return jsonify({"prediction": prediction})
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)