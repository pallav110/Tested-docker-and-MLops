---

# 🚀 MLOps Demo: Model Deployment with Docker & MLflow

Welcome to the **MLOps Demo** — a hands-on project showcasing how to deploy a Machine Learning model using **Docker** and **MLflow**, complete with a REST API for predictions. This setup ensures seamless tracking, reproducibility, and portability, essential pillars of **MLOps**.

---

## 📌 Project Overview

This project demonstrates:

* ✅ **Model Tracking & Packaging** with MLflow
* 🐳 **Containerized Deployment** using Docker
* 🌐 **Serving the Model via REST API** (FastAPI/Gunicorn)
* 🧪 Health check & prediction endpoints
* 📁 Persistent logging using mounted volumes

---

## 🛠️ Tech Stack

| Tool     | Role                                         |
| -------- | -------------------------------------------- |
| Python   | Core programming language                    |
| MLflow   | Model tracking, packaging, and deployment    |
| Docker   | Containerization for reproducibility         |
| Gunicorn | Production-ready WSGI server for Python apps |
| FastAPI  | Lightweight and fast web framework           |
| SQLite   | Lightweight database for MLflow backend      |

---

## 📂 Directory Structure

```
mlops-demo/
│
├── app.py                   # REST API logic (FastAPI/Gunicorn)
├── inference.py             # Model loading and prediction
├── train.py                 # Model training and MLflow logging
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container setup
├── mlruns/                  # MLflow tracking folder (mounted)
└── mlflow.db                # SQLite backend store (mounted)
```

---

## 🧪 Endpoints

Once deployed, the API provides:

* `GET /` → API metadata and status
* `GET /health` → Health check endpoint
* `POST /predict` → Make predictions with your model

---

## 🚀 How to Run

### 1. 🏗️ Build Docker Image

```bash
docker build -t mlops-demo .
```

### 2. 🧳 Run Docker Container with Volumes

```bash
docker run -p 5001:5000 \
  -v "C:/your/path/mlruns:/app/mlruns" \
  -v "C:/your/path/mlflow.db:/app/mlflow.db" \
  mlops-demo
```

### 3. 🌐 Access the API

Visit: [http://localhost:5001](http://localhost:5001)

Use tools like **Postman** or **cURL** to test the `/predict` endpoint.

---

## 📈 Sample Output (on visiting `/`)

```json
{
  "api_version": "1.0",
  "endpoints": {
    "/": "API information (this message)",
    "/health": "Health check endpoint",
    "/predict": "Prediction endpoint (POST)"
  },
  "model_loaded": true,
  "status": "online"
}
```

---

## 🎥 Suggested Demo Topics

You can showcase this as a part of:

* 🔹 **Model Deployment in MLOps**
* 🔹 **MLflow + Docker Integration**
* 🔹 **End-to-End ML Inference Pipeline**
* 🔹 **API-based Model Serving**

---

## 🙌 Credits

Built by **Pallav Sharma** as a demonstration of **real-world MLOps practices**.

---

