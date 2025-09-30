# End-to-End MLOps Pipeline: Real-Time YouTube Sentiment Analysis via AWS CI/CD

## Core Technical Goals:

* **Real-Time Architecture**: Design a system for low-latency data fetching and prediction.
* **ML Model Development**: Leverage scikit-learn for training and production of the sentiment classification model.
* **Reproducibility (Nix & DVC)**: Utilize Nix for a declarative environment and DVC for versioning data and model artifacts.
* **Experiment Tracking (MLflow)**: Implement MLflow for logging parameters, metrics, and managing model versions.
* **Containerization (Docker)**: Package the model and Flask/FastAPI service into Docker containers for consistency.
* **API Development**: Build a Flask/FastAPI endpoint to serve real-time predictions and aggregated sentiment metrics.
* **Automated CI/CD**: Automated building, testing, and deployment to AWS EC2/ECR via GitHub Actions.

