# End-to-End MLOps Pipeline: Real-Time YouTube Sentiment Analysis via AWS CI/CD

## Core Technical Goals:

* **Real-Time Architecture**: Design a system for low-latency data fetching and prediction.
* **ML Model Development**: Leverage scikit-learn for training and production of the sentiment classification model.
* **Reproducibility (Nix & DVC)**: Utilize Nix for a declarative environment and DVC for versioning data and model artifacts.
* **Experiment Tracking (MLflow)**: Implement MLflow for logging parameters, metrics, and managing model versions.
* **Containerization (Docker)**: Package the model and Flask/FastAPI service into Docker containers for consistency.
* **API Development**: Build a Flask/FastAPI endpoint to serve real-time predictions and aggregated sentiment metrics.
* **Automated CI/CD**: Automated building, testing, and deployment to AWS EC2/ECR via GitHub Actions.

  
## MLflow on AWS
### A.MLflow on AWS Setup:
1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port
Run the following command on EC2 machine
```bash
* sudo apt update
* sudo apt update
* sudo apt install python3-pip
* sudo apt install pipenv
* sudo apt install virtualenv
* mkdir mlflow
* cd mlflow
* pipenv install mlflow
* pipenv install awscli
* pipenv install boto3
* pipenv shell

# Then set aws credentials
aws configure
# Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://[YOUR-S3-BUCKET-NAME]/
# open Public IPv4 DNS to the port 5000
#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-xx-xx-xx-xx.compute-1.amazonaws.com:5000/
```
### B.MLflow Server Installation & Remote Access
The MLflow server dependencies were installed and configured on the remote EC2 machine. This section documents the process for accessing and starting the server remotely:
```bash
* ssh ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com
* tmux
* aws configure
* cd mlflow
* pipenv shell
* mlflow server -h 0.0.0.0 --workers 1 --default-artifact-root s3://mlflow-bucket-first-25
```
* **Note**:
  * Configure AWS Credentials once to set the credentials that grant the EC2 instance environment access to the S3 artifact bucket
  * The flag --workers 1 is explicitly used here for stability and resource management on a single, potentially low-resource EC2 instance




