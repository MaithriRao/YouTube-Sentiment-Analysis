import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from dotenv import load_dotenv
import googleapiclient.discovery
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv() 

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") 

if YOUTUBE_API_KEY:
    # FIX: Added flush=True to force output immediately.
    print(f"*** API KEY STATUS: Key successfully loaded. Length: {len(YOUTUBE_API_KEY)} characters.", flush=True)
else:
    print("*** API KEY STATUS: FATAL! YOUTUBE_API_KEY is NOT set in the environment.", flush=True)

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# # Load the model and vectorizer from the AWS model registry and local storage
# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     # Set MLflow tracking URI to your server
#     load_dotenv()   # Setting MLflow tracking URI
#     print("Starting load_model_and_vectorizer()")
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     model = model.get_raw_model() # type is  lightgbm.sklearn.LGBMClassifier 
#     with open(vectorizer_path, 'rb') as file:
#         vectorizer = pickle.load(file)
#     print("Done load_model_and_vectorizer()")
#     return model, vectorizer

# # Initialize the model and vectorizer
# model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "3", "./tfidf_vectorizer.pkl")  # ("my_model", "version_", "./tfidf_vectorizer.pkl")

def load_model(model_path, vectorizer_path): # alternatively from local directory in case the instance is not active
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        # Re-raise the exception after printing a failure notice
        print(f"Error loading ML assets: {e}", flush=True)
        raise e

# Initialize the model and vectorizer
try:
    # CRITICAL: This needs to be runnable for Gunicorn to start workers.
    model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")
    # New print statement to confirm model load
    print("*** ML STATUS: Model and Vectorizer loaded successfully.", flush=True)
except Exception as e:
    # If model loading fails, the app cannot run, but we want the print to show up.
    print("*** ML STATUS: FATAL! Application cannot start without model files.", flush=True)
    # The app will likely crash here if the assets are genuinely missing.

# Initialize the model and vectorizer
# model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  


def get_comments_from_youtube(video_id, max_results=50):
    if not YOUTUBE_API_KEY:
        print("Error: YOUTUBE_API_KEY is missing, cannot call YouTube API.", flush=True)
        return []

    try:
        # Initialize YouTube API client
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        
        comments_list = []
        next_page_token = None
        
        # We limit to 1 page to minimize quota use and speed up response
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token,
            order="relevance"
        )
        response = request.execute()

        # CRITICAL DEBUG: Check the raw response before processing
        items = response.get("items", [])
        print(f"*** YOUTUBE RESPONSE DEBUG *** Items received: {len(items)}", flush=True)

        for item in items:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments_list.append(comment)
            
        print(f"Successfully fetched {len(comments_list)} comments.", flush=True)
        return comments_list

    except googleapiclient.errors.HttpError as e:
        # CRITICAL FIX: Extract the status code and print the full error content.
        status_code = e.resp.status
        error_content = e.content.decode()
        error_message = f"*** YOUTUBE API REJECTION *** Status: {status_code}. Content: {error_content}"
        print(error_message, flush=True)
        
        # We can now rely on the detailed printout above.
        return []
    except Exception as e:
        print(f"An unexpected error occurred during comment fetching: {e}", flush=True)
        return []

# --- Preprocessing and Prediction Logic ---

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|\S+\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
    return text

def predict_sentiment(comments):
    if not comments:
        return None
    
    preprocessed_comments = [preprocess_text(comment) for comment in comments]
    
    # 1. Vectorize the comments
    # Vectorizer must be fitted on the training data and loaded here.
    comment_vectors = vectorizer.transform(preprocessed_comments)
    
    # 2. Predict the classes (0 or 1)
    predictions = model.predict(comment_vectors)
    
    # 3. Calculate metrics
    total_comments = len(predictions)
    positive_count = int(sum(predictions)) # 1 is positive
    negative_count = total_comments - positive_count # 0 is negative
    
    positive_rate = round(positive_count / total_comments * 100, 2)
    negative_rate = round(negative_count / total_comments * 100, 2)
    
    return {
        "status": "success",
        "total_comments_analyzed": total_comments,
        "positive_comments": positive_count,
        "negative_comments": negative_count,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "raw_predictions": predictions.tolist() # For debugging/completeness
    }

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    return "Welcome to our YouTube Sentiment Analysis API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        
        if not video_id:
            return jsonify({"error": "Missing video_id parameter"}), 400

        # Step 1: Fetch comments
        comments = get_comments_from_youtube(video_id)
        
        if not comments:
            # This is the error the user is receiving
            print(f"Warning: No comments retrieved for video ID {video_id}.", flush=True)
            return jsonify({"error": "No comments provided (Key invalid, video private, or comments disabled)"}), 404
        
        # Step 2: Predict sentiment
        analysis_result = predict_sentiment(comments)
        
        return jsonify(analysis_result)

    except Exception as e:
        print(f"--- FATAL ERROR in /predict endpoint ---", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": "An internal server error occurred during prediction."}), 500

# @app.route('/')
# def home():
#     return "Welcome to our flask api"

# @app.route('/predict', methods=['POST'])


# def predict():
#     data = request.json
#     comments = data.get('comments')
#     print("i am the comment: ",comments)
#     print("i am the comment type: ",type(comments))
    
#     if not comments:
#         return jsonify({"error": "No comments provided"}), 400

#     try:
#         # Preprocess each comment before vectorizing
#         preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
#         # Transform comments using the vectorizer
#         transformed_comments = vectorizer.transform(preprocessed_comments)

#         # Convert the sparse matrix to dense format
#         dense_comments = transformed_comments.toarray()  # Convert to dense array
        
#         # Make predictions
#         predictions = model.predict(dense_comments).tolist()  # Convert to list
    
#     # Convert predictions to strings for consistency
#     # predictions = [str(pred) for pred in predictions]
#     except Exception as e:
#        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
#     # Return the response with original comments and predicted sentiments
#     response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
#     return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    print("Comments are", comments_data)
    print("Type of the comments", type(comments_data))
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) # port=5000 while running from local setup
