#importing libraires
from flask import Flask, render_template, request
import pickle
from components.Data_Transformation import text_preprocessing  # Import the function from preprocessing.py

app = Flask(__name__)

# Load the model components
try:
    tfidf = pickle.load(open("E:/DataScience_projects/Sentiment_Analysis/components/artifacts/vectorizer.pkl", "rb"))
    model = pickle.load(open("E:/DataScience_projects/Sentiment_Analysis/components/artifacts/model.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    tfidf, model = None, None

@app.route('/', methods=["GET", "POST"])
def home():
    review = ""
    prediction = "Awaiting review..."
    isPositive = None  # Default value for GET request

    if request.method == "POST":
        review = request.form['text']
        transformed_review = text_preprocessing(review)
        vectorized_review = tfidf.transform([transformed_review])
        result = model.predict(vectorized_review)

        if result == 1:
            prediction = "Positive"
            isPositive = True
        else:
            prediction = "Negative"
            isPositive = False

    return render_template('index.html', review=review, prediction=prediction, isPositive=isPositive)

if __name__ == "__main__":
    app.run(debug=True)
