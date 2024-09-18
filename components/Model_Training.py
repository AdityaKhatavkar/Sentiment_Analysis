import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def model_trainer(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predicted_sentiment = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_sentiment)
    
    # Save the model to a file
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return accuracy

