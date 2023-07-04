from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel


app = FastAPI()

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the request body model
class PredictionRequest(BaseModel):
    target: int

# K-Nearest Neighbors endpoint
@app.post("/knn")
def knn_prediction(request: PredictionRequest):
    target = request.target

    if not 0 <= target < len(X):
        return {"error": "Invalid target index"}

    # Create and fit the K-Nearest Neighbors model
    knn = KNeighborsClassifier()
    knn.fit(X, y)

    # Make a prediction
    prediction = knn.predict([X[target]])
    return {"prediction": iris.target_names[prediction[0]]}

# Random Forest endpoint
@app.post("/randomforest")
def random_forest_prediction(request: PredictionRequest):
    target = request.target

    if not 0 <= target < len(X):
        return {"error": "Invalid target index"}

    # Create and fit the Random Forest model
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # Make a prediction
    prediction = rf.predict([X[target]])
    return {"prediction": iris.target_names[prediction[0]]}

# Logistic Regression endpoint
@app.post("/logisticregression")
def logistic_regression_prediction(request: PredictionRequest):
    target = request.target

    if not 0 <= target < len(X):
        return {"error": "Invalid target index"}

    # Create and fit the Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(X, y)

    # Make a prediction
    prediction = logreg.predict([X[target]])
    return {"prediction": iris.target_names[prediction[0]]}
