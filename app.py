from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# Paths for saving data and model
DATA_PATH = "data.csv"
MODEL_PATH = "random_forest_model.pkl"

# Global variables
model = None
imputer = None


@app.route("/")
def index():
    """
    Root endpoint to provide API information.
    """
    return {
        "message": "Predictive Analysis for Manufacturing Operations",
        "endpoints": {
            "POST /upload": "Upload a CSV file with machine defect data.",
            "POST /train": "Train the Random Forest Model using uploaded data.",
            "POST /predict": "Make predictions using the trained model.",
        }
    }


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a CSV file and save it locally.
    """
    try:
        if "file" not in request.files:
            return {"error": "No file part in the request."}, 400

        file = request.files["file"]

        if file.filename == "":
            return {"error": "No file selected for uploading."}, 400

        # Save file locally
        data = pd.read_csv(file)
        data.to_csv(DATA_PATH, index=False)

        # Get all columns except the target column for prediction
        columns = list(data.columns)
        if "Downtime_Flag" in columns:
            columns.remove("Downtime_Flag")
        return {
            "columns": columns,
            "predictor": "Downtime_Flag",
            "message": "File uploaded successfully!",
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/train", methods=["POST"])
def train_model():
    """
    Train a Random Forest model using the uploaded dataset with all relevant features.
    """
    global model, imputer
    try:
        if not os.path.exists(DATA_PATH):
            return {"error": "No data uploaded. Please upload a CSV file first."}, 400

        data = pd.read_csv(DATA_PATH)

        if "Downtime_Flag" not in data.columns:
            return {"error": "Target variable 'Downtime_Flag' not found in the dataset."}, 400

        # Drop irrelevant columns like 'Date', 'Machine_ID', and 'Downtime_Flag' (target variable)
        X = data.drop(columns=["Downtime_Flag", "Date", "Machine_ID"])

        # Create binary target (1 for Machine_Failure, 0 for anything else)
        y = (data["Downtime_Flag"] == "Machine_Failure").astype(int)

        # Create and fit the imputer to handle missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the Random Forest model
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Save the model and the imputer together in a single file
        joblib.dump((model, imputer), MODEL_PATH)

        return {
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make a prediction using the trained model and input JSON data.
    """
    global model, imputer
    try:
        if model is None or imputer is None:
            if os.path.exists(MODEL_PATH):
                model, imputer = joblib.load(MODEL_PATH)  # Load both model and imputer
            else:
                return {"error": "Model not found. Please train the model first using the /train endpoint."}, 400

        input_data = request.json
        if not input_data:
            return {"error": "Invalid input format. Please provide JSON data."}, 400

        data = pd.read_csv(DATA_PATH)

        # Get the columns used for prediction (remove target column and any unwanted columns like Date and Machine_ID)
        columns = list(data.columns)
        if "Downtime_Flag" in columns:
            columns.remove("Downtime_Flag")
        if "Date" in columns:
            columns.remove("Date")
        if "Machine_ID" in columns:
            columns.remove("Machine_ID")

        # Convert input JSON to DataFrame and ensure the correct order of features
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=columns, fill_value=None)

        # Impute missing numeric features (if any) with the mean of the corresponding columns
        input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0])

        # Return the response in the desired format
        return {
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": round(confidence, 2),
        }
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True)
