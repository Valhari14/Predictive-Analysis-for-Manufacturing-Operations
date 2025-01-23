# Predictive Analysis for Manufacturing Operations

## Objective

This project develops a RESTful API to predict machine downtime and production defects using advanced machine learning techniques. The system provides decision-makers with actionable insights to minimize production disruptions and optimize manufacturing processes.

## Dataset Features

The dataset includes critical machine operation parameters:

| Parameter | Description |
|-----------|-------------|
| Date | Operation date |
| Machine_ID | Unique machine identifier |
| Hydraulic_Pressure(bar) | Hydraulic system pressure |
| Coolant_Pressure(bar) | Coolant system pressure |
| Air_System_Pressure(bar) | Air system pressure |
| Coolant_Temperature | Coolant temperature |
| Hydraulic_Oil_Temperature(°C) | Hydraulic oil temperature |
| Spindle_Bearing_Temperature(°C) | Spindle bearings temperature |
| Spindle_Vibration(m) | Spindle vibration |
| Tool_Vibration(m) | Tool vibration |
| Spindle_Speed(RPM) | Spindle rotational speed |
| Voltage(volts) | Machine voltage |
| Torque(Nm) | Torque measurement |
| Cutting(kN) | Cutting force |
| Run_time(hours) | Machine running time |
| Downtime_Flag | Target variable (Machine_Failure or No_Failure) |

## Technology Stack

- **Language**: Python 3.x
- **API Framework**: Flask
- **Machine Learning**: Scikit-learn
- **Command-line Tool**: cURL (for API testing)

## Prerequisites

- Python 3.8+
- pip (Python package manager)

### Dependencies
- Flask
- scikit-learn
- pandas
- joblib
- numpy

## Installation

Open your terminal or command prompt:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Flask API Server
```bash
python app.py
```
🌐 API will be running at: `http://127.0.0.1:5000`

## API Endpoints

Open your command prompt and navigate to the project directory:

### 1. Upload Data (POST /upload)
**Command:**
```bash
curl -X POST -F "file=@manufacturing_downtime.csv" http://127.0.0.1:5000/upload
```

**Response:**
```json
{
  "columns": [
    "Date",
    "Machine_ID",
    "Hydraulic_Pressure(bar)",
    "Coolant_Pressure(bar)",
    "Air_System_Pressure(bar)",
    "Coolant_Temperature",
    "Hydraulic_Oil_Temperature(?C)",
    "Spindle_Bearing_Temperature(?C)",
    "Spindle_Vibration(?m)",
    "Tool_Vibration(?m)",
    "Spindle_Speed(RPM)",
    "Voltage(volts)",
    "Torque(Nm)",
    "Cutting(kN)",
    "Run_time(hours)"
  ],
  "message": "File uploaded successfully!",
  "predictor": "Downtime_Flag"
}
```

### 2. Train Model (POST /train)
**Command:**
```bash
curl -X POST http://127.0.0.1:5000/train
```

**Response:**
```json
{
  "accuracy": 0.988,
  "classification_report": {
    "0": {
      "f1-score": 0.9878542510121457,
      "precision": 0.9878542510121457,
      "recall": 0.9878542510121457,
      "support": 247.0
    },
    "1": {
      "f1-score": 0.9881422924901185,
      "precision": 0.9881422924901185,
      "recall": 0.9881422924901185,
      "support": 253.0
    },
    "accuracy": 0.988,
    "macro avg": {
      "f1-score": 0.9879982717511322,
      "precision": 0.9879982717511322,
      "recall": 0.9879982717511322,
      "support": 500.0
    },
    "weighted avg": {
      "f1-score": 0.988,
      "precision": 0.988,
      "recall": 0.988,
      "support": 500.0
    }
  },
  "confusion_matrix": [
    [244, 3],
    [3, 250]
  ],
  "message": "Model trained successfully!"
}
```

### 3. Predict Downtime (POST /predict)
**Command:**
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"Hydraulic_Pressure(bar)\": 71.99, \"Coolant_Pressure(bar)\": 4.19194524, \"Air_System_Pressure(bar)\": 6.220352142, \"Coolant_Temperature\": 6.8, \"Hydraulic_Oil_Temperature(?C)\": 44.2, \"Spindle_Bearing_Temperature(?C)\": 40, \"Spindle_Vibration(?m)\": 0.717, \"Tool_Vibration(?m)\": 24.459, \"Spindle_Speed(RPM)\": 26526, \"Voltage(volts)\": 399, \"Torque(Nm)\": 28.37456166, \"Cutting(kN)\": 2.35, \"Run_time(hours)\": 16.85493031}" http://127.0.0.1:5000/predict
```

**Response:**
```json
{
  "Confidence": 0.99,
  "Downtime": "Yes"
}
```

## Model Performance Analysis

### Key Metrics
- **Overall Accuracy**: 98.8%
- **Confusion Matrix**:
  - True Negatives: 244
  - False Positives: 3
  - False Negatives: 3
  - True Positives: 250

### Classification Performance
- **Class 0 (No Downtime)**:
  - Precision: 0.988
  - Recall: 0.988
  - F1-Score: 0.988

- **Class 1 (Downtime)**:
  - Precision: 0.988
  - Recall: 0.988
  - F1-Score: 0.988

## Prediction Example
In the sample prediction, the model:
- Predicted: Downtime = "Yes"
- Confidence: 99%

