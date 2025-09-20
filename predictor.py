import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and preprocessors with CORRECT filenames
model = joblib.load("multi_svm_model.pkl")  # Keep this if you saved it as this
scaler = joblib.load("scaler.pkl") 
le_knee = joblib.load("le_knee_condition.pkl")
le_severity = joblib.load("le_severity_level.pkl") 
le_treatment = joblib.load("le_treatment_advised.pkl")

# Rest of your code stays the same...
def predict_knee_condition(rms_amplitude, spectral_entropy, zero_crossing_rate, mean_frequency, peak_frequency=20.0):
    features = np.array([[rms_amplitude, peak_frequency, spectral_entropy, zero_crossing_rate, mean_frequency]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    knee_condition = le_knee.inverse_transform([prediction[0][0]])[0]
    severity_level = le_severity.inverse_transform([prediction[0][1]])[0]
    treatment_advised = le_treatment.inverse_transform([prediction[0][2]])[0]
    return knee_condition, severity_level, treatment_advised
