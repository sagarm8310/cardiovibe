
import numpy as np  
import pandas as pd  
import scipy.signal as signal  
import joblib  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
  
# Generate Synthetic PPG Dataset with Normal & Abnormal Cases  
def generate_ppg_dataset(samples=500):  
    data = []  
    labels = []  
    for i in range(samples):  
        t = np.linspace(0, 10, 1000)  
        if i % 2 == 0:  
            # Normal PPG signal  
            ppg_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.05, len(t))  
            label = 0  # Normal  
        else:  
            # Abnormal PPG signal (Irregularities in wave)  
            ppg_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))  
            label = 1  # Abnormal  
          
        features = extract_features(ppg_signal)  
        if features:  
            data.append(features)  
            labels.append(label)  
    return np.array(data), np.array(labels)  
  
# Feature Extraction  
def extract_features(ppg_signal, fs=100):  
    peaks, _ = signal.find_peaks(ppg_signal, distance=fs//2)  
    rr_intervals = np.diff(peaks) / fs  # Time between peaks  
      
    if len(rr_intervals) < 2:  
        return None  
      
    hrv = np.std(rr_intervals)  # Heart rate variability  
    bp_estimate = np.mean(rr_intervals) * 100  # Simulated BP estimation  
      
    return [hrv, bp_estimate]  
  
# Train AI Model  
def train_ai_model():  
    X, y = generate_ppg_dataset()  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
      
    model = RandomForestClassifier()  
    model.fit(X_train, y_train)  
      
    y_pred = model.predict(X_test)  
    acc = accuracy_score(y_test, y_pred)  
    print(f"Model Accuracy: {acc:.2f}")  
      
    joblib.dump(model, "cardiovibe_model.pkl")  
  
# Load and Use AI Model  
def analyze_ppg(ppg_signal):  
    model = joblib.load("cardiovibe_model.pkl")  
    features = extract_features(ppg_signal)  
      
    if features is None:  
        return "Insufficient data"  
      
    prediction = model.predict([features])[0]  
    return "Abnormal Cardiovascular Activity Detected!" if prediction == 1 else "Normal Cardiovascular Activity"  
  
# Run the System  
t, ppg_signal = np.linspace(0, 10, 1000), 0.5 * np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 1000)) + 0.3 * np.random.randn(1000)  # Simulated abnormal PPG  
train_ai_model()  
result = analyze_ppg(ppg_signal)  
print(result)
