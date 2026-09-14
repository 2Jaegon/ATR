import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

class DetectionAgent:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False
        
    def train(self, normal_data):
        """정상 데이터로 모델 학습"""
        features = normal_data[['temperature', 'vibration']]
        self.model.fit(features)
        self.is_trained = True
        print("Detection Agent trained successfully on normal data.")
        
    def detect(self, data):
        """새로운 데이터에서 이상 탐지"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        features = data[['temperature', 'vibration']]
        # predict returns 1 for inliers, -1 for outliers
        predictions = self.model.predict(features)
        # convert to 0 (normal), 1 (anomaly)
        anomaly_labels = np.where(predictions == -1, 1, 0)
        
        results = data.copy()
        results['detected_anomaly'] = anomaly_labels
        
        # Calculate decision function scores (lower = more abnormal)
        scores = self.model.decision_function(features)
        results['anomaly_score'] = scores
        
        return results

if __name__ == "__main__":
    # Test the agent
    data_path = 'data/sensor_data.csv'
    try:
        df = pd.read_csv(data_path)
        
        # Assume first 200 rows are mostly normal for training
        train_data = df.iloc[:200]
        test_data = df
        
        agent = DetectionAgent(contamination=0.05)
        agent.train(train_data)
        
        results = agent.detect(test_data)
        
        # Evaluation
        # True anomalies (including false alarms for now, as both look abnormal to the sensor)
        actual_anomalies = results[results['label'] > 0]
        detected_anomalies = results[results['detected_anomaly'] == 1]
        
        print(f"Total rows: {len(results)}")
        print(f"Actual Anomalies (including false alarms): {len(actual_anomalies)}")
        print(f"Detected Anomalies: {len(detected_anomalies)}")
        
        # How many true anomalies were caught?
        caught = results[(results['label'] > 0) & (results['detected_anomaly'] == 1)]
        print(f"True Anomalies caught: {len(caught)}")
        
    except FileNotFoundError:
        print("Run generate_data.py first to create the dataset.")
