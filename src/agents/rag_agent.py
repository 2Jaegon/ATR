import pandas as pd
import numpy as np

class MockRAGAgent:
    def __init__(self):
        # 가상의 과거 정비 이력 DB
        self.history_db = [
            {
                "id": 1,
                "pattern_desc": "High temperature and high vibration",
                "temp_mean": 65.0,
                "vib_mean": 1.2,
                "cause": "Bearing wear",
                "action": "Replace bearing",
                "success_rate": 0.95
            },
            {
                "id": 2,
                "pattern_desc": "High temperature, normal vibration",
                "temp_mean": 60.0,
                "vib_mean": 0.6,
                "cause": "Cooling fan failure",
                "action": "Check and replace cooling fan",
                "success_rate": 0.88
            },
            {
                "id": 3,
                "pattern_desc": "Normal temperature, high vibration",
                "temp_mean": 50.0,
                "vib_mean": 1.5,
                "cause": "Misalignment",
                "action": "Realign the shaft",
                "success_rate": 0.92
            },
            {
                "id": 4,
                "pattern_desc": "Short spike in temperature (False Alarm)",
                "temp_mean": 55.0,
                "vib_mean": 0.5,
                "cause": "Sensor Glitch",
                "action": "Ignore or recalibrate sensor",
                "success_rate": 0.99
            }
        ]
        
    def search_similar_pattern(self, anomaly_data):
        """
        Phase 1: 임시로 이상 데이터의 평균값을 기반으로 가장 유사한 과거 사례를 검색.
        (실제로는 시계열 임베딩 간의 코사인 유사도를 계산해야 함)
        """
        target_temp = anomaly_data['temperature'].mean()
        target_vib = anomaly_data['vibration'].mean()
        
        best_match = None
        min_distance = float('inf')
        
        for record in self.history_db:
            # Simple Euclidean distance in feature space
            dist = np.sqrt((record['temp_mean'] - target_temp)**2 + (record['vib_mean'] - target_vib)**2)
            if dist < min_distance:
                min_distance = dist
                best_match = record
                
        return best_match
    
    def generate_action_report(self, anomaly_data):
        match = self.search_similar_pattern(anomaly_data)
        
        report = f"""
        [Action Report]
        - 유사 과거 사례 ID: {match['id']}
        - 예상 원인: {match['cause']}
        - 권장 조치: {match['action']}
        - 조치 성공률: {match['success_rate']*100}%
        """
        return report

if __name__ == "__main__":
    # Test Mock RAG
    mock_anomaly = pd.DataFrame({
        'temperature': [64.0, 65.5, 66.0],
        'vibration': [1.1, 1.2, 1.3]
    })
    
    agent = MockRAGAgent()
    print(agent.generate_action_report(mock_anomaly))
