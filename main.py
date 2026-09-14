import pandas as pd
from src.data.generate_data import generate_sensor_data
from src.agents.detection_agent import DetectionAgent
from src.agents.rag_agent import MockRAGAgent
import os

def run_phase1():
    print("--- Phase 1: Proof of Concept ---")
    
    # 1. 데이터 생성
    data_path = 'data/sensor_data.csv'
    if not os.path.exists(data_path):
        print("Generating mock sensor data...")
        generate_sensor_data(num_samples=1000, save_path=data_path)
        
    df = pd.read_csv(data_path)
    
    # 2. 감지 에이전트 초기화 및 학습
    print("\n--- Detection Agent Initializing ---")
    # 앞쪽 데이터는 정상이라고 가정하고 학습
    train_data = df.iloc[:200]
    test_data = df
    
    detector = DetectionAgent(contamination=0.05)
    detector.train(train_data)
    
    # 3. 이상 탐지 실행
    print("\n--- Running Detection on data stream ---")
    results = detector.detect(test_data)
    
    # detected_anomaly == 1 인 구간들 추출
    anomalies = results[results['detected_anomaly'] == 1]
    print(f"Total anomalies detected: {len(anomalies)}")
    
    # 4. RAG 에이전트를 통한 분석 (Main Agent 역할 대행)
    print("\n--- RAG Agent Analysis (Action Report) ---")
    rag = MockRAGAgent()
    
    if len(anomalies) > 0:
        # 이상치 중 첫 번째 그룹(군집)을 추출하여 RAG에 전달한다고 가정
        first_anomaly_idx = anomalies.index[0]
        # 해당 시점 근처의 데이터를 수집
        event_data = anomalies.loc[first_anomaly_idx:first_anomaly_idx+10]
        
        print("Found anomaly event. Requesting Action Report from RAG...")
        report = rag.generate_action_report(event_data)
        print(report)
    else:
        print("No anomalies detected.")

if __name__ == "__main__":
    run_phase1()
