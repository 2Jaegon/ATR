import pandas as pd
from src.workflow import compile_workflow
from src.agents.detection_agent import DetectionAgent
import warnings
import time
warnings.filterwarnings('ignore')

def run_streaming_simulation():
    print("=== Phase 5: Continuous Time-Series Streaming Simulation ===")
    
    detector = DetectionAgent()
    detector.train()
    
    print("\n[Simulator] Loading continuous dataset (100,000 rows)...")
    df = pd.read_csv("data/training_core.csv")
    
    # 훈련에 사용하지 않은 테스트 구간 (전체 데이터의 절반 이후부터)
    test_start_idx = int(len(df) * 0.5)
    test_stream = df.iloc[test_start_idx:].reset_index(drop=True)
    
    print("[Simulator] Starting sensor data stream...")
    
    app = compile_workflow()
    
    window_size = 5 # 5초(5행) 윈도우 단위로 검사
    
    for i in range(0, len(test_stream), window_size):
        window = test_stream.iloc[i:i+window_size]
        
        # 모델은 정답(label)을 보지 않고 순수 17개 센서값만으로 판단
        is_anomaly = detector.detect(window)
        
        # 실시간 진행 상황 출력 (너무 빠르지 않게 조절)
        current_time = window['time'].iloc[-1] if not window.empty else "Unknown"
        if i % 100 == 0:
            print(f"Streaming... [Row {i+20000}] Time: {current_time} | Status: {'Normal' if not is_anomaly else 'ANOMALY DETECTED'}")
        
        if is_anomaly:
            print(f"\n==================================================")
            print(f"[ALERT] Anomaly Detected at Time: {current_time} !!!")
            print(f"==================================================")
            
            # 실제 정답 확인 (우리가 평가하기 위한 용도)
            true_label = window['label'].max()
            print(f"[Evaluation] True Label at this window: {'Fault(1)' if true_label == 1 else 'Normal(0) (False Alarm)'}")
            
            print("\n[System] Triggering Multi-Agent Workflow for Root Cause Analysis...")
            
            initial_state = {
                "sensor_data": window,
                "anomaly_detected": True,
                "is_false_alarm": False, # 초기값
                "action_report": "",
                "engineer_feedback": "Machine triggered anomaly alert. Please verify.",
                "next_step": "rag"
            }
            
            final_state = app.invoke(initial_state)
            
            print("\n=== Final Agent Outcome ===")
            print(f"Action Report:\n{final_state.get('action_report')}")
            print(f"Is False Alarm Evaluated by Maintenance Agent:\n{final_state.get('is_false_alarm')}")
            
            # 시뮬레이션 종료 (한 번의 이상 상황만 테스트)
            print("\n[Simulator] Stopping stream after resolving anomaly.")
            break

if __name__ == "__main__":
    run_streaming_simulation()
