import pandas as pd
from src.workflow import compile_workflow
from src.agents.detection_agent import DetectionAgent
from src.data.data_loader import StreamingDataLoader
import os
import warnings
warnings.filterwarnings('ignore')

def run_dynamic_streaming_simulation():
    print("=== Phase 5: Dynamic Raw Data Streaming Simulation ===")
    
    base_dir = "data/phm_data_challenge_2018/train"
    raw_data_file = os.path.join(base_dir, "05_M02_DC_train.csv")
    fault_data_file = os.path.join(base_dir, "train_faults", "05_M02_train_fault_data.csv")
    
    if not os.path.exists(raw_data_file):
        print(f"Error: {raw_data_file} not found. Please extract the PHM dataset first.")
        return
        
    print(f"[Simulator] Initializing Data Loader on raw file: {raw_data_file}")
    # chunksize를 크게 잡아서 하드디스크 I/O 병목을 줄입니다.
    loader = StreamingDataLoader(raw_data_file, fault_data_file, chunksize=50000)
    
    detector = DetectionAgent()
    app = compile_workflow()
    
    # 제너레이터에서 첫 번째 청크(50,000줄)를 뽑아와서 모델을 초기 학습시킵니다.
    print("[Simulator] Fetching first chunk for initial Model Training...")
    data_iter = iter(loader)
    first_chunk = next(data_iter)
    
    detector.train(first_chunk)
    
    print("\n[Simulator] Starting continuous real-time stream...")
    window_size = 5 # 5행(초) 단위로 감시
    row_count = 50000
    anomaly_triggered = False
    
    # 두 번째 청크부터 끝까지 스트리밍
    for chunk in data_iter:
        if anomaly_triggered:
            break
            
        # 청크 내부를 5행씩 잘라서 실시간 스트리밍 흉내를 냅니다.
        for i in range(0, len(chunk), window_size):
            window = chunk.iloc[i:i+window_size]
            is_anomaly = detector.detect(window)
            
            current_time = window['time'].iloc[-1] if not window.empty else "Unknown"
            
            # 로그 출력이 너무 많아지지 않도록 1만 줄에 한 번씩만 정상 상태 출력
            if row_count % 10000 == 0:
                print(f"Streaming... [Row {row_count}] Time: {current_time} | Status: Normal")
            
            row_count += window_size
            
            if is_anomaly:
                print(f"\n==================================================")
                print(f"[ALERT] Anomaly Detected at Time: {current_time} !!!")
                print(f"==================================================")
                
                true_label = window['label'].max()
                print(f"[Evaluation] True Label at this window: {'Fault(1)' if true_label == 1 else 'Normal(0) (False Positive)'}")
                
                print("\n[System] Triggering Multi-Agent Workflow for Root Cause Analysis...")
                
                initial_state = {
                    "sensor_data": window,
                    "anomaly_detected": True,
                    "is_false_alarm": False,
                    "action_report": "",
                    "engineer_feedback": "Raw stream triggered anomaly alert.",
                    "next_step": "rag"
                }
                
                final_state = app.invoke(initial_state)
                
                print("\n=== Final Agent Outcome ===")
                print(f"Action Report:\n{final_state.get('action_report')}")
                
                anomaly_triggered = True
                print("\n[Simulator] Stopping stream after handling the first anomaly.")
                break

if __name__ == "__main__":
    run_dynamic_streaming_simulation()
