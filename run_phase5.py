import os
import warnings
warnings.filterwarnings('ignore')

from src.workflow import compile_workflow
from src.agents.detection_agent import DetectionAgent
from src.data.data_loader import RawDataLoader

def run_strict_orchestration():
    print("=== Phase 5: Strict Agent Orchestration Simulation ===")
    
    base_dir = "data/phm_data_challenge_2018/train"
    raw_data_file = os.path.join(base_dir, "05_M02_DC_train.csv")
    fault_data_file = os.path.join(base_dir, "train_faults", "05_M02_train_fault_data.csv")
    
    if not os.path.exists(raw_data_file):
        print(f"Error: {raw_data_file} not found.")
        return
        
    print(f"[Main] Initializing Raw Data Stream from: {raw_data_file}")
    
    # 원본 데이터를 5행(초) 단위로 그냥 읽어옵니다. (데이터 조작 없음)
    streamer = RawDataLoader(raw_data_file, chunksize=5)
    
    # 1. Detection Agent (오직 감지만 수행하되, 내부에 ML 두뇌 탑재)
    detector = DetectionAgent()
    detector.train(raw_data_file, fault_data_file)
    
    # 2. Main Agent (워크플로우 총괄)
    app = compile_workflow()
    
    print("\n[Main] Starting continuous sensor monitoring...")
    
    row_count = 0
    
    # 센서 데이터를 계속 읽습니다.
    for chunk in streamer:
        # 감지 에이전트에게 센서값을 보여주고 상태를 묻습니다.
        status = detector.detect(chunk)
        current_time = chunk['time'].iloc[-1]
        
        row_count += len(chunk)
        
        if status == "normal":
            if row_count % 5000 == 0:
                print(f"Streaming... [Row {row_count}] Time: {current_time} | Status: [NORMAL]")
            continue
            
        # 노란점(이상치) 또는 빨간점(오류) 감지 시
        print(f"\n==================================================")
        if status == "yellow":
            print(f"[YELLOW DOT] Anomaly Detected at Time: {current_time}")
            event_msg = "이상치(Anomaly)가 감지되었습니다. 원인 분석이 필요합니다."
        elif status == "red":
            print(f"[RED DOT] Fault Detected at Time: {current_time}")
            event_msg = "치명적 오류(Fault)가 발생했습니다. 긴급 정비 이력 검색이 필요합니다."
        print(f"==================================================")
        
        # Main Agent가 RAG와 정비 에이전트를 깨워 워크플로우를 돌립니다.
        print("\n[Main Agent] Triggering RAG & Maintenance Workflow...")
        
        initial_state = {
            "sensor_data": chunk,
            "anomaly_detected": True,
            "is_false_alarm": False,
            "action_report": "",
            "engineer_feedback": event_msg,
            "next_step": "rag" # RAG 에이전트로 넘김
        }
        
        final_state = app.invoke(initial_state)
        
        print("\n=== [Maintenance Agent] Final Action Report ===")
        print(final_state.get('action_report'))
        print("===============================================\n")
        
        # 시뮬레이션 종료 (한 번의 이벤트만 처리 후 종료)
        print("[Main] Orchestration cycle completed. Stopping stream.")
        break

if __name__ == "__main__":
    run_strict_orchestration()
