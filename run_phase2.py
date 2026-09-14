from src.workflow import compile_workflow
import pprint

def run_phase2():
    print("=== Phase 2: LangGraph & Local LLM (Qwen) PoC ===")
    
    # 1. 워크플로우 컴파일
    app = compile_workflow()
    
    # 2. 초기 상태 (감지 에이전트가 이상치를 감지하여 전달했다고 가정)
    # Phase 1에서 생성했던 임의의 이상치 평균 데이터
    initial_state = {
        "anomaly_data": {
            "temperature": [64.0, 65.0, 66.0],
            "vibration": [1.1, 1.2, 1.3]
        }
    }
    
    print("\n[Start Execution]")
    # 3. 그래프 실행
    for output in app.stream(initial_state):
        # 각 노드가 실행될 때마다 결과 상태를 간단히 출력
        for key, value in output.items():
            print(f"Finished node: '{key}'")
            # 전체 상태 출력이 너무 길어질 수 있으므로 스킵 (노드 내부에서 print 처리함)
            
    print("\n[Final State Evaluation Result]")
    # 스트림이 끝난 후 최종 상태를 얻기 위해 invoke 사용 혹은 상태 객체 직접 조회 (간이 출력용)
    final_state = app.invoke(initial_state)
    print(final_state.get("evaluation_result", "No evaluation result."))
    print(f"Is False Alarm? : {final_state.get('is_false_alarm')}")

if __name__ == "__main__":
    run_phase2()
