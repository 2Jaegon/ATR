from langgraph.graph import StateGraph, END
from src.agents.graph_state import AgentState
from src.agents.main_agent import MainAgent
from src.agents.maintenance_agent import MaintenanceAgent
from src.agents.rag_agent import OracleRAGAgent
import pandas as pd

def compile_workflow():
    # 그래프 초기화
    workflow = StateGraph(AgentState)
    
    # 에이전트 인스턴스화
    rag_agent = OracleRAGAgent()
    main_agent = MainAgent(model_name="qwen2.5:7b")
    maintenance_agent = MaintenanceAgent(model_name="qwen2.5:7b")
    
    # 1. RAG Node: 이상치 데이터를 받아 과거 이력 검색
    def rag_node(state: AgentState):
        print("--- [Node: RAG Agent] Searching history ---")
        anomaly = state.get("anomaly_data")
        # DataFrame 형태로 변환 (기존 RAG 에이전트 호환용)
        df_anomaly = pd.DataFrame(anomaly)
        history = rag_agent.search_similar_pattern(df_anomaly)
        return {"retrieved_history": [history], "next_step": "main_agent"}
        
    # 2. Main Node: 검색된 이력을 바탕으로 Action Report 생성
    def main_node(state: AgentState):
        return main_agent.generate_action_report(state)
        
    # 3. Engineer Node (Simulated): 엔지니어가 리포트를 읽고 실제 정비 후 결과를 입력하는 단계
    def engineer_node(state: AgentState):
        print("--- [Node: Field Engineer] Performing maintenance and writing feedback ---")
        # Phase 2에서는 시뮬레이션을 위해 하드코딩된 피드백 주입 (추후 UI 연동 가능)
        feedback = "현장 점검 결과 베어링 문제는 아니었으며, 센서 연결부 이물질로 인한 노이즈(가성 불량)로 확인되어 청소 후 조치 완료함."
        print(f"Engineer Feedback: {feedback}")
        return {"engineer_feedback": feedback, "next_step": "maintenance_agent"}
        
    # 4. Maintenance Node: 피드백 평가
    def maintenance_node(state: AgentState):
        return maintenance_agent.evaluate_feedback(state)
        
    # 5. DB Update Node: 최종 평가를 Oracle DB에 저장
    def db_update_node(state: AgentState):
        print("--- [Node: DB Update] Saving evaluation to Oracle DB ---")
        is_false_alarm = state.get("is_false_alarm")
        feedback = state.get("engineer_feedback")
        
        # 실제 환경에서는 여기서 oracledb connection을 열어 INSERT를 수행합니다.
        # 예시: 
        # sql = "INSERT INTO maintenance_history (pattern_desc, is_false_alarm) VALUES (:1, :2)"
        # cursor.execute(sql, (feedback, int(is_false_alarm)))
        # conn.commit()
        
        if is_false_alarm:
            print("Action: 해당 시계열 패턴을 '가성 불량'으로 라벨링하여 Oracle DB(maintenance_history)에 저장.")
        else:
            print("Action: 해당 정비 이력 및 성공 여부를 Oracle DB(maintenance_history)에 업데이트.")
        return {"next_step": "end"}

    # 노드 등록
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("main_node", main_node)
    workflow.add_node("engineer_node", engineer_node)
    workflow.add_node("maintenance_node", maintenance_node)
    workflow.add_node("db_update_node", db_update_node)
    
    # 엣지 연결 (순차적 흐름)
    workflow.set_entry_point("rag_node")
    workflow.add_edge("rag_node", "main_node")
    workflow.add_edge("main_node", "engineer_node")
    workflow.add_edge("engineer_node", "maintenance_node")
    workflow.add_edge("maintenance_node", "db_update_node")
    workflow.add_edge("db_update_node", END)
    
    # 컴파일
    app = workflow.compile()
    return app
