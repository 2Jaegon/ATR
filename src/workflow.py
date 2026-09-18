from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import pandas as pd

from src.agents.graph_state import AgentState
from src.agents.main_agent import MainAgent
from src.agents.maintenance_agent import MaintenanceAgent
from src.agents.rag_agent import OracleRAGAgent

def compile_workflow():
    # 그래프 초기화
    workflow = StateGraph(AgentState)
    
    # 에이전트 인스턴스화
    rag_agent = OracleRAGAgent()
    main_agent = MainAgent(model_name="qwen2.5:1.5b")
    maintenance_agent = MaintenanceAgent(model_name="qwen2.5:1.5b")
    
    # 1. RAG Node: 이상치 데이터를 받아 과거 이력 검색
    def rag_node(state: AgentState):
        print("--- [Node: RAG Agent] Searching history ---")
        anomaly = state.get("anomaly_data")
        df_anomaly = pd.DataFrame([anomaly]) # dict to DataFrame
        history = rag_agent.search_similar_pattern(df_anomaly)
        return {"retrieved_history": [history], "next_step": "main_agent"}
        
    # 2. Main Node: 검색된 이력을 바탕으로 Action Report 생성
    def main_node(state: AgentState):
        return main_agent.generate_action_report(state)
        
    # 3. Engineer Node: (Human-in-the-loop 대기 지점)
    def engineer_node(state: AgentState):
        print("--- [Node: Field Engineer] (This node should be skipped via update_state) ---")
        return {"next_step": "maintenance_agent"}
        
    # 4. Maintenance Node: 피드백 평가
    def maintenance_node(state: AgentState):
        return maintenance_agent.evaluate_feedback(state)
        
    # 5. DB Update Node: 최종 평가를 SQLite DB에 저장
    def db_update_node(state: AgentState):
        print("--- [Node: DB Update] Saving evaluation to SQLite DB ---")
        is_false_alarm = state.get("is_false_alarm")
        feedback = state.get("engineer_feedback")
        action_report = state.get("action_report")
        
        try:
            conn = sqlite3.connect("factory_logs.db")
            cursor = conn.cursor()
            
            # 최소한의 정보만 로깅 (실제 환경에서는 17개 센서값도 로깅)
            sql = """
            INSERT INTO maintenance_history (pattern_desc, action_taken, cause, is_false_alarm)
            VALUES (?, ?, ?, ?)
            """
            cursor.execute(sql, ("Anomaly Detected", feedback, action_report, int(is_false_alarm)))
            conn.commit()
            
            if is_false_alarm:
                print(f"Action: '{feedback}' -> 가성 불량(False Alarm)으로 라벨링하여 저장.")
            else:
                print(f"Action: '{feedback}' -> 실제 정비 이력 및 성공 여부 저장.")
        except Exception as e:
            print(f"DB Insert Failed: {e}")
        finally:
            conn.close()
            
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
    
    # 체크포인터(SQLite)를 붙여서 Human-in-the-loop 상태 영속화
    # SqliteSaver.from_conn_string은 Checkpointer를 생성하고 관련 테이블도 자동 생성함
    conn_string = "factory_logs.db"
    # conn = sqlite3.connect(conn_string, check_same_thread=False)
    # memory = SqliteSaver(conn) 
    # v3부터는 connection 직접 넘기기가 권장됨. FastAPI 백그라운드 스레드 고려:
    import sqlite3
    conn = sqlite3.connect(conn_string, check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory, interrupt_before=["engineer_node"])
    return app

