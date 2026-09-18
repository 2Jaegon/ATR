"""
ATR Oculus 대시보드 서버
========================
- State-Aware 동적 앙상블 탐지 에이전트 연동
- WebSocket을 통한 실시간 센서 + 문맥 데이터 스트리밍
- 동적 가중치 및 POT 임계값 정보를 클라이언트에 전달
"""
import asyncio
import json
import os
import random
import time
import uuid
import io
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

app = FastAPI()

# 정적 파일 서빙 (명시적 /static 경로)
app.mount("/static", StaticFiles(directory="static"), name="static_explicit")

# 업로드 디렉토리
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("[Server] ATR Oculus 대시보드 서버 초기화 중...")


class DownloadReportsRequest(BaseModel):
    report_ids: list[str]

@app.get("/api/action-reports")
async def get_action_reports():
    return JSONResponse({"reports": reports_history})

@app.post("/api/download-action-reports")
async def download_selected_action_reports(req: DownloadReportsRequest):
    has_font = register_korean_font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    font_name = 'KoreanFont' if has_font else 'Helvetica'

    selected = [r for r in reports_history if r.get("id") in req.report_ids]
    if not selected:
        return JSONResponse({"error": "No reports selected"}, status_code=400)

    for idx, report in enumerate(selected):
        if idx > 0:
            c.showPage()
        
        c.setFillColor(colors.HexColor('#212529'))
        c.rect(0, height - 60, width, 60, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(font_name, 15)
        c.drawString(40, height - 38, 'ATR Oculus - 반도체 플라즈마 에칭 종합 액션 리포트')

        c.setFillColor(colors.HexColor('#495057'))
        c.setFont(font_name, 9)
        ts = report.get("timestamp", "")
        status_val = report.get("status", "nominal")
        status_kr = "위험 (Critical)" if status_val == "red" else ("주의 (Warning)" if status_val == "yellow" else "정상 가동 (Nominal)")
        c.drawString(40, height - 85, f'발행 일시: {ts}   |   상태: {status_kr}   |   지속 시간: {report.get("duration", 0)}s')

        c.setStrokeColor(colors.HexColor('#DEE2E6'))
        c.setLineWidth(1)
        c.line(40, height - 95, width - 40, height - 95)

        y = height - 125
        c.setFont(font_name, 12)
        c.setFillColor(colors.HexColor('#212529'))
        c.drawString(40, y, '1. 설비 진단 개요 (Diagnosis Query)')
        y -= 20
        c.setFont(font_name, 9)
        c.setFillColor(colors.HexColor('#495057'))
        c.drawString(50, y, f'• 분석 쿼리: {report.get("query", "")}')
        y -= 16
        c.drawString(50, y, f'• 종합 평균 이상 점수: {report.get("score", 0.0):.4f}')
        y -= 30

        c.setFont(font_name, 12)
        c.setFillColor(colors.HexColor('#212529'))
        c.drawString(40, y, '2. AI 수석 엔지니어 분석 보고서 (Qwen LLM Analysis)')
        y -= 20
        c.setFont(font_name, 9)
        c.setFillColor(colors.HexColor('#495057'))
        ai_text = report.get("ai_report", "")
        for line in ai_text.split('\n'):
            while len(line) > 60:
                c.drawString(50, y, line[:60])
                line = line[60:]
                y -= 15
                if y < 50:
                    c.showPage()
                    c.setFont(font_name, 9)
                    y = height - 50
            if line.strip():
                c.drawString(50, y, line)
                y -= 15
                if y < 50:
                    c.showPage()
                    c.setFont(font_name, 9)
                    y = height - 50

    c.save()
    buf.seek(0)
    out_name = f"ATR_Merged_Reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        buf, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )

# ============================================================
# 라우팅: 사용자(Operator) 화면 / 관리자(Admin) 화면
# ============================================================

@app.get("/")
async def get_operator():
    """사용자(현장 엔지니어) 화면"""
    return FileResponse("static/operator.html")

@app.get("/admin")
async def get_admin():
    """관리자(AI/데이터 엔지니어) 화면"""
    return FileResponse("static/admin.html")

@app.get("/studio")
async def get_studio():
    """Advanced Sensor 스타일 다중 플롯 센서 분석 스튜디오"""
    return FileResponse("static/sensor_studio.html")

# ============================================================
# REST API: 정비 리포트 업로드 / 다운로드
# ============================================================

@app.post("/api/upload-report")
async def upload_report(file: UploadFile = File(...)):
    """정비 리포트 파일 업로드"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    print(f"[Server] 정비 리포트 업로드 완료: {safe_name} ({len(contents)} bytes)")
    return JSONResponse({
        "status": "success",
        "filename": safe_name,
        "size_bytes": len(contents)
    })

@app.get("/api/reports")
async def list_reports():
    """업로드된 리포트 목록 조회"""
    files = []
    if os.path.exists(UPLOAD_DIR):
        for f in sorted(os.listdir(UPLOAD_DIR), reverse=True):
            filepath = os.path.join(UPLOAD_DIR, f)
            files.append({
                "filename": f,
                "size_bytes": os.path.getsize(filepath),
                "uploaded_at": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
    return JSONResponse({"reports": files})

@app.get("/api/download-report/{filename}")
async def download_report(filename: str):
    """업로드된 과거 정비 리포트 파일 다운로드"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename)
    return JSONResponse({"error": "파일을 찾을 수 없습니다."}, status_code=404)

# 최신 액션 리포트 메모리 캐시 (PDF 출력용)
reports_history = []

latest_action_report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "status": "nominal",
    "score": 0.0,
    "recipe_step": 1,
    "query": "플라즈마 에칭 챔버 MFC-3 유량 드리프트 및 RF 정합 매칭 네트워크 진단",
    "sop_list": [
        ("SOP-SEC-72", "RF 매칭 네트워크 정전용량 편차 시 C14 가변 캐패시터 유전체 열화 점검 및 CF4 반응 가스 잔류물 세정 절차"),
        ("SOP-SEC-19", "웨이퍼 교체 시 배치 인터체인지 동안 15분 N2 고속 퍼지 실행 규정")
    ],
    "actions": [
        "차기 웨이퍼(#W-8943) 투입 전 퍼지 사이클 3분 자동 수행 버튼 인가.",
        "오후 16:00 정기 점검 시 하부 가스 포트 필터 차압계 육안 확인.",
        "긴급 라인 중단 불필요 (현재 안전 한계 82% 여유 보유)."
    ],
    "ai_report": None
}

font_registered = False
def register_korean_font():
    global font_registered
    if font_registered:
        return True
    font_candidates = [
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/gulim.ttc',
        'C:/Windows/Fonts/batang.ttc'
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('KoreanFont', fp))
                font_registered = True
                return True
            except Exception as e:
                print(f"[PDF] Font error with {fp}: {e}")
    return False

@app.get("/api/download-action-report")
async def download_action_report():
    """현재 화면의 액션 리포트(SOP 지침 + AI 분석 결과)를 전문 산업용 PDF로 생성하여 즉시 다운로드"""
    has_font = register_korean_font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    font_name = 'KoreanFont' if has_font else 'Helvetica'

    # 상단 헤더 배너
    c.setFillColor(colors.HexColor('#212529'))
    c.rect(0, height - 60, width, 60, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(font_name, 15)
    c.drawString(40, height - 38, 'ATR Oculus - 반도체 플라즈마 에칭 설비 액션 리포트')

    # 메타데이터 정보
    c.setFillColor(colors.HexColor('#495057'))
    c.setFont(font_name, 9)
    ts = latest_action_report.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    status_val = latest_action_report.get("status", "nominal")
    status_kr = "위험 (Critical)" if status_val == "red" else ("주의 (Warning)" if status_val == "yellow" else "정상 가동 (Nominal)")
    c.drawString(40, height - 85, f'발행 일시: {ts}   |   설비 노드: Fab 04 Plasma Etch (Node 84-Alpha)   |   상태: {status_kr}')

    c.setStrokeColor(colors.HexColor('#DEE2E6'))
    c.setLineWidth(1)
    c.line(40, height - 95, width - 40, height - 95)

    y = height - 125

    # 1. 설비 진단 개요
    c.setFont(font_name, 12)
    c.setFillColor(colors.HexColor('#212529'))
    c.drawString(40, y, '1. 설비 진단 개요 (Diagnosis Query)')
    y -= 20
    c.setFont(font_name, 9)
    c.setFillColor(colors.HexColor('#495057'))
    query_text = latest_action_report.get("query", "")
    c.drawString(50, y, f'• 분석 쿼리: {query_text}')
    y -= 16
    score = latest_action_report.get("score", 0.0)
    c.drawString(50, y, f'• 종합 이상 점수: {score:.4f} (공정 스텝: Step {latest_action_report.get("recipe_step", 1)})')
    y -= 30

    # 2. 관련 표준운영절차(SOP)
    c.setFont(font_name, 12)
    c.setFillColor(colors.HexColor('#212529'))
    c.drawString(40, y, '2. 관련 표준운영절차 지침 (Standard Operating Procedures)')
    y -= 20
    c.setFont(font_name, 9)
    c.setFillColor(colors.HexColor('#495057'))
    for sop_id, desc in latest_action_report.get("sop_list", []):
        c.drawString(50, y, f'• [{sop_id}] {desc[:60]}')
        y -= 15
        if len(desc) > 60:
            c.drawString(65, y, desc[60:120])
            y -= 15
    y -= 15

    # 3. 권장 정비 지침
    c.setFont(font_name, 12)
    c.setFillColor(colors.HexColor('#212529'))
    c.drawString(40, y, '3. 현장 엔지니어 권장 조치 지침 (Action Guide)')
    y -= 20
    c.setFont(font_name, 9)
    c.setFillColor(colors.HexColor('#495057'))
    for idx, act in enumerate(latest_action_report.get("actions", []), 1):
        c.drawString(50, y, f'{idx}. {act}')
        y -= 18
    y -= 15

    # 4. AI 수석 엔지니어 분석 보고서
    c.setFont(font_name, 12)
    c.setFillColor(colors.HexColor('#212529'))
    c.drawString(40, y, '4. AI 수석 엔지니어 분석 보고서 (Qwen LLM Analysis)')
    y -= 20
    c.setFont(font_name, 9)
    c.setFillColor(colors.HexColor('#495057'))
    ai_text = latest_action_report.get("ai_report")
    if not ai_text:
        ai_text = "현재 공정은 정상 범위 내에서 가동 중입니다. 이상 징후 발생 시 Qwen 경량 모델의 심층 분석 리포트가 수록됩니다."
    
    for line in ai_text.split('\n'):
        while len(line) > 60:
            c.drawString(50, y, line[:60])
            line = line[60:]
            y -= 15
            if y < 50:
                c.showPage()
                c.setFont(font_name, 9)
                y = height - 50
        if line.strip():
            c.drawString(50, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                c.setFont(font_name, 9)
                y = height - 50

    # 푸터
    c.setFont(font_name, 8)
    c.setFillColor(colors.HexColor('#ADB5BD'))
    c.drawString(40, 30, 'CONFIDENTIAL — ATR Oculus Fab 04 Plasma Etching Automated Maintenance Report')

    c.showPage()
    c.save()
    buf.seek(0)
    
    out_name = f"ATR_Action_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        buf, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )

# ============================================================


class DownloadReportsRequest(BaseModel):
    report_ids: list[str]

@app.get("/api/action-reports")
async def get_action_reports():
    return JSONResponse({"reports": reports_history})

@app.post("/api/download-action-reports")
async def download_selected_action_reports(req: DownloadReportsRequest):
    has_font = register_korean_font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    font_name = 'KoreanFont' if has_font else 'Helvetica'

    selected = [r for r in reports_history if r.get("id") in req.report_ids]
    if not selected:
        return JSONResponse({"error": "No reports selected"}, status_code=400)

    for idx, report in enumerate(selected):
        if idx > 0:
            c.showPage()
        
        c.setFillColor(colors.HexColor('#212529'))
        c.rect(0, height - 60, width, 60, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(font_name, 15)
        c.drawString(40, height - 38, 'ATR Oculus - 반도체 플라즈마 에칭 종합 액션 리포트')

        c.setFillColor(colors.HexColor('#495057'))
        c.setFont(font_name, 9)
        ts = report.get("timestamp", "")
        status_val = report.get("status", "nominal")
        status_kr = "위험 (Critical)" if status_val == "red" else ("주의 (Warning)" if status_val == "yellow" else "정상 가동 (Nominal)")
        c.drawString(40, height - 85, f'발행 일시: {ts}   |   상태: {status_kr}   |   지속 시간: {report.get("duration", 0)}s')

        c.setStrokeColor(colors.HexColor('#DEE2E6'))
        c.setLineWidth(1)
        c.line(40, height - 95, width - 40, height - 95)

        y = height - 125
        c.setFont(font_name, 12)
        c.setFillColor(colors.HexColor('#212529'))
        c.drawString(40, y, '1. 설비 진단 개요 (Diagnosis Query)')
        y -= 20
        c.setFont(font_name, 9)
        c.setFillColor(colors.HexColor('#495057'))
        c.drawString(50, y, f'• 분석 쿼리: {report.get("query", "")}')
        y -= 16
        c.drawString(50, y, f'• 종합 평균 이상 점수: {report.get("score", 0.0):.4f}')
        y -= 30

        c.setFont(font_name, 12)
        c.setFillColor(colors.HexColor('#212529'))
        c.drawString(40, y, '2. AI 수석 엔지니어 분석 보고서 (Qwen LLM Analysis)')
        y -= 20
        c.setFont(font_name, 9)
        c.setFillColor(colors.HexColor('#495057'))
        ai_text = report.get("ai_report", "")
        for line in ai_text.split('\n'):
            while len(line) > 60:
                c.drawString(50, y, line[:60])
                line = line[60:]
                y -= 15
                if y < 50:
                    c.showPage()
                    c.setFont(font_name, 9)
                    y = height - 50
            if line.strip():
                c.drawString(50, y, line)
                y -= 15
                if y < 50:
                    c.showPage()
                    c.setFont(font_name, 9)
                    y = height - 50

    c.save()
    buf.seek(0)
    out_name = f"ATR_Merged_Reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        buf, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )

# ============================================================
# REST API: RAG 기반 Chat 서비스
# ============================================================
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """사용자 질문에 대해 현재 설비 상태(최신 Action Report)를 기반으로 답변"""
    try:
        # qwen2.5:1.5b 모델 사용 (MainAgent와 동일)
        llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.3)
        
        # 최신 장비 상태(RAG Context) 가져오기
        status_kr = "위험" if latest_action_report.get("status") == "red" else ("주의" if latest_action_report.get("status") == "yellow" else "정상")
        score = latest_action_report.get("score", 0.0)
        ai_report = latest_action_report.get("ai_report", "없음")
        
        system_prompt = (
            "당신은 공장 설비의 센서 데이터를 분석하고 오퍼레이터를 돕는 AI 어시스턴트(ATR Oculus RAG Agent)입니다.\n"
            "현장 엔지니어가 설비 상태나 정비 지침에 대해 질문하면 친절하고 전문적으로 답변하세요.\n\n"
            "[현재 설비 문맥 정보 (Context)]\n"
            f"- 현재 상태: {status_kr} (이상 점수: {score:.3f})\n"
            f"- 최근 AI 분석 리포트 요약: {ai_report}\n"
            "- 당신은 위 정보를 바탕으로 현재 설비의 상황을 인지하고 있어야 합니다."
        )
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=req.message)
        ])
        
        return JSONResponse({"status": "success", "reply": response.content})
    except Exception as e:
        print(f"[Chat API Error] {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# WebSocket & 비동기 큐 디커플링 아키텍처
# ============================================================

from src.agents.detection_agent import DetectionAgent
from src.workflow import compile_workflow
import pandas as pd

# 전역 DetectionAgent 인스턴스
detector = DetectionAgent(seq_len=5)
detector.load_weights()

# 전역 LangGraph 워크플로우 인스턴스 (Human-in-the-loop)
graph_app = compile_workflow()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
telemetry_queue = asyncio.Queue(maxsize=100)

# 팝업 도배 방지용 플래그 및 쿨다운
is_waiting_for_feedback = False
last_llm_time = 0.0
LLM_COOLDOWN = 30.0 # 30초 쿨다운

class FeedbackRequest(BaseModel):
    thread_id: str
    feedback: str

@app.post("/api/feedback")
async def receive_feedback(req: FeedbackRequest):
    """엔지니어 피드백 수신 -> LangGraph 재개(Phase B: 평가 및 DB 저장)"""
    global is_waiting_for_feedback
    config = {"configurable": {"thread_id": req.thread_id}}
    state = graph_app.get_state(config)
    
    if not state or not state.next:
        return JSONResponse({"status": "error", "message": "해당 알람에 대한 대기 중인 워크플로우가 없습니다."})
        
    def _resume():
        # engineer_node의 결과를 외부에서 주입
        graph_app.update_state(config, {"engineer_feedback": req.feedback}, as_node="engineer_node")
        # 나머지 노드(maintenance_node -> db_update_node) 실행
        graph_app.invoke(None, config)
        
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _resume)
    
    is_waiting_for_feedback = False # 피드백 완료 후 팝업 제한 해제
    return JSONResponse({"status": "success", "message": "피드백이 성공적으로 AI에 반영되어 평가/저장되었습니다."})

async def run_llm_agent(anomaly_data, thread_id, status, final_score, duration_ticks=0):
    loop = asyncio.get_event_loop()
    config = {"configurable": {"thread_id": thread_id}}
    
    msg = "장비 이상 징후 감지. AI가 원인 분석을 시작합니다..." if status == 'yellow' else "임계치 초과(Fault) 감지! AI가 즉각 원인 분석을 시작합니다..."
    await manager.broadcast_json({
        "type": "report",
        "status": status,
        "message": f"[{'경고' if status=='yellow' else '긴급'}] {msg} (점수: {final_score:.3f})"
    })
    
    def _run():
        graph_app.invoke({"anomaly_data": anomaly_data}, config)
        state = graph_app.get_state(config)
        return state.values.get("action_report", "LLM 분석 완료")
        
    action_report = await loop.run_in_executor(None, _run)
    
    # 캐시 갱신 및 히스토리 저장
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_action_report["ai_report"] = action_report
    latest_action_report["status"] = status
    latest_action_report["score"] = final_score
    latest_action_report["timestamp"] = ts
    
    new_report = {
        "id": thread_id,
        "timestamp": ts,
        "status": status,
        "score": final_score,
        "query": latest_action_report["query"],
        "ai_report": action_report,
        "duration": int(duration_ticks * 0.5) # 2Hz
    }
    reports_history.append(new_report)


    await manager.broadcast_json({
        "type": "llm_report",
        "thread_id": thread_id,
        "status": status,
        "report": action_report,
        "report_id": thread_id,
        "duration": int(duration_ticks * 0.5)
    })

async def producer():
    """백그라운드에서 데이터를 큐에 적재 (데이터 수집 디커플링) - 다중 파일 순차 재생"""
    import glob
    test_dir = "data/phm_data_challenge_2018/test"
    
    # Get all csv files, excluding temp files
    csv_files = sorted([f for f in glob.glob(os.path.join(test_dir, "*.csv")) if not os.path.basename(f).startswith("~$")])
    
    if not csv_files:
        print(f"[Producer] Error: No valid CSV files found in {test_dir}.")
        return
        
    while True:
        for csv_path in csv_files:
            print(f"[Producer] Starting stream for {os.path.basename(csv_path)}...")
            chunk_iter = pd.read_csv(csv_path, chunksize=5)
            for chunk in chunk_iter:
                chunk = chunk.ffill().bfill()
                await telemetry_queue.put(chunk)
                await asyncio.sleep(0.5)
            print(f"[Producer] Finished {os.path.basename(csv_path)}. Waiting 3 seconds before next file...")
            await asyncio.sleep(3)

async def consumer():
    """큐에서 데이터를 꺼내 추론 후 모든 클라이언트에게 브로드캐스트"""
    while True:
        chunk = await telemetry_queue.get()
        result = detector.detect(chunk)
        
        raw_sensors = chunk.iloc[-1].to_dict()
        base_pressure = raw_sensors.get('FLOWCOOLPRESSURE', 0)
        
        message = {
            "type": "telemetry",
            "status": result["status"],
            "final_score": result["final_score"],
            "scores": result["scores"],
            "sensor_value": float(base_pressure),
            "raw_sensors": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) 
                            for k, v in raw_sensors.items()},
            "dynamic_weights": result["weights"],
            "recipe_step": result["recipe_step"],
            "thresholds": result["thresholds"]
        }
        
        await manager.broadcast_json(message)
        

        # --- Aggregation Logic ---
        global anomaly_buffer, is_anomaly_active
        if 'anomaly_buffer' not in globals():
            globals()['anomaly_buffer'] = []
            globals()['is_anomaly_active'] = False

        if result["status"] in ['yellow', 'red']:
            is_anomaly_active = True
            anomaly_buffer.append(result)
            
            # If buffer gets too large (e.g. 30 ticks = 15s), force generate
            if len(anomaly_buffer) >= 30:
                thread_id = str(uuid.uuid4())
                # Aggregate data
                avg_score = sum(r['final_score'] for r in anomaly_buffer) / len(anomaly_buffer)
                anomaly_data = {
                    "temp_mean": float(chunk['ETCHBEAMCURRENT'].mean()) if 'ETCHBEAMCURRENT' in chunk.columns else 0.0,
                    "vib_mean": float(chunk['FLOWCOOLPRESSURE'].mean()) if 'FLOWCOOLPRESSURE' in chunk.columns else 0.0,
                    "status": result["status"],
                    "duration_ticks": len(anomaly_buffer)
                }
                asyncio.create_task(run_llm_agent(anomaly_data, thread_id, result["status"], avg_score, len(anomaly_buffer)))
                anomaly_buffer = []
                is_anomaly_active = False
                
        else:
            # Transition to green
            if is_anomaly_active and len(anomaly_buffer) > 0:
                thread_id = str(uuid.uuid4())
                avg_score = sum(r['final_score'] for r in anomaly_buffer) / len(anomaly_buffer)
                anomaly_data = {
                    "temp_mean": float(chunk['ETCHBEAMCURRENT'].mean()) if 'ETCHBEAMCURRENT' in chunk.columns else 0.0,
                    "vib_mean": float(chunk['FLOWCOOLPRESSURE'].mean()) if 'FLOWCOOLPRESSURE' in chunk.columns else 0.0,
                    "status": "yellow", # summarizing the past anomaly
                    "duration_ticks": len(anomaly_buffer)
                }
                asyncio.create_task(run_llm_agent(anomaly_data, thread_id, "yellow", avg_score, len(anomaly_buffer)))
                anomaly_buffer = []
                is_anomaly_active = False
            
        telemetry_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(producer())
    asyncio.create_task(consumer())

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    await websocket.send_json({
        "type": "init",
        "weights": detector.default_weights,
        "yellow_threshold": detector.yellow_threshold,
        "red_threshold": detector.red_threshold,
        "pot_method": detector.pot_method
    })
    
    try:
        while True:
            # 클라이언트 측의 메시지 대기 (연결 유지)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("[Server] 클라이언트 연결 종료")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
