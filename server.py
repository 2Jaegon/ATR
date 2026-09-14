import asyncio
import json
import os
import random
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

app = FastAPI()

# 정적 파일 서빙 (명시적 /static 경로)
app.mount("/static", StaticFiles(directory="static"), name="static_explicit")

# 업로드 디렉토리
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("[Server] ATR Oculus 대시보드 서버 초기화 중...")

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
    """리포트 파일 다운로드"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename)
    return JSONResponse({"error": "파일을 찾을 수 없습니다."}, status_code=404)

# ============================================================
# WebSocket: 실시간 텔레메트리 스트리밍
# ============================================================

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 초기 설정 전송
    await websocket.send_json({
        "type": "init",
        "weights": {"TR": 0.40, "GNN": 0.35, "LSTM": 0.25},
        "yellow_threshold": 0.6,
        "red_threshold": 0.8
    })
    
    print("[Server] 클라이언트 연결! 스트리밍 시작...")
    try:
        base_pressure = 50.0
        
        while True:
            base_pressure += random.uniform(-0.5, 0.5)
            
            tr_score = random.uniform(0.1, 0.4)
            gnn_score = random.uniform(0.1, 0.4)
            lstm_score = random.uniform(0.1, 0.4)
            
            # 이벤트 시뮬레이션
            chance = random.random()
            if chance < 0.03:
                tr_score += 0.6; gnn_score += 0.5; lstm_score += 0.6
                base_pressure += random.uniform(10.0, 15.0)
            elif chance < 0.1:
                tr_score += 0.3; gnn_score += 0.3; lstm_score += 0.4
                base_pressure -= random.uniform(5.0, 8.0)
                
            final_score = 0.4*tr_score + 0.35*gnn_score + 0.25*lstm_score
            
            status = "normal"
            if final_score > 0.8:
                status = "red"
            elif final_score > 0.6:
                status = "yellow"
                
            await websocket.send_json({
                "type": "telemetry",
                "status": status,
                "final_score": final_score,
                "scores": {"TR": tr_score, "GNN": gnn_score, "LSTM": lstm_score},
                "sensor_value": float(base_pressure)
            })
            
            if status == 'yellow':
                await websocket.send_json({
                    "type": "report",
                    "status": "yellow",
                    "message": "[경고] Flowcool Pressure 비정상 하강 감지. 과거 이력 분석: 밸브 미세 누수 의심. SOP-PMT-12 절차에 따라 밸브 점검을 권고합니다."
                })
            elif status == 'red':
                await websocket.send_json({
                    "type": "report",
                    "status": "red",
                    "message": "[긴급] 압력 한계선 돌파! 즉각적인 장비 가동 중단(Interlock) 및 펌프 교체 요망. SOP-EMG-01 비상 대응 절차를 즉시 개시하십시오."
                })
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        print("[Server] 클라이언트 연결 종료")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
