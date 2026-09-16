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

from src.agents.detection_agent import DetectionAgent
import pandas as pd

# 전역 DetectionAgent 인스턴스 (State-Aware 동적 앙상블)
detector = DetectionAgent(seq_len=5)
detector.load_weights()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 초기 설정 전송 (POT 임계값 + 기본 가중치 포함)
    await websocket.send_json({
        "type": "init",
        "weights": detector.default_weights,
        "yellow_threshold": detector.yellow_threshold,
        "red_threshold": detector.red_threshold,
        "pot_method": detector.pot_method
    })

    print("[Server] 클라이언트 연결! 실제 데이터 스트리밍 시작...")
    try:
        # 실제 데이터 스트리밍 (05_M02_DC_train.csv 파일 사용)
        csv_path = "data/phm_data_challenge_2018/train/05_M02_DC_train.csv"
        if not os.path.exists(csv_path):
            print(f"[Server] 오류: {csv_path} 파일을 찾을 수 없습니다.")
            return

        chunk_iter = pd.read_csv(csv_path, chunksize=5)

        for chunk in chunk_iter:
            chunk = chunk.ffill().bfill()

            # State-Aware DetectionAgent를 통해 추론
            # (문맥 변수 포함된 DataFrame 그대로 전달)
            result = detector.detect(chunk)

            raw_sensors = chunk.iloc[-1].to_dict()
            base_pressure = raw_sensors.get('FLOWCOOLPRESSURE', 0)

            await websocket.send_json({
                "type": "telemetry",
                "status": result["status"],
                "final_score": result["final_score"],
                "scores": result["scores"],
                "sensor_value": float(base_pressure),
                "raw_sensors": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) 
                                for k, v in raw_sensors.items()},
                # 동적 가중치 & 문맥 정보 (대시보드 표시용)
                "dynamic_weights": result["weights"],
                "recipe_step": result["recipe_step"],
                "thresholds": result["thresholds"]
            })

            if result["status"] == 'yellow':
                await websocket.send_json({
                    "type": "report",
                    "status": "yellow",
                    "message": f"[경고] 장비 센서 이상 징후 감지. 최종 점수: {result['final_score']:.3f}. 주의가 필요합니다."
                })
            elif result["status"] == 'red':
                await websocket.send_json({
                    "type": "report",
                    "status": "red",
                    "message": f"[긴급] 임계치 초과(Fault) 감지! 최종 점수: {result['final_score']:.3f}. 즉각 정비 요망."
                })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("[Server] 클라이언트 연결 종료")

# numpy 타입 JSON 직렬화 지원
import numpy as np

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
