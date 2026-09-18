// --- ATR Oculus: Operator Dashboard JS ---

// 1. Lucide Icons
lucide.createIcons();

// (Dark mode logic removed for industrial minimalist theme)


// 3. Sensor Canvas
const canvas = document.getElementById('sensorChart');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
 const rect = canvas.getBoundingClientRect();
 const dpr = window.devicePixelRatio || 1;
 canvas.width = rect.width * dpr;
 canvas.height = rect.height * dpr;
 ctx.scale(dpr, dpr);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

let step = 0;
const pointCount = 120;

const SENSOR_FEATURES = [
 'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
 'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
 'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
 'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
 'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
 'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
];

// 공정 센서별 고유 공학적 스팬 (Industrial Calibration Spans)
const SENSOR_SPANS = {
 'IONGAUGEPRESSURE': { min: -0.15, max: 0.15 },
 'ETCHBEAMVOLTAGE': { min: -1.20, max: 1.80 },
 'ETCHBEAMCURRENT': { min: -1.20, max: 2.00 },
 'ETCHSUPPRESSORVOLTAGE': { min: -1.20, max: 1.80 },
 'ETCHSUPPRESSORCURRENT': { min: -1.20, max: 2.10 },
 'FLOWCOOLFLOWRATE': { min: -3.00, max: 1.20 },
 'FLOWCOOLPRESSURE': { min: -2.50, max: 0.80 },
 'ETCHGASCHANNEL1READBACK': { min: -2.00, max: 2.80 },
 'ETCHPBNGASREADBACK': { min: -3.00, max: 1.20 },
 'FIXTURETILTANGLE': { min: -1.80, max: 2.20 },
 'ROTATIONSPEED': { min: -0.05, max: 0.05 },
 'ACTUALROTATIONANGLE': { min: -0.20, max: 0.05 },
 'FIXTURESHUTTERPOSITION': { min: -0.20, max: 3.20 },
 'ETCHSOURCEUSAGE': { min: -0.25, max: -0.05 },
 'ETCHAUXSOURCETIMER': { min: -0.05, max: 0.08 },
 'ETCHAUX2SOURCETIMER': { min: 0.10, max: 0.30 },
 'ACTUALSTEPDURATION': { min: -1.00, max: 4.50 }
};

const SENSOR_COLORS = [
 '#DC2626', '#EA580C', '#D97706', '#CA8A04', '#65A30D',
 '#16A34A', '#059669', '#0D9488', '#0891B2', '#0284C7',
 '#2563EB', '#4F46E5', '#7C3AED', '#9333EA', '#C026D3',
 '#DB2777', '#4B5563'
];

const sensorData = {};
const sensorVisible = {};
const defaultVisible = [
 'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
 'FLOWCOOLFLOWRATE', 'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 
 'ETCHPBNGASREADBACK', 'FIXTURETILTANGLE'
];

// 초기 상태: 빈 배열로 시작 (1초부터 차근히 데이터가 누적되어 나옴)
function resetSensorBuffers() {
 SENSOR_FEATURES.forEach((feat, idx) => {
 sensorData[feat] = {
 color: SENSOR_COLORS[idx],
 values: [], // 1초부터 차근히 누적
 latestVal: 0
 };
 if (sensorVisible[feat] === undefined) {
 sensorVisible[feat] = defaultVisible.includes(feat);
 }
 });
 const windowStatus = document.getElementById('chart-window-status');
 if (windowStatus) windowStatus.textContent = '관측 시간: 0s / 60s';
 drawChart();
}

resetSensorBuffers();

// 동적 토글 버튼 및 실시간 수치 표시 초기화
function initLegend() {
 const legendContainer = document.getElementById('sensor-legend-container');
 if (!legendContainer) return;
 legendContainer.innerHTML = '';

 SENSOR_FEATURES.forEach(feat => {
 const btn = document.createElement('button');
 btn.id = `btn-sensor-${feat}`;

 const updateBtn = () => {
 const isVis = sensorVisible[feat];
 const valStr = sensorData[feat].latestVal !== undefined ? Number(sensorData[feat].latestVal).toFixed(2) : '-';
 btn.className = `px-1.5 py-0.5 text-[9px] font-mono rounded border transition-all flex items-center space-x-1 ${
 isVis ? 'bg-gray-100 border-gray-400 text-gray-900 font-bold shadow-sm' : 'bg-white border-gray-200 text-gray-400 opacity-60 hover:opacity-100'
 }`;
 btn.innerHTML = `<span class="w-1.5 h-1.5 rounded-full inline-block flex-shrink-0" style="background-color: ${isVis ? sensorData[feat].color : '#D1D5DB'}"></span><span class="font-sans">${feat.substring(0, 8)}:</span><span>${valStr}</span>`;
 };
 updateBtn();

 btn.onclick = () => {
 sensorVisible[feat] = !sensorVisible[feat];
 updateBtn();
 drawChart();
 };
 legendContainer.appendChild(btn);
 });
}

if (document.readyState === 'loading') {
 document.addEventListener('DOMContentLoaded', () => {
 initLegend();
 lucide.createIcons();
 });
} else {
 initLegend();
 lucide.createIcons();
}

function drawChart() {
 const w = canvas.getBoundingClientRect().width;
 const h = canvas.getBoundingClientRect().height;
 ctx.clearRect(0, 0, w, h);

 // Grid lines
 ctx.strokeStyle = 'rgba(0, 0, 0, 0.04)';
 ctx.lineWidth = 1;
 for (let y = 20; y < h; y += 40) {
 ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
 }
 for (let x = 0; x < w; x += 60) {
 ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
 }

 // 활성화된 센서 렌더링
 SENSOR_FEATURES.forEach((feat) => {
 if (!sensorVisible[feat]) return;

 const data = sensorData[feat].values;
 if (!data || data.length === 0) return; // 아직 데이터가 없으면 렌더링 안 함

 const span = SENSOR_SPANS[feat] || { min: -2.0, max: 2.0 };

 // 안전 마진: 데이터가 사전에 정의된 스팬을 벗어나면 유연하게 확장
 let sMin = span.min;
 let sMax = span.max;
 for (let i = 0; i < data.length; i++) {
 const v = data[i];
 if (typeof v === 'number' && !isNaN(v)) {
 if (v < sMin) sMin = v;
 if (v > sMax) sMax = v;
 }
 }

 let spanRange = sMax - sMin;
 if (spanRange <= 1e-4) spanRange = 1.0;

 const stepX = w / (pointCount - 1);

 // 데이터가 1개만 있을 때는 첫 위치(x=0)에 점(Dot)을 표시
 if (data.length === 1) {
 const val = data[0];
 let norm = (val - sMin) / spanRange;
 norm = Math.max(0.02, Math.min(0.98, norm));
 const y = h - (norm * (h * 0.84) + h * 0.08);

 ctx.fillStyle = sensorData[feat].color;
 ctx.beginPath();
 ctx.arc(0, y, 3, 0, Math.PI * 2);
 ctx.fill();
 return;
 }

 // 2개 이상일 때 선으로 연결하여 1초부터 점진적으로 차오름
 ctx.strokeStyle = sensorData[feat].color;
 ctx.lineWidth = 1.6;
 ctx.lineJoin = 'round';
 ctx.lineCap = 'round';
 ctx.beginPath();

 let started = false;

 for (let i = 0; i < data.length; i++) {
 const val = data[i];
 if (typeof val !== 'number' || isNaN(val)) continue;

 const x = i * stepX;

 let norm = (val - sMin) / spanRange;
 norm = Math.max(0.02, Math.min(0.98, norm));
 const y = h - (norm * (h * 0.84) + h * 0.08);

 if (!started) {
 ctx.moveTo(x, y);
 started = true;
 } else {
 ctx.lineTo(x, y);
 }
 }
 ctx.stroke();
 });
}

// 4. DOM Elements
const globalStatus = document.getElementById('global-status-indicator');
const alertLog = document.getElementById('alert-log');
const ragConsole = document.getElementById('rag-console');
const downloadBtn = document.getElementById('download-report-btn');
const fileInput = document.getElementById('file-upload-input');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadBtn = document.getElementById('upload-btn');
const chartWindowStatus = document.getElementById('chart-window-status');

// 5. WebSocket Connection
const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);

ws.onopen = () => {
 addAlertLog('시스템 연결 완료', 'ATR Oculus 실시간 스트림이 연결되었습니다.', 'normal');
};

ws.onmessage = (event) => {
 const msg = JSON.parse(event.data);

 if (msg.type === 'telemetry') {
 const raw = msg.raw_sensors || {};

 // 1초부터 차근히 1개씩 푸시 (초기 버퍼 임의 채움 없음)
 let currentLen = 0;
 SENSOR_FEATURES.forEach(feat => {
 let v = Number(raw[feat]);
 if (isNaN(v)) v = sensorData[feat].latestVal || 0;

 sensorData[feat].values.push(v);
 if (sensorData[feat].values.length > pointCount) {
 sensorData[feat].values.shift(); // 60초(120포인트) 도달 후 연속 스크롤
 }
 sensorData[feat].latestVal = v;
 currentLen = sensorData[feat].values.length;
 });

 // 관측 윈도우 경과 시간 동적 갱신
 if (chartWindowStatus) {
 if (currentLen < pointCount) {
 const elapsedSec = Math.max(1, Math.round(currentLen * 0.5));
 chartWindowStatus.textContent = `관측 시간: ${elapsedSec}s / 60s`;
 } else {
 chartWindowStatus.textContent = `관측 윈도우: T-60s → T-0s (연속 스트리밍)`;
 }
 }

 // 범례 버튼에 최신 실시간 수치 갱신
 SENSOR_FEATURES.forEach(feat => {
 const btn = document.getElementById(`btn-sensor-${feat}`);
 if (btn) {
 const isVis = sensorVisible[feat];
 const valStr = sensorData[feat].latestVal.toFixed(2);
 btn.innerHTML = `<span class="w-1.5 h-1.5 rounded-full inline-block flex-shrink-0" style="background-color: ${isVis ? sensorData[feat].color : '#D1D5DB'}"></span><span class="font-sans">${feat.substring(0, 8)}:</span><span>${valStr}</span>`;
 }
 });

 drawChart();

 // 상단 KPI 카드 실시간 갱신
 const kpiVacuum = document.getElementById('kpi-vacuum');
 if (kpiVacuum && raw['IONGAUGEPRESSURE'] !== undefined) {
 kpiVacuum.innerHTML = `${Number(raw['IONGAUGEPRESSURE']).toFixed(3)} <span class="text-xs font-normal text-gray-400 font-sans">Torr</span>`;
 }
 const kpiBeamV = document.getElementById('kpi-beam-v');
 if (kpiBeamV && raw['ETCHBEAMVOLTAGE'] !== undefined) {
 kpiBeamV.innerHTML = `${Number(raw['ETCHBEAMVOLTAGE']).toFixed(2)} <span class="text-xs font-normal text-gray-400 font-sans">V</span>`;
 }
 const kpiBeamI = document.getElementById('kpi-beam-i');
 if (kpiBeamI && raw['ETCHBEAMCURRENT'] !== undefined) {
 kpiBeamI.innerHTML = `${Number(raw['ETCHBEAMCURRENT']).toFixed(2)} <span class="text-xs font-normal text-gray-400 font-sans">mA</span>`;
 }
 const kpiFlowCool = document.getElementById('kpi-flowcool');
 if (kpiFlowCool && raw['FLOWCOOLPRESSURE'] !== undefined) {
 kpiFlowCool.innerHTML = `${Number(raw['FLOWCOOLPRESSURE']).toFixed(2)} <span class="text-xs font-normal text-gray-400 font-sans">Torr</span>`;
 }

 // 하단 보조 인디케이터 실시간 갱신
 const subRecipe = document.getElementById('sub-recipe-step');
 if (subRecipe && msg.recipe_step !== undefined) {
 subRecipe.textContent = `Step ${msg.recipe_step}`;
 }
 const subAnomaly = document.getElementById('sub-anomaly-score');
 if (subAnomaly && msg.final_score !== undefined) {
 subAnomaly.textContent = msg.final_score.toFixed(4);
 subAnomaly.className = `text-sm font-semibold font-mono mt-0.5 ${
 msg.status === 'red' ? 'text-red-600' : msg.status === 'yellow' ? 'text-amber-600' : 'text-gray-800 '
 }`;
 }
 const subTilt = document.getElementById('sub-tilt-angle');
 if (subTilt && raw['FIXTURETILTANGLE'] !== undefined) {
 subTilt.textContent = `${Number(raw['FIXTURETILTANGLE']).toFixed(3)}°`;
 }

 // Update global status
 if (msg.status === 'red') {
 if (globalStatus) {
 globalStatus.textContent = '위험 (Critical)';
 globalStatus.className = 'text-red-500 font-semibold text-xs';
 }
 addAlertLog('[위험 감지]', `최종 이상 점수: ${msg.final_score.toFixed(3)} — 즉각 조치 필요`, 'critical');
 } else if (msg.status === 'yellow') {
 if (globalStatus) {
 globalStatus.textContent = '주의 (Warning)';
 globalStatus.className = 'text-amber-500 font-semibold text-xs';
 }
 addAlertLog('[이상 징후]', `최종 이상 점수: ${msg.final_score.toFixed(3)} — 모니터링 강화`, 'warning');
 } else {
 if (globalStatus) {
 globalStatus.textContent = '정상 가동 (Nominal)';
 globalStatus.className = 'text-emerald-500 font-semibold text-xs';
 }
 }
 }

 if (msg.type === 'report') {
 addRagEntry(msg.message, msg.status);
 }

 if (msg.type === 'llm_report') {
 // 사용자를 방해하는 자동 팝업 모달을 띄우지 않고, 우측 액션 리포트 패널에 직접 반영
 const modal = document.getElementById('llm-modal');
 const content = document.getElementById('llm-report-content');
 
 if (modal && content) {
 content.textContent = msg.report;
 modal.dataset.threadId = msg.thread_id; // 필요 시 엔지니어가 수동으로 피드백 작성할 수 있도록 저장
 }
 
 // 액션 리포트 콘솔에 AI 수석 엔지니어 분석 결과 등록
 addRagEntry(`[Qwen LLM 분석 보고서]\n${msg.report}`, msg.status);
 }
};

// --- LLM Modal Logic ---
const llmModal = document.getElementById('llm-modal');
const closeLlmBtn = document.getElementById('close-llm-modal');
const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
const feedbackInput = document.getElementById('engineer-feedback-input');

function closeLlmModal() {
 if (!llmModal) return;
 
 const threadId = llmModal.dataset.threadId;
 if (threadId) {
 // 백엔드의 is_waiting_for_feedback 플래그를 리셋하기 위해 무시 피드백 전송
 fetch('/api/feedback', {
 method: 'POST',
 headers: { 'Content-Type': 'application/json' },
 body: JSON.stringify({ thread_id: threadId, feedback: "알람 확인 후 무시 (Dismissed)" })
 }).catch(() => {});
 llmModal.dataset.threadId = '';
 }

 llmModal.classList.add('opacity-0');
 setTimeout(() => llmModal.classList.add('hidden'), 300);
}

if (closeLlmBtn) {
 closeLlmBtn.addEventListener('click', closeLlmModal);
}

// 엔지니어가 원할 때 수동으로 피드백을 입력할 수 있도록 열기 버튼 연결
const openFeedbackBtn = document.getElementById('open-feedback-btn');
if (openFeedbackBtn) {
 openFeedbackBtn.addEventListener('click', () => {
 if (llmModal) {
 llmModal.classList.remove('hidden');
 setTimeout(() => llmModal.classList.remove('opacity-0'), 10);
 }
 });
}

if (submitFeedbackBtn) {
 submitFeedbackBtn.addEventListener('click', async () => {
 const feedbackText = feedbackInput.value.trim();
 if (!feedbackText) {
 alert("피드백 내용을 입력해 주세요.");
 return;
 }
 
 const threadId = llmModal.dataset.threadId;
 if (!threadId) return;

 submitFeedbackBtn.disabled = true;
 submitFeedbackBtn.innerHTML = "<span>전송 중...</span>";

 try {
 const res = await fetch('/api/feedback', {
 method: 'POST',
 headers: { 'Content-Type': 'application/json' },
 body: JSON.stringify({ thread_id: threadId, feedback: feedbackText })
 });
 const data = await res.json();
 
 if (data.status === 'success') {
 alert("피드백 전송 완료! (가성 불량 여부 판단 후 DB에 저장됩니다)");
 addAlertLog('[피드백 반영]', feedbackText, 'normal');
 feedbackInput.value = "";
 closeLlmModal();
 } else {
 alert("전송 실패: " + data.message);
 }
 } catch (err) {
 alert("서버 연결 오류가 발생했습니다.");
 } finally {
 submitFeedbackBtn.disabled = false;
 submitFeedbackBtn.innerHTML = `<i data-lucide="send" class="w-3.5 h-3.5"></i><span>평가 및 DB 저장</span>`;
 lucide.createIcons();
 }
 });
}

// 6. Alert Log (이상 탐지 로그)
function addAlertLog(title, message, level) {
 if (!alertLog) return;
 const emptyPlaceholder = document.getElementById('alert-log-empty');
 if (emptyPlaceholder) {
 emptyPlaceholder.remove();
 }
 const now = new Date();
 const timeStr = now.toLocaleTimeString('ko-KR');

 let borderColor = 'border-industrial-300';
 let dotColor = 'bg-industrial-400';
 if (level === 'warning') {
 borderColor = 'border-yellow-400';
 dotColor = 'bg-yellow-400';
 } else if (level === 'critical') {
 borderColor = 'border-red-500';
 dotColor = 'bg-red-500';
 }

 const entry = document.createElement('div');
 entry.className = `border-l-4 ${borderColor} bg-industrial-50 pl-3 py-2 text-xs mb-1`;
 entry.innerHTML = `
 <div class="flex items-center gap-2 mb-0.5">
 <span class="w-2 h-2 rounded-none ${dotColor}"></span>
 <span class="font-bold text-industrial-900">${title}</span>
 <span class="ml-auto text-industrial-500 font-mono text-[10px]">${timeStr}</span>
 </div>
 <p class="text-industrial-700 pl-4">${message}</p>
 `;

 alertLog.prepend(entry);
 while (alertLog.children.length > 50) {
 alertLog.removeChild(alertLog.lastChild);
 }
}

// 7. RAG Console (액션 리포트)
function addRagEntry(message, status) {
 if (!ragConsole) return;
 const now = new Date();
 const timeStr = now.toLocaleTimeString('ko-KR');

 let textColor = 'text-industrial-700';
 if (status === 'yellow') textColor = 'text-yellow-700 font-bold';
 else if (status === 'red') textColor = 'text-red-700 font-bold';

 const entry = document.createElement('div');
 entry.className = 'py-2 border-b border-industrial-200 text-xs';
 entry.innerHTML = `
 <span class="${textColor} block mb-0.5">[${timeStr}] RAG System</span>
 <p class="text-industrial-900 leading-relaxed">${message}</p>
 `;

 ragConsole.appendChild(entry);
 ragConsole.scrollTop = ragConsole.scrollHeight;
}

// 8. File Upload
if (fileInput) {
 fileInput.addEventListener('change', (e) => {
 if (e.target.files && e.target.files[0]) {
 const file = e.target.files[0];
 if (fileNameDisplay) {
 fileNameDisplay.innerHTML = `<span class="text-blue-600 font-semibold">선택됨: ${file.name}</span> (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
 }
 }
 });
}

if (uploadBtn) {
 uploadBtn.addEventListener('click', async () => {
 if (!fileInput.files || !fileInput.files[0]) {
 alert('업로드할 정비 리포트 파일을 먼저 선택해 주세요.');
 return;
 }

 const formData = new FormData();
 formData.append('file', fileInput.files[0]);

 try {
 uploadBtn.textContent = '업로드 중...';
 uploadBtn.disabled = true;

 const res = await fetch('/api/upload-report', { method: 'POST', body: formData });
 const data = await res.json();

 alert(`'${fileInput.files[0].name}' 정비 리포트가 성공적으로 업로드되었습니다.`);
 addAlertLog('[리포트 업로드]', `${fileInput.files[0].name} 업로드 완료`, 'normal');
 } catch (err) {
 alert('업로드 중 오류가 발생했습니다: ' + err.message);
 } finally {
 uploadBtn.textContent = '업로드';
 uploadBtn.disabled = false;
 fileInput.value = '';
 if (fileNameDisplay) fileNameDisplay.textContent = 'PDF, XLSX, DOCX (최대 25MB)';
 }
 });
}

// 9. Report Download - 현재 화면의 액션 리포트를 즉시 PDF로 다운로드
if (downloadBtn) {
 downloadBtn.addEventListener('click', () => {
 window.open('/api/download-action-report', '_blank');
 });
}


// --- Sidebar Logic Removed ---



// --- Action Report History Logic ---
const reportHistoryList = document.getElementById('report-history-list');
const downloadSelectedBtn = document.getElementById('download-selected-reports-btn');
const reportListEmpty = document.getElementById('report-list-empty');

function addReportToList(report) {
    if (reportListEmpty) reportListEmpty.style.display = 'none';
    if (!reportHistoryList) return;

    const div = document.createElement('div');
    div.className = 'flex items-start space-x-2 p-2 bg-gray-50/80 border border-gray-200/60 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer';
    
    let statusClass = 'bg-gray-100 text-gray-600';
    let statusText = '정상';
    if (report.status === 'red') {
        statusClass = 'bg-red-100 text-red-700 border border-red-200';
        statusText = '위험';
    } else if (report.status === 'yellow') {
        statusClass = 'bg-amber-100 text-amber-700 border border-amber-200';
        statusText = '주의';
    }

    const timeStr = new Date().toLocaleTimeString('ko-KR', { hour12: false });
    
    // Summary snippet
    let snippet = report.report ? report.report.substring(0, 40) + '...' : '분석 결과';

    div.innerHTML = `
        <div class="pt-1">
            <input type="checkbox" class="report-checkbox w-3.5 h-3.5 text-blue-600 border-gray-300 rounded focus:ring-blue-500" value="${report.report_id}">
        </div>
        <div class="flex-1 min-w-0" onclick="this.previousElementSibling.firstElementChild.click()">
            <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono text-gray-500">${timeStr}</span>
                <span class="px-1.5 py-0.5 rounded text-[9px] font-bold ${statusClass}">${statusText}</span>
            </div>
            <p class="text-[11px] text-gray-800 mt-1 truncate" title="${snippet}">${snippet}</p>
        </div>
    `;
    
    reportHistoryList.prepend(div);
    updateDownloadBtnState();
    
    const checkboxes = div.querySelectorAll('.report-checkbox');
    checkboxes.forEach(cb => cb.addEventListener('change', updateDownloadBtnState));
}

function updateDownloadBtnState() {
    if (!downloadSelectedBtn) return;
    const checked = document.querySelectorAll('.report-checkbox:checked');
    if (checked.length > 0) {
        downloadSelectedBtn.disabled = false;
        downloadSelectedBtn.classList.remove('cursor-not-allowed', 'opacity-50');
        downloadSelectedBtn.querySelector('span').textContent = `선택 항목 다운로드 (${checked.length}개)`;
    } else {
        downloadSelectedBtn.disabled = true;
        downloadSelectedBtn.classList.add('cursor-not-allowed', 'opacity-50');
        downloadSelectedBtn.querySelector('span').textContent = `선택 항목 다운로드 (PDF)`;
    }
}

if (downloadSelectedBtn) {
    downloadSelectedBtn.addEventListener('click', async () => {
        const checked = Array.from(document.querySelectorAll('.report-checkbox:checked')).map(cb => cb.value);
        if (checked.length === 0) return;
        
        downloadSelectedBtn.disabled = true;
        downloadSelectedBtn.querySelector('span').textContent = 'PDF 생성 중...';
        
        try {
            const res = await fetch('/api/download-action-reports', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ report_ids: checked })
            });
            
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                // Get filename from Content-Disposition if possible
                const disp = res.headers.get('Content-Disposition');
                let filename = 'ATR_Merged_Reports.pdf';
                if (disp && disp.indexOf('filename=') !== -1) {
                    filename = disp.split('filename=')[1].replace(/"/g, '');
                }
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            } else {
                alert('리포트 다운로드에 실패했습니다.');
            }
        } catch (err) {
            alert('서버 연결 오류');
        } finally {
            updateDownloadBtnState();
        }
    });
}

// --- Chat Service Logic ---
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');

function appendChatMessage(sender, text) {
    if (!chatMessages) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex items-start space-x-2 mt-3';
    
    if (sender === 'user') {
        msgDiv.innerHTML = `
            <div class="flex-1"></div>
            <div class="bg-blue-600 rounded-lg rounded-tr-none px-3 py-2 text-[11px] text-white shadow-sm max-w-[90%] whitespace-pre-wrap">${text}</div>
            <div class="w-6 h-6 rounded bg-blue-100 border border-blue-200 flex items-center justify-center flex-shrink-0">
                <i data-lucide="user" class="w-3.5 h-3.5 text-blue-700"></i>
            </div>
        `;
    } else {
        msgDiv.innerHTML = `
            <div class="w-6 h-6 rounded bg-industrial-100 border border-industrial-200 flex items-center justify-center flex-shrink-0">
                <i data-lucide="bot" class="w-3.5 h-3.5 text-industrial-700"></i>
            </div>
            <div class="bg-industrial-50 rounded-lg rounded-tl-none px-3 py-2 text-[11px] text-industrial-800 border border-industrial-200 max-w-[90%] whitespace-pre-wrap">${text}</div>
        `;
    }
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    lucide.createIcons();
}

if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        
        appendChatMessage('user', text);
        chatInput.value = '';
        chatInput.disabled = true;
        
        // Show loading indicator
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.className = 'flex items-start space-x-2 mt-3';
        loadingDiv.innerHTML = `
            <div class="w-6 h-6 rounded bg-industrial-100 border border-industrial-200 flex items-center justify-center flex-shrink-0">
                <i data-lucide="bot" class="w-3.5 h-3.5 text-industrial-700"></i>
            </div>
            <div class="bg-industrial-50 rounded-lg rounded-tl-none px-3 py-2 text-[11px] text-industrial-500 border border-industrial-200">
                <span class="animate-pulse">답변을 생성 중입니다...</span>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            
            // Remove loading indicator
            const lDiv = document.getElementById(loadingId);
            if (lDiv) lDiv.remove();
            
            if (data.status === 'success') {
                appendChatMessage('ai', data.reply);
            } else {
                appendChatMessage('ai', '죄송합니다. 오류가 발생했습니다: ' + data.message);
            }
        } catch (err) {
            const lDiv = document.getElementById(loadingId);
            if (lDiv) lDiv.remove();
            appendChatMessage('ai', '서버와 연결할 수 없습니다.');
        } finally {
            chatInput.disabled = false;
            chatInput.focus();
        }
    });
}
