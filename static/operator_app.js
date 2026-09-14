// --- ATR Oculus: Operator Dashboard JS ---

// 1. Lucide Icons
lucide.createIcons();

// 2. Dark Mode Toggle
const themeToggleBtn = document.getElementById('themeToggleBtn');
const themeToggleText = document.getElementById('themeToggleText');
const themeIcon = document.getElementById('themeIcon');
const htmlEl = document.documentElement;

function setTheme(isDark) {
  if (isDark) {
    htmlEl.classList.add('dark');
    if (themeToggleText) themeToggleText.textContent = 'Light';
    if (themeIcon) themeIcon.setAttribute('data-lucide', 'sun');
    localStorage.setItem('atr_theme', 'dark');
  } else {
    htmlEl.classList.remove('dark');
    if (themeToggleText) themeToggleText.textContent = 'Dark';
    if (themeIcon) themeIcon.setAttribute('data-lucide', 'moon');
    localStorage.setItem('atr_theme', 'light');
  }
  lucide.createIcons();
}

const savedTheme = localStorage.getItem('atr_theme');
setTheme(savedTheme === 'dark');

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    const isCurrentlyDark = htmlEl.classList.contains('dark');
    setTheme(!isCurrentlyDark);
  });
}

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
const seriesRF = Array(pointCount).fill(0.5);
const seriesPiezo = Array(pointCount).fill(0.4);
const seriesThermal = Array(pointCount).fill(0.65);

function drawChart() {
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  ctx.clearRect(0, 0, w, h);

  const isDark = htmlEl.classList.contains('dark');

  // Grid lines
  ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.04)';
  ctx.lineWidth = 1;
  for (let y = 30; y < h; y += 45) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let x = 0; x < w; x += 75) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  function renderSeries(data, color, lineWidth = 2) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    const stepX = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
      const x = i * stepX;
      const y = h - (data[i] * (h * 0.7) + h * 0.15);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  renderSeries(seriesRF, '#10B981', 2.2);
  renderSeries(seriesPiezo, '#3B82F6', 2);
  renderSeries(seriesThermal, '#F59E0B', 2);
}

// 4. DOM Elements
const globalStatus = document.getElementById('global-status-indicator');
const alertLog = document.getElementById('alert-log');
const ragConsole = document.getElementById('rag-console');
const downloadBtn = document.getElementById('download-report-btn');
const fileInput = document.getElementById('file-upload-input');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadBtn = document.getElementById('upload-btn');

// 5. WebSocket Connection
const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);

ws.onopen = () => {
  addAlertLog('시스템 연결 완료', 'ATR Oculus 실시간 스트림이 연결되었습니다.', 'normal');
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'telemetry') {
    // Update chart data
    let norm = (msg.sensor_value - 10) / 80;
    norm = Math.max(0, Math.min(1, norm));

    seriesRF.shift();
    seriesRF.push(norm);
    seriesPiezo.shift();
    seriesPiezo.push(Math.max(0, Math.min(1, norm * 0.8 + (Math.random() - 0.5) * 0.1)));
    seriesThermal.shift();
    seriesThermal.push(Math.max(0, Math.min(1, norm * 1.2 + (Math.random() - 0.5) * 0.1)));

    drawChart();

    // Update global status
    if (msg.status === 'red') {
      if (globalStatus) {
        globalStatus.textContent = '위험 (Critical)';
        globalStatus.className = 'text-red-500 font-semibold text-xs';
      }
      addAlertLog('⚠️ 위험 감지', `최종 이상 점수: ${msg.final_score.toFixed(3)} — 즉각 조치 필요`, 'critical');
    } else if (msg.status === 'yellow') {
      if (globalStatus) {
        globalStatus.textContent = '주의 (Warning)';
        globalStatus.className = 'text-amber-500 font-semibold text-xs';
      }
      addAlertLog('⚡ 이상 징후', `최종 이상 점수: ${msg.final_score.toFixed(3)} — 모니터링 강화`, 'warning');
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
};

// 6. Alert Log (이상 탐지 로그)
function addAlertLog(title, message, level) {
  if (!alertLog) return;
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ko-KR');

  let borderColor = 'border-emerald-400';
  let dotColor = 'bg-emerald-400';
  let levelText = '정상';
  if (level === 'warning') {
    borderColor = 'border-amber-400';
    dotColor = 'bg-amber-400';
    levelText = '주의';
  } else if (level === 'critical') {
    borderColor = 'border-red-400';
    dotColor = 'bg-red-400';
    levelText = '위험';
  }

  const entry = document.createElement('div');
  entry.className = `border-l-2 ${borderColor} pl-3 py-2 text-xs`;
  entry.innerHTML = `
    <div class="flex items-center gap-2 mb-0.5">
      <span class="w-1.5 h-1.5 rounded-full ${dotColor}"></span>
      <span class="font-semibold text-neutral-800 dark:text-neutral-200">${title}</span>
      <span class="ml-auto text-neutral-400 font-mono text-[10px]">${timeStr}</span>
    </div>
    <p class="text-neutral-500 dark:text-neutral-400 pl-3.5">${message}</p>
  `;

  alertLog.prepend(entry); // 최신이 위에
  // 최대 50개 유지
  while (alertLog.children.length > 50) {
    alertLog.removeChild(alertLog.lastChild);
  }
}

// 7. RAG Console (액션 리포트)
function addRagEntry(message, status) {
  if (!ragConsole) return;
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ko-KR');

  let textColor = 'text-emerald-600 dark:text-emerald-400';
  if (status === 'yellow') textColor = 'text-amber-600 dark:text-amber-400';
  else if (status === 'red') textColor = 'text-red-600 dark:text-red-400';

  const entry = document.createElement('div');
  entry.className = 'py-2 border-b border-neutral-100 dark:border-neutral-800 text-xs';
  entry.innerHTML = `
    <span class="${textColor} font-semibold block mb-0.5">[${timeStr}] RAG 분석 결과</span>
    <p class="text-neutral-600 dark:text-neutral-400 leading-relaxed">${message}</p>
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
        fileNameDisplay.innerHTML = `<span class="text-blue-600 dark:text-blue-400 font-semibold">선택됨: ${file.name}</span> (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
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
      addAlertLog('📎 리포트 업로드', `${fileInput.files[0].name} 업로드 완료`, 'normal');
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

// 9. Report Download
if (downloadBtn) {
  downloadBtn.addEventListener('click', () => {
    window.open('/api/reports', '_blank');
  });
}
