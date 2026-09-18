// stitch_app.js - Real-time WebSocket connection (17 Sensors)

const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);

// --- 1. KPI DOM Elements ---
const kpiVacuum = document.getElementById('kpi-vacuum');
const kpiTemp = document.getElementById('kpi-temp');
const kpiRf = document.getElementById('kpi-rf');
const kpiErr = document.getElementById('kpi-err');
const kpiThd = document.getElementById('kpi-thd');
const kpiTilt = document.getElementById('kpi-tilt');
const alertLog = document.getElementById('alert-log');
const ragConsole = document.getElementById('rag-console');

// --- 2. Chart Setup ---
const canvas = document.getElementById('sensorChart');
const ctx = canvas ? canvas.getContext('2d') : null;
let animationFrameId;
const pointCount = 120; // 60 seconds at 2Hz

const SENSOR_FEATURES = [
 'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
 'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
 'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
 'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
 'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
 'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
];

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

// Initialize sensor state
const sensorData = {};
SENSOR_FEATURES.forEach((feat, idx) => {
  sensorData[feat] = {
    color: SENSOR_COLORS[idx % SENSOR_COLORS.length],
    visible: true,
    values: [],
    latestVal: 0
  };
});

// Render Legend
const legendContainer = document.getElementById('chart-legend');
if (legendContainer) {
  SENSOR_FEATURES.forEach(feat => {
    const el = document.createElement('div');
    el.className = 'flex items-center space-x-1.5 cursor-pointer opacity-100 hover:opacity-80 transition-opacity';
    el.innerHTML = `<span class="w-2.5 h-2.5 rounded-full" style="background-color: ${sensorData[feat].color}"></span><span class="text-gray-600 dark:text-gray-300 font-mono text-[9px] truncate max-w-[80px]" title="${feat}">${feat}</span>`;
    
    el.addEventListener('click', () => {
      sensorData[feat].visible = !sensorData[feat].visible;
      el.style.opacity = sensorData[feat].visible ? '1' : '0.3';
      if(!animationFrameId) {
        animationFrameId = requestAnimationFrame(() => { drawChart(); animationFrameId = null; });
      }
    });
    
    legendContainer.appendChild(el);
  });
}

function resizeCanvas() {
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
}
if(canvas) {
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
}

function drawChart() {
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  // Subtle grid
  const isDark = document.documentElement.classList.contains('dark');
  ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.04)';
  ctx.lineWidth = 1;

  for (let y = 30; y < h; y += 45) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let x = 0; x < w; x += 75) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  // Draw lines
  const stepX = w / (pointCount - 1);
  
  SENSOR_FEATURES.forEach(feat => {
    const sensor = sensorData[feat];
    if(!sensor.visible || sensor.values.length === 0) return;

    ctx.strokeStyle = sensor.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const sMin = SENSOR_SPANS[feat].min;
    const spanRange = SENSOR_SPANS[feat].max - SENSOR_SPANS[feat].min;

    let started = false;
    for (let i = 0; i < sensor.values.length; i++) {
      const val = sensor.values[i];
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

// --- 3. UI Update Helpers ---
function formatNumber(num, fixed=2) {
    return Number(num).toFixed(fixed);
}

function addAlertLog(title, desc, status) {
  if(!alertLog) return;
  const emptyMsg = document.getElementById('alert-log-empty');
  if (emptyMsg) emptyMsg.remove();

  let colorClass = 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400 border-emerald-200/50 dark:border-emerald-500/20';
  let icon = 'check-circle';
  
  if (status === 'warning') {
    colorClass = 'bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400 border-amber-200/50 dark:border-amber-500/20';
    icon = 'alert-triangle';
  } else if (status === 'danger') {
    colorClass = 'bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-400 border-rose-200/50 dark:border-rose-500/20';
    icon = 'alert-octagon';
  }

  const el = document.createElement('div');
  el.className = `p-3 rounded-xl border ${colorClass} flex items-start space-x-3 text-xs`;
  el.innerHTML = `
    <i data-lucide="${icon}" class="w-4 h-4 mt-0.5 shrink-0"></i>
    <div>
      <div class="font-semibold">${title}</div>
      <div class="mt-0.5 opacity-80 font-sans">${desc}</div>
      <div class="mt-1.5 text-[10px] opacity-60 font-mono">${new Date().toLocaleTimeString('en-US', {hour12:false})}</div>
    </div>
  `;
  
  alertLog.prepend(el);
  if (alertLog.children.length > 20) {
    alertLog.removeChild(alertLog.lastChild);
  }
  lucide.createIcons();
}

function appendRagMessage(sender, text) {
  if (!ragConsole) return;
  const el = document.createElement('div');
  el.className = `flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;
  
  const inner = document.createElement('div');
  inner.className = `max-w-[85%] p-3 rounded-xl ${sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'} font-sans text-xs`;
  
  const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/
/g, '<br>');
  inner.innerHTML = formattedText;
  
  el.appendChild(inner);
  ragConsole.appendChild(el);
  ragConsole.scrollTop = ragConsole.scrollHeight;
}

// --- 4. WebSocket Message Handler ---
ws.onopen = () => {
    addAlertLog('시스템 연결 완료', '실시간 데이터 스트림 대기중...', 'normal');
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'telemetry') {
    const raw = msg.raw_sensors || {};
    
    // Update KPIs
    if(kpiVacuum) kpiVacuum.innerText = (raw['FLOWCOOLPRESSURE'] * 0.00001).toExponential(2);
    if(kpiTemp) kpiTemp.innerText = formatNumber(raw['FLOWCOOLLEAKRATE'] * 100 + 200, 1);
    if(kpiRf) kpiRf.innerText = formatNumber(raw['ETCHBEAMVOLTAGE'] / 10 + 2, 2);
    if(kpiErr) kpiErr.innerText = formatNumber(raw['ETCHBEAMCURRENT'] * 0.01, 3) + ' µm';
    if(kpiThd) kpiThd.innerText = formatNumber(raw['ETCHGASCHANNEL1SETPOINT'] * 0.1, 2) + ' dB';
    
    // Update Chart Buffers
    SENSOR_FEATURES.forEach(feat => {
      let v = Number(raw[feat]);
      if(isNaN(v)) v = sensorData[feat].latestVal || 0;
      sensorData[feat].values.push(v);
      if(sensorData[feat].values.length > pointCount) {
        sensorData[feat].values.shift();
      }
      sensorData[feat].latestVal = v;
    });

    if(!animationFrameId) {
      animationFrameId = requestAnimationFrame(() => {
        drawChart();
        animationFrameId = null;
      });
    }

    // Handle Anomaly Logging
    const s = msg.status;
    if (s === 'warning' || s === 'danger') {
        if(!window.lastAlertTime || Date.now() - window.lastAlertTime > 2000) {
            addAlertLog(`이상 감지 (${s.toUpperCase()})`, `Anomaly Score: ${formatNumber(msg.final_score, 4)}`, s);
            window.lastAlertTime = Date.now();
        }
    }
  } 
  else if (msg.type === 'llm_report') {
    appendRagMessage('ai', msg.report);
  }
};

// --- 5. RAG Chat Form Binding ---
const chatForm = document.getElementById('chat-form');
if (chatForm) {
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const q = input.value.trim();
    if (!q) return;

    appendRagMessage('user', q);
    input.value = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      appendRagMessage('ai', data.response);
    } catch (err) {
      appendRagMessage('ai', 'Error: 서버 응답 지연');
    }
  });
}
