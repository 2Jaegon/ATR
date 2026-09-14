// --- 1. DARK MODE TOGGLE LOGIC ---
const themeToggleBtn = document.getElementById('themeToggleBtn');
const themeToggleText = document.getElementById('themeToggleText');
const htmlEl = document.documentElement;

function applyTheme(isDark) {
  if (isDark) {
    htmlEl.classList.add('dark');
    if (themeToggleText) themeToggleText.textContent = 'Light';
    localStorage.setItem('atr_theme', 'dark');
  } else {
    htmlEl.classList.remove('dark');
    if (themeToggleText) themeToggleText.textContent = 'Dark';
    localStorage.setItem('atr_theme', 'light');
  }
}

const savedTheme = localStorage.getItem('atr_theme');
if (savedTheme === 'dark') {
  applyTheme(true);
} else {
  applyTheme(false); // Clean white tone light mode default
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    const isCurrentlyDark = htmlEl.classList.contains('dark');
    applyTheme(!isCurrentlyDark);
  });
}

// --- 2. LIVE DATA SENSOR CANVAS ---
const canvas = document.getElementById('sensorChart');
const ctx = canvas.getContext('2d');

let dpr = window.devicePixelRatio || 1;
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const pointCount = 120;
const seriesRF = Array(pointCount).fill(0.5);
const seriesPiezo = Array(pointCount).fill(0.4);
const seriesThermal = Array(pointCount).fill(0.65);

function drawChart() {
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  ctx.clearRect(0, 0, w, h);

  const isDark = htmlEl.classList.contains('dark');

  // Apple-style clean minimal grid lines
  ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)';
  ctx.lineWidth = 1;

  for (let i = 1; i <= 3; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  function renderSeries(data, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const step = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      // data expected to be roughly 0.0 to 1.0
      const y = h - (data[i] * (h * 0.7) + h * 0.15);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  renderSeries(seriesRF, '#10B981');     // Emerald
  renderSeries(seriesPiezo, '#6366F1');  // Indigo
  renderSeries(seriesThermal, '#F59E0B'); // Amber
}

// 3. WebSocket Connection
const scoreTrEl = document.getElementById('score-tr');
const scoreGnnEl = document.getElementById('score-gnn');
const scoreLstmEl = document.getElementById('score-lstm');
const finalScoreEl = document.getElementById('final-score');
const finalCircle = document.getElementById('final-score-circle');
const globalStatus = document.getElementById('global-status-indicator');
const ragConsole = document.getElementById('rag-console');

const maxCircumference = 2 * Math.PI * 64; // ~402.12

const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);

ws.onopen = () => {
  addLogEntry('SYSTEM CONNECTED', 'ATR Oculus Core v2.9.4 initialized.', 'blue');
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  if (msg.type === 'telemetry') {
    // Normalize sensor values from 10~90 roughly to 0.0~1.0
    let norm = (msg.sensor_value - 10) / 80;
    if (norm > 1.0) norm = 1.0;
    if (norm < 0.0) norm = 0.0;

    seriesRF.shift();
    seriesRF.push(norm);

    seriesPiezo.shift();
    seriesPiezo.push(Math.max(0, Math.min(1, norm * 0.8 + (Math.random() - 0.5)*0.1)));

    seriesThermal.shift();
    seriesThermal.push(Math.max(0, Math.min(1, norm * 1.2 + (Math.random() - 0.5)*0.1)));

    drawChart();

    // Update Scores
    if (scoreTrEl) scoreTrEl.innerText = msg.scores.TR.toFixed(4);
    if (scoreGnnEl) scoreGnnEl.innerText = msg.scores.GNN.toFixed(4);
    if (scoreLstmEl) scoreLstmEl.innerText = msg.scores.LSTM.toFixed(4);

    // Final score 0-100 logic (Inverted for visual: 100 is healthy)
    let visualScore = Math.max(0, 100 - (msg.final_score * 100));
    if (finalScoreEl) finalScoreEl.innerText = Math.floor(visualScore);

    // Ring SVG
    if (finalCircle) {
      const offset = maxCircumference - (visualScore / 100) * maxCircumference;
      finalCircle.style.strokeDashoffset = offset;
    }

    // Status UI Updates
    if (msg.status === 'red') {
      if (globalStatus) {
        globalStatus.innerText = "System Critical";
        globalStatus.className = "text-red-500 font-semibold";
      }
      if (finalCircle) finalCircle.style.stroke = "#EF4444";
    } else if (msg.status === 'yellow') {
      if (globalStatus) {
        globalStatus.innerText = "Attention Required";
        globalStatus.className = "text-amber-500 font-semibold";
      }
      if (finalCircle) finalCircle.style.stroke = "#F59E0B";
    } else {
      if (globalStatus) {
        globalStatus.innerText = "System Nominal";
        globalStatus.className = "text-emerald-500 font-semibold";
      }
      if (finalCircle) finalCircle.style.stroke = "#10B981";
    }
  }

  if (msg.type === 'report') {
    const typeColor = msg.status === 'red' ? 'red' : msg.status === 'yellow' ? 'amber' : 'emerald';
    addLogEntry('RAG INTELLIGENCE', msg.message, typeColor);
  }
};

function addLogEntry(title, msg, typeColor) {
  if (!ragConsole) return;
  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
  
  let borderColor = 'border-emerald-500';
  let textColor = 'text-emerald-600 dark:text-emerald-400';
  
  if (typeColor === 'amber') {
    borderColor = 'border-amber-500';
    textColor = 'text-amber-600 dark:text-amber-400';
  } else if (typeColor === 'red') {
    borderColor = 'border-red-500';
    textColor = 'text-red-600 dark:text-red-400';
  } else if (typeColor === 'blue') {
    borderColor = 'border-blue-500';
    textColor = 'text-blue-600 dark:text-blue-400';
  }

  const logEntry = document.createElement('div');
  logEntry.className = `border-l-2 ${borderColor} pl-3 py-1 text-neutral-600 dark:text-neutral-400 animate-fadeIn mb-3 text-xs font-mono`;
  logEntry.innerHTML = `
    <span class="${textColor} block font-semibold mb-0.5">[${timeStr}] ${title}</span>
    <p class="leading-relaxed opacity-90">${msg}</p>
  `;

  ragConsole.appendChild(logEntry);
  ragConsole.scrollTop = ragConsole.scrollHeight;
}
