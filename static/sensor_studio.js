// ============================================================
// ATR Oculus — ATR Sensor Telemetry Studio JS
// ============================================================

lucide.createIcons();

// --- 1. 17 Sensor Definitions & Calibration Spans ---
const SENSOR_FEATURES = [
  'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
  'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
  'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
  'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
  'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
  'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
];

const SENSOR_META = {
  'IONGAUGEPRESSURE':        { name: '이온 게이지 진공도', unit: 'Torr', group: 'flow', span: { min: -0.15, max: 0.15 } },
  'ETCHBEAMVOLTAGE':         { name: '에칭 빔 전압',       unit: 'V',    group: 'beam', span: { min: -1.20, max: 1.80 } },
  'ETCHBEAMCURRENT':         { name: '에칭 빔 전류',       unit: 'mA',   group: 'beam', span: { min: -1.20, max: 2.00 } },
  'ETCHSUPPRESSORVOLTAGE':   { name: '서프레서 전압',     unit: 'V',    group: 'beam', span: { min: -1.20, max: 1.80 } },
  'ETCHSUPPRESSORCURRENT':   { name: '서프레서 전류',     unit: 'mA',   group: 'beam', span: { min: -1.20, max: 2.10 } },
  'FLOWCOOLFLOWRATE':        { name: '냉각수 유량',       unit: 'sccm', group: 'flow', span: { min: -3.00, max: 1.20 } },
  'FLOWCOOLPRESSURE':        { name: '냉각 배면 압력',     unit: 'Torr', group: 'flow', span: { min: -2.50, max: 0.80 } },
  'ETCHGASCHANNEL1READBACK': { name: '에칭 가스 채널 1',  unit: 'sccm', group: 'flow', span: { min: -2.00, max: 2.80 } },
  'ETCHPBNGASREADBACK':      { name: 'PBN 가스 유량',     unit: 'sccm', group: 'flow', span: { min: -3.00, max: 1.20 } },
  'FIXTURETILTANGLE':        { name: '서브스트레이트 틸트', unit: '°',    group: 'mech', span: { min: -1.80, max: 2.20 } },
  'ROTATIONSPEED':           { name: '웨이퍼 회전 속도',   unit: 'rpm',  group: 'mech', span: { min: -0.05, max: 0.05 } },
  'ACTUALROTATIONANGLE':     { name: '실제 회전 각도',     unit: '°',    group: 'mech', span: { min: -0.20, max: 0.05 } },
  'FIXTURESHUTTERPOSITION':  { name: '셔터 개폐 위치',     unit: 'mm',   group: 'mech', span: { min: -0.20, max: 3.20 } },
  'ETCHSOURCEUSAGE':         { name: '소스 누적 사용량',   unit: 'hrs',  group: 'misc', span: { min: -0.25, max: -0.05 } },
  'ETCHAUXSOURCETIMER':      { name: '보조 소스 타이머 1', unit: 'sec',  group: 'misc', span: { min: -0.05, max: 0.08 } },
  'ETCHAUX2SOURCETIMER':     { name: '보조 소스 타이머 2', unit: 'sec',  group: 'misc', span: { min: 0.10,  max: 0.30 } },
  'ACTUALSTEPDURATION':      { name: '공정 스텝 지속 시간', unit: 'sec',  group: 'misc', span: { min: -1.00, max: 4.50 } }
};

const SENSOR_COLORS = [
  '#DC2626', '#EA580C', '#D97706', '#CA8A04', '#65A30D',
  '#16A34A', '#059669', '#0D9488', '#0891B2', '#0284C7',
  '#2563EB', '#4F46E5', '#7C3AED', '#9333EA', '#C026D3',
  '#DB2777', '#4B5563'
];

// 센서별 고유 컬러 매핑
const SENSOR_COLOR_MAP = {};
SENSOR_FEATURES.forEach((feat, idx) => {
  SENSOR_COLOR_MAP[feat] = SENSOR_COLORS[idx];
});

// --- 2. Live Telemetry Data Buffer ---
const MAX_BUFFER_POINTS = 600; // 최대 300초 (2Hz 기준 600개)
let timeWindowSec = 60;        // 기본 60초 (120포인트)
let isPaused = false;

const sensorBuffers = {};
const latestValues = {};
SENSOR_FEATURES.forEach(feat => {
  sensorBuffers[feat] = [];
  latestValues[feat] = 0;
});

// 동기화 커서 상태
let syncHoverNormX = null; // 0.0 ~ 1.0 (마우스 위치)
let hoveredPlotId = null;

// --- 3. Plot Manager State ---
let plotIdCounter = 1;
let currentLayout = '2x1';

// 초기 플롯 창 구성: 2x1 분할 레이아웃
let plots = [
  {
    id: 'plot-1',
    title: '플롯 1 (전압 및 전류)',
    sensors: ['ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT']
  },
  {
    id: 'plot-2',
    title: '플롯 2 (진공 및 유량)',
    sensors: ['FLOWCOOLPRESSURE', 'FLOWCOOLFLOWRATE', 'IONGAUGEPRESSURE']
  }
];

// --- 4. DOM Elements ---
const plotsGridContainer = document.getElementById('plots-grid-container');
const sensorListContainer = document.getElementById('sensor-list-container');
const sensorSearchInput = document.getElementById('sensor-search-input');
const timeWindowSelect = document.getElementById('time-window-select');
const pauseStreamBtn = document.getElementById('pause-stream-btn');
const pauseIcon = document.getElementById('pause-icon');
const pauseText = document.getElementById('pause-text');
const studioStreamStatus = document.getElementById('studio-stream-status');
const addPlotBtn = document.getElementById('add-plot-btn');
const crosshairTooltip = document.getElementById('crosshair-tooltip');

// Modal Elements
const addSensorModal = document.getElementById('add-sensor-modal');
const modalPlotTitle = document.getElementById('modal-plot-title');
const modalSensorList = document.getElementById('modal-sensor-list');
const closeSensorModalBtn = document.getElementById('close-sensor-modal-btn');
let modalTargetPlotId = null;

// --- 5. Left Sidebar: Sensor Explorer Render ---
let currentGroupFilter = 'all';

function renderSensorList() {
  if (!sensorListContainer) return;
  const searchQuery = (sensorSearchInput ? sensorSearchInput.value.trim().toLowerCase() : '');
  sensorListContainer.innerHTML = '';

  let visibleCount = 0;

  SENSOR_FEATURES.forEach(feat => {
    const meta = SENSOR_META[feat];
    if (currentGroupFilter !== 'all' && meta.group !== currentGroupFilter) return;
    if (searchQuery && !feat.toLowerCase().includes(searchQuery) && !meta.name.toLowerCase().includes(searchQuery)) return;

    visibleCount++;
    const item = document.createElement('div');
    item.draggable = true;
    item.dataset.sensor = feat;
    item.className = 'sensor-drag-item group flex items-center justify-between p-2 rounded-lg bg-industrial-50 hover:bg-industrial-100 border border-industrial-200 transition-all cursor-grab';

    const color = SENSOR_COLOR_MAP[feat];
    const val = latestValues[feat] !== undefined ? Number(latestValues[feat]).toFixed(2) : '-';

    item.innerHTML = `
      <div class="flex items-center space-x-2 min-w-0 pr-1">
        <i data-lucide="grip-vertical" class="w-3.5 h-3.5 text-industrial-400 group-hover:text-industrial-600 flex-shrink-0"></i>
        <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${color}"></span>
        <div class="min-w-0">
          <div class="text-[11px] font-bold font-mono text-industrial-900 truncate">${feat}</div>
          <div class="text-[10px] text-industrial-500 truncate">${meta.name}</div>
        </div>
      </div>
      <div class="flex items-center space-x-1.5 flex-shrink-0 pl-1">
        <span id="live-val-${feat}" class="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-white border border-industrial-200 text-industrial-800">${val} <span class="text-[9px] font-normal text-industrial-400 font-sans">${meta.unit}</span></span>
        <button class="quick-add-btn opacity-0 group-hover:opacity-100 p-1 hover:bg-industrial-200 rounded text-industrial-600 transition-opacity" data-sensor="${feat}" title="활성 플롯에 추가">
          <i data-lucide="plus" class="w-3 h-3"></i>
        </button>
      </div>
    `;

    // HTML5 Drag and Drop
    item.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', feat);
      e.dataTransfer.effectAllowed = 'copy';
      item.classList.add('opacity-50');
    });

    item.addEventListener('dragend', () => {
      item.classList.remove('opacity-50');
    });

    // Quick Add Button
    const qBtn = item.querySelector('.quick-add-btn');
    if (qBtn) {
      qBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (plots.length > 0) {
          const target = plots[0];
          if (!target.sensors.includes(feat)) {
            target.sensors.push(feat);
            renderPlots();
          }
        }
      });
    }

    sensorListContainer.appendChild(item);
  });

  const countBadge = document.getElementById('sensor-count-badge');
  if (countBadge) {
    countBadge.textContent = `${visibleCount}/17`;
  }

  lucide.createIcons({ root: sensorListContainer });
}

// Group Filter Tabs
document.querySelectorAll('.filter-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-tab').forEach(b => {
      b.classList.remove('text-industrial-900', 'font-bold', 'bg-industrial-100');
      b.classList.add('text-industrial-500');
    });
    btn.classList.remove('text-industrial-500');
    btn.classList.add('text-industrial-900', 'font-bold', 'bg-industrial-100');
    currentGroupFilter = btn.dataset.group;
    renderSensorList();
  });
});

if (sensorSearchInput) {
  sensorSearchInput.addEventListener('input', renderSensorList);
}

// --- 6. Layout Grid Configuration ---
function updateLayoutClass() {
  if (!plotsGridContainer) return;
  plotsGridContainer.className = 'flex-1 grid gap-3 w-full h-full overflow-hidden';

  switch (currentLayout) {
    case '1x1':
      plotsGridContainer.classList.add('grid-cols-1', 'grid-rows-1');
      break;
    case '1x2':
      plotsGridContainer.classList.add('grid-cols-1', 'grid-rows-2');
      break;
    case '2x1':
      plotsGridContainer.classList.add('grid-cols-2', 'grid-rows-1');
      break;
    case '2x2':
      plotsGridContainer.classList.add('grid-cols-2', 'grid-rows-2');
      break;
    case '3x1':
      plotsGridContainer.classList.add('grid-cols-1', 'grid-rows-3');
      break;
    default:
      if (plots.length <= 1) plotsGridContainer.classList.add('grid-cols-1', 'grid-rows-1');
      else if (plots.length === 2) plotsGridContainer.classList.add('grid-cols-2', 'grid-rows-1');
      else plotsGridContainer.classList.add('grid-cols-2', 'grid-rows-2');
      break;
  }
}

document.querySelectorAll('.layout-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.layout-btn').forEach(b => {
      b.classList.remove('bg-white', 'text-industrial-900', 'border-industrial-300', 'shadow-xs');
      b.classList.add('text-industrial-600', 'border-transparent');
    });
    btn.classList.remove('text-industrial-600', 'border-transparent');
    btn.classList.add('bg-white', 'text-industrial-900', 'border-industrial-300', 'shadow-xs');

    const layout = btn.dataset.layout;
    currentLayout = layout;

    // 프리셋에 맞게 플롯 개수 자동 맞춤
    let targetCount = 1;
    if (layout === '1x1') targetCount = 1;
    else if (layout === '1x2' || layout === '2x1') targetCount = 2;
    else if (layout === '2x2') targetCount = 4;
    else if (layout === '3x1') targetCount = 3;

    while (plots.length < targetCount) {
      plotIdCounter++;
      plots.push({
        id: `plot-${plotIdCounter}`,
        title: `플롯 ${plots.length + 1}`,
        sensors: []
      });
    }
    if (plots.length > targetCount) {
      plots = plots.slice(0, targetCount);
    }

    renderPlots();
  });
});

if (addPlotBtn) {
  addPlotBtn.addEventListener('click', () => {
    plotIdCounter++;
    plots.push({
      id: `plot-${plotIdCounter}`,
      title: `플롯 ${plots.length + 1}`,
      sensors: []
    });
    currentLayout = 'custom';
    document.querySelectorAll('.layout-btn').forEach(b => {
      b.classList.remove('bg-white', 'text-industrial-900', 'border-industrial-300', 'shadow-xs');
      b.classList.add('text-industrial-600', 'border-transparent');
    });
    renderPlots();
  });
}

// --- 7. Render Subplot Cards ---
function renderPlots() {
  if (!plotsGridContainer) return;
  plotsGridContainer.innerHTML = '';
  updateLayoutClass();

  plots.forEach((plot, plotIdx) => {
    const card = document.createElement('div');
    card.id = `card-${plot.id}`;
    card.className = 'plot-card bg-white rounded-xl border border-industrial-200 shadow-xs flex flex-col overflow-hidden relative transition-colors';

    // Sensor chips list
    let chipsHtml = '';
    if (plot.sensors.length === 0) {
      chipsHtml = `<span class="text-[11px] text-industrial-400 font-sans italic">센서 미할당 (좌측에서 드래그하여 추가)</span>`;
    } else {
      plot.sensors.forEach(feat => {
        const color = SENSOR_COLOR_MAP[feat] || '#6B7280';
        chipsHtml += `
          <span class="inline-flex items-center space-x-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold text-industrial-800 bg-industrial-100 border border-industrial-200">
            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${color}"></span>
            <span>${feat}</span>
            <button class="remove-sensor-btn hover:text-red-600 ml-0.5" data-plot-id="${plot.id}" data-sensor="${feat}" title="이 플롯에서 센서 제거">
              <i data-lucide="x" class="w-2.5 h-2.5"></i>
            </button>
          </span>
        `;
      });
    }

    card.innerHTML = `
      <!-- Plot Header Bar -->
      <div class="px-3 py-1.5 bg-industrial-50/80 border-b border-industrial-200 flex items-center justify-between flex-shrink-0">
        <div class="flex items-center space-x-2 min-w-0 pr-2">
          <span class="text-xs font-bold text-industrial-900 font-mono tracking-tight flex-shrink-0">${plot.title}</span>
          <div class="flex items-center space-x-1 overflow-x-auto py-0.5 no-scrollbar">
            ${chipsHtml}
          </div>
        </div>

        <!-- Controls -->
        <div class="flex items-center space-x-1 flex-shrink-0">
          <button class="add-sensor-to-plot-btn p-1 text-industrial-600 hover:text-industrial-900 hover:bg-industrial-200 rounded transition-colors" data-plot-id="${plot.id}" title="센서 추가">
            <i data-lucide="plus" class="w-3.5 h-3.5"></i>
          </button>
          <button class="clear-plot-btn p-1 text-industrial-500 hover:text-industrial-800 hover:bg-industrial-200 rounded transition-colors" data-plot-id="${plot.id}" title="센서 전체 비우기">
            <i data-lucide="rotate-ccw" class="w-3.5 h-3.5"></i>
          </button>
          ${plots.length > 1 ? `
            <button class="close-plot-btn p-1 text-industrial-400 hover:text-red-600 hover:bg-industrial-200 rounded transition-colors" data-plot-id="${plot.id}" title="이 플롯 창 닫기">
              <i data-lucide="x" class="w-3.5 h-3.5"></i>
            </button>
          ` : ''}
        </div>
      </div>

      <!-- Plot Canvas Area -->
      <div class="flex-1 relative w-full h-full bg-white overflow-hidden" id="canvas-wrapper-${plot.id}">
        <canvas id="canvas-${plot.id}" class="w-full h-full block cursor-crosshair"></canvas>
        <div class="plot-drop-overlay absolute inset-0 hidden border-2 border-dashed border-blue-500 bg-blue-50/70 flex items-center justify-center font-bold text-xs text-blue-700 pointer-events-none">
          센서를 여기에 드롭하여 추가
        </div>
      </div>
    `;

    // Drag and Drop into Plot Card
    card.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      const overlay = card.querySelector('.plot-drop-overlay');
      if (overlay) overlay.classList.remove('hidden');
    });

    card.addEventListener('dragleave', (e) => {
      if (!card.contains(e.relatedTarget)) {
        const overlay = card.querySelector('.plot-drop-overlay');
        if (overlay) overlay.classList.add('hidden');
      }
    });

    card.addEventListener('drop', (e) => {
      e.preventDefault();
      const overlay = card.querySelector('.plot-drop-overlay');
      if (overlay) overlay.classList.add('hidden');

      const feat = e.dataTransfer.getData('text/plain');
      if (feat && SENSOR_FEATURES.includes(feat)) {
        if (!plot.sensors.includes(feat)) {
          plot.sensors.push(feat);
          renderPlots();
        }
      }
    });

    // Remove Sensor Button
    card.querySelectorAll('.remove-sensor-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const s = btn.dataset.sensor;
        plot.sensors = plot.sensors.filter(x => x !== s);
        renderPlots();
      });
    });

    // Clear Plot Button
    const clearBtn = card.querySelector('.clear-plot-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        plot.sensors = [];
        renderPlots();
      });
    }

    // Close Plot Button
    const closeBtn = card.querySelector('.close-plot-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        plots = plots.filter(p => p.id !== plot.id);
        renderPlots();
      });
    }

    // Add Sensor Modal Open
    const addBtn = card.querySelector('.add-sensor-to-plot-btn');
    if (addBtn) {
      addBtn.addEventListener('click', () => {
        openAddSensorModal(plot.id);
      });
    }

    // Canvas Events (Synchronized Crosshair)
    const cvs = card.querySelector(`#canvas-${plot.id}`);
    if (cvs) {
      cvs.addEventListener('mousemove', (e) => {
        const rect = cvs.getBoundingClientRect();
        syncHoverNormX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        hoveredPlotId = plot.id;
        updateCrosshairTooltip(e.clientX, e.clientY, plot);
        drawAllCanvases();
      });

      cvs.addEventListener('mouseleave', () => {
        syncHoverNormX = null;
        hoveredPlotId = null;
        if (crosshairTooltip) crosshairTooltip.classList.add('hidden');
        drawAllCanvases();
      });
    }

    plotsGridContainer.appendChild(card);
  });

  lucide.createIcons({ root: plotsGridContainer });
  resizeAllCanvases();
  drawAllCanvases();
}

// --- 8. Modal: Quick Add Sensor to Plot ---
function openAddSensorModal(plotId) {
  modalTargetPlotId = plotId;
  const targetPlot = plots.find(p => p.id === plotId);
  if (!targetPlot || !addSensorModal || !modalSensorList) return;

  modalPlotTitle.textContent = `${targetPlot.title} 센서 추가`;
  modalSensorList.innerHTML = '';

  SENSOR_FEATURES.forEach(feat => {
    const isAlreadyAdded = targetPlot.sensors.includes(feat);
    const meta = SENSOR_META[feat];
    const color = SENSOR_COLOR_MAP[feat];

    const row = document.createElement('button');
    row.type = 'button';
    row.className = `w-full flex items-center justify-between p-2 rounded-lg border text-left text-xs transition-colors ${
      isAlreadyAdded ? 'bg-industrial-100 border-industrial-300 opacity-50 cursor-default' : 'bg-white hover:bg-industrial-50 border-industrial-200 cursor-pointer'
    }`;
    row.innerHTML = `
      <div class="flex items-center space-x-2">
        <span class="w-2 h-2 rounded-full" style="background-color: ${color}"></span>
        <span class="font-bold font-mono text-industrial-900">${feat}</span>
        <span class="text-[10px] text-industrial-500">(${meta.name})</span>
      </div>
      <span class="text-[10px] font-mono ${isAlreadyAdded ? 'text-industrial-400' : 'text-blue-600 font-semibold'}">${isAlreadyAdded ? '추가됨' : '+ 추가'}</span>
    `;

    if (!isAlreadyAdded) {
      row.addEventListener('click', () => {
        targetPlot.sensors.push(feat);
        addSensorModal.classList.add('hidden');
        renderPlots();
      });
    }

    modalSensorList.appendChild(row);
  });

  addSensorModal.classList.remove('hidden');
}

if (closeSensorModalBtn) {
  closeSensorModalBtn.addEventListener('click', () => {
    if (addSensorModal) addSensorModal.classList.add('hidden');
  });
}

// --- 9. Synchronized Crosshair Tooltip Overlay ---
function updateCrosshairTooltip(clientX, clientY, plot) {
  if (!crosshairTooltip || syncHoverNormX === null) return;
  if (plot.sensors.length === 0) {
    crosshairTooltip.classList.add('hidden');
    return;
  }

  const targetPoints = timeWindowSec * 2;
  let linesHtml = `<div class="font-bold text-[11px] text-industrial-900 border-b border-industrial-200 pb-1 mb-1">${plot.title}</div>`;

  plot.sensors.forEach(feat => {
    const buf = sensorBuffers[feat];
    const color = SENSOR_COLOR_MAP[feat];
    const meta = SENSOR_META[feat];
    if (buf && buf.length > 0) {
      const idx = Math.min(buf.length - 1, Math.max(0, Math.floor(syncHoverNormX * (buf.length - 1))));
      const val = Number(buf[idx]).toFixed(3);
      linesHtml += `
        <div class="flex items-center justify-between space-x-3 text-[10px]">
          <span class="flex items-center space-x-1">
            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${color}"></span>
            <span class="font-semibold text-industrial-700">${feat}:</span>
          </span>
          <span class="font-bold font-mono text-industrial-900">${val} <span class="font-normal text-[9px] text-industrial-400 font-sans">${meta.unit}</span></span>
        </div>
      `;
    }
  });

  crosshairTooltip.innerHTML = linesHtml;
  crosshairTooltip.style.left = `${clientX + 15}px`;
  crosshairTooltip.style.top = `${clientY + 15}px`;
  crosshairTooltip.classList.remove('hidden');
}

// --- 10. Canvas Rendering Engine ---
function resizeAllCanvases() {
  plots.forEach(plot => {
    const canvas = document.getElementById(`canvas-${plot.id}`);
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
  });
}

window.addEventListener('resize', () => {
  resizeAllCanvases();
  drawAllCanvases();
});

function drawAllCanvases() {
  plots.forEach(plot => {
    drawPlotCanvas(plot);
  });
}

function drawPlotCanvas(plot) {
  const canvas = document.getElementById(`canvas-${plot.id}`);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;

  ctx.clearRect(0, 0, w, h);

  // 1. Grid Lines (ATR Sensor Standard)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.04)';
  ctx.lineWidth = 1;
  for (let y = 20; y < h; y += 30) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let x = 0; x < w; x += 50) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  // 2. Empty State
  if (plot.sensors.length === 0) {
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('좌측 센서 탐색기에서 센서를 드래그하여 이 플롯에 놓으세요', w / 2, h / 2);
    return;
  }

  const targetPoints = timeWindowSec * 2; // e.g. 60s * 2Hz = 120 pts

  // 3. Draw Each Assigned Curve
  plot.sensors.forEach(feat => {
    const buf = sensorBuffers[feat];
    if (!buf || buf.length === 0) return;

    const data = buf.slice(-targetPoints);
    if (data.length === 0) return;

    const color = SENSOR_COLOR_MAP[feat] || '#2563EB';
    const span = SENSOR_META[feat]?.span || { min: -1.0, max: 1.0 };
    const spanRange = (span.max - span.min) || 1.0;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const step = (w - 10) / Math.max(1, (targetPoints - 1));
    const paddingX = 5;

    for (let i = 0; i < data.length; i++) {
      const x = paddingX + (i * step);
      let norm = (data[i] - span.min) / spanRange;
      norm = Math.max(0.02, Math.min(0.98, norm));
      const y = h - (norm * (h - 20) + 10);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // 4. Synchronized Vertical Crosshair Line
  if (syncHoverNormX !== null) {
    const crossX = syncHoverNormX * w;
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(crossX, 0);
    ctx.lineTo(crossX, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 5. Time Axis Readouts (Bottom Right / Left)
  ctx.fillStyle = '#9CA3AF';
  ctx.font = '9px Menlo, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`T-${timeWindowSec}s`, 6, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText('T-0s (Now)', w - 6, h - 4);
}

// --- 11. Controls Handlers ---
// Time Window Select
if (timeWindowSelect) {
  timeWindowSelect.addEventListener('change', (e) => {
    timeWindowSec = parseInt(e.target.value, 10) || 60;
    drawAllCanvases();
  });
}

// Pause/Resume Stream
if (pauseStreamBtn) {
  pauseStreamBtn.addEventListener('click', () => {
    isPaused = !isPaused;
    if (isPaused) {
      pauseText.textContent = '재생';
      if (pauseIcon) pauseIcon.setAttribute('data-lucide', 'play');
      if (studioStreamStatus) {
        studioStreamStatus.textContent = 'PAUSED';
        studioStreamStatus.parentElement.className = 'inline-flex items-center px-2 py-1 rounded text-[11px] font-mono font-semibold bg-amber-50 text-amber-700 border border-amber-200';
      }
    } else {
      pauseText.textContent = '일시정지';
      if (pauseIcon) pauseIcon.setAttribute('data-lucide', 'pause');
      if (studioStreamStatus) {
        studioStreamStatus.textContent = 'LIVE 2Hz';
        studioStreamStatus.parentElement.className = 'inline-flex items-center px-2 py-1 rounded text-[11px] font-mono font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200';
      }
    }
    lucide.createIcons();
    drawAllCanvases();
  });
}

// --- 12. WebSocket Real-time Ingestion ---
let ws;
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[ATR Sensor Studio] WebSocket Connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'telemetry') {
        const raw = msg.raw_sensors || {};

        // 17 센서 버퍼 및 최신 수치 갱신
        SENSOR_FEATURES.forEach(feat => {
          if (raw[feat] !== undefined) {
            const val = Number(raw[feat]);
            latestValues[feat] = val;
            sensorBuffers[feat].push(val);
            if (sensorBuffers[feat].length > MAX_BUFFER_POINTS) {
              sensorBuffers[feat].shift();
            }

            // 좌측 리스트의 실시간 수치 뱃지 갱신
            const badge = document.getElementById(`live-val-${feat}`);
            if (badge) {
              const meta = SENSOR_META[feat];
              badge.innerHTML = `${val.toFixed(2)} <span class="text-[9px] font-normal text-industrial-400 font-sans">${meta.unit}</span>`;
            }
          }
        });

        // 일시정지 상태가 아닐 때만 캔버스 재렌더링
        if (!isPaused) {
          drawAllCanvases();
        }
      }
    } catch (e) {
      console.error('[ATR Sensor Studio] WS Parse Error:', e);
    }
  };

  ws.onclose = () => {
    console.log('[ATR Sensor Studio] WS Disconnected. Reconnecting in 2s...');
    setTimeout(connectWebSocket, 2000);
  };
}

// --- 13. Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  renderSensorList();
  renderPlots();
  connectWebSocket();
});
