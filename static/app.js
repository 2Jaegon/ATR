const MAX_POINTS = 50; // 차트에 표시할 최대 데이터 갯수

// Chart.js 초기화
const ctx = document.getElementById('sensorChart').getContext('2d');
const sensorChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'FLOWCOOLPRESSURE',
            data: [],
            borderColor: 'rgba(0, 255, 204, 0.8)',
            backgroundColor: 'rgba(0, 255, 204, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0, // 기본 점 숨김
            pointBackgroundColor: [],
            pointBorderColor: [],
            pointRadiusArray: [] // 동적 크기
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: { display: false },
        },
        scales: {
            x: { 
                display: false,
                grid: { display: false }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#888899', font: { family: 'JetBrains Mono' } }
            }
        },
        elements: {
            point: {
                radius: function(context) {
                    return context.dataset.pointRadiusArray[context.dataIndex] || 0;
                }
            }
        }
    }
});

// UI 요소
const statusText = document.getElementById('main-status-text');
const bgGlow = document.getElementById('bg-glow-main');
const finalScoreEl = document.getElementById('final-score');
const ringProgress = document.getElementById('ring-progress');
const ragConsole = document.getElementById('rag-console');

const scoreTr = document.getElementById('score-tr');
const scoreGnn = document.getElementById('score-gnn');
const scoreLstm = document.getElementById('score-lstm');

const thYellow = document.getElementById('th-yellow');
const thRed = document.getElementById('th-red');

// WebSocket 연결
const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);

ws.onopen = () => {
    addLogEntry('[SYSTEM] WebSocket Connected. Waiting for telemetry...', 'system');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    if (msg.type === 'init') {
        thYellow.innerText = msg.yellow_threshold.toFixed(3);
        thRed.innerText = msg.red_threshold.toFixed(3);
        addLogEntry(`[SYSTEM] Initialized. Weights -> TR:${msg.weights.TR.toFixed(2)} GNN:${msg.weights.GNN.toFixed(2)} LSTM:${msg.weights.LSTM.toFixed(2)}`, 'system');
    }
    
    if (msg.type === 'telemetry') {
        updateChart(msg.sensor_value, msg.status);
        updateScores(msg.scores, msg.final_score, msg.status);
    }
    
    if (msg.type === 'report') {
        addLogEntry(`[RAG REPORT] ${msg.message}`, msg.status === 'red' ? 'urgent' : 'report');
    }
};

function updateChart(value, status) {
    const ds = sensorChart.data.datasets[0];
    
    sensorChart.data.labels.push('');
    ds.data.push(value);
    
    // 상태에 따른 포인트 스타일링
    if (status === 'yellow') {
        ds.pointBackgroundColor.push('#ffcc00');
        ds.pointBorderColor.push('#ffcc00');
        ds.pointRadiusArray.push(6);
    } else if (status === 'red') {
        ds.pointBackgroundColor.push('#ff3333');
        ds.pointBorderColor.push('#ff3333');
        ds.pointRadiusArray.push(8);
    } else {
        ds.pointBackgroundColor.push('rgba(0,0,0,0)');
        ds.pointBorderColor.push('rgba(0,0,0,0)');
        ds.pointRadiusArray.push(0);
    }
    
    if (ds.data.length > MAX_POINTS) {
        sensorChart.data.labels.shift();
        ds.data.shift();
        ds.pointBackgroundColor.shift();
        ds.pointBorderColor.shift();
        ds.pointRadiusArray.shift();
    }
    
    // 차트 선 색상 동적 변경
    if (status === 'red') {
        ds.borderColor = 'rgba(255, 51, 51, 0.8)';
        ds.backgroundColor = 'rgba(255, 51, 51, 0.1)';
    } else if (status === 'yellow') {
        ds.borderColor = 'rgba(255, 204, 0, 0.8)';
        ds.backgroundColor = 'rgba(255, 204, 0, 0.1)';
    } else {
        ds.borderColor = 'rgba(0, 255, 204, 0.8)';
        ds.backgroundColor = 'rgba(0, 255, 204, 0.1)';
    }
    
    sensorChart.update();
}

function updateScores(scores, finalScore, status) {
    scoreTr.innerText = scores.TR.toFixed(3);
    scoreGnn.innerText = scores.GNN.toFixed(3);
    scoreLstm.innerText = scores.LSTM.toFixed(3);
    finalScoreEl.innerText = finalScore.toFixed(3);
    
    // 링 게이지 업데이트 (0~1 기준)
    const maxDash = 283;
    const progress = Math.min(finalScore, 1.0);
    const offset = maxDash - (progress * maxDash);
    ringProgress.style.strokeDashoffset = offset;
    
    // 글로벌 테마 상태 업데이트
    document.body.className = '';
    bgGlow.className = 'bg-glow';
    
    if (status === 'red') {
        document.body.classList.add('state-red');
        bgGlow.classList.add('bg-glow-red');
        statusText.innerText = 'CRITICAL FAULT DETECTED';
    } else if (status === 'yellow') {
        document.body.classList.add('state-yellow');
        bgGlow.classList.add('bg-glow-yellow');
        statusText.innerText = 'ANOMALY WARNING';
    } else {
        bgGlow.classList.add('bg-glow-blue');
        statusText.innerText = 'SYSTEM NORMAL';
    }
}

function addLogEntry(text, type) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
    
    entry.innerHTML = `<span style="color:#556">[${timeStr}]</span> ${text}`;
    
    ragConsole.appendChild(entry);
    ragConsole.scrollTop = ragConsole.scrollHeight;
}
