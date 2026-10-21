/**
 * app.js
 * ------
 * Interactive drawing canvas, image upload, and prediction pipeline renderer.
 * Author: Naravula Prudhvi Sri Bhanu Vivek
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabDraw = document.getElementById('tabDraw');
  const tabUpload = document.getElementById('tabUpload');
  const drawSection = document.getElementById('drawSection');
  const uploadSection = document.getElementById('uploadSection');

  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  const btnPredictCanvas = document.getElementById('btnPredictCanvas');
  const btnClearCanvas = document.getElementById('btnClearCanvas');
  const btnUndoCanvas = document.getElementById('btnUndoCanvas');
  const brushButtons = document.querySelectorAll('.brush-btn');

  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const previewBox = document.getElementById('previewBox');
  const previewImage = document.getElementById('previewImage');
  const btnChangeImage = document.getElementById('btnChangeImage');
  const btnPredictImage = document.getElementById('btnPredictImage');

  const resultsCard = document.getElementById('resultsCard');
  const heroSequence = document.getElementById('heroSequence');
  const statDigits = document.getElementById('statDigits');
  const statConfidence = document.getElementById('statConfidence');
  const statLatency = document.getElementById('statLatency');
  const linesContainer = document.getElementById('linesContainer');

  // Canvas State
  let isDrawing = false;
  let strokeWidth = 18;
  let hasDrawn = false;
  let undoStack = [];
  let currentStroke = [];

  // -------------------------------------------------------------------------
  // 1. High DPI Canvas Setup
  // -------------------------------------------------------------------------
  function setupCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = 640;
    const height = 320;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = '100%';
    canvas.style.maxWidth = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.scale(dpr, dpr);
    resetCanvasBackground();
    saveCanvasState();
  }

  function resetCanvasBackground() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function saveCanvasState() {
    if (undoStack.length >= 15) {
      undoStack.shift();
    }
    undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  }

  setupCanvas();
  window.addEventListener('resize', () => {
    // Keep aspect ratio
  });

  // -------------------------------------------------------------------------
  // 2. Canvas Drawing Logic (Mouse & Touch)
  // -------------------------------------------------------------------------
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = (canvas.width / (window.devicePixelRatio || 1)) / rect.width;
    const scaleY = (canvas.height / (window.devicePixelRatio || 1)) / rect.height;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  let strokePoints = [];

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    const coords = getCanvasCoords(e);
    strokePoints = [coords];

    ctx.strokeStyle = '#FFFFFF';
    ctx.fillStyle = '#FFFFFF';
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.arc(coords.x, coords.y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCanvasCoords(e);
    strokePoints.push(coords);

    if (strokePoints.length >= 3) {
      const prev = strokePoints[strokePoints.length - 2];
      const curr = strokePoints[strokePoints.length - 1];
      const midX = (prev.x + curr.x) / 2;
      const midY = (prev.y + curr.y) / 2;
      ctx.quadraticCurveTo(prev.x, prev.y, midX, midY);
      ctx.stroke();
    } else {
      ctx.lineTo(coords.x, coords.y);
      ctx.stroke();
    }
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    if (strokePoints.length >= 2) {
      const last = strokePoints[strokePoints.length - 1];
      ctx.lineTo(last.x, last.y);
      ctx.stroke();
    }
    ctx.closePath();
    saveCanvasState();
    strokePoints = [];
  }

  // Mouse Listeners
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseleave', stopDrawing);

  // Touch Listeners (Mobile & Tablet Support)
  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', stopDrawing);
  canvas.addEventListener('touchcancel', stopDrawing);

  // Brush Size Selection
  brushButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      brushButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      strokeWidth = parseInt(btn.dataset.size, 10);
    });
  });

  // Clear Canvas
  btnClearCanvas.addEventListener('click', () => {
    resetCanvasBackground();
    hasDrawn = false;
    undoStack = [];
    saveCanvasState();
    hideResults();
  });

  // Undo Stroke
  btnUndoCanvas.addEventListener('click', () => {
    if (undoStack.length > 1) {
      undoStack.pop(); // Remove current
      const prevState = undoStack[undoStack.length - 1];
      ctx.putImageData(prevState, 0, 0);
    } else {
      resetCanvasBackground();
      hasDrawn = false;
      undoStack = [];
      saveCanvasState();
    }
  });

  // Preset Demonstrations
  document.querySelectorAll('.preset-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const demo = chip.dataset.demo;
      drawPresetDigits(demo);
    });
  });

  function drawPresetDigits(text) {
    resetCanvasBackground();
    hasDrawn = true;
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 96px "Plus Jakarta Sans", -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const totalChars = text.length;
    // Spacing between characters so they do not fuse into one single blob
    const charSpacing = totalChars > 2 ? 80 : 105;
    const totalWidth = (totalChars - 1) * charSpacing;
    const startX = 320 - totalWidth / 2;

    for (let i = 0; i < totalChars; i++) {
      const charX = startX + i * charSpacing;
      ctx.fillText(text[i], charX, 160);
    }

    saveCanvasState();
    showToast(`Loaded "${text}" sample on canvas. Click Recognize to test!`);
  }

  // -------------------------------------------------------------------------
  // 3. Tab Navigation (Draw vs Upload)
  // -------------------------------------------------------------------------
  tabDraw.addEventListener('click', () => {
    tabDraw.classList.add('active');
    tabUpload.classList.remove('active');
    drawSection.style.display = 'block';
    uploadSection.style.display = 'none';
  });

  tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabDraw.classList.remove('active');
    drawSection.style.display = 'none';
    uploadSection.style.display = 'block';
  });

  // -------------------------------------------------------------------------
  // 4. Image Upload & Drag-and-Drop
  // -------------------------------------------------------------------------
  let selectedFile = null;

  dropzone.addEventListener('click', () => imageInput.click());

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleFileSelected(files[0]);
    } else {
      showToast('Please drop a valid image file.', 'error');
    }
  });

  imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropzone.style.display = 'none';
      previewBox.classList.add('active');
    };
    reader.readAsDataURL(file);
  }

  btnChangeImage.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    previewBox.classList.remove('active');
    dropzone.style.display = 'block';
    hideResults();
  });

  // -------------------------------------------------------------------------
  // 5. Predict from Canvas
  // -------------------------------------------------------------------------
  btnPredictCanvas.addEventListener('click', async () => {
    // Check if anything has been drawn
    if (!hasDrawn && !isCanvasDrawnOn()) {
      showToast('Please draw at least one digit on the canvas first.', 'error');
      return;
    }

    setLoading(btnPredictCanvas, true, 'Recognizing...');
    try {
      const dataUrl = canvas.toDataURL('image/png');
      const response = await fetch('/api/predict-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        showToast(data.message || data.detail || 'Prediction failed', 'error');
        return;
      }

      renderResults(data);
      showToast('Sequence recognized successfully!', 'success');
    } catch (err) {
      showToast(`Network error: ${err.message}`, 'error');
    } finally {
      setLoading(btnPredictCanvas, false, '🔍 Recognize Sequence');
    }
  });

  function isCanvasDrawnOn() {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < imgData.data.length; i += 4) {
      if (imgData.data[i] > 30) return true; // Found white/bright pixel
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // 6. Predict from Uploaded Image
  // -------------------------------------------------------------------------
  btnPredictImage.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please choose an image file first.', 'error');
      return;
    }

    setLoading(btnPredictImage, true, 'Analyzing...');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/api/predict-image', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        showToast(data.message || data.detail || 'Prediction failed', 'error');
        return;
      }

      renderResults(data);
      showToast('Image analyzed successfully!', 'success');
    } catch (err) {
      showToast(`Network error: ${err.message}`, 'error');
    } finally {
      setLoading(btnPredictImage, false, '🔍 Recognize Uploaded Image');
    }
  });

  // -------------------------------------------------------------------------
  // 7. Results Renderer
  // -------------------------------------------------------------------------
  function renderResults(data) {
    heroSequence.textContent = data.sequence || 'None';
    statDigits.textContent = `${data.total_digits} digit${data.total_digits === 1 ? '' : 's'}`;

    // Average confidence calculation
    let totalConf = 0;
    let count = 0;
    data.lines.forEach((line) => {
      line.digits.forEach((d) => {
        totalConf += d.confidence;
        count++;
      });
    });
    const avgConf = count > 0 ? (totalConf / count) * 100 : 0;
    statConfidence.textContent = `${avgConf.toFixed(1)}%`;
    statLatency.textContent = `${data.latency_ms} ms`;

    // Render Line by Line
    linesContainer.innerHTML = '';
    data.lines.forEach((line) => {
      const lineGroup = document.createElement('div');
      lineGroup.className = 'line-group';

      const lineHeader = document.createElement('div');
      lineHeader.className = 'line-header';
      lineHeader.innerHTML = `
        <div class="line-tag">📍 Line ${line.line_number}: <strong>${line.line_text}</strong></div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">${line.digits.length} digit(s)</div>
      `;
      lineGroup.appendChild(lineHeader);

      const grid = document.createElement('div');
      grid.className = 'line-digits-grid';

      line.digits.forEach((d, idx) => {
        const card = document.createElement('div');
        card.className = 'digit-card';

        // Probability bars HTML
        let probBarsHtml = '';
        d.probabilities.forEach((p, probIdx) => {
          const isWinner = probIdx === d.digit;
          const pct = (p * 100).toFixed(1);
          probBarsHtml += `
            <div class="prob-bar-row ${isWinner ? 'winner' : ''}">
              <span class="prob-digit-label">${probIdx}</span>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${pct}%"></div>
              </div>
              <span style="font-size: 0.65rem; color: var(--text-muted); min-width: 32px; text-align: right;">${pct}%</span>
            </div>
          `;
        });

        card.innerHTML = `
          <div class="digit-idx-badge">Position #${idx + 1}</div>
          <div class="digit-crop-preview" title="28x28 MNIST centered tensor">
            <img src="${d.preview_base64}" alt="Digit ${d.digit}" />
          </div>
          <div class="digit-prediction-number">${d.digit}</div>
          <div class="digit-conf-pill">${d.confidence_pct}% conf</div>
          <div>
            <button type="button" class="prob-toggle-btn">View Probabilities</button>
            <div class="prob-chart-drawer">
              ${probBarsHtml}
            </div>
          </div>
        `;

        // Toggle probability accordion
        const toggleBtn = card.querySelector('.prob-toggle-btn');
        const drawer = card.querySelector('.prob-chart-drawer');
        toggleBtn.addEventListener('click', () => {
          drawer.classList.toggle('open');
          toggleBtn.textContent = drawer.classList.contains('open') ? 'Hide Probabilities' : 'View Probabilities';
        });

        grid.appendChild(card);
      });

      lineGroup.appendChild(grid);
      linesContainer.appendChild(lineGroup);
    });

    resultsCard.classList.add('active');
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideResults() {
    resultsCard.classList.remove('active');
  }

  // -------------------------------------------------------------------------
  // 8. Utilities: Loading state & Toast Notifications
  // -------------------------------------------------------------------------
  function setLoading(btn, isLoading, text) {
    if (isLoading) {
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span> ${text}`;
    } else {
      btn.disabled = false;
      btn.innerHTML = text;
    }
  }

  function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'error' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.25s ease-out';
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }
});
