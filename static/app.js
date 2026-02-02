(() => {
  const root = document.documentElement;
  const overlay = document.getElementById('loadingOverlay');
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const preview = document.getElementById('preview');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const resultContent = document.getElementById('resultContent');
  const tipsList = document.getElementById('tipsList');
  const themeToggle = document.getElementById('themeToggle');

  const THEME_KEY = 'dermaware-theme';

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }

  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') {
      setTheme(saved);
      return;
    }
    setTheme('light');
  }

  themeToggle.addEventListener('click', () => {
    const current = root.getAttribute('data-theme') || 'light';
    setTheme(current === 'light' ? 'dark' : 'light');
  });

  function showOverlay(show) {
    overlay.classList.toggle('active', !!show);
    overlay.setAttribute('aria-hidden', show ? 'false' : 'true');
  }

  function setPreview(file) {
    const url = URL.createObjectURL(file);
    preview.innerHTML = `<img src="${url}" alt="preview">`;
    preview.style.display = 'block';
  }

  function clearResult() {
    resultContent.innerHTML = `<p class="muted">No result yet. Upload an image and analyze.</p>`;
    tipsList.innerHTML = '';
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#60a5fa';
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '';
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      setPreview(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) setPreview(fileInput.files[0]);
  });

  async function analyze() {
    const mode = document.querySelector('input[name="mode"]:checked')?.value || 'online';
    const file = fileInput.files?.[0];
    if (!file) {
      alert('Please select an image file first.');
      return;
    }
    showOverlay(true);
    analyzeBtn.disabled = true;
    clearResult();
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      form.append('mode', mode);
      const res = await fetch('/classify', { method: 'POST', body: form });
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      resultContent.innerHTML = `<p class="muted">Analysis failed. Please try again.</p>`;
      console.error(err);
    } finally {
      analyzeBtn.disabled = false;
      showOverlay(false);
    }
  }

  function pct(x) {
    if (typeof x !== 'number') return '—';
    return `${(x * 100).toFixed(1)}%`;
  }

  function renderResult(data) {
    const label = data.label || 'Unknown';
    const confidence = pct(data.confidence);
    const tagalog = data.tagalog_term ? ` • ${data.tagalog_term}` : '';
    resultContent.innerHTML = `
      <div class="result-line"><strong>${label}</strong><span class="muted"> (${confidence})${tagalog}</span></div>
      ${data.explanation ? `<p>${data.explanation}</p>` : ''}
      ${data.all_results ? renderAllResults(data.all_results) : ''}
    `;
    const todos = [...(data.dos || []), ...(data.donts || [])];
    tipsList.innerHTML = todos.map(t => `<li>${t}</li>`).join('');
  }

  function renderAllResults(all) {
    const entries = Object.entries(all).sort((a, b) => b[1] - a[1]).slice(0, 5);
    return `
      <div class="muted" style="margin-top:8px">Top predictions:</div>
      <div style="display:grid;gap:6px;margin-top:6px">
        ${entries.map(([k, v]) => `
          <div style="display:flex;justify-content:space-between">
            <span>${k.replace(/_/g, ' ')}</span>
            <span class="muted">${pct(v)}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  analyzeBtn.addEventListener('click', analyze);
  initTheme();
})(); 

