// Shared helpers
const API_BASE = window.location.origin;

function getToken() {
  return localStorage.getItem('rag_token');
}

function setToken(token) {
  localStorage.setItem('rag_token', token);
}

function clearToken() {
  localStorage.removeItem('rag_token');
}

function requireAuth() {
  const token = getToken();
  if (!token) {
    window.location.href = '/ui/login.html';
    return null;
  }
  return token;
}

async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (resp.status === 401) {
    clearToken();
    window.location.href = '/ui/login.html';
    throw new Error('Unauthorized');
  }
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return resp;
}

// ---------- Login page ----------
function initLogin() {
  const form = document.getElementById('login-form');
  const err = document.getElementById('login-error');
  if (!form) return;

  // If already logged in, go to chat
  if (getToken()) {
    window.location.href = '/ui/chat.html';
    return;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    err.classList.remove('visible');
    const username = form.username.value.trim();
    const password = form.password.value;
    if (!username || !password) return;

    const btn = form.querySelector('button');
    btn.disabled = true;
    btn.textContent = 'Signing in…';

    try {
      const body = new URLSearchParams({ username, password });
      const resp = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
      });
      if (!resp.ok) throw new Error('Incorrect username or password');
      const data = await resp.json();
      setToken(data.access_token);
      window.location.href = '/ui/chat.html';
    } catch (ex) {
      err.textContent = ex.message;
      err.classList.add('visible');
      btn.disabled = false;
      btn.textContent = 'Sign in';
    }
  });
}

// ---------- Chat page ----------
function initChat() {
  const messagesEl = document.getElementById('chat-messages');
  if (!messagesEl) return;
  if (!requireAuth()) return;

  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const logoutBtn = document.getElementById('logout-btn');

  logoutBtn.addEventListener('click', () => {
    clearToken();
    window.location.href = '/ui/login.html';
  });

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    appendMessage('user', question);
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    const assistantEl = appendMessage('assistant', '');
    const contentEl = assistantEl.querySelector('.content');

    try {
      const token = getToken();
      const resp = await fetch(`${API_BASE}/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let sources = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const raw of events) {
          if (!raw.trim()) continue;
          const lines = raw.split('\n');
          let eventType = 'message';
          let data = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim();
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (!data) continue;

          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }

          if (eventType === 'token') {
            contentEl.textContent += parsed.token;
            messagesEl.scrollTop = messagesEl.scrollHeight;
          } else if (eventType === 'done') {
            sources = parsed.sources || [];
          } else if (eventType === 'error') {
            contentEl.textContent += `\n[Error: ${parsed.error}]`;
          }
        }
      }

      if (sources.length) {
        const srcEl = document.createElement('div');
        srcEl.className = 'sources';
        srcEl.innerHTML = '<strong>Sources:</strong> ' +
          sources.map(s => `${escapeHtml(s.filename || 'unknown')} — page ${s.page || '?'}`).join(', ');
        assistantEl.appendChild(srcEl);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } catch (ex) {
      contentEl.textContent = `Error: ${ex.message}`;
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });

  input.focus();
}

function appendMessage(role, text) {
  const messagesEl = document.getElementById('chat-messages');
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `<div class="content"></div>`;
  el.querySelector('.content').textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ---------- Ingest page ----------
async function initIngest() {
  const startBtn = document.getElementById('start-btn');
  if (!startBtn) return;
  if (!requireAuth()) return;

  const dropZone = document.getElementById('upload-drop');
  const fileInput = document.getElementById('file-input');
  const uploadList = document.getElementById('upload-list');
  const filesTable = document.getElementById('files-table');
  const refreshBtn = document.getElementById('refresh-btn');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const fileList = document.getElementById('file-list');
  const summary = document.getElementById('summary');
  const errEl = document.getElementById('ingest-error');

  async function refreshFiles() {
    try {
      const resp = await apiFetch('/files/list');
      const data = await resp.json();
      renderFiles(data.files || []);
    } catch (ex) {
      filesTable.innerHTML = `<div class="empty">Error: ${escapeHtml(ex.message)}</div>`;
    }
  }

  function renderFiles(files) {
    if (!files.length) {
      filesTable.innerHTML = '<div class="empty">No files yet. Upload one above.</div>';
      return;
    }
    filesTable.innerHTML = '';
    for (const f of files) {
      const row = document.createElement('div');
      row.className = 'file-item';
      row.innerHTML = `
        <span class="name">${escapeHtml(f.name)}</span>
        <span class="size">${formatBytes(f.size)}</span>
        <span class="actions">
          <button class="btn-danger" data-name="${escapeHtml(f.name)}">Delete</button>
        </span>
      `;
      row.querySelector('button').addEventListener('click', () => deleteFile(f.name));
      filesTable.appendChild(row);
    }
  }

  async function deleteFile(name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
      await apiFetch(`/files/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await refreshFiles();
    } catch (ex) {
      alert(`Delete failed: ${ex.message}`);
    }
  }

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragging'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles([...fileInput.files]);
    fileInput.value = '';
  });

  async function uploadFiles(files) {
    for (const file of files) {
      const item = document.createElement('div');
      item.className = 'upload-item';
      item.innerHTML = `<span>${escapeHtml(file.name)}</span><span class="status uploading">uploading…</span>`;
      uploadList.appendChild(item);
      const status = item.querySelector('.status');

      try {
        const token = getToken();
        const form = new FormData();
        form.append('file', file);
        const resp = await fetch(`${API_BASE}/files/upload`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` },
          body: form,
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
          throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        status.textContent = 'done';
        status.className = 'status done';
      } catch (ex) {
        status.textContent = 'failed';
        status.className = 'status error';
        item.title = ex.message;
      }
    }
    await refreshFiles();
    setTimeout(() => { uploadList.innerHTML = ''; }, 3000);
  }

  refreshBtn.addEventListener('click', refreshFiles);

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    fileList.innerHTML = '';
    summary.classList.remove('visible');
    errEl.classList.remove('visible');
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting…';

    const token = getToken();
    try {
      const resp = await fetch(`${API_BASE}/ingest/stream`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const rows = new Map();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const raw of events) {
          if (!raw.trim()) continue;
          const lines = raw.split('\n');
          let eventType = 'message';
          let data = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim();
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (!data) continue;
          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }

          if (eventType === 'start') {
            progressText.textContent = `${parsed.total} file(s) to process`;
          } else if (eventType === 'file') {
            updateFileRow(rows, fileList, parsed);
            const pct = Math.round((parsed.index / parsed.total) * 100);
            progressFill.style.width = `${pct}%`;
            progressText.textContent = `${parsed.index} / ${parsed.total} — ${pct}%`;
          } else if (eventType === 'done') {
            progressFill.style.width = '100%';
            progressText.textContent = 'Complete';
            document.getElementById('sum-docs').textContent = parsed.documents_processed;
            document.getElementById('sum-created').textContent = parsed.chunks_created;
            document.getElementById('sum-stored').textContent = parsed.chunks_stored;
            document.getElementById('sum-total').textContent = parsed.total_chunks;
            summary.classList.add('visible');
          } else if (eventType === 'error') {
            errEl.textContent = parsed.error;
            errEl.classList.add('visible');
          }
        }
      }
    } catch (ex) {
      errEl.textContent = ex.message;
      errEl.classList.add('visible');
    } finally {
      startBtn.disabled = false;
    }
  });

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(2)} MB`;
  }

  await refreshFiles();
}

function updateFileRow(rows, container, data) {
  let row = rows.get(data.filename);
  if (!row) {
    row = document.createElement('div');
    row.className = 'file-row';
    row.innerHTML = `<span class="name"></span><span class="stage"></span>`;
    container.appendChild(row);
    rows.set(data.filename, row);
  }
  row.querySelector('.name').textContent = data.filename;
  const stageEl = row.querySelector('.stage');
  stageEl.textContent = data.stage;
  stageEl.className = `stage ${data.stage}`;
}

async function initSettings() {
  const saveBtn = document.getElementById('save-btn');
  if (!saveBtn) return;
  if (!requireAuth()) return;

  const providerSelect = document.getElementById('LLM_PROVIDER');
  const sections = document.querySelectorAll('.provider-section');
  const errorEl = document.getElementById('settings-error');
  const successEl = document.getElementById('settings-success');
  const testResult = document.getElementById('test-result');
  const testBtn = document.getElementById('test-btn');
  const resetBtn = document.getElementById('reset-btn');

  // All editable field IDs
  const FIELDS = [
    'LLM_PROVIDER', 'LLAMACPP_BASE_URL', 'LLAMACPP_MODEL', 'LLAMACPP_TIMEOUT',
    'OPENROUTER_API_KEY', 'OPENROUTER_MODEL', 'OPENROUTER_BASE_URL',
    'GEMINI_API_KEY', 'GEMINI_MODEL',
    'TOP_K', 'RERANK_TOP_K', 'RELEVANCE_THRESHOLD',
    'HYBRID_ENABLED', 'RERANKER_ENABLED',
    'QUERY_EXPANSION_ENABLED', 'QUERY_EXPANSION_MODE',
    'MAX_HISTORY_MESSAGES',
    'RATE_LIMIT_PER_MINUTE', 'RATE_LIMIT_PER_DAY',
  ];

  function showProviderSection(provider) {
    sections.forEach(sec => {
      sec.hidden = sec.dataset.provider !== provider;
    });
  }

  providerSelect.addEventListener('change', () => showProviderSection(providerSelect.value));

  async function loadSettings() {
    try {
      const resp = await apiFetch('/settings');
      const data = await resp.json();
      const s = data.settings || {};
      for (const k of FIELDS) {
        const el = document.getElementById(k);
        if (!el) continue;
        if (s[k] !== undefined && s[k] !== null) {
          el.value = s[k];
        }
      }
      showProviderSection(providerSelect.value || 'llamacpp');
    } catch (ex) {
      errorEl.textContent = `Failed to load: ${ex.message}`;
      errorEl.classList.add('visible');
    }
  }

  saveBtn.addEventListener('click', async () => {
    errorEl.classList.remove('visible');
    successEl.classList.remove('visible');
    const values = {};
    for (const k of FIELDS) {
      const el = document.getElementById(k);
      if (!el) continue;
      const v = el.value.trim();
      if (v === '') continue;
      // Skip masked secrets
      if (v.startsWith('****')) continue;
      values[k] = v;
    }
    try {
      const resp = await apiFetch('/settings', {
        method: 'PUT',
        body: JSON.stringify({ values }),
      });
      const data = await resp.json();
      successEl.textContent = `Saved ${data.count} setting(s). Restart the API to fully apply.`;
      successEl.classList.add('visible');
      setTimeout(() => successEl.classList.remove('visible'), 5000);
    } catch (ex) {
      errorEl.textContent = `Save failed: ${ex.message}`;
      errorEl.classList.add('visible');
    }
  });

  testBtn.addEventListener('click', async () => {
    testResult.classList.remove('visible', 'ok', 'error');
    testResult.textContent = 'Testing…';
    testResult.classList.add('visible');
    try {
      const resp = await apiFetch('/settings/test', { method: 'POST' });
      const data = await resp.json();
      testResult.classList.add(data.status === 'ok' ? 'ok' : 'error');
      testResult.textContent = `${data.provider}: ${data.status} — ${typeof data.detail === 'object' ? JSON.stringify(data.detail) : data.detail}`;
    } catch (ex) {
      testResult.classList.add('error');
      testResult.textContent = `Test failed: ${ex.message}`;
    }
  });

  resetBtn.addEventListener('click', async () => {
    if (!confirm('Reset runtime overrides and fall back to .env values?')) return;
    try {
      await apiFetch('/settings/reset', { method: 'POST' });
      successEl.textContent = 'Runtime overrides cleared. Reload the page to see .env values.';
      successEl.classList.add('visible');
      await loadSettings();
    } catch (ex) {
      errorEl.textContent = `Reset failed: ${ex.message}`;
      errorEl.classList.add('visible');
    }
  });

  await loadSettings();
}
// ---------- Bootstrap ----------
document.addEventListener('DOMContentLoaded', () => {
  initLogin();
  initChat();
  initIngest();
  initSettings();
});