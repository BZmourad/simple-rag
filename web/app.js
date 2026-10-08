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
function initIngest() {
  const startBtn = document.getElementById('start-btn');
  if (!startBtn) return;
  if (!requireAuth()) return;

  const logoutBtn = document.getElementById('logout-btn');
  logoutBtn.addEventListener('click', () => {
    clearToken();
    window.location.href = '/ui/login.html';
  });

  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const fileList = document.getElementById('file-list');
  const summary = document.getElementById('summary');
  const errEl = document.getElementById('ingest-error');

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
      let total = 0;
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
            total = parsed.total;
            progressText.textContent = `${total} file(s) to process`;
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

// ---------- Bootstrap ----------
document.addEventListener('DOMContentLoaded', () => {
  initLogin();
  initChat();
  initIngest();
});