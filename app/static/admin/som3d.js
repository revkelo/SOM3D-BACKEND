// ========================
// Config API + Auth
// ========================
const urlParams = new URLSearchParams(location.search);
const apiParam = urlParams.get('api');
if (apiParam) localStorage.setItem('som3d_api', apiParam);

function apiBase() {
  const saved = localStorage.getItem('som3d_api');
  if (saved) return saved.replace(/\/$/, '');
  if (/:8000$/.test(location.origin)) return '';
  const u = new URL(location.href);
  return `${u.protocol}//${u.hostname}:8000`;
}
function api(path) { return apiBase() + path; }
function token() { return localStorage.getItem('som3d_token'); }
function authHeaders() { return { Authorization: 'Bearer ' + token() }; }

async function ensureAuth() {
  const t = token();
  if (!t) { location.href = '/static/admin/login.html'; return false; }
  try {
    const res = await fetch(api('/auth/me'), { headers: authHeaders() });
    if (!res.ok) throw 0;
    return true;
  } catch {
    localStorage.removeItem('som3d_token');
    location.href = '/static/admin/login.html';
    return false;
  }
}

// ========================
// Drag & Drop .zip bonito
// ========================
const dropZone = document.getElementById('drop_zone');
const fileInput = document.getElementById('file_input');
const fileState = document.getElementById('file_state');
const fileNameEl = document.getElementById('file_name');
const fileSizeEl = document.getElementById('file_size');
const fileMsg = document.getElementById('file_msg');
const changeFileBtn = document.getElementById('change_file_btn');

const progressWrap = document.getElementById('upload_progress_wrap');
const progressBar  = document.getElementById('upload_progress_bar');
const progressText = document.getElementById('upload_progress_text');

function fmtBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0; let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`;
}

function showFileState(file) {
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = fmtBytes(file.size);
  fileState.classList.remove('hidden');
  fileMsg.textContent = '';
}

function clearFile() {
  fileInput.value = '';
  fileState.classList.add('hidden');
  fileMsg.textContent = '';
}

function acceptZip(file) {
  return file && /\.zip$/i.test(file.name);
}

dropZone.addEventListener('click', () => fileInput.click());
changeFileBtn.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('border-cyan-400', 'bg-black/30');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('border-cyan-400', 'bg-black/30');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('border-cyan-400', 'bg-black/30');
  const file = e.dataTransfer.files?.[0];
  if (!file) return;
  if (!acceptZip(file)) { fileMsg.textContent = 'Archivo inválido: debe ser .zip'; return; }
  fileInput.files = e.dataTransfer.files;
  showFileState(file);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) { clearFile(); return; }
  if (!acceptZip(file)) { clearFile(); fileMsg.textContent = 'Archivo inválido: debe ser .zip'; return; }
  showFileState(file);
});

function setUploadProgress(pct) {
  progressWrap.classList.remove('hidden');
  progressBar.style.width = pct + '%';
  progressText.textContent = Math.max(0, Math.min(100, pct)).toFixed(0) + '%';
}

// ========================
// UI helpers
// ========================
function progressCell(p) {
  const pct = Math.max(0, Math.min(100, Number(p || 0)));
  return `
    <div class="flex items-center">
      <div class="w-40 h-2 bg-gray-800 rounded">
        <div class="h-2 bg-cyan-500 rounded" style="width:${pct}%"></div>
      </div>
      <span class="ml-2 text-xs">${pct}%</span>
    </div>`;
}

// ========================
// Buscar paciente por cédula
// ========================
async function searchByCedula() {
  const input = document.getElementById('search_cedula');
  const select = document.getElementById('som_id_paciente');
  const msg = document.getElementById('search_msg');
  msg.textContent = '';

  const ced = (input?.value || '').trim();
  if (!ced) {
    select.innerHTML = '<option value="">-- Ingresa cédula para buscar --</option>';
    return;
  }

  try {
    const res = await fetch(api('/patients?doc_numero=' + encodeURIComponent(ced)), { headers: authHeaders() });
    if (!res.ok) throw new Error('Error buscando');
    const rows = await res.json();
    if (!rows.length) {
      select.innerHTML = '<option value="">-- Sin resultados --</option>';
      msg.textContent = 'Sin resultados';
      return;
    }
    select.innerHTML = rows.map(p =>
      `<option value="${p.id_paciente}">${p.id_paciente} - ${p.nombres} ${p.apellidos} (${p.doc_numero ?? ''})</option>`
    ).join('');
    msg.textContent = `${rows.length} resultado(s)`;
  } catch (e) {
    msg.textContent = 'Error buscando';
  }
}

// ========================
// Crear Job (con progreso real y validación)
// ========================
const submitBtn = document.getElementById('submit_btn');

async function submitSomJob(ev) {
  ev.preventDefault();
  const form = ev.target;
  const msg = document.getElementById('som_msg');
  msg.textContent = '';
  msg.classList.remove('text-green-400', 'text-red-400');

  const pid = document.getElementById('som_id_paciente')?.value?.trim();
  if (!pid) { msg.textContent = 'Debes seleccionar un paciente'; return; }

  const file = fileInput.files?.[0];
  if (!file) { msg.textContent = 'Selecciona un .zip'; fileInput.click(); return; }

  // construir FormData
  const fd = new FormData();
  fd.append('file', file);
  Array.from(form.elements).forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') {
      if (el.checked) fd.append(el.name, 'on'); // backend acepta 'on'
    } else if (el.tagName === 'SELECT' || el.tagName === 'INPUT') {
      if (el.name !== 'file') fd.append(el.name, el.value);
    }
  });

  submitBtn.disabled = true; submitBtn.textContent = 'Subiendo...';
  setUploadProgress(0);

  try {
    // XHR para poder reportar progreso
    const xhr = new XMLHttpRequest();
    xhr.open('POST', api('/som3d/jobs'));
    xhr.setRequestHeader('Authorization', 'Bearer ' + token());

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = (e.loaded / e.total) * 100;
        setUploadProgress(pct);
      }
    };

    const resData = await new Promise((resolve, reject) => {
      xhr.onload = () => {
        try {
          const json = JSON.parse(xhr.responseText || '{}');
          if (xhr.status >= 200 && xhr.status < 300) resolve(json);
          else reject(new Error(json?.detail?.message || json?.detail || 'Error al crear job'));
        } catch {
          if (xhr.status >= 200 && xhr.status < 300) resolve({});
          else reject(new Error('Error al crear job'));
        }
      };
      xhr.onerror = () => reject(new Error('Error de red'));
      xhr.send(fd);
    });

    msg.classList.add('text-green-400');
    msg.textContent = `Job creado: ${resData.job_id ?? '(sin id)'}`;

    // ocultar barra un momento después
    setTimeout(() => progressWrap.classList.add('hidden'), 1000);

    await loadJobs();
  } catch (err) {
    msg.classList.add('text-red-400');
    msg.textContent = err.message || String(err);
  } finally {
    submitBtn.disabled = false; submitBtn.textContent = 'Crear Job';
  }
}

// ========================
// Listado de Jobs
// ========================
let jobsPoll = null;

async function loadJobs() {
  const list = document.getElementById('jobs_list');
  const msg = document.getElementById('jobs_msg');
  if (!list) return;

  list.innerHTML = '<tr><td colspan="5" class="py-3">Cargando...</td></tr>';
  try {
    const res = await fetch(api('/som3d/jobs/mine'), { headers: authHeaders() });
    if (!res.ok) { list.innerHTML = '<tr><td colspan="5" class="py-3">Error</td></tr>'; return; }

    const data = await res.json();
    const rows = (data.jobs || []).sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')));

    if (!rows.length) { list.innerHTML = '<tr><td colspan="5" class="py-3 text-gray-400">Sin jobs</td></tr>'; msg.textContent = ''; return; }

    list.innerHTML = rows.map(j => `
      <tr>
        <td class="py-2 pr-3 font-mono text-xs">${j.job_id}</td>
        <td class="py-2 pr-3">${j.status}</td>
        <td class="py-2 pr-3">${j.phase || ''}</td>
        <td class="py-2 pr-3">${progressCell(j.percent)}</td>
        <td class="py-2 pr-3">
          <button class="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs" onclick="openLogsPanel('${j.job_id}')">Ver logs</button>
        </td>
      </tr>`).join('');
    msg.textContent = `${rows.length} job(s)`;
  } catch {
    list.innerHTML = '<tr><td colspan="5" class="py-3">Error</td></tr>';
  }
}

// ========================
// Logs
// ========================
let logsJobId = null;
let logsTimer = null;

async function updateLogs() {
  const pre = document.getElementById('logs_pre');
  const meta = document.getElementById('logs_meta');
  if (!logsJobId) return;

  try {
    const res = await fetch(api(`/som3d/jobs/${logsJobId}/log?tail=200`), { headers: authHeaders() });
    if (!res.ok) throw 0;
    const data = await res.json();
    const lines = (data?.lines || []).join('\n');
    pre.textContent = lines || '(Sin logs)';

    let status = data?.status;
    let phase = data?.phase;

    if (!status || !phase) {
      try {
        const jres = await fetch(api(`/som3d/jobs/${logsJobId}`), { headers: authHeaders() });
        if (jres.ok) {
          const job = await jres.json();
          status = status || job?.status;
          phase  = phase  || job?.phase;
        }
      } catch {}
    }
    meta.textContent = `Estado: ${status ?? '—'} · Fase: ${phase ?? '—'}`;
  } catch {
    pre.textContent = '(Error obteniendo logs)';
    meta.textContent = 'Estado: — · Fase: —';
  }
}

function openLogsPanel(jobId) {
  logsJobId = jobId;
  document.getElementById('logs_title').textContent = `Logs ${jobId}`;
  document.getElementById('logs_panel').classList.remove('hidden');
  updateLogs();

  const auto = document.getElementById('logs_auto');
  if (auto) {
    auto.checked = true;
    if (logsTimer) clearInterval(logsTimer);
    logsTimer = setInterval(updateLogs, 25000);
  }
}

function closeLogsPanel() {
  logsJobId = null;
  document.getElementById('logs_panel').classList.add('hidden');
  if (logsTimer) { clearInterval(logsTimer); logsTimer = null; }
  const auto = document.getElementById('logs_auto');
  if (auto) auto.checked = false;
}

// ========================
// Boot
// ========================
window.addEventListener('DOMContentLoaded', async () => {
  const ok = await ensureAuth();
  if (!ok) return;

  document.getElementById('btn_search_cedula')?.addEventListener('click', searchByCedula);
  document.getElementById('som_form')?.addEventListener('submit', submitSomJob);
  document.getElementById('jobs_refresh')?.addEventListener('click', loadJobs);
  document.getElementById('logs_update')?.addEventListener('click', updateLogs);
  document.getElementById('logs_close')?.addEventListener('click', closeLogsPanel);

  // Cerrar con ESC
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !document.getElementById('logs_panel').classList.contains('hidden')) {
      closeLogsPanel();
    }
  });

  // Respaldo: asegurar que toda la tarjeta togglea el checkbox
  document.querySelectorAll('.task-card').forEach(card => {
    card.addEventListener('click', (ev) => {
      // si se hace click en enlaces/inputs internos, dejamos el comportamiento nativo
      if (ev.target.tagName === 'INPUT' || ev.target.closest('input')) return;
      const cb = card.querySelector('input[type="checkbox"]');
      if (cb) cb.checked = !cb.checked;
    });
  });

  const auto = document.getElementById('logs_auto');
  auto?.addEventListener('change', (e) => {
    if (e.target.checked) {
      if (logsTimer) clearInterval(logsTimer);
      logsTimer = setInterval(updateLogs, 25000);
    } else {
      if (logsTimer) { clearInterval(logsTimer); logsTimer = null; }
    }
  });

  await loadJobs();

  if (window.jobsPoll) clearInterval(window.jobsPoll);
  window.jobsPoll = setInterval(loadJobs, 25000);
});

// Exponer para inline onclick
window.openLogsPanel = openLogsPanel;
window.closeLogsPanel = closeLogsPanel;
