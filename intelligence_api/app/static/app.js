// Purplle Intelligence — interactive dashboard.
// Streams the annotated mp4 with the live event feed timeline-synced to playback.

const STORE_ID = document.getElementById('store-id').textContent.trim();

const state = {
  cameras: [],
  current: null,
  events: [],
  preferAnnotated: true,
  lastFiredId: null,
};

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(url + ' -> ' + r.status);
  return r.json();
}

const fmtType = (t) => (t || '').replace(/_/g, ' ');
const shortId = (s, n=6) => (s || '').slice(-n);

// ============================================================================
// Cameras
// ============================================================================
async function initCameras() {
  const data = await fetchJSON('/dashboard/cameras');
  state.cameras = data.cameras;
  const nav = document.getElementById('cam-nav');
  const sel = document.getElementById('cam-select');
  const upSel = document.getElementById('upload-cam');
  nav.innerHTML = ''; sel.innerHTML = ''; upSel.innerHTML = '';

  for (const c of state.cameras) {
    const btn = document.createElement('button');
    btn.dataset.cam = c.camera_id;
    btn.innerHTML = `<span>${c.camera_id}</span>` +
      (c.annotated_url ? '' : '<span class="ml-2 inline-block w-1.5 h-1.5 rounded-full bg-amber-400" title="no annotated mp4"></span>');
    btn.onclick = () => selectCamera(c.camera_id);
    nav.appendChild(btn);

    const opt = document.createElement('option');
    opt.value = c.camera_id; opt.textContent = c.camera_id;
    sel.appendChild(opt);
    upSel.appendChild(opt.cloneNode(true));
  }
  sel.onchange = (ev) => selectCamera(ev.target.value);
  document.getElementById('cam-toggle-annot').onchange = (ev) => {
    state.preferAnnotated = ev.target.checked;
    if (state.current) loadVideoSource();
  };
  if (state.cameras.length) selectCamera(state.cameras[0].camera_id);
}

function selectCamera(camId) {
  const cam = state.cameras.find(c => c.camera_id === camId);
  if (!cam) return;
  state.current = cam;
  state.lastFiredId = null;
  document.getElementById('cam-select').value = camId;
  for (const b of document.querySelectorAll('#cam-nav button')) {
    b.classList.toggle('active', b.dataset.cam === camId);
  }
  loadVideoSource();
  loadEvents();
}

function loadVideoSource() {
  const cam = state.current;
  const v = document.getElementById('cam-video');
  const tag = document.getElementById('cam-source-tag');
  let src = cam.clip_url;
  let label = 'raw clip';
  if (state.preferAnnotated && cam.annotated_url) {
    src = cam.annotated_url;
    label = 'annotated · boxes / IDs / zones';
    tag.classList.add('text-brand-300', 'border-brand-700/60', 'bg-brand-500/10');
    tag.classList.remove('text-amber-300', 'border-amber-700/60', 'bg-amber-500/10');
  } else {
    tag.classList.remove('text-brand-300', 'border-brand-700/60', 'bg-brand-500/10');
    tag.classList.add('text-amber-300', 'border-amber-700/60', 'bg-amber-500/10');
    if (state.preferAnnotated && !cam.annotated_url) {
      label = 'no annotated mp4 yet';
    }
  }
  v.src = src;
  tag.textContent = label;
  v.ontimeupdate = onVideoTime;
  v.onloadedmetadata = renderTimelineMarks;
  document.getElementById('ev-toast-stack').innerHTML = '';
}

// ============================================================================
// Events
// ============================================================================
async function loadEvents() {
  const cam = state.current;
  const list = document.getElementById('ev-list');
  list.innerHTML = '<div class="text-center text-slate-500 text-xs py-6">loading events…</div>';
  try {
    const data = await fetchJSON(`/dashboard/events?camera_id=${cam.camera_id}&limit=5000`);
    state.events = data.events;
    document.getElementById('ev-count').textContent =
      `${state.events.length} event${state.events.length === 1 ? '' : 's'}`;
    updateCamStats();
    renderTimelineMarks();
    renderFeed();
    if (!state.events.length) {
      list.innerHTML = '<div class="text-center text-slate-500 text-xs py-6">no events for this camera</div>';
    }
  } catch (e) {
    list.innerHTML = `<div class="text-center text-rose-400 text-xs py-6">error: ${e.message}</div>`;
  }
}

function updateCamStats() {
  const evs = state.events;
  document.getElementById('cam-stat-total').textContent = evs.length;
  const visitors = new Set(evs.map(e => e.visitor_id));
  document.getElementById('cam-stat-vis').textContent = visitors.size;
  const visStaff = new Map();
  for (const e of evs) visStaff.set(e.visitor_id, e.is_staff);
  let staff = 0, cust = 0;
  for (const v of visStaff.values()) v ? staff++ : cust++;
  document.getElementById('cam-stat-mix').textContent = `${staff} / ${cust}`;
}

function renderFeed() {
  const list = document.getElementById('ev-list');
  list.innerHTML = '';
  for (const ev of state.events) {
    const row = document.createElement('div');
    row.className = 'ev-row' + (ev.is_staff ? ' staff' : '');
    row.dataset.eventId = ev.event_id;
    row.innerHTML = `
      <div class="ev-left">
        <span class="ev-dot"></span>
        <span class="ev-type">${fmtType(ev.event_type)}</span>
      </div>
      <div class="ev-meta">
        <span class="ev-zone">${ev.zone_id || '—'}</span>
        <span class="ev-vid">${shortId(ev.visitor_id)} · ${ev.is_staff ? 'STAFF' : 'cust'}</span>
      </div>
      <span class="ev-time">${(ev.offset_ms/1000).toFixed(1)}s</span>`;
    list.appendChild(row);
  }
}

function renderTimelineMarks() {
  const v = document.getElementById('cam-video');
  const tl = document.getElementById('ev-timeline');
  tl.innerHTML = '';
  const evs = state.events;
  if (!evs.length) return;
  const dur = (v.duration && isFinite(v.duration)) ? v.duration * 1000 : null;
  const maxOffset = dur || Math.max(...evs.map(e => e.offset_ms), 1);
  for (const ev of evs) {
    const m = document.createElement('div');
    m.className = 'mark' + (ev.is_staff ? ' staff' : '');
    m.dataset.eventId = ev.event_id;
    m.style.left = `${Math.min(99.5, (ev.offset_ms / maxOffset) * 100)}%`;
    m.title = `${fmtType(ev.event_type)} @ ${(ev.offset_ms/1000).toFixed(1)}s`;
    tl.appendChild(m);
  }
}

function onVideoTime() {
  const v = document.getElementById('cam-video');
  const cur = (v.currentTime || 0) * 1000;

  // Highlight all events whose fire window contains the playhead.
  const FIRE_WINDOW_MS = 1500;
  let firedRow = null;
  let firstFiredEv = null;

  for (const row of document.querySelectorAll('#ev-list .ev-row')) {
    const id = row.dataset.eventId;
    const ev = state.events.find(e => e.event_id === id);
    if (!ev) continue;
    const inFire = Math.abs(ev.offset_ms - cur) < FIRE_WINDOW_MS && ev.offset_ms <= cur + FIRE_WINDOW_MS;
    row.classList.toggle('fired', inFire);
    if (inFire) { firedRow = row; if (!firstFiredEv) firstFiredEv = ev; }
  }
  for (const m of document.querySelectorAll('#ev-timeline .mark')) {
    const id = m.dataset.eventId;
    const ev = state.events.find(e => e.event_id === id);
    if (!ev) continue;
    m.classList.toggle('fired', Math.abs(ev.offset_ms - cur) < FIRE_WINDOW_MS);
  }
  if (firedRow) firedRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

  // Show a toast for the most recent freshly-fired event (deduped via lastFiredId).
  if (firstFiredEv && firstFiredEv.event_id !== state.lastFiredId) {
    state.lastFiredId = firstFiredEv.event_id;
    showToast(firstFiredEv);
  }
}

function showToast(ev) {
  const stack = document.getElementById('ev-toast-stack');
  const t = document.createElement('div');
  t.className = 'ev-toast' + (ev.is_staff ? ' staff' : '');
  t.innerHTML = `
    <span class="t-type">${ev.is_staff ? '★ STAFF' : 'CUST'} · ${fmtType(ev.event_type)}</span>
    <span class="t-meta">${ev.zone_id || '—'} · ${shortId(ev.visitor_id)} · t=${(ev.offset_ms/1000).toFixed(1)}s</span>`;
  stack.appendChild(t);
  while (stack.children.length > 3) stack.removeChild(stack.firstChild);
  setTimeout(() => {
    t.classList.add('fading');
    setTimeout(() => t.remove(), 280);
  }, 2400);
}

// ============================================================================
// Top KPIs / funnel / heatmap / anomalies
// ============================================================================
async function refreshAll() {
  try {
    const [m, f, h, a, hl] = await Promise.all([
      fetchJSON(`/stores/${STORE_ID}/metrics`),
      fetchJSON(`/stores/${STORE_ID}/funnel`),
      fetchJSON(`/stores/${STORE_ID}/heatmap`),
      fetchJSON(`/stores/${STORE_ID}/anomalies`),
      fetchJSON(`/health`),
    ]);
    document.getElementById('m-uv').textContent   = m.unique_visitors;
    document.getElementById('m-sess').textContent = m.sessions;
    document.getElementById('m-conv').textContent = (m.conversion_rate * 100).toFixed(1) + '%';
    document.getElementById('m-q').textContent =
      `${m.queue_depth_now ?? '–'} / ${m.queue_depth_avg.toFixed(1)}`;
    document.getElementById('m-aban').textContent = (m.abandonment_rate * 100).toFixed(1) + '%';

    const fEl = document.getElementById('funnel'); fEl.innerHTML = '';
    const top = f.stages[0]?.count || 1;
    for (const s of f.stages) {
      const li = document.createElement('li');
      const w = Math.max(3, (s.count / top) * 100);
      li.innerHTML = `
        <div class="row"><span class="name">${s.stage}</span><span class="count">${s.count}</span></div>
        <div class="bar" style="width:${w}%"></div>
        <span class="drop">drop ${s.drop_off_pct}%</span>`;
      fEl.appendChild(li);
    }

    const hEl = document.getElementById('heatmap'); hEl.innerHTML = '';
    const maxIntensity = Math.max(...h.zones.map(z => z.intensity), 1);
    for (const z of h.zones.slice(0, 10)) {
      const li = document.createElement('li');
      li.style.setProperty('--heat', Math.min(0.5, z.intensity / maxIntensity * 0.45).toFixed(2));
      li.innerHTML = `
        <span class="name">${z.zone_id}</span>
        <span class="val">${z.intensity}</span>
        <span class="meta">visits ${z.visits} · avg dwell ${(z.avg_dwell_ms/1000).toFixed(1)}s</span>`;
      hEl.appendChild(li);
    }
    if (h.data_confidence === 'low') {
      const li = document.createElement('li');
      li.innerHTML = `<span class="name">low confidence</span><span class="val">${h.sessions_in_window}</span>`;
      hEl.appendChild(li);
    }

    const aEl = document.getElementById('anomalies'); aEl.innerHTML = '';
    if (a.anomalies.length === 0) {
      aEl.innerHTML = '<li class="low"><span class="msg">No anomalies in current window.</span></li>';
    }
    for (const an of a.anomalies) {
      const li = document.createElement('li');
      li.className = an.severity;
      li.innerHTML = `
        <div class="head">
          <span class="badge">${an.severity}</span>
          <span class="type">${an.type}</span>
        </div>
        <span class="msg">${an.message}</span>
        <span class="act">→ ${an.suggested_action}</span>`;
      aEl.appendChild(li);
    }

    const pill = document.getElementById('health-pill');
    const txt  = pill.querySelector('.health-text');
    const stale = hl.stores.some(s => s.stale_feed);
    if (hl.status === 'ok' && !stale) {
      pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/30';
      pill.querySelector('span:first-child').className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse-fast';
      txt.textContent = 'live';
    } else {
      pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30';
      pill.querySelector('span:first-child').className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse-fast';
      txt.textContent = stale ? 'stale feed' : hl.status;
    }
  } catch (e) {
    const pill = document.getElementById('health-pill');
    pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-300 border border-rose-500/30';
    pill.querySelector('span:first-child').className = 'w-2 h-2 rounded-full bg-rose-400 animate-pulse-fast';
    pill.querySelector('.health-text').textContent = 'offline';
  }
}

// ============================================================================
// Upload
// ============================================================================
document.getElementById('upload-file').addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  document.getElementById('upload-fname').textContent = f ? f.name : 'Choose video file…';
});

document.getElementById('upload-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const fd = new FormData();
  const fileEl = document.getElementById('upload-file');
  if (!fileEl.files[0]) return;
  fd.append('file', fileEl.files[0]);
  fd.append('camera_id', document.getElementById('upload-cam').value);
  document.getElementById('upload-status').textContent = 'uploading…';
  try {
    const job = await fetchJSON('/uploads', { method: 'POST', body: fd });
    document.getElementById('upload-status').textContent =
      `job ${job.job_id} queued (${job.camera_id}) — running detection pipeline…`;
    pollJob(job.job_id);
  } catch (e) {
    document.getElementById('upload-status').textContent = 'upload failed: ' + e.message;
  }
});

async function pollJob(jobId) {
  const interval = setInterval(async () => {
    try {
      const j = await fetchJSON(`/uploads/${jobId}`);
      const txt = `job ${j.job_id}: ${j.status}` +
        (j.status === 'done' ? ` — ${j.accepted} accepted, ${j.duplicates} dup, ${j.failed} failed` : '') +
        (j.error ? ` — ${j.error}` : '');
      document.getElementById('upload-status').textContent = txt;
      if (j.status === 'done' || j.status === 'failed') {
        clearInterval(interval);
        await initCameras();
        if (j.camera_id) selectCamera(j.camera_id);
        refreshAll();
      }
    } catch (e) { /* ignore */ }
  }, 2000);
}

// ============================================================================
// boot
// ============================================================================
initCameras().then(refreshAll);
setInterval(refreshAll, 5000);
