// Purplle Intelligence — multi-store dashboard with router and live processing.

const state = {
  stores: [],
  activeStore: null,
  cameras: [],
  current: null,
  events: [],
  preferAnnotated: true,
  lastFiredId: null,
  refreshTimer: null,
  job: null,
  jobES: null,
  jobPreviewTimer: null,
};

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(url + ' -> ' + r.status);
  return r.json();
}
const fmtType = (t) => (t || '').replace(/_/g, ' ');
const shortId = (s, n = 6) => (s || '').slice(-n);

// ============================================================================
// Router
// ============================================================================
const ROUTES = ['/store/ST1008', '/store/ST2009', '/process'];
function currentRoute() {
  const h = (location.hash || '').replace(/^#/, '');
  if (!h) return '/store/ST1008';
  return h;
}
function navigate(route) {
  if (location.hash !== '#' + route) location.hash = route;
  else applyRoute();
}
function applyRoute() {
  const r = currentRoute();
  for (const el of document.querySelectorAll('.nav-item')) {
    el.classList.toggle('active', el.dataset.route === r);
  }
  for (const p of document.querySelectorAll('.page')) p.classList.add('hidden');
  if (r.startsWith('/store/')) {
    document.getElementById('page-store').classList.remove('hidden');
    const sid = r.split('/')[2];
    activateStore(sid);
  } else if (r === '/process') {
    document.getElementById('page-process').classList.remove('hidden');
    if (state.refreshTimer) { clearInterval(state.refreshTimer); state.refreshTimer = null; }
    initProcessPage();
  }
}
window.addEventListener('hashchange', applyRoute);

// ============================================================================
// Stores list
// ============================================================================
async function loadStores() {
  try {
    const data = await fetchJSON('/dashboard/stores');
    state.stores = data.stores;
  } catch (e) {
    state.stores = [
      { store_id: 'ST1008', name: 'Brigade Bangalore', cameras: [] },
      { store_id: 'ST2009', name: 'Store 2', cameras: [] },
    ];
  }
}

// ============================================================================
// Store dashboard
// ============================================================================
async function activateStore(storeId) {
  state.activeStore = storeId;
  const meta = state.stores.find(s => s.store_id === storeId) || { store_id: storeId, name: storeId };
  document.getElementById('store-tag').textContent = storeId === 'ST1008' ? 'Store 1' : 'Store 2';
  document.getElementById('store-name').textContent = meta.name || storeId;
  document.getElementById('store-meta').textContent =
    `${meta.event_count ? meta.event_count.toLocaleString() : '0'} events · ${meta.cameras?.length || 0} cameras`;
  document.getElementById('store-id').textContent = storeId;
  await initCameras(storeId);
  await refreshAll(storeId);
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = setInterval(() => refreshAll(state.activeStore), 5000);
}

async function initCameras(storeId) {
  const data = await fetchJSON(`/dashboard/cameras?store_id=${storeId}`);
  state.cameras = data.cameras;
  const nav = document.getElementById('cam-nav');
  const sel = document.getElementById('cam-select');
  nav.innerHTML = ''; sel.innerHTML = '';
  for (const c of state.cameras) {
    const btn = document.createElement('button');
    btn.dataset.cam = c.camera_id;
    btn.innerHTML = `<span>${c.camera_id}</span>` +
      (c.annotated_url ? '' : '<span class="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-amber-400" title="no annotated mp4"></span>');
    btn.onclick = () => selectCamera(c.camera_id);
    nav.appendChild(btn);
    const opt = document.createElement('option');
    opt.value = c.camera_id; opt.textContent = c.camera_id;
    sel.appendChild(opt);
  }
  sel.onchange = (ev) => selectCamera(ev.target.value);
  document.getElementById('cam-toggle-annot').onchange = (ev) => {
    state.preferAnnotated = ev.target.checked;
    if (state.current) loadVideoSource();
  };
  if (state.cameras.length) selectCamera(state.cameras[0].camera_id);
  else {
    document.getElementById('cam-video').removeAttribute('src');
    document.getElementById('ev-list').innerHTML =
      '<div class="text-center text-slate-500 text-xs py-6">no cameras for this store</div>';
  }
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
    if (state.preferAnnotated && !cam.annotated_url) label = 'no annotated mp4 yet';
  }
  v.src = src;
  tag.textContent = label;
  v.ontimeupdate = onVideoTime;
  v.onloadedmetadata = renderTimelineMarks;
  document.getElementById('ev-toast-stack').innerHTML = '';
}

async function loadEvents() {
  const cam = state.current;
  const list = document.getElementById('ev-list');
  list.innerHTML = '<div class="text-center text-slate-500 text-xs py-6">loading events…</div>';
  try {
    const data = await fetchJSON(`/dashboard/events?camera_id=${cam.camera_id}&store_id=${state.activeStore}&limit=5000`);
    state.events = data.events;
    document.getElementById('ev-count').textContent =
      `${state.events.length} event${state.events.length === 1 ? '' : 's'}`;
    updateCamStats();
    renderTimelineMarks();
    renderFeed();
    if (!state.events.length) list.innerHTML = '<div class="text-center text-slate-500 text-xs py-6">no events for this camera</div>';
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
    row.style.cursor = 'pointer';
    row.title = 'Click to jump video to this event';
    const md = ev.metadata || {};
    const demo = [md.gender, md.age_bucket].filter(Boolean).join(' ');
    const demoHtml = demo ? `<span class="ev-demo">${demo}</span>` : '';
    row.innerHTML = `
      <div class="ev-left"><span class="ev-dot"></span><span class="ev-type">${fmtType(ev.event_type)}</span></div>
      <div class="ev-meta"><span class="ev-zone">${ev.zone_id || '—'}</span><span class="ev-vid">${shortId(ev.visitor_id)} · ${ev.is_staff ? 'STAFF' : 'cust'}${demo ? ' · ' : ''}${demo}</span></div>
      <span class="ev-time">${(ev.offset_ms / 1000).toFixed(1)}s</span>`;
    row.addEventListener('click', () => seekToOffset(ev.offset_ms));
    list.appendChild(row);
  }
}

function seekToOffset(offsetMs) {
  const v = document.getElementById('cam-video');
  if (!v) return;
  const target = Math.max(0, offsetMs / 1000);
  const apply = () => {
    try { v.currentTime = target; v.play().catch(() => {}); }
    catch (e) { console.warn('seek failed', e); }
  };
  if (v.readyState >= 1 && isFinite(v.duration)) apply();
  else v.addEventListener('loadedmetadata', apply, { once: true });
}

function renderTimelineMarks() {
  const v = document.getElementById('cam-video');
  const tl = document.getElementById('ev-timeline');
  tl.innerHTML = '';
  if (!state.events.length) return;
  const dur = (v.duration && isFinite(v.duration)) ? v.duration * 1000 : null;
  const maxOffset = dur || Math.max(...state.events.map(e => e.offset_ms), 1);
  for (const ev of state.events) {
    const m = document.createElement('div');
    m.className = 'mark' + (ev.is_staff ? ' staff' : '');
    m.dataset.eventId = ev.event_id;
    m.style.left = `${Math.min(99.5, (ev.offset_ms / maxOffset) * 100)}%`;
    m.style.cursor = 'pointer';
    m.title = `${fmtType(ev.event_type)} @ ${(ev.offset_ms / 1000).toFixed(1)}s — click to seek`;
    m.addEventListener('click', (e) => { e.stopPropagation(); seekToOffset(ev.offset_ms); });
    tl.appendChild(m);
  }
  if (!tl.dataset.seekBound) {
    tl.addEventListener('click', (e) => {
      if (e.target !== tl) return;
      const rect = tl.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const dur = (v.duration && isFinite(v.duration)) ? v.duration : (maxOffset / 1000);
      seekToOffset(ratio * dur * 1000);
    });
    tl.dataset.seekBound = '1';
  }
}

function onVideoTime() {
  const v = document.getElementById('cam-video');
  const cur = (v.currentTime || 0) * 1000;
  const W = 1500;
  let firedRow = null, firstFiredEv = null;
  for (const row of document.querySelectorAll('#ev-list .ev-row')) {
    const id = row.dataset.eventId;
    const ev = state.events.find(e => e.event_id === id);
    if (!ev) continue;
    const inFire = Math.abs(ev.offset_ms - cur) < W && ev.offset_ms <= cur + W;
    row.classList.toggle('fired', inFire);
    if (inFire) { firedRow = row; if (!firstFiredEv) firstFiredEv = ev; }
  }
  for (const m of document.querySelectorAll('#ev-timeline .mark')) {
    const id = m.dataset.eventId;
    const ev = state.events.find(e => e.event_id === id);
    if (!ev) continue;
    m.classList.toggle('fired', Math.abs(ev.offset_ms - cur) < W);
  }
  if (firedRow) firedRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (firstFiredEv && firstFiredEv.event_id !== state.lastFiredId) {
    state.lastFiredId = firstFiredEv.event_id;
    showToast(firstFiredEv);
  }
}
function showToast(ev) {
  const stack = document.getElementById('ev-toast-stack');
  const t = document.createElement('div');
  t.className = 'ev-toast' + (ev.is_staff ? ' staff' : '');
  const md = ev.metadata || {};
  const demo = [md.gender, md.age_bucket].filter(Boolean).join(' ');
  const demoStr = demo ? ` · ${demo}` : '';
  t.innerHTML = `<span class="t-type">${ev.is_staff ? '★ STAFF' : 'CUST'} · ${fmtType(ev.event_type)}</span><span class="t-meta">${ev.zone_id || '—'} · ${shortId(ev.visitor_id)}${demoStr} · t=${(ev.offset_ms / 1000).toFixed(1)}s</span>`;
  stack.appendChild(t);
  while (stack.children.length > 3) stack.removeChild(stack.firstChild);
  setTimeout(() => { t.classList.add('fading'); setTimeout(() => t.remove(), 280); }, 2400);
}

// ============================================================================
// Top KPIs / funnel / heatmap / anomalies
// ============================================================================
async function refreshAll(storeId) {
  if (!storeId) return;
  try {
    const [m, f, h, a, d, hl] = await Promise.all([
      fetchJSON(`/stores/${storeId}/metrics`),
      fetchJSON(`/stores/${storeId}/funnel`),
      fetchJSON(`/stores/${storeId}/heatmap`),
      fetchJSON(`/stores/${storeId}/anomalies`),
      fetchJSON(`/stores/${storeId}/demographics`),
      fetchJSON(`/health`),
    ]);
    document.getElementById('m-uv').textContent = m.unique_visitors;
    document.getElementById('m-sess').textContent = m.sessions;
    document.getElementById('m-conv').textContent = (m.conversion_rate * 100).toFixed(1) + '%';
    document.getElementById('m-q').textContent = `${m.queue_depth_now ?? '–'} / ${m.queue_depth_avg.toFixed(1)}`;
    document.getElementById('m-aban').textContent = (m.abandonment_rate * 100).toFixed(1) + '%';

    const fEl = document.getElementById('funnel'); fEl.innerHTML = '';
    const top = f.stages[0]?.count || 1;
    for (const s of f.stages) {
      const li = document.createElement('li');
      const w = Math.max(3, (s.count / top) * 100);
      li.innerHTML = `<div class="row"><span class="name">${s.stage}</span><span class="count">${s.count}</span></div><div class="bar" style="width:${w}%"></div><span class="drop">drop ${s.drop_off_pct}%</span>`;
      fEl.appendChild(li);
    }

    const hEl = document.getElementById('heatmap'); hEl.innerHTML = '';
    const maxIntensity = Math.max(...h.zones.map(z => z.intensity), 1);
    for (const z of h.zones.slice(0, 10)) {
      const li = document.createElement('li');
      li.style.setProperty('--heat', Math.min(0.5, z.intensity / maxIntensity * 0.45).toFixed(2));
      li.innerHTML = `<span class="name">${z.zone_id}</span><span class="val">${z.intensity}</span><span class="meta">visits ${z.visits} · avg dwell ${(z.avg_dwell_ms / 1000).toFixed(1)}s</span>`;
      hEl.appendChild(li);
    }
    if (h.data_confidence === 'low') {
      const li = document.createElement('li');
      li.innerHTML = `<span class="name">low confidence</span><span class="val">${h.sessions_in_window}</span>`;
      hEl.appendChild(li);
    }

    const aEl = document.getElementById('anomalies'); aEl.innerHTML = '';
    if (a.anomalies.length === 0) aEl.innerHTML = '<li class="low"><span class="msg">No anomalies in current window.</span></li>';
    for (const an of a.anomalies) {
      const li = document.createElement('li'); li.className = an.severity;
      li.innerHTML = `<div class="head"><span class="badge">${an.severity}</span><span class="type">${an.type}</span></div><span class="msg">${an.message}</span><span class="act">→ ${an.suggested_action}</span>`;
      aEl.appendChild(li);
    }

    setHealthPill(hl);
    renderDemographics(d);
  } catch (e) {
    setHealthPill(null);
  }
}

function renderDemographics(d) {
  const total = d.classified_visitors || 0;
  const cov = d.total_visitors
    ? `${total}/${d.total_visitors} classified`
    : `${total} classified`;
  const topCov = document.getElementById('demo-top-coverage');
  const botCov = document.getElementById('demo-bot-summary');
  if (topCov) topCov.textContent = cov;
  if (botCov) botCov.textContent = cov;

  // Gender bars
  const palette = { F: 'bg-fuchsia-500', M: 'bg-blue-500' };
  const palLight = { F: 'text-fuchsia-300', M: 'text-blue-300' };
  const labelMap = { F: 'Female', M: 'Male' };
  const genderEntries = Object.entries(d.gender || {}).sort((a, b) => b[1] - a[1]);
  const gTotal = genderEntries.reduce((s, [, v]) => s + v, 0) || 1;
  const renderGenderInto = (el, big) => {
    if (!el) return;
    el.innerHTML = '';
    if (genderEntries.length === 0) {
      el.innerHTML = '<div class="text-xs text-slate-500">No demographic data yet.</div>';
      return;
    }
    for (const [k, v] of genderEntries) {
      const pct = (v / gTotal) * 100;
      const colour = palette[k] || 'bg-slate-500';
      const tcol = palLight[k] || 'text-slate-300';
      const row = document.createElement('div');
      row.innerHTML = `
        <div class="flex items-center justify-between text-xs mb-1">
          <span class="${tcol} font-semibold">${labelMap[k] || k}</span>
          <span class="font-mono text-slate-400">${v} <span class="text-slate-500">· ${pct.toFixed(0)}%</span></span>
        </div>
        <div class="h-${big ? '2.5' : '1.5'} rounded-full bg-slate-800/70 overflow-hidden">
          <div class="${colour} h-full transition-all" style="width:${pct}%"></div>
        </div>`;
      el.appendChild(row);
    }
  };
  renderGenderInto(document.getElementById('demo-top-gender'), false);
  renderGenderInto(document.getElementById('demo-bot-gender'), true);

  // Age bars
  const order = (d.age_bucket_order && d.age_bucket_order.length)
    ? d.age_bucket_order
    : ['child', 'teen', '20s', '30s', '40s', '50s', '60+'];
  const ages = d.age_buckets || {};
  const aMax = Math.max(...order.map(k => ages[k] || 0), 1);
  const renderAgeInto = (barsEl, labelsEl, big) => {
    if (!barsEl) return;
    barsEl.innerHTML = '';
    if (labelsEl) labelsEl.innerHTML = '';
    for (const k of order) {
      const v = ages[k] || 0;
      const h = (v / aMax) * 100;
      const col = document.createElement('div');
      col.className = 'flex-1 flex flex-col items-center gap-1 min-w-0';
      col.innerHTML = `
        <div class="text-[10px] font-mono text-slate-400">${v}</div>
        <div class="w-full bg-slate-800/70 rounded-md overflow-hidden flex items-end" style="height:${big ? '110px' : '40px'}">
          <div class="w-full bg-gradient-to-t from-fuchsia-600 to-brand-400 transition-all" style="height:${h}%"></div>
        </div>
        ${big ? '' : `<div class="text-[9px] text-slate-500 truncate w-full text-center">${k}</div>`}`;
      barsEl.appendChild(col);
      if (big && labelsEl) {
        const lbl = document.createElement('div');
        lbl.className = 'flex-1 text-center font-mono';
        lbl.textContent = k;
        labelsEl.appendChild(lbl);
      }
    }
  };
  renderAgeInto(document.getElementById('demo-top-age'), null, false);
  renderAgeInto(document.getElementById('demo-bot-age'),
                document.getElementById('demo-bot-age-labels'), true);
}

function setHealthPill(hl) {
  const pill = document.getElementById('health-pill');
  const txt = pill.querySelector('.health-text');
  const dot = pill.querySelector('span:first-child');
  if (!hl) { dot.className = 'w-2 h-2 rounded-full bg-rose-400 animate-pulse-fast'; txt.textContent = 'offline'; return; }
  const stale = hl.stores.some(s => s.stale_feed);
  if (hl.status === 'ok' && !stale) { dot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse-fast'; txt.textContent = 'live'; }
  else { dot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse-fast'; txt.textContent = stale ? 'stale feed' : hl.status; }
}

// ============================================================================
// Process Video page
// ============================================================================
function initProcessPage() {
  const storeSel = document.getElementById('upload-store');
  if (!storeSel.options.length) {
    storeSel.innerHTML = '';
    for (const s of state.stores) {
      const o = document.createElement('option'); o.value = s.store_id; o.textContent = `${s.store_id} — ${s.name}`;
      storeSel.appendChild(o);
    }
    storeSel.value = state.stores[0]?.store_id || '';
    storeSel.onchange = repopulateUploadCameras;
    repopulateUploadCameras();
  }
  const drop = document.getElementById('upload-drop');
  const fileInput = document.getElementById('upload-file');
  if (drop.dataset.bound !== '1') {
    drop.dataset.bound = '1';
    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
    drop.addEventListener('dragenter', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
    drop.addEventListener('dragover',  (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; drop.classList.add('dragover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('dragover');
      const f = e.dataTransfer.files[0];
      if (!f) return;
      try {
        const dt = new DataTransfer();
        dt.items.add(f);
        fileInput.files = dt.files;
      } catch (_) { /* Safari/older — fall back to keeping reference */ }
      document.getElementById('upload-fname').textContent = f.name;
    });
    fileInput.addEventListener('change', (ev) => {
      const f = ev.target.files[0];
      document.getElementById('upload-fname').textContent = f ? f.name : 'Drop video here or click to browse';
    });
    document.getElementById('upload-form').addEventListener('submit', onSubmitProcess);
    // Prevent page-level drop from opening the file in a new tab.
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop',     (e) => e.preventDefault());
  }
}

async function repopulateUploadCameras() {
  const sid = document.getElementById('upload-store').value;
  const sel = document.getElementById('upload-cam');
  sel.innerHTML = '<option>loading…</option>';
  try {
    const data = await fetchJSON(`/dashboard/cameras?store_id=${sid}`);
    sel.innerHTML = '';
    for (const c of data.cameras) {
      const o = document.createElement('option'); o.value = c.camera_id; o.textContent = c.camera_id;
      sel.appendChild(o);
    }
  } catch (e) {
    sel.innerHTML = '<option>error</option>';
  }
}

async function onSubmitProcess(ev) {
  ev.preventDefault();
  const fileEl = document.getElementById('upload-file');
  if (!fileEl.files[0]) return;
  const fd = new FormData();
  fd.append('file', fileEl.files[0]);
  fd.append('camera_id', document.getElementById('upload-cam').value);

  resetProcessUI();
  setProcStatus('uploading', 'uploading');
  appendLog(`uploading ${fileEl.files[0].name} (${(fileEl.files[0].size / 1024 / 1024).toFixed(1)} MB)…`, 'log-step');
  try {
    const job = await fetchJSON('/uploads', { method: 'POST', body: fd });
    state.job = job;
    document.getElementById('proc-job').textContent = `job ${job.job_id}`;
    appendLog(`queued job ${job.job_id} (${job.camera_id})`, 'log-step');
    streamJob(job.job_id);
    startPreviewPoll(job.job_id);
  } catch (e) {
    setProcStatus('failed', 'failed');
    appendLog(`upload failed: ${e.message}`, 'log-err');
  }
}

function resetProcessUI() {
  document.getElementById('proc-log').innerHTML = '';
  document.getElementById('proc-bar').style.width = '0%';
  document.getElementById('proc-frames').textContent = '0 frames';
  document.getElementById('proc-step').textContent = 'starting…';
  document.getElementById('proc-rec').classList.remove('hidden');
  document.getElementById('proc-overlay').classList.add('hidden');
  document.getElementById('proc-final').classList.add('hidden');
  document.getElementById('proc-final').removeAttribute('src');
  document.getElementById('upload-submit').disabled = true;
}

function streamJob(jobId) {
  if (state.jobES) state.jobES.close();
  const es = new EventSource(`/uploads/${jobId}/stream`);
  state.jobES = es;
  es.addEventListener('log', (ev) => {
    try {
      const line = JSON.parse(ev.data);
      const cls =
        line.includes('it [') ? 'log-progress' :
        line.startsWith('[run]') || line.startsWith('[init]') || line.startsWith('[done]') || line.startsWith('[ok]') ? 'log-step' :
        line.startsWith('[save]') || line.startsWith('[ffmpeg]') ? 'log-step' :
        line.toLowerCase().includes('warn') ? 'log-warn' :
        line.toLowerCase().includes('error') || line.toLowerCase().includes('failed') ? 'log-err' : '';
      appendLog(line, cls);
    } catch (e) { /* ignore */ }
  });
  es.addEventListener('status', (ev) => {
    try {
      const j = JSON.parse(ev.data);
      if (j.frames_processed) {
        document.getElementById('proc-frames').textContent = `${j.frames_processed.toLocaleString()} frames`;
        const pct = Math.min(95, j.frames_processed / 30); // rough estimate; pipeline doesn't report total
        document.getElementById('proc-bar').style.width = pct + '%';
      }
      if (j.current_step) {
        document.getElementById('proc-step').textContent = j.current_step;
      }
      if (j.status) setProcStatus(j.status, j.status);
    } catch (e) { /* ignore */ }
  });
  es.addEventListener('done', (ev) => {
    try {
      const j = JSON.parse(ev.data);
      stopPreviewPoll();
      es.close();
      document.getElementById('upload-submit').disabled = false;
      document.getElementById('proc-rec').classList.add('hidden');
      if (j.status === 'done') {
        setProcStatus('done', 'done');
        document.getElementById('proc-bar').style.width = '100%';
        document.getElementById('proc-step').textContent =
          `done · ${j.accepted ?? 0} accepted, ${j.duplicates ?? 0} dup, ${j.failed ?? 0} failed`;
        appendLog(`✓ pipeline finished — ${j.accepted} events ingested`, 'log-step');
        if (j.annotated_url) {
          const fv = document.getElementById('proc-final');
          fv.src = j.annotated_url + '?t=' + Date.now();
          fv.classList.remove('hidden');
          document.getElementById('proc-preview').style.opacity = '0.3';
        }
      } else {
        setProcStatus('failed', 'failed');
        appendLog(`✗ pipeline failed: ${j.error || 'unknown'}`, 'log-err');
      }
    } catch (e) { /* ignore */ }
  });
  es.onerror = () => {
    // SSE will naturally retry; just log once.
    appendLog('(stream interrupted, retrying…)', 'log-warn');
  };
}

function startPreviewPoll(jobId) {
  stopPreviewPoll();
  const img = document.getElementById('proc-preview');
  state.jobPreviewTimer = setInterval(() => {
    const u = `/uploads/${jobId}/preview.jpg?t=${Date.now()}`;
    img.src = u;
  }, 600);
  document.getElementById('proc-overlay').classList.add('hidden');
}
function stopPreviewPoll() {
  if (state.jobPreviewTimer) { clearInterval(state.jobPreviewTimer); state.jobPreviewTimer = null; }
}

function setProcStatus(label, cls) {
  const pill = document.getElementById('proc-status-pill');
  pill.textContent = label;
  pill.className = 'ml-2 px-2.5 py-1 rounded-md text-[11px] font-medium border ' + (cls || '');
  if (!cls) pill.classList.add('bg-slate-800/60', 'text-slate-400', 'border-slate-700');
}

function appendLog(line, cls) {
  const el = document.getElementById('proc-log');
  const div = document.createElement('div');
  if (cls) div.className = cls;
  div.textContent = line;
  el.appendChild(div);
  if (el.children.length > 600) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

// ============================================================================
// boot
// ============================================================================
(async function boot() {
  await loadStores();
  applyRoute();
})();
