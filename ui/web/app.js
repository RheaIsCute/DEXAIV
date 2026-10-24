// DEXAIV thin client. It holds no engine state: everything shown comes from the events
// stream (state / meters / perf), and every control sends one command.
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const SCREENS = ["live", "models", "performance", "settings"];
const METER_MIN_DB = -70;
const PLAY_ICON = "M8 5v14l11-7z";
const STOP_ICON = "M9 7h6a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z";

let state = null;
let perf = null;
let history = [];
let meters = { running: false, in_db: -120, out_db: -120, gate_open: false, speaking: false };
let modelsKey = "";
let pendingIndexModel = null;
let lastScreen = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const signed = (n) => (n > 0 ? "+" : "") + Math.round(n);
const formantText = (st) => (st === 0 ? "0 st" : (st > 0 ? "+" : "") + st.toFixed(1) + " st");
const fixed = (n, d = 1) => (n == null ? "–" : Number(n).toFixed(d));

// ---------------------------------------------------------------- transport

async function cmd(name, args = {}) {
  let reply;
  try {
    const res = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Vox-Token": TOKEN },
      body: JSON.stringify({ cmd: name, args }),
    });
    reply = await res.json();
  } catch (err) {
    toast("The engine is not answering (" + err.message + ")", "error");
    return null;
  }
  if (!reply.ok) {
    toast(reply.error, "error");
    return null;
  }
  return reply.result === undefined || reply.result === null ? {} : reply.result;
}

async function upload(file) {
  const res = await fetch("/api/upload?name=" + encodeURIComponent(file.name), {
    method: "POST", headers: { "X-Vox-Token": TOKEN }, body: file,
  });
  const reply = await res.json();
  if (!reply.ok) throw new Error(reply.error);
  return reply.result.name;
}

function connect() {
  const source = new EventSource("/api/events?token=" + encodeURIComponent(TOKEN));
  source.addEventListener("state", (e) => { state = JSON.parse(e.data); history = state.perf_history || history; render(); });
  source.addEventListener("meters", (e) => { meters = JSON.parse(e.data); renderMeters(); });
  source.addEventListener("perf", (e) => {
    perf = JSON.parse(e.data);
    history.push({ t: perf.t, p50: perf.p50, p99: perf.p99 });
    if (history.length > 120) history.shift();
    renderPerf();
    renderAutoPitch();
  });
  source.addEventListener("toast", (e) => { const t = JSON.parse(e.data); toast(t.message, t.kind); });
  source.onerror = () => { $("status-sub").textContent = "Lost the engine connection — retrying"; };
}

// A slider sends while it moves, but never faster than the engine needs.
function throttled(fn, ms = 60) {
  let timer = null, latest = null;
  return (...args) => {
    latest = args;
    if (timer) return;
    timer = setTimeout(() => { timer = null; fn(...latest); }, ms);
  };
}

function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 9000 : 5000);
}

// ------------------------------------------------------------------- fitting

// The design frame is 1200x780. A bigger window stretches the layout; a smaller one scales
// the whole page down instead of clipping it.
const DESIGN_W = 1200, DESIGN_H = 780;
function fit() {
  const app = document.querySelector(".app");
  const scale = Math.min(1, window.innerWidth / DESIGN_W, window.innerHeight / DESIGN_H);
  app.style.width = window.innerWidth / scale + "px";
  app.style.height = window.innerHeight / scale + "px";
  app.style.transform = scale < 1 ? "scale(" + scale + ")" : "";
}

// ------------------------------------------------------------------ routing

function route() {
  const want = location.hash.replace("#", "");
  const screen = SCREENS.includes(want) ? want : "live";
  for (const name of SCREENS) $("screen-" + name).hidden = name !== screen;
  document.querySelectorAll(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.screen === screen));
  closePopovers();
  // Opening Settings re-scans devices, unless audio is open (PortAudio can't re-init then).
  if (screen === "settings" && lastScreen !== "settings") refreshDevices(true);
  lastScreen = screen;
}

function engineLive() {
  const status = state && state.engine.status;
  return status === "running" || status === "starting" || status === "stopping";
}

function refreshDevices(auto) {
  if (auto && (!state || engineLive())) return;
  cmd("refresh_devices");
}

// ----------------------------------------------------------------- rendering

// Do not fight the user: a control being dragged or edited keeps its own value.
function setValue(el, value) {
  if (document.activeElement === el && el.matches(":active, select")) return;
  if (String(el.value) !== String(value)) el.value = value;
}

function setToggle(id, rowId, info) {
  const el = $(id);
  el.checked = !!info.enabled;
  el.disabled = !info.available;
  if (rowId) {
    $(rowId).classList.toggle("off", !info.available);
    $(rowId).title = info.available ? info.reason || "" : info.reason || "Not built into the engine yet";
  }
}

// The ceiling slider's left end (190) means "off"; the engine takes 0 for off.
const CEILING_OFF = 190;
const ceilingText = (hz) => (hz > 0 ? Math.round(hz) + " Hz" : "off");

function render() {
  if (!state) return;
  const { engine, model, params, gate, devices, tts, shape } = state;
  const status = engine.status;
  const running = status === "running";
  const gpu = state.app.gpu.replace(/^NVIDIA GeForce /, "");
  $("app-foot").textContent = "v" + state.app.version + " · " + gpu;

  // power + status
  const transitional = status === "starting" || status === "stopping";
  $("power").disabled = transitional || !model;
  $("power-label").textContent = running || status === "starting" ? "STOP" : "START";
  $("power-icon").setAttribute("d", running || status === "starting" ? STOP_ICON : PLAY_ICON);
  renderStatus();
  if (!running) { $("latency").textContent = "–"; $("dropouts").textContent = "0 dropouts"; }

  // model card
  if (model) {
    $("model-chip").textContent = model.name[0].toUpperCase();
    $("model-chip").classList.add("on");
    $("model-name").textContent = model.name;
    const index = model.index ? "index paired" : "no index";
    $("model-spec").textContent = "RVC " + model.version + " · " + model.sr / 1000 + " kHz · " + index;
  } else {
    $("model-chip").textContent = "–";
    $("model-chip").classList.remove("on");
    $("model-name").textContent = "No model";
    $("model-spec").textContent = "Import a .pth on the Models screen";
  }

  // pitch
  const calibrated = !!state.profile.calibrated;
  const sing = !!params && params.preset === "sing";
  if (params) {
    setValue($("pitch"), params.pitch);
    if (document.activeElement !== $("pitch")) $("pitch-value").textContent = signed(params.pitch);
    const badge = $("pitch-badge");
    badge.hidden = !params.pitch_is_calibrated;
    badge.textContent = sing ? "sing default" : calibrated ? "auto · calibrated" : "profile default";
    badge.disabled = true;
    document.querySelectorAll("#preset [data-preset]").forEach((b) => b.classList.toggle("on", b.dataset.preset === params.preset));
    setValue($("formant"), params.formant || 0);
    if (document.activeElement !== $("formant")) $("formant-value").textContent = formantText(params.formant || 0);
    setValue($("index-rate"), Math.round(params.index_rate * 100));
    $("index-rate-value").textContent = params.index_inert ? "no effect" : fixed(params.index_rate, 2);
    $("index-rate").disabled = !!params.index_inert;
    $("index-rate").closest("label").title = params.index_note || "";
    $("index-rate").closest("label").classList.toggle("off", !!params.index_inert);
    setValue($("protect"), Math.round(params.protect * 100));
    $("protect-value").textContent = fixed(params.protect, 2);
    $("auto-pitch").checked = !!params.auto_pitch;
    $("pitch").closest("label").title = params.auto_pitch ? "With auto pitch on, this offset applies only until half a second of voice was heard" : "";
    setValue($("target-hz"), Math.round(params.target_hz));
    if (document.activeElement !== $("target-hz")) $("target-value").textContent = Math.round(params.target_hz) + " Hz";
    setValue($("pitch-range"), Math.round(params.pitch_range * 100));
    if (document.activeElement !== $("pitch-range")) $("range-value").textContent = Math.round(params.pitch_range * 100) + " %";
    $("target-row").classList.toggle("off", !params.auto_pitch);
    $("range-row").classList.toggle("off", !params.auto_pitch);
    renderAutoPitch();
    setValue($("ceiling"), params.ceiling_hz > 0 ? params.ceiling_hz : CEILING_OFF);
    if (document.activeElement !== $("ceiling")) $("ceiling-value").textContent = ceilingText(params.ceiling_hz);
    setValue($("tracker"), params.f0_method);
    const lo = Math.round(params.f0_min_hz), hi = Math.round(params.f0_max_hz);
    $("clamp-note").textContent = calibrated
      ? "Your range is clamped to " + lo + "–" + hi + " Hz from calibration, so octave errors on low notes are filtered out."
      : "No calibration yet: the full " + lo + "–" + hi + " Hz range is tracked.";
  }
  $("pitch").disabled = !params;
  $("formant").disabled = !params;
  document.querySelectorAll("#preset [data-preset]").forEach((b) => { b.disabled = !params; });
  const natural = !!params && params.preset === "natural";
  $("conversion-badge").hidden = !calibrated || sing || natural;
  $("preset-badge").hidden = !(sing || natural);
  $("preset-badge").textContent = natural ? "Natural preset" : "Sing preset";

  // gate
  setValue($("gate-threshold"), gate.threshold_dbfs);
  $("gate-value").textContent = Math.round(gate.threshold_dbfs) + " dB";
  $("gate-tick").style.left = meterPct(gate.threshold_dbfs) + "%";
  $("gate-tick").hidden = !gate.enabled;
  $("gate-on").checked = gate.enabled;
  $("gate-on-2").checked = gate.enabled;
  const level = state.input_level || { auto: true, gain_db: 0 };
  setValue($("input-gain"), level.gain_db);
  $("input-gain").disabled = !!level.trim_locked;
  $("input-gain").closest("label").title = level.trim_locked
    ? "Sing preset: trim 0 dB and auto level at most +" + Math.round(level.max_gain_db) + " dB, so held notes do not hit the limiter"
    : "Trim on top of auto level (or the whole input gain with auto level off)";
  $("auto-level").checked = level.auto;
  renderInputGain();
  renderVolume();
  renderRecord();
  $("noise-floor").textContent = calibrated
    ? "Threshold set from your calibration noise floor (" + Math.round(gate.noise_floor_dbfs) + " dB)"
    : "Not calibrated yet: the threshold is the profile default";
  setToggle("suppression", "suppression-row", state.suppression);
  const sup = state.suppression;
  $("suppression-mix-row").hidden = !(sup.available && sup.enabled);
  setValue($("suppression-mix"), Math.round((sup.mix ?? 1) * 100));
  if (document.activeElement !== $("suppression-mix")) $("suppression-mix-value").textContent = Math.round((sup.mix ?? 1) * 100) + " %";
  setToggle("monitor", "monitor-row", state.monitor);
  setToggle("autotune", "autotune-row", state.auto_tune);

  // speak bar
  $("tts-voice").textContent = "voice " + tts.voice;
  $("tts-pitch").textContent = tts.last ? "pitch " + signed(tts.last.pitch) + " auto" : "pitch auto";
  $("tts-note").textContent = !tts.available ? "TTS unavailable: " + tts.reason
    : running ? "mic muted while it speaks" : "plays on the output device while stopped";
  $("tts-note").className = tts.available ? "good-text" : "warn-text";
  const rendering = engine.busy === "Rendering speech" || (engine.busy || "").startsWith("Previewing");
  $("speak-go").disabled = !tts.available || !model;
  $("speak-go").textContent = rendering ? "…" : tts.speaking ? "Stop" : "Speak";
  if (!$("history").hidden) renderHistory();

  // devices
  $("mic-name").textContent = devices.input_file ? "File: " + devices.input_file
    : devices.input ? devices.input.name + " (WASAPI)" : "No input device";
  $("out-name").textContent = devices.output ? devices.output.name : "No output device";
  fillDevices($("mic"), devices.inputs, devices.input);
  fillDevices($("out"), devices.outputs, devices.output);
  $("mic").disabled = !!devices.input_file;
  $("refresh-devices").disabled = engineLive() || !!engine.busy;
  $("refresh-devices").title = engineLive() ? "Stop the engine to refresh devices" : "Refresh audio devices";
  $("exclusive").checked = state.exclusive;

  // engine column
  setValue($("hop"), shape.block_ms);
  $("hop-value").textContent = Math.round(shape.block_ms) + " ms";
  setValue($("context"), shape.extra_ms);
  $("context-value").textContent = Math.round(shape.extra_ms) + " ms";
  $("graphs").checked = state.graphs;
  $("graphs").disabled = !state.graphs_available;

  // typed speech column
  setValue($("variation"), Math.round(tts.variation * 100));
  if (document.activeElement !== $("variation")) $("variation-value").textContent = fixed(tts.variation, 2);
  setValue($("laughter"), tts.laughter);
  $("contour").checked = tts.contour > 1;
  const slang = tts.slang;
  $("slang-note").textContent = slang.error ? "Not loaded: " + slang.error
    : slang.exists ? slang.user_entries + " word" + (slang.user_entries === 1 ? "" : "s") + " of your own, on top of the built-in list"
    : "Only the built-in list. Edit to add your own words.";
  $("slang-note").className = slang.error ? "hint warn-text" : "hint";

  // settings header
  const when = state.profile.calibrated ? "calibrated " + shortDate(state.profile.calibrated) : "not calibrated yet";
  $("profile-lead").innerHTML = "Profile <b>" + esc(state.profile.name) + "</b> — " + esc(when) + ". Most of this is set for you.";

  renderModels();
  renderPerf();
}

function renderStatus() {
  if (!state) return;
  const { engine, tts } = state;
  const dot = $("status-dot");
  let title = "Stopped", sub = state.model ? "Press START to go live" : "Import a voice model first", cls = "dot";
  if (engine.status === "starting") { title = "Starting"; sub = "Loading the model and capturing the engine"; cls = "dot busy"; }
  else if (engine.status === "stopping") { title = "Stopping"; sub = ""; cls = "dot busy"; }
  else if (engine.status === "error") { title = "Error"; sub = engine.error || ""; cls = "dot error"; }
  else if (engine.status === "running") {
    title = "Live";
    if (engine.busy && engine.busy !== "Rendering speech") { sub = engine.busy; cls = "dot busy"; }
    else if (engine.recapturing) { sub = "Re-capturing · muted"; cls = "dot busy"; }
    else if (meters.speaking || tts.speaking) { sub = "Speaking typed text · mic muted"; cls = "dot live"; }
    else if (!state.gate.enabled) { sub = "Gate off · always converting"; cls = "dot live"; }
    else if (meters.gate_open) { sub = "Gate open · speaking"; cls = "dot live"; }
    else { sub = "Gate closed"; cls = "dot dim"; }
  }
  if (engine.status !== "running" && engine.busy) sub = engine.busy;
  dot.className = cls;
  $("status-title").textContent = title;
  $("status-sub").textContent = sub;
  $("status-sub").title = sub;
}

const meterPct = (db) => Math.max(0, Math.min(100, (100 * (db - METER_MIN_DB)) / -METER_MIN_DB));
const meterText = (db) => (db == null || db <= -100 ? "–" : fixed(db, 1) + " dB");

function renderMeters() {
  $("meter-in").style.width = meterPct(meters.in_db ?? -120) + "%";
  $("meter-out").style.width = meterPct(meters.out_db ?? -120) + "%";
  $("meter-in-db").textContent = meters.running ? meterText(meters.in_db) : "–";
  $("meter-out-db").textContent = meters.running ? meterText(meters.out_db) : "–";
  renderInputGain();
  renderRecord();
  renderStatus();
}

// Record: only while live; the label counts the take (seconds come with the meters).
function renderRecord() {
  if (!state) return;
  const live = state.engine.status === "running";
  const on = live && !!(state.recording && state.recording.active);
  const btn = $("record");
  btn.disabled = !live || state.engine.busy === "Saving recording";
  btn.classList.toggle("on", on);
  btn.title = live ? (on ? "Stop and save the take (opens it in Explorer)"
    : "Record the mic and the converted voice to WAV files") : "Start the engine to record";
  const s = Math.floor(meters.rec_s || 0);
  $("record-label").textContent = on ? "Stop · " + Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0")
    : state.engine.busy === "Saving recording" ? "Saving…" : "Record";
}

// Auto pitch: the transposition it is applying right now (from the perf snapshot).
function renderAutoPitch() {
  const el = $("auto-pitch-now");
  const on = state && state.params && state.params.auto_pitch;
  const running = state && state.engine.status === "running";
  const st = perf && perf.auto_pitch_st;
  el.textContent = !on || !running ? "" : st == null ? "listening" : "now " + signed(Math.round(st));
}

// The trim the slider sets, plus (live, with auto level on) the total gain being applied.
function renderInputGain() {
  const trim = +$("input-gain").value;
  const locked = !!(state && state.input_level && state.input_level.trim_locked);
  const text = signed(trim) + " dB" + (locked ? " · Sing" : "");
  const live = meters.running && $("auto-level").checked && meters.in_gain_db != null;
  $("input-gain-value").textContent = live ? text + " · now " + signed(Math.round(meters.in_gain_db)) : text;
}

// Output and typed-speech volume (dB) and the typed-speech level match.
function renderVolume() {
  const v = state && state.volume;
  if (!v) return;
  setValue($("output-volume"), v.output_db);
  setValue($("typed-volume"), v.typed_db);
  if (document.activeElement !== $("output-volume")) $("output-volume-value").textContent = signed(Math.round(v.output_db)) + " dB";
  if (document.activeElement !== $("typed-volume")) $("typed-volume-value").textContent = signed(Math.round(v.typed_db)) + " dB";
  $("auto-match").checked = v.auto_match;
  const from = { live: "your live voice", "last session": "your last live session", default: "a default level" }[v.match_source] || v.match_source;
  $("match-level").textContent = v.auto_match ? "Typed clips matched to " + Math.round(v.match_dbfs) + " dBFS (" + from + ")" : "";
}

function fillDevices(select, list, current) {
  const key = JSON.stringify([list, current && current.name]);
  if (select.dataset.key === key) return;
  select.dataset.key = key;
  select.innerHTML = list.map((d) =>
    '<option value="' + esc(d.name) + '"' + (current && d.name === current.name ? " selected" : "") + ">"
    + esc(d.name) + (d.missing ? " (not connected)" : d.cable ? " (virtual cable)" : "") + "</option>").join("") || "<option>No devices found</option>";
}

function shortDate(iso) {
  const date = new Date(iso);
  return isNaN(date) ? iso : date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// models -----------------------------------------------------------------------

function renderModels() {
  const models = state.models;
  const key = JSON.stringify(models);
  if (key === modelsKey) return;
  modelsKey = key;
  const grid = $("model-grid");
  grid.innerHTML = models.map((m) => {
    const tags = [];
    if (m.active) tags.push("in use");
    if (m.test_voice) tags.push("test voice");
    if (m.index) tags.push(Math.round(m.index_vectors / 1000) + "k vectors");
    if (m.speakers > 1 && !m.test_voice) tags.push(m.speakers + " speakers");
    const options = m.index_options.map((o) =>
      '<option value="' + esc(o.file) + '"' + (o.selected ? " selected" : "") + (o.selectable ? "" : " disabled") + ">" + esc(o.label) + "</option>").join("")
      + '<option value="\::browse::">Browse…</option>';
    return '<article class="card" data-model="' + esc(m.name) + '">'
      + '<div class="row center" style="gap:12px"><div class="chip chip-md' + (m.active ? " on" : "") + '">' + esc(m.name[0].toUpperCase()) + "</div>"
      + '<div class="minw0"><div class="card-name" title="' + esc(m.name) + '">' + esc(m.name) + "</div>"
      + '<div class="card-spec">RVC ' + esc(m.version) + " · " + m.sr / 1000 + " kHz · " + (m.f0 ? "f0" : "no f0") + "</div></div></div>"
      + '<div class="tags">' + tags.map((t) => '<span class="tag">' + esc(t) + "</span>").join("") + "</div>"
      + '<label class="col" style="gap:5px"><span class="mini-label">Index</span><select class="mono" data-act="index">' + options + "</select></label>"
      + '<div class="card-actions"><button class="btn-use' + (m.active ? " current" : "") + '" data-act="use"' + (m.active ? " disabled" : "") + ">" + (m.active ? "In use" : "Use") + "</button>"
      + '<button class="icon-btn" data-act="preview" aria-label="Preview this voice" title="Preview this voice"><svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></button></div>'
      + "</article>";
  }).join("") || '<div class="empty">No voices yet. Drop a .pth here to import your first one.</div>';
  const total = models.reduce((sum, m) => sum + m.size_mb, 0);
  const folder = state.models_dir.split(/[\\/]/).filter(Boolean).pop();
  $("models-foot").textContent = models.length + (models.length === 1 ? " model · " : " models · ")
    + Math.round(total) + " MB on disk · " + folder + "\\";
}

// A .pth named like this says nothing about the voice (training exports are often just
// "model.pth"): the folder it came in names it, else the user does.
const GENERIC_PTH = /^(model|g_\d+|g|weights?|voice|checkpoint)$/i;

function voiceName(pth, folder) {
  const stem = pth.name.replace(/\.pth$/i, "");
  if (folder) return folder;
  if (!GENERIC_PTH.test(stem)) return null; // the server names it after the file
  let answer = null;
  try { answer = window.prompt("\"" + pth.name + "\" doesn't say which voice it is. Name it:", ""); } catch (_) { answer = null; }
  return answer && answer.trim() ? answer.trim() : undefined; // undefined: cancelled
}

async function importFiles(files, folder = null) {
  const list = [...files];
  const pths = list.filter((f) => f.name.toLowerCase().endsWith(".pth"));
  const indexes = list.filter((f) => f.name.toLowerCase().endsWith(".index"));
  if (!pths.length) { toast("Drop a .pth (an .index alone goes in through a model's Index menu)", "error"); return; }
  const zone = $("dropzone");
  zone.classList.add("busy");
  try {
    for (const pth of pths) {
      // An index dropped with exactly one .pth belongs to it.
      const index = pths.length === 1 && indexes.length === 1 ? indexes[0] : null;
      const name = voiceName(pth, pths.length === 1 ? folder : null);
      if (name === undefined) { toast("Import of " + pth.name + " cancelled", "warn"); continue; }
      $("dropzone-text").textContent = "Copying " + (name || pth.name) + (index ? " + " + index.name : "") + "…";
      const sent = await upload(pth);
      const sentIndex = index ? await upload(index) : null;
      await cmd("import_model", { pth: sent, index: sentIndex, name });
    }
    if (indexes.length && pths.length !== 1) toast("Indexes are paired only when dropped with exactly one .pth", "warn");
  } catch (err) {
    toast("Import failed: " + err.message, "error");
  } finally {
    zone.classList.remove("busy");
    $("dropzone-text").innerHTML = "Drop a voice's <b>folder</b>, a <b>.pth</b>, or <b>.pth + .index together</b>";
  }
}

// A dropped folder: its .pth / .index files, named after the folder. Entries must be taken
// from the DataTransfer synchronously, inside the drop event; reading them can wait.
function dropEntries(dt) {
  return [...(dt.items || [])].map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
}

function entryFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

async function folderFiles(dir) {
  const reader = dir.createReader();
  const all = [];
  for (;;) { // readEntries returns the listing in batches until an empty one
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    all.push(...batch);
  }
  const wanted = all.filter((e) => e.isFile && /\.(pth|index)$/i.test(e.name));
  return Promise.all(wanted.map(entryFile));
}

async function importDrop(entries, files) {
  const dirs = entries.filter((e) => e.isDirectory);
  if (!dirs.length) { await importFiles(files); return; }
  for (const dir of dirs) {
    const inside = await folderFiles(dir);
    if (!inside.some((f) => /\.pth$/i.test(f.name))) { toast(dir.name + ": no .pth inside", "error"); continue; }
    await importFiles(inside, dir.name);
  }
  const loose = [...files].filter((f) => /\.(pth|index)$/i.test(f.name));
  if (loose.length) await importFiles(loose);
}

// performance ------------------------------------------------------------------

function clock(seconds) {
  const s = Math.floor(seconds);
  const pad = (n) => String(n).padStart(2, "0");
  return pad(Math.floor(s / 3600)) + ":" + pad(Math.floor(s / 60) % 60) + ":" + pad(s % 60);
}

function renderPerf() {
  if (!state) return;
  const running = state.engine.status === "running";
  const p = running ? perf : null;
  const hop = p ? p.hop_ms : state.shape.block_ms;
  $("perf-lead").textContent = running
    ? "Live, last 60 s · CUDA graphs " + (state.engine.mode === "graphs" ? "on" : "off") + " · hop " + Math.round(hop) + " ms"
    : "Stopped — start the engine to see live numbers";
  $("perf-budget").textContent = "budget: hop " + fixed(hop, 1) + " ms";
  $("chart-budget").textContent = "hop budget " + Math.round(hop) + " ms";
  const latency = p && p.latency_ms != null ? Math.round(p.latency_ms) : "–";
  $("latency").textContent = latency;
  $("perf-latency").textContent = latency;
  const under = p ? p.underruns : 0;
  $("dropouts").textContent = under + (under === 1 ? " dropout" : " dropouts");
  $("perf-underruns").textContent = under;
  $("perf-underruns").className = "stat " + (under ? "bad" : "good");
  $("perf-headroom").textContent = p && p.headroom_pct != null ? Math.round(p.headroom_pct) : "–";

  const stages = p ? p.stages : [];
  $("stages").innerHTML = stages.map((s) => {
    const pct = Math.max(0, Math.min(100, (100 * (s.p50 || 0)) / hop));
    return '<div class="stage"><div class="stage-name">' + esc(s.label) + '</div><div class="stage-bar"><div class="stage-fill'
      + (s.p99 > hop ? " over" : "") + '" style="width:' + pct.toFixed(1) + '%"></div></div><div class="stage-vals">'
      + fixed(s.p50) + " / " + fixed(s.p99) + " ms</div></div>";
  }).join("") || '<div class="stage-empty">No hops measured yet.</div>';
  $("perf-total").textContent = p ? fixed(p.p50) + " / " + fixed(p.p99) + " ms" : "– / – ms";

  // 60 s chart: the budget line sits at y=30; the scale leaves 25% above it for overruns.
  const points = (key) => {
    const rows = running ? history.filter((h) => h[key] != null) : [];
    if (!rows.length) return "";
    const end = rows[rows.length - 1].t;
    return rows.map((h) => {
      const x = 620 - (620 * (end - h.t)) / 60;
      const y = Math.max(2, 150 - (h[key] * 120) / hop);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
  };
  $("chart-p99").setAttribute("points", points("p99"));
  $("chart-p50").setAttribute("points", points("p50"));
  // The line labels ride on the newest value of each line, as in the mockup.
  const newest = running && history.length ? history[history.length - 1] : null;
  for (const key of ["p50", "p99"]) {
    const tag = $("chart-" + key + "-tag");
    tag.hidden = !newest || newest[key] == null;
    if (tag.hidden) continue;
    const y = Math.max(2, 150 - (newest[key] * 120) / hop);
    tag.style.top = Math.max(12, (100 * y) / 150 + (key === "p50" ? 12 : 0)) + "%";
    tag.style.right = "36px";
  }

  $("sys-gpu").textContent = p && p.gpu_pct != null ? Math.round(p.gpu_pct) + " % · " + fixed(p.vram_gb) + " GB VRAM" : "–";
  $("sys-cpu").textContent = p && p.cpu_pct != null ? Math.round(p.cpu_pct) + " %" : "–";
  $("sys-ring").textContent = p ? Math.round(p.out_fill_ms) + " ms fill" : "–";
  const gateEl = $("sys-gate");
  gateEl.textContent = !running ? "–" : meters.gate_open ? "open" : "closed";
  gateEl.className = "num" + (running && meters.gate_open ? " good-text" : "");
  $("sys-session").textContent = p ? clock(p.t) : "–";
}

// popovers ---------------------------------------------------------------------

function closePopovers() { $("history").hidden = true; $("voices").hidden = true; }

function renderHistory() {
  const items = state.tts.history;
  $("history").innerHTML = items.map((h) =>
    '<button class="pop-item" data-id="' + h.id + '" title="' + esc(h.spoken && h.spoken !== h.text ? "Said as: " + h.spoken + " · click to play again" : "Play again") + '"><svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>'
    + '<span class="pop-text">' + esc(h.text) + '</span><span class="pop-meta">' + fixed(h.seconds) + " s · " + signed(h.pitch) + "</span></button>").join("")
    || '<div class="pop-empty">Nothing spoken yet. Clips you type appear here to replay.</div>';
}

async function openVoices() {
  const pop = $("voices");
  pop.className = "popover left";
  pop.innerHTML = '<div class="pop-empty">Loading voices…</div>';
  pop.hidden = false;
  const voices = await cmd("tts_voices");
  if (!voices) { pop.hidden = true; return; }
  pop.innerHTML = voices.map((v) => '<button class="pop-item' + (v === state.tts.voice ? " current" : "") + '" data-voice="' + esc(v) + '"><span class="pop-text num">' + esc(v) + "</span></button>").join("");
}

// ------------------------------------------------------------------- wiring

function wire() {
  window.addEventListener("hashchange", route);
  $("power").addEventListener("click", () => {
    const status = state && state.engine.status;
    cmd(status === "running" || status === "starting" ? "stop" : "start");
  });

  const sendPitch = throttled((v) => cmd("set_param", { name: "pitch", value: v }));
  $("pitch").addEventListener("input", (e) => { $("pitch-value").textContent = signed(+e.target.value); sendPitch(+e.target.value); });
  // Formant: each value is its own engine shape (a first visit pauses the voice briefly), so
  // it is sent when the slider is let go, not on every step it passes.
  $("formant").addEventListener("input", (e) => { $("formant-value").textContent = formantText(+e.target.value); });
  $("formant").addEventListener("change", (e) => cmd("set_param", { name: "formant", value: +e.target.value }));
  $("formant").addEventListener("dblclick", () => {
    $("formant").value = 0;
    $("formant-value").textContent = formantText(0);
    cmd("set_param", { name: "formant", value: 0 });
  });
  const sendGate = throttled((v) => cmd("set_gate", { threshold_dbfs: v }));
  $("gate-threshold").addEventListener("input", (e) => {
    const v = +e.target.value;
    $("gate-value").textContent = v + " dB";
    $("gate-tick").style.left = meterPct(v) + "%";
    sendGate(v);
  });
  for (const id of ["gate-on", "gate-on-2"]) $(id).addEventListener("change", (e) => cmd("set_gate", { enabled: e.target.checked }));
  const sendGain = throttled((v) => cmd("set_input_level", { gain_db: v }));
  $("input-gain").addEventListener("input", (e) => { renderInputGain(); sendGain(+e.target.value); });
  $("auto-level").addEventListener("change", (e) => cmd("set_input_level", { auto: e.target.checked }));
  const sendOutput = throttled((v) => cmd("set_volume", { output_db: v }));
  $("output-volume").addEventListener("input", (e) => { $("output-volume-value").textContent = signed(+e.target.value) + " dB"; sendOutput(+e.target.value); });
  const sendTyped = throttled((v) => cmd("set_volume", { typed_db: v }));
  $("typed-volume").addEventListener("input", (e) => { $("typed-volume-value").textContent = signed(+e.target.value) + " dB"; sendTyped(+e.target.value); });
  $("auto-match").addEventListener("change", (e) => cmd("set_volume", { auto_match: e.target.checked }));
  $("record").addEventListener("click", () => cmd("record", { enabled: !(state && state.recording && state.recording.active) }));
  const sendIndex = throttled((v) => cmd("set_param", { name: "index_rate", value: v }));
  $("index-rate").addEventListener("input", (e) => { $("index-rate-value").textContent = (e.target.value / 100).toFixed(2); sendIndex(e.target.value / 100); });
  const sendProtect = throttled((v) => cmd("set_param", { name: "protect", value: v }));
  $("protect").addEventListener("input", (e) => { $("protect-value").textContent = (e.target.value / 100).toFixed(2); sendProtect(e.target.value / 100); });
  const sendCeiling = throttled((v) => cmd("set_param", { name: "ceiling", value: v }));
  $("ceiling").addEventListener("input", (e) => {
    const hz = +e.target.value <= CEILING_OFF ? 0 : +e.target.value;
    $("ceiling-value").textContent = ceilingText(hz);
    sendCeiling(hz);
  });
  $("auto-pitch").addEventListener("change", (e) => cmd("set_auto_pitch", { enabled: e.target.checked }));
  const sendTarget = throttled((v) => cmd("set_param", { name: "target_hz", value: v }));
  $("target-hz").addEventListener("input", (e) => { $("target-value").textContent = e.target.value + " Hz"; sendTarget(+e.target.value); });
  const sendRange = throttled((v) => cmd("set_param", { name: "pitch_range", value: v }));
  $("pitch-range").addEventListener("input", (e) => { $("range-value").textContent = e.target.value + " %"; sendRange(e.target.value / 100); });
  $("preset").addEventListener("click", (e) => {
    const b = e.target.closest("[data-preset]");
    if (b && !b.classList.contains("on")) cmd("set_preset", { preset: b.dataset.preset });
  });
  $("suppression").addEventListener("change", (e) => cmd("set_suppression", { enabled: e.target.checked }));
  const sendMix = throttled((v) => cmd("set_suppression", { mix: v }));
  $("suppression-mix").addEventListener("input", (e) => { $("suppression-mix-value").textContent = e.target.value + " %"; sendMix(e.target.value / 100); });

  // Restart-class controls send once, when the user lets go.
  $("hop").addEventListener("input", (e) => { $("hop-value").textContent = e.target.value + " ms"; });
  $("hop").addEventListener("change", (e) => cmd("set_shape", { block_ms: +e.target.value }));
  $("context").addEventListener("input", (e) => { $("context-value").textContent = e.target.value + " ms"; });
  $("context").addEventListener("change", (e) => cmd("set_shape", { extra_ms: +e.target.value }));
  $("tracker").addEventListener("change", (e) => cmd("set_shape", { f0_method: e.target.value }));
  $("graphs").addEventListener("change", (e) => cmd("set_graphs", { enabled: e.target.checked }));
  $("exclusive").addEventListener("change", (e) => cmd("set_exclusive", { enabled: e.target.checked }));
  $("refresh-devices").addEventListener("click", () => refreshDevices(false));
  $("mic").addEventListener("change", (e) => cmd("set_device", { kind: "input", name: e.target.value }));
  $("out").addEventListener("change", (e) => cmd("set_device", { kind: "output", name: e.target.value }));
  $("calibrate").addEventListener("click", () => cmd("calibrate"));
  const sendVariation = throttled((v) => cmd("set_tts", { variation: v }));
  $("variation").addEventListener("input", (e) => { $("variation-value").textContent = (e.target.value / 100).toFixed(2); sendVariation(e.target.value / 100); });
  $("laughter").addEventListener("change", (e) => cmd("set_tts", { laughter: e.target.value }));
  $("contour").addEventListener("change", (e) => cmd("set_tts", { contour: e.target.checked ? 1.15 : 1.0 }));
  $("slang-edit").addEventListener("click", async () => {
    const r = await cmd("open_slang");
    if (r && r.path) toast((r.created ? "Created " : "Opened ") + r.path + ". Save it, then press Reload.");
  });
  $("slang-reload").addEventListener("click", () => cmd("reload_slang"));
  $("export").addEventListener("click", () => cmd("export_report"));

  // type to speak
  const say = () => {
    if (state && state.tts.speaking && !$("speak-text").value.trim()) { cmd("cancel_speech"); return; }
    const text = $("speak-text").value.trim();
    if (!text) return;
    $("speak-text").value = "";
    cmd("speak", { text });
  };
  $("speak-go").addEventListener("click", say);
  $("speak-text").addEventListener("keydown", (e) => { if (e.key === "Enter") say(); });
  $("history-open").addEventListener("click", (e) => {
    e.stopPropagation();
    const pop = $("history");
    const open = pop.hidden;
    closePopovers();
    if (open) { renderHistory(); pop.hidden = false; }
  });
  $("history").addEventListener("click", (e) => {
    const item = e.target.closest("[data-id]");
    if (item) cmd("replay", { id: +item.dataset.id });
  });
  $("tts-voice").addEventListener("click", (e) => {
    e.stopPropagation();
    const open = $("voices").hidden;
    closePopovers();
    if (open) openVoices();
  });
  $("voices").addEventListener("click", (e) => {
    const item = e.target.closest("[data-voice]");
    if (item) { cmd("set_tts", { voice: item.dataset.voice }); closePopovers(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".popover")) closePopovers(); });

  // models
  $("import-open").addEventListener("click", () => $("import-files").click());
  $("import-files").addEventListener("change", (e) => { importFiles(e.target.files); e.target.value = ""; });
  $("model-grid").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]");
    if (!act || act.tagName === "SELECT") return;
    const model = act.closest("[data-model]").dataset.model;
    if (act.dataset.act === "use") cmd("select_model", { name: model });
    if (act.dataset.act === "preview") cmd("preview", { model });
  });
  $("model-grid").addEventListener("change", (e) => {
    if (e.target.dataset.act !== "index") return;
    const model = e.target.closest("[data-model]").dataset.model;
    if (e.target.value === "\::browse::") {
      pendingIndexModel = model;
      modelsKey = "";  // put the menu back on the real choice
      renderModels();
      $("index-file").click();
    } else {
      cmd("set_index", { model, file: e.target.value });
    }
  });
  $("index-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file || !pendingIndexModel) return;
    try {
      toast("Copying " + file.name + "…");
      await cmd("add_index", { model: pendingIndexModel, index: await upload(file) });
    } catch (err) { toast("Could not add the index: " + err.message, "error"); }
  });

  // whole-window drag and drop
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; $("drop-veil").hidden = false; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", (e) => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth) $("drop-veil").hidden = true; });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    $("drop-veil").hidden = true;
    location.hash = "#models";
    importDrop(dropEntries(e.dataTransfer), [...e.dataTransfer.files]);
  });
}

window.addEventListener("resize", fit);
fit();
wire();
route();
connect();
