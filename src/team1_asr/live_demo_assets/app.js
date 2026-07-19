const ui = {
  serverDot: document.querySelector("#server-dot"),
  modeLabel: document.querySelector("#mode-label"),
  modeChip: document.querySelector("#mode-chip"),
  policyLabel: document.querySelector("#policy-label"),
  sampleSwitcher: document.querySelector("#sample-switcher"),
  conditionLabel: document.querySelector("#condition-label"),
  referenceText: document.querySelector("#reference-text"),
  canvas: document.querySelector("#signal-canvas"),
  video: document.querySelector("#video-preview"),
  controls: document.querySelector("#signal-controls"),
  audio: document.querySelector("#audio-confidence"),
  audioValue: document.querySelector("#audio-confidence-value"),
  mouth: document.querySelector("#mouth-confidence"),
  mouthValue: document.querySelector("#mouth-confidence-value"),
  roi: document.querySelector("#roi-ready"),
  run: document.querySelector("#run-button"),
  buttonState: document.querySelector("#button-state"),
  elapsed: document.querySelector("#elapsed"),
  trace: document.querySelector("#trace-list"),
  score: document.querySelector("#score-value"),
  threshold: document.querySelector("#threshold-label"),
  meterValue: document.querySelector("#meter-value"),
  meterThreshold: document.querySelector("#meter-threshold"),
  resultBand: document.querySelector("#result-band"),
  selectedSystem: document.querySelector("#selected-system"),
  resultText: document.querySelector("#result-text"),
  comparison: document.querySelector("#comparison"),
  whisperText: document.querySelector("#whisper-text"),
  avecText: document.querySelector("#avec-text"),
};

const stageLabels = {
  whisper: "Whisper",
  features: "Features",
  router: "Router",
  avec: "AVEC",
  result: "Result",
  error: "Error",
};

let state = null;
let selectedSample = null;
let routeStarted = 0;
let timer = null;
let waveformPhase = 0;

async function loadState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    state = await response.json();
    ui.serverDot.classList.add("online");
    ui.modeLabel.textContent = state.mode === "real" ? "Real backends online" : "Presentation runtime";
    ui.modeChip.textContent = state.mode === "real" ? "Real" : "Demo";
    ui.controls.hidden = state.mode === "real";
    ui.policyLabel.textContent = `${state.policy.type} policy · ${state.policy.audio_system} / ${state.policy.av_system}`;
    const threshold = Number(state.policy.threshold);
    ui.threshold.textContent = Number.isFinite(threshold)
      ? `Threshold ${threshold.toFixed(2)}`
      : "Policy gate";
    ui.meterThreshold.style.left = Number.isFinite(threshold)
      ? `${Math.max(0, Math.min(100, threshold * 100))}%`
      : "50%";
    renderSamples();
    selectSample(state.samples[0].id);
  } catch (error) {
    ui.modeLabel.textContent = "Server unavailable";
    showError(error.message);
  }
}

function renderSamples() {
  ui.sampleSwitcher.replaceChildren();
  ui.sampleSwitcher.style.gridTemplateColumns = `repeat(${Math.min(state.samples.length, 3)}, minmax(0, 1fr))`;
  for (const sample of state.samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-option";
    button.role = "tab";
    button.dataset.sampleId = sample.id;

    const title = document.createElement("strong");
    title.textContent = sample.label;
    const condition = document.createElement("span");
    condition.textContent = sample.condition;
    button.append(title, condition);
    button.addEventListener("click", () => selectSample(sample.id));
    ui.sampleSwitcher.append(button);
  }
}

function selectSample(sampleId) {
  selectedSample = state.samples.find((sample) => sample.id === sampleId);
  if (!selectedSample) return;
  for (const button of ui.sampleSwitcher.children) {
    const active = button.dataset.sampleId === sampleId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  ui.conditionLabel.textContent = selectedSample.condition;
  ui.referenceText.textContent = selectedSample.reference || "No reference transcript";
  const features = selectedSample.features || {};
  if (features.audio_avg_logprob_mean != null) {
    ui.audio.value = features.audio_avg_logprob_mean;
  }
  if (features.mouth_landmark_confidence_mean != null) {
    ui.mouth.value = features.mouth_landmark_confidence_mean;
  }
  ui.roi.checked = features.roi_ready !== false;
  syncControls();
  resetRoute();
  configureMedia();
  drawWaveform();
}

function configureMedia() {
  const realVideo = state.mode === "real" && selectedSample.has_video;
  ui.video.hidden = !realVideo;
  ui.canvas.hidden = realVideo;
  if (realVideo) {
    ui.video.src = `/api/media/${encodeURIComponent(selectedSample.id)}/video`;
  } else {
    ui.video.removeAttribute("src");
  }
}

function syncControls() {
  ui.audioValue.textContent = Number(ui.audio.value).toFixed(2);
  ui.mouthValue.textContent = Number(ui.mouth.value).toFixed(2);
  drawWaveform();
}

function drawWaveform() {
  if (ui.canvas.hidden) return;
  const canvas = ui.canvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  ctx.fillStyle = "#202c2f";
  ctx.fillRect(0, 0, width, height);

  const confidence = Math.max(0, Math.min(1, (Number(ui.audio.value) + 2) / 2));
  const center = height * 0.5;
  const bars = Math.max(45, Math.floor(width / 6));
  ctx.fillStyle = confidence > 0.55 ? "#42bbc3" : "#f05d44";
  for (let index = 0; index < bars; index += 1) {
    const x = (index / bars) * width;
    const harmonic = Math.sin(index * 0.51 + waveformPhase) * 0.55;
    const texture = Math.sin(index * 1.73 + 1.4) * 0.25;
    const envelope = 0.25 + 0.75 * Math.sin((index / bars) * Math.PI);
    const amplitude = Math.max(4, Math.abs(harmonic + texture) * envelope * height * 0.34);
    ctx.fillRect(x, center - amplitude, Math.max(2, width / bars - 2), amplitude * 2);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.moveTo(0, center + 0.5);
  ctx.lineTo(width, center + 0.5);
  ctx.stroke();
}

async function runRoute() {
  if (!selectedSample || ui.run.disabled) return;
  resetRoute();
  ui.run.disabled = true;
  ui.buttonState.textContent = "Running";
  routeStarted = performance.now();
  timer = window.setInterval(updateElapsed, 50);
  animateSignal();

  const overrides = state.mode === "mock"
    ? {
        audio_avg_logprob_mean: Number(ui.audio.value),
        mouth_landmark_confidence_mean: Number(ui.mouth.value),
        roi_ready: ui.roi.checked,
      }
    : {};

  try {
    const response = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: selectedSample.id, overrides }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`Route request failed with ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() || "";
      for (const line of lines) {
        if (line.trim()) handleEvent(JSON.parse(line));
      }
      if (done) break;
    }
    if (pending.trim()) handleEvent(JSON.parse(pending));
  } catch (error) {
    showError(error.message);
  } finally {
    window.clearInterval(timer);
    timer = null;
    updateElapsed();
    ui.run.disabled = false;
    ui.buttonState.textContent = "Ready";
  }
}

function handleEvent(event) {
  appendTrace(event);
  const stage = event.stage === "features" ? "router" : event.stage;
  const node = document.querySelector(`.pipeline-node[data-stage="${stage}"]`);
  if (node) {
    node.classList.remove("running", "complete", "skipped");
    node.classList.add(event.status);
  }
  updateLinks(event);

  if (event.stage === "router") {
    const score = Number(event.score);
    ui.score.textContent = Number.isFinite(score) ? score.toFixed(3) : "Gate";
    ui.meterValue.style.width = Number.isFinite(score)
      ? `${Math.max(0, Math.min(100, score * 100))}%`
      : event.selects_avec ? "100%" : "0%";
  }
  if (event.stage === "result" && event.result) {
    renderResult(event.result);
  }
}

function appendTrace(event) {
  const empty = document.querySelector("#trace-empty");
  if (empty) empty.remove();
  const item = document.createElement("div");
  item.className = `trace-item ${event.status || ""}`;

  const stage = document.createElement("span");
  stage.className = "trace-stage";
  stage.textContent = stageLabels[event.stage] || event.stage;

  const copy = document.createElement("div");
  copy.className = "trace-copy";
  const title = document.createElement("strong");
  title.textContent = event.message || "Stage update";
  copy.append(title);
  const detailText = traceDetail(event);
  if (detailText) {
    const detail = document.createElement("p");
    detail.textContent = detailText;
    copy.append(detail);
  }

  const status = document.createElement("span");
  status.className = "trace-status";
  status.textContent = event.status || "update";
  item.append(stage, copy, status);
  ui.trace.append(item);
  ui.trace.scrollTop = ui.trace.scrollHeight;
}

function traceDetail(event) {
  if (event.transcript) return `“${event.transcript}”`;
  if (event.stage === "router" && event.selected_system) {
    return `Selected ${event.selected_system} · score ${formatNumber(event.score)} · threshold ${formatNumber(event.threshold)}`;
  }
  if (event.features) {
    const audio = formatNumber(event.features.audio_avg_logprob_mean);
    const mouth = formatNumber(event.features.mouth_landmark_confidence_mean);
    return `Whisper log p ${audio} · mouth confidence ${mouth}`;
  }
  if (event.elapsed_seconds != null) return `${Number(event.elapsed_seconds).toFixed(2)} seconds`;
  return "";
}

function renderResult(result) {
  ui.selectedSystem.textContent = result.selected_system;
  ui.resultText.textContent = result.transcript.text || "No speech detected.";
  ui.whisperText.textContent = result.whisper_transcript.text || "No speech detected.";
  ui.avecText.textContent = result.avec_transcript
    ? result.avec_transcript.text || "No speech detected."
    : "Skipped by router";
  ui.comparison.hidden = false;
  for (const node of document.querySelectorAll(".pipeline-node")) {
    node.classList.remove("selected");
  }
  const selectedStage = result.avec_transcript ? "avec" : "whisper";
  document.querySelector(`.pipeline-node[data-stage="${selectedStage}"]`)?.classList.add("selected");
  ui.resultBand.dataset.system = selectedStage;
}

function updateLinks(event) {
  const links = document.querySelectorAll(".pipeline-link");
  if (event.stage === "whisper" && event.status === "running") links[0]?.classList.add("active");
  if (event.stage === "router" && event.status === "complete") {
    links[0]?.classList.remove("active");
    if (event.selects_avec) links[1]?.classList.add("active");
  }
  if (event.stage === "avec" && event.status !== "running") links[1]?.classList.remove("active");
}

function resetRoute() {
  routeStarted = 0;
  window.clearInterval(timer);
  timer = null;
  ui.elapsed.textContent = "0.00 s";
  ui.score.textContent = "Waiting";
  ui.meterValue.style.width = "0";
  ui.selectedSystem.textContent = "No route yet";
  ui.resultText.textContent = "The final transcription will appear here.";
  ui.comparison.hidden = true;
  ui.trace.innerHTML = '<div class="trace-empty" id="trace-empty"><span class="trace-line"></span><p>Run a sample to stream the inference trace.</p></div>';
  for (const node of document.querySelectorAll(".pipeline-node")) {
    node.classList.remove("running", "complete", "skipped", "selected");
  }
  for (const link of document.querySelectorAll(".pipeline-link")) link.classList.remove("active");
}

function updateElapsed() {
  if (!routeStarted) return;
  ui.elapsed.textContent = `${((performance.now() - routeStarted) / 1000).toFixed(2)} s`;
}

function animateSignal() {
  if (ui.run.disabled && !ui.canvas.hidden) {
    waveformPhase += 0.12;
    drawWaveform();
    requestAnimationFrame(animateSignal);
  }
}

function showError(message) {
  handleEvent({ stage: "error", status: "error", message });
  ui.selectedSystem.textContent = "Route failed";
  ui.resultText.textContent = message;
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "n/a";
}

ui.audio.addEventListener("input", syncControls);
ui.mouth.addEventListener("input", syncControls);
ui.roi.addEventListener("change", drawWaveform);
ui.run.addEventListener("click", runRoute);
window.addEventListener("resize", drawWaveform);
loadState();
