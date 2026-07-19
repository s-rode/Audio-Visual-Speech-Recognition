const ui = {
  serverDot: document.querySelector("#server-dot"),
  modeLabel: document.querySelector("#mode-label"),
  modeChip: document.querySelector("#mode-chip"),
  policyLabel: document.querySelector("#policy-label"),
  samplesMode: document.querySelector("#samples-mode"),
  captureMode: document.querySelector("#capture-mode"),
  sampleOnly: document.querySelectorAll("[data-sample-only]"),
  sampleSwitcher: document.querySelector("#sample-switcher"),
  conditionLabel: document.querySelector("#condition-label"),
  durationLabel: document.querySelector("#duration-label"),
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
  captureControls: document.querySelector("#capture-controls"),
  enableCamera: document.querySelector("#enable-camera"),
  record: document.querySelector("#record-button"),
  recordState: document.querySelector("#record-state"),
  deviceDot: document.querySelector("#device-dot"),
  deviceTitle: document.querySelector("#device-title"),
  deviceCopy: document.querySelector("#device-copy"),
  mediaStage: document.querySelector(".media-stage"),
  inputTitle: document.querySelector("#input-title"),
  liveAudioStrip: document.querySelector("#live-audio-strip"),
  liveWaveform: document.querySelector("#live-waveform"),
  captureInspector: document.querySelector("#capture-inspector"),
  inspectorTabs: document.querySelectorAll("[data-inspector-view]"),
  inspectorPanels: document.querySelectorAll("[data-inspector-panel]"),
  sourceCapture: document.querySelector("#source-capture"),
  processedCapture: document.querySelector("#processed-capture"),
  processedTab: document.querySelector("#processed-tab"),
  captureAudio: document.querySelector("#capture-audio"),
  captureWaveform: document.querySelector("#capture-waveform"),
  artifactStatus: document.querySelector("#artifact-status"),
  telemetry: document.querySelector("#telemetry-grid"),
  transcriptSelection: document.querySelector("#transcript-selection"),
  whisperTranscriptCard: document.querySelector("#whisper-transcript-card"),
  whisperTranscriptStatus: document.querySelector("#whisper-transcript-status"),
  whisperLiveTranscript: document.querySelector("#whisper-live-transcript"),
  whisperSegments: document.querySelector("#whisper-segments"),
  avecTranscriptCard: document.querySelector("#avec-transcript-card"),
  avecTranscriptStatus: document.querySelector("#avec-transcript-status"),
  avecLiveTranscript: document.querySelector("#avec-live-transcript"),
  avecSegments: document.querySelector("#avec-segments"),
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
  capture: "Capture",
  preprocess: "Preprocess",
  media: "Artifacts",
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
let inputMode = "capture";
let mediaStream = null;
let mediaRecorder = null;
let audioContext = null;
let audioAnalyser = null;
let audioSource = null;
let liveWaveformFrame = null;
let capturePlaybackFrame = null;
let captureWaveform = [];
let artifactMetrics = {};
let routeFeatures = {};

async function loadState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    state = await response.json();
    ui.serverDot.classList.add("online");
    ui.modeLabel.textContent = state.mode === "real" ? "Real backends online" : "Presentation runtime";
    ui.modeChip.textContent = state.mode === "real" ? "Real" : "Demo";
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
    setInputMode("capture");
  } catch (error) {
    ui.modeLabel.textContent = "Server unavailable";
    showError(error.message);
  }
}

function setInputMode(mode) {
  inputMode = mode;
  const capture = mode === "capture";
  ui.samplesMode.classList.toggle("active", !capture);
  ui.captureMode.classList.toggle("active", capture);
  ui.samplesMode.setAttribute("aria-selected", String(!capture));
  ui.captureMode.setAttribute("aria-selected", String(capture));
  for (const element of ui.sampleOnly) {
    element.hidden = capture;
  }
  ui.controls.hidden = capture || state.mode === "real";
  ui.captureControls.hidden = !capture;
  ui.inputTitle.textContent = capture ? "Live input" : "Route a sample";
  resetRoute();

  if (capture) {
    ui.canvas.hidden = true;
    ui.video.hidden = false;
    ui.video.controls = false;
    ui.video.muted = true;
    ui.video.classList.add("live-preview");
    ui.video.removeAttribute("src");
    ui.video.load();
    ui.durationLabel.textContent = "5 s window";
    ui.conditionLabel.textContent = mediaStream ? "Camera ready" : "Camera offline";
    if (mediaStream) ui.video.srcObject = mediaStream;
  } else {
    stopCamera();
    ui.video.classList.remove("live-preview");
    configureMedia();
    ui.conditionLabel.textContent = selectedSample.condition;
  }
}

async function enableCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera and microphone capture is unavailable in this browser.");
    return;
  }
  ui.enableCamera.disabled = true;
  setDeviceState("Requesting browser permission", "Waiting for camera and microphone", "");
  try {
    stopCamera();
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: {
        width: { ideal: 640 },
        height: { ideal: 360 },
        frameRate: { ideal: 25, max: 30 },
        facingMode: "user",
      },
    });
    ui.video.removeAttribute("src");
    ui.video.srcObject = mediaStream;
    await ui.video.play();
    startAudioMonitor(mediaStream);
    for (const track of mediaStream.getTracks()) {
      track.addEventListener("ended", () => {
        if (mediaStream && mediaStream.getTracks().every((item) => item.readyState === "ended")) {
          stopCamera();
        }
      });
    }
    ui.record.disabled = false;
    ui.recordState.textContent = "Ready";
    ui.enableCamera.textContent = "Reset devices";
    ui.conditionLabel.textContent = "Camera ready";
    setDeviceState("Camera and microphone ready", "Five-second synchronized window", "ready");
  } catch (error) {
    stopCamera();
    showError(error.message || "Camera and microphone permission was denied.");
  } finally {
    ui.enableCamera.disabled = false;
  }
}

function stopCamera() {
  stopAudioMonitor();
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  mediaRecorder = null;
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) track.stop();
  }
  mediaStream = null;
  ui.video.srcObject = null;
  ui.record.disabled = true;
  ui.recordState.textContent = "Offline";
  ui.conditionLabel.textContent = "Camera offline";
  setDeviceState("Camera and microphone off", "Browser permission required", "");
}

function setDeviceState(title, copy, status) {
  ui.deviceTitle.textContent = title;
  ui.deviceCopy.textContent = copy;
  ui.deviceDot.classList.remove("ready", "recording");
  if (status) ui.deviceDot.classList.add(status);
}

function startAudioMonitor(stream) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  stopAudioMonitor();
  audioContext = new AudioContextClass();
  audioAnalyser = audioContext.createAnalyser();
  audioAnalyser.fftSize = 1024;
  audioAnalyser.smoothingTimeConstant = 0.72;
  audioSource = audioContext.createMediaStreamSource(stream);
  audioSource.connect(audioAnalyser);
  audioContext.resume().catch(() => {});
  ui.liveAudioStrip.hidden = false;
  drawLiveWaveform();
}

function stopAudioMonitor() {
  if (liveWaveformFrame) cancelAnimationFrame(liveWaveformFrame);
  liveWaveformFrame = null;
  if (audioSource) audioSource.disconnect();
  audioSource = null;
  audioAnalyser = null;
  if (audioContext) audioContext.close().catch(() => {});
  audioContext = null;
  ui.liveAudioStrip.hidden = true;
}

function drawLiveWaveform() {
  if (!audioAnalyser) return;
  const values = new Uint8Array(audioAnalyser.fftSize);
  audioAnalyser.getByteTimeDomainData(values);
  const canvas = ui.liveWaveform;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#8fe2d8";
  context.lineWidth = 2;
  context.beginPath();
  for (let index = 0; index < values.length; index += 1) {
    const x = (index / (values.length - 1)) * canvas.width;
    const y = (values[index] / 255) * canvas.height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  liveWaveformFrame = requestAnimationFrame(drawLiveWaveform);
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
  if (inputMode === "capture") return;
  const realVideo = state.mode === "real" && selectedSample.has_video;
  ui.video.srcObject = null;
  ui.video.controls = realVideo;
  ui.video.muted = !realVideo;
  ui.durationLabel.textContent = realVideo ? "Recorded sample" : "3.2 s";
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
    await readEventStream(response);
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

async function recordCapture() {
  if (!mediaStream || mediaRecorder || ui.record.disabled) return;
  resetRoute();
  ui.record.disabled = true;
  ui.enableCamera.disabled = true;
  ui.samplesMode.disabled = true;
  ui.captureMode.disabled = true;
  ui.mediaStage.classList.add("recording");
  setDeviceState("Recording live window", "Speak while facing the camera", "recording");
  routeStarted = performance.now();
  timer = window.setInterval(updateElapsed, 50);

  try {
    const mimeType = preferredCaptureType();
    const options = mimeType ? { mimeType, videoBitsPerSecond: 1_500_000 } : undefined;
    const chunks = [];
    const recorder = new MediaRecorder(mediaStream, options);
    mediaRecorder = recorder;
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    const stopped = new Promise((resolve, reject) => {
      recorder.addEventListener("stop", resolve, { once: true });
      recorder.addEventListener(
        "error",
        (event) => reject(event.error || new Error("Media recording failed.")),
        { once: true },
      );
    });

    recorder.start(250);
    for (let remaining = 5; remaining > 0; remaining -= 1) {
      ui.recordState.textContent = `${remaining}s`;
      ui.durationLabel.textContent = `${remaining} s remaining`;
      await delay(1000);
      if (recorder.state === "inactive") break;
    }
    if (recorder.state !== "inactive") recorder.stop();
    await stopped;

    const blobType = recorder.mimeType || mimeType || "video/webm";
    const recording = new Blob(chunks, { type: blobType });
    if (!recording.size) throw new Error("The browser produced an empty recording.");
    ui.recordState.textContent = "Routing";
    ui.durationLabel.textContent = formatBytes(recording.size);
    ui.conditionLabel.textContent = "Uploading synchronized capture";

    const response = await fetch("/api/live-route", {
      method: "POST",
      headers: { "Content-Type": recording.type },
      body: recording,
    });
    await readEventStream(response);
  } catch (error) {
    showError(error.message);
  } finally {
    mediaRecorder = null;
    window.clearInterval(timer);
    timer = null;
    updateElapsed();
    ui.mediaStage.classList.remove("recording");
    ui.record.disabled = !mediaStream;
    ui.enableCamera.disabled = false;
    ui.samplesMode.disabled = false;
    ui.captureMode.disabled = false;
    ui.recordState.textContent = mediaStream ? "Ready" : "Offline";
    ui.durationLabel.textContent = "5 s window";
    if (mediaStream) {
      ui.conditionLabel.textContent = "Camera ready";
      setDeviceState("Camera and microphone ready", "Five-second synchronized window", "ready");
    }
  }
}

function preferredCaptureType() {
  const candidates = [
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp9,opus",
    "video/webm",
    "video/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function readEventStream(response) {
  if (!response.ok) {
    const body = await response.text();
    let message = `Route request failed with ${response.status}`;
    try {
      message = JSON.parse(body).error || message;
    } catch {
      if (body.trim()) message = body.trim();
    }
    throw new Error(message);
  }
  if (!response.body) throw new Error("Route response has no event stream.");

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
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function formatBytes(bytes) {
  return bytes >= 1_000_000
    ? `${(bytes / 1_000_000).toFixed(1)} MB`
    : `${Math.ceil(bytes / 1000)} KB`;
}

function showCaptureMedia(media) {
  artifactMetrics = media;
  routeFeatures = {};
  ui.captureInspector.hidden = false;
  ui.sourceCapture.src = media.source_url;
  ui.processedCapture.src = media.processed_url;
  ui.captureAudio.src = media.audio_url;
  ui.sourceCapture.load();
  ui.processedCapture.load();
  ui.captureAudio.load();
  ui.processedTab.textContent = media.processed_label || "AVEC input";
  ui.artifactStatus.textContent = media.processed_distinct
    ? "Mouth ROI ready"
    : "Feature inputs ready";
  captureWaveform = Array.isArray(media.waveform) ? media.waveform : [];
  setInspectorView("source");
  drawCaptureWaveform();
  renderTelemetry();
}

function setInspectorView(view) {
  for (const tab of ui.inspectorTabs) {
    const active = tab.dataset.inspectorView === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  }
  for (const panel of ui.inspectorPanels) {
    panel.hidden = panel.dataset.inspectorPanel !== view;
  }
  if (view === "audio") drawCaptureWaveform();
}

function drawCaptureWaveform() {
  const canvas = ui.captureWaveform;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const center = height / 2;
  context.fillStyle = "#202c2f";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(255, 255, 255, 0.2)";
  context.beginPath();
  context.moveTo(0, center + 0.5);
  context.lineTo(width, center + 0.5);
  context.stroke();
  if (!captureWaveform.length) return;
  const barWidth = width / captureWaveform.length;
  context.fillStyle = "#42bbc3";
  for (let index = 0; index < captureWaveform.length; index += 1) {
    const amplitude = Math.max(1, captureWaveform[index] * (height * 0.43));
    context.fillRect(index * barWidth, center - amplitude, Math.max(1, barWidth - 1), amplitude * 2);
  }
  if (Number.isFinite(ui.captureAudio.duration) && ui.captureAudio.duration > 0) {
    const playhead = (ui.captureAudio.currentTime / ui.captureAudio.duration) * width;
    context.fillStyle = "#f05d44";
    context.fillRect(playhead, 0, 2, height);
  }
}

function animateCapturePlayback() {
  drawCaptureWaveform();
  if (!ui.captureAudio.paused) {
    capturePlaybackFrame = requestAnimationFrame(animateCapturePlayback);
  }
}

function renderTelemetry() {
  const metrics = [
    ["Capture size", artifactMetrics.bytes, (value) => formatBytes(value)],
    ["Audio duration", artifactMetrics.duration_seconds, (value) => Number(value).toFixed(2) + " s"],
    ["Sample rate", artifactMetrics.sample_rate, (value) => Math.round(value / 1000) + " kHz"],
    ["Input RMS", routeFeatures.audio_rms_dbfs ?? artifactMetrics.rms_dbfs, (value) => Number(value).toFixed(1) + " dBFS"],
    ["Whisper log p", routeFeatures.audio_avg_logprob_mean, (value) => Number(value).toFixed(3)],
    ["No speech", routeFeatures.audio_no_speech_prob_max, (value) => Number(value).toFixed(3)],
    ["Spectral flatness", routeFeatures.audio_spectral_flatness, (value) => Number(value).toFixed(3)],
    ["Mouth motion", routeFeatures.video_mouth_motion_mean, (value) => Number(value).toFixed(3)],
    [artifactMetrics.visual_metrics_mode === "presentation" ? "Visual gate (demo)" : "Face detection", routeFeatures.face_detection_rate, (value) => (Number(value) * 100).toFixed(0) + "%"],
    ["A/V correlation", routeFeatures.audio_video_sync_correlation, (value) => Number(value).toFixed(3)],
    ["A/V lag", routeFeatures.audio_video_sync_lag_ms, (value) => Number(value).toFixed(0) + " ms"],
    ["ROI status", routeFeatures.roi_status, (value) => String(value)],
  ].filter(([, value]) => value !== null && value !== undefined);

  ui.telemetry.replaceChildren();
  for (const [label, value, formatter] of metrics) {
    const item = document.createElement("div");
    item.className = "telemetry-item";
    const name = document.createElement("span");
    name.textContent = label;
    const output = document.createElement("strong");
    output.textContent = formatter(value);
    output.title = output.textContent;
    item.append(name, output);
    ui.telemetry.append(item);
  }
}

function clearCaptureInspector() {
  if (capturePlaybackFrame) cancelAnimationFrame(capturePlaybackFrame);
  capturePlaybackFrame = null;
  for (const media of [ui.sourceCapture, ui.processedCapture, ui.captureAudio]) {
    media.pause();
    media.removeAttribute("src");
    media.load();
  }
  captureWaveform = [];
  artifactMetrics = {};
  routeFeatures = {};
  ui.telemetry.replaceChildren();
  ui.captureInspector.hidden = true;
}

function transcriptUi(system) {
  return system === "whisper"
    ? {
        card: ui.whisperTranscriptCard,
        status: ui.whisperTranscriptStatus,
        text: ui.whisperLiveTranscript,
        segments: ui.whisperSegments,
      }
    : {
        card: ui.avecTranscriptCard,
        status: ui.avecTranscriptStatus,
        text: ui.avecLiveTranscript,
        segments: ui.avecSegments,
      };
}

function updateTranscriptEvent(event) {
  if (event.stage !== "whisper" && event.stage !== "avec") return;
  const target = transcriptUi(event.stage);
  target.card.classList.remove("running", "complete", "skipped");
  target.card.classList.add(event.status);
  if (event.status === "running") {
    target.status.textContent = "Transcribing";
    target.text.textContent = event.stage === "whisper"
      ? "Listening to the extracted microphone audio..."
      : "Fusing audio with the visual stream...";
  } else if (event.status === "complete") {
    target.status.textContent = "Complete";
    target.text.textContent = event.transcript || "No speech detected.";
  } else if (event.status === "skipped") {
    target.status.textContent = "Skipped";
    target.text.textContent = "Skipped by the routing policy.";
  }
}

function renderTranscriptResult(system, transcript) {
  if (!transcript) return;
  const target = transcriptUi(system);
  target.status.textContent = "Complete";
  target.text.textContent = transcript.text || "No speech detected.";
  target.card.classList.remove("running", "skipped");
  target.card.classList.add("complete");
  target.segments.replaceChildren();
  for (const segment of transcript.segments || []) {
    const row = document.createElement("div");
    row.className = "transcript-segment";
    const timestamp = document.createElement("time");
    timestamp.textContent = formatTimestamp(segment.start) + " - " + formatTimestamp(segment.end);
    const text = document.createElement("p");
    text.textContent = segment.text || "No speech";
    const confidence = document.createElement("span");
    confidence.textContent = segment.avg_logprob == null
      ? ""
      : "log p " + Number(segment.avg_logprob).toFixed(2);
    row.append(timestamp, text, confidence);
    target.segments.append(row);
  }
}

function resetTranscriptMonitor() {
  ui.transcriptSelection.textContent = "No selection";
  for (const system of ["whisper", "avec"]) {
    const target = transcriptUi(system);
    target.card.classList.remove("running", "complete", "skipped", "selected");
    target.status.textContent = "Waiting";
    target.segments.replaceChildren();
  }
  ui.whisperLiveTranscript.textContent = "Waiting for the audio first pass.";
  ui.avecLiveTranscript.textContent = "Waiting for the routing decision.";
}

function formatTimestamp(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? seconds.toFixed(2) + "s" : "--";
}

function handleEvent(event) {
  appendTrace(event);
  updateTranscriptEvent(event);
  if (event.stage === "media" && event.media) showCaptureMedia(event.media);
  if (event.features) {
    routeFeatures = { ...routeFeatures, ...event.features };
    renderTelemetry();
  }
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
  if (event.stage === "error") {
    ui.selectedSystem.textContent = "Route failed";
    ui.resultText.textContent = event.message;
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
  if (event.bytes != null) return formatBytes(event.bytes);
  if (event.media) return "Captured video, extracted WAV, and router video available";
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
  renderTranscriptResult("whisper", result.whisper_transcript);
  renderTranscriptResult("avec", result.avec_transcript);
  const selectedTranscript = String(result.selected_system || "").toLowerCase();
  ui.whisperTranscriptCard.classList.toggle("selected", selectedTranscript === "whisper");
  ui.avecTranscriptCard.classList.toggle("selected", selectedTranscript === "avec");
  ui.transcriptSelection.textContent = result.selected_system + " selected";
  if (result.features) {
    routeFeatures = { ...routeFeatures, ...result.features };
    renderTelemetry();
  }
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
  resetTranscriptMonitor();
  clearCaptureInspector();
  ui.trace.innerHTML = '<div class="trace-empty" id="trace-empty"><span class="trace-line"></span><p>Start an input to stream the inference trace.</p></div>';
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
ui.samplesMode.addEventListener("click", () => setInputMode("samples"));
ui.captureMode.addEventListener("click", () => setInputMode("capture"));
ui.enableCamera.addEventListener("click", enableCamera);
ui.record.addEventListener("click", recordCapture);
ui.captureAudio.addEventListener("play", animateCapturePlayback);
ui.captureAudio.addEventListener("pause", drawCaptureWaveform);
ui.captureAudio.addEventListener("seeked", drawCaptureWaveform);
for (const tab of ui.inspectorTabs) {
  tab.addEventListener("click", () => setInspectorView(tab.dataset.inspectorView));
}
window.addEventListener("resize", () => {
  drawWaveform();
  drawCaptureWaveform();
});
window.addEventListener("beforeunload", stopCamera);
loadState();
