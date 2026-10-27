'use strict';
const sessionToken = sessionStorage.getItem('token');
const loginURL = '/login?next=/test-voice';
if (!sessionToken) location.replace(loginURL);

const mic = document.getElementById('voice-mic');
const micLabel = document.getElementById('voice-mic-label');
const status = document.getElementById('voice-status');
const modelState = document.getElementById('model-state');
const history = document.getElementById('caption-history');
const partial = document.getElementById('caption-partial');
const empty = document.getElementById('caption-empty');
const copy = document.getElementById('voice-copy');
const clear = document.getElementById('voice-clear');
const timer = document.getElementById('voice-time');
const captions = document.querySelector('.voice-captions');
const phrases = new Map();
let recording = false, stopping = false, busy = false, ready = false;
let stream, context, source, worklet, sink, capture, clock, started;
let queue = [], run = 0, activeRequest, flushResolve;
let drainResolve = [];

function message(text, error = false) {
  status.textContent = text;
  status.classList.toggle('voice-error', error);
}
function authHeaders(extra = {}) {
  return { Authorization: 'Bearer ' + sessionToken, ...extra };
}
async function authenticatedFetch(path, options = {}) {
  const response = await fetch(path, { ...options, headers: authHeaders(options.headers) });
  if (response.status === 401) {
    releaseMicrophone();
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('username');
    location.replace(loginURL);
    throw new Error('برای ادامه دوباره وارد شوید.');
  }
  return response;
}
function render() {
  history.replaceChildren();
  const recent = [...phrases.values()].slice(-200);
  const finals = recent.filter((p) => p.final && p.text);
  for (const phrase of finals) {
    const p = document.createElement('p');
    p.textContent = phrase.text;
    history.appendChild(p);
  }
  partial.textContent = recent.filter((p) => !p.final && p.text).map((p) => p.text).join('\n');
  empty.hidden = Boolean(finals.length || partial.textContent);
  copy.disabled = empty.hidden === false;
  clear.disabled = recording || stopping || busy || empty.hidden === false;
  captions.scrollTop = captions.scrollHeight;
}
function textValue() {
  return [...phrases.values()].map((p) => p.text).filter(Boolean).join('\n');
}
function enqueue(packet, generation) {
  packet.generation = generation;
  const index = queue.findIndex((p) => p.phrase === packet.phrase && p.generation === generation);
  if (index >= 0) queue[index] = packet;
  else queue.push(packet);
  drain();
}
async function recognize(packet) {
  for (let retry = 0; retry < 4; retry++) {
    const controller = new AbortController();
    activeRequest = controller;
    const timeout = setTimeout(() => controller.abort(), 32000);
    let response;
    try {
      response = await authenticatedFetch('/api/live/transcribe', {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
        body: packet.bytes, signal: controller.signal,
      });
    } finally {
      clearTimeout(timeout);
      if (activeRequest === controller) activeRequest = null;
    }
    if (response.status === 429 && retry < 3) {
      await new Promise((resolve) => setTimeout(resolve, 500 * (retry + 1)));
      continue;
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'پردازش صدا انجام نشد.');
    return data;
  }
}
async function drain() {
  if (busy) return;
  busy = true;
  try {
    while (queue.length) {
      const packet = queue.shift();
      if (packet.generation !== run) continue;
      try {
        const data = await recognize(packet);
        if (packet.generation !== run) continue;
        const key = `${run}:${packet.phrase}`;
        phrases.set(key, { text: data.text || '', final: packet.final });
        while (phrases.size > 200) phrases.delete(phrases.keys().next().value);
        render();
        if (recording && !stopping) message('دارم می‌شنوم؛ به صحبت ادامه بده.');
      } catch (error) {
        message(error.name === 'AbortError' ? 'دریافت نوشته طول کشید؛ دوباره امتحان کن.' : error.message, true);
      }
    }
  } finally {
    busy = false;
    render();
    drainResolve.splice(0).forEach((resolve) => resolve());
  }
}
function waitForRecognition() {
  return busy || queue.length ? new Promise((resolve) => drainResolve.push(resolve)) : Promise.resolve();
}
function releaseMicrophone() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (source) { try { source.disconnect(); } catch (_) {} }
  if (worklet) { try { worklet.disconnect(); } catch (_) {} }
  if (sink) { try { sink.disconnect(); } catch (_) {} }
  if (context && context.state !== 'closed') context.close().catch(() => {});
  stream = source = worklet = sink = context = null;
  clearInterval(clock);
}
async function startRecording() {
  mic.disabled = true;
  message('در حال روشن کردن میکروفن…');
  run++;
  const generation = run;
  // Keep text from previous runs, but settle any unsuccessful partial as a finished line.
  for (const phrase of phrases.values()) phrase.final = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true,
    } });
    context = new AudioContext();
    await context.resume();
    await context.audioWorklet.addModule('/static/hossein-pcm-worklet.js?v=1');
    capture = new HosseinCapture(context.sampleRate, (packet) => enqueue(packet, generation));
    source = context.createMediaStreamSource(stream);
    worklet = new AudioWorkletNode(context, 'hossein-pcm', { outputChannelCount: [1] });
    sink = context.createGain();
    sink.gain.value = 0;
    worklet.port.onmessage = (event) => {
      if (event.data === 'flushed') { if (flushResolve) flushResolve(); }
      else if (generation === run && recording) capture.push(event.data);
    };
    source.connect(worklet).connect(sink).connect(context.destination);
    recording = true;
    stream.getAudioTracks()[0].addEventListener('ended', () => { if (recording && !stopping) stopRecording(); });
    mic.classList.add('recording');
    mic.setAttribute('aria-pressed', 'true');
    mic.setAttribute('aria-label', 'توقف میکروفن');
    micLabel.textContent = 'توقف میکروفن';
    started = performance.now();
    timer.textContent = '۰۰:۰۰';
    clock = setInterval(() => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      timer.textContent = `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`.replace(/\d/g, (d) => '۰۱۲۳۴۵۶۷۸۹'[d]);
    }, 500);
    message('دارم می‌شنوم؛ صحبت کن. برای پایان دوباره روی میکروفن بزن.');
    render();
  } catch (error) {
    releaseMicrophone();
    recording = false;
    const descriptions = {
      NotAllowedError: 'اجازهٔ میکروفن داده نشده است؛ از تنظیمات مرورگر اجازه بده و دوباره امتحان کن.',
      NotFoundError: 'میکروفنی پیدا نشد؛ اتصال میکروفن را بررسی کن.',
      NotReadableError: 'میکروفن در دسترس نیست؛ برنامهٔ دیگری را که از آن استفاده می‌کند ببند.',
    };
    message(descriptions[error.name] || 'روشن کردن میکروفن ممکن نشد؛ دوباره امتحان کن.', true);
  } finally { mic.disabled = false; }
}
async function stopRecording() {
  if (!recording || stopping) return;
  stopping = true;
  mic.disabled = true;
  micLabel.textContent = 'تکمیل نوشته…';
  message('میکروفن خاموش شد؛ آخرین بخش صحبت را می‌نویسم…');
  source.disconnect();
  await new Promise((resolve) => {
    const timeout = setTimeout(() => { flushResolve = null; resolve(); }, 600);
    flushResolve = () => { clearTimeout(timeout); flushResolve = null; resolve(); };
    worklet.port.postMessage('flush');
  });
  capture.flush();
  recording = false;
  releaseMicrophone();
  mic.classList.remove('recording');
  mic.setAttribute('aria-pressed', 'false');
  await waitForRecognition();
  stopping = false;
  mic.disabled = false;
  mic.setAttribute('aria-label', 'شروع میکروفن');
  micLabel.textContent = 'شروع دوباره';
  if (!status.classList.contains('voice-error')) message('تمام شد. می‌توانی نوشته را کپی کنی یا دوباره صحبت کنی.');
  render();
}
async function checkReady() {
  if (!sessionToken) return;
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    message('برای استفاده از میکروفن، این صفحه را با آدرس امن HTTPS باز کن.', true);
    modelState.textContent = 'اتصال امن لازم است';
    return;
  }
  if (!window.AudioWorkletNode) {
    message('این مرورگر از ضبط زنده پشتیبانی نمی‌کند؛ از نسخهٔ تازهٔ Chrome یا Edge استفاده کن.', true);
    return;
  }
  try {
    const response = await authenticatedFetch('/api/live/status');
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || 'مدل آماده نیست.');
    ready = true;
    mic.disabled = false;
    modelState.textContent = 'مدل صدای حسین آماده است';
    modelState.classList.add('ready');
    message('آماده‌ام؛ برای شروع روی میکروفن بزن.');
  } catch (error) {
    modelState.textContent = 'در انتظار اتصال به مدل';
    message(error.message, true);
    setTimeout(checkReady, 10000);
  }
}
mic.addEventListener('click', () => { if (ready && !stopping) recording ? stopRecording() : startRecording(); });
clear.addEventListener('click', () => { if (!recording && !stopping && !busy) { phrases.clear(); render(); message('نوشته پاک شد. برای شروع روی میکروفن بزن.'); } });
copy.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(textValue()); message('نوشته کپی شد.'); }
  catch (_) { message('کپی خودکار ممکن نشد؛ نوشته را انتخاب و کپی کن.', true); }
});
window.addEventListener('pagehide', () => {
  run++;
  recording = false;
  queue = [];
  if (activeRequest) activeRequest.abort();
  releaseMicrophone();
});
checkReady();
