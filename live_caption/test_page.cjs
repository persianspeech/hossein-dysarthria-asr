/* Exercise the actual page controller with a virtual microphone and delayed ASR replies. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const tick = () => new Promise((resolve) => setImmediate(resolve));
class Element {
  constructor() {
    this.children = []; this.events = {}; this.hidden = false; this.textContent = '';
    const classes = new Set();
    this.classList = { add: (c) => classes.add(c), remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c), toggle: (c, on) => on ? classes.add(c) : classes.delete(c) };
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  replaceChildren() { this.children = []; }
  appendChild(child) { this.children.push(child); }
  setAttribute(name, value) { this[name] = value; }
}
async function main() {
  const elements = new Map();
  const get = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const pending = [], tracks = [], contexts = [];
  let currentWorklet, clipboard, denyPermission = false;
  class AudioContext {
    constructor() { this.sampleRate = 48000; this.state = 'running'; contexts.push(this); }
    async resume() {}
    async close() { this.state = 'closed'; }
    audioWorklet = { addModule: async () => {} };
    createMediaStreamSource() { return { connect: () => currentWorklet, disconnect: () => {} }; }
    createGain() { return { gain: { value: 1 }, connect: () => {}, disconnect: () => {} }; }
  }
  class Worklet {
    constructor() { currentWorklet = this; }
    port = { onmessage: null, postMessage: () => queueMicrotask(() => this.port.onmessage({ data: 'flushed' })) };
    connect(sink) { return sink; }
    disconnect() {}
  }
  const environment = {
    console, URLSearchParams, AudioContext, AudioWorkletNode: Worklet, AbortController,
    performance, setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {},
    sessionStorage: { getItem: () => 'test-session', removeItem: () => {} },
    location: { replace: () => assert.fail('Unexpected login redirect') },
    document: { getElementById: get, querySelector: get, createElement: () => new Element() },
    navigator: { mediaDevices: { getUserMedia: async () => {
      if (denyPermission) throw Object.assign(new Error('denied'), { name: 'NotAllowedError' });
      const track = { stopped: false, stop() { this.stopped = true; }, addEventListener() {} };
      tracks.push(track);
      return { getTracks: () => [track], getAudioTracks: () => [track] };
    } }, clipboard: { writeText: async (text) => { clipboard = text; } } },
    fetch: async (path, options) => {
      if (path.endsWith('/status')) return { ok: true, status: 200, json: async () => ({ ready: true }) };
      return new Promise((resolve) => pending.push({ options, resolve }));
    },
  };
  environment.window = { isSecureContext: true, AudioWorkletNode: Worklet, addEventListener() {} };
  const context = vm.createContext(environment);
  vm.runInContext(fs.readFileSync('static/hossein-capture.js', 'utf8'), context);
  vm.runInContext(fs.readFileSync('static/hossein-test.js', 'utf8'), context);
  await tick();
  assert.equal(get('voice-mic').disabled, false);
  await vm.runInContext('startRecording()', context);
  currentWorklet.port.onmessage({ data: new Float32Array(48000 * 2).fill(0.1) });
  assert.equal(pending.length, 1);
  currentWorklet.port.onmessage({ data: new Float32Array(48000 * 2).fill(0.1) });
  const stopped = vm.runInContext('stopRecording()', context);
  await tick();
  assert.equal(tracks[0].stopped, true, 'stop releases physical capture before waiting for inference');
  assert.equal(get('voice-mic').disabled, true, 'cannot restart while final text is pending');
  const response = (text) => ({ ok: true, status: 200, json: async () => ({ text }) });
  pending[0].resolve(response('متن موقت'));
  await tick();
  assert.equal(pending.length, 2, 'queued partial is replaced by latest final');
  assert.equal(pending[1].options.body.byteLength, 16000 * 4 * 2, 'final request contains all four seconds');
  pending[1].resolve(response('صدای حسین'));
  await stopped;
  assert.equal(vm.runInContext('textValue()', context), 'صدای حسین', 'partial is revised, never duplicated');
  assert.equal(get('voice-mic').disabled, false);
  assert.equal(get('voice-mic')['aria-pressed'], 'false');
  assert.equal(contexts[0].state, 'closed');
  await get('voice-copy').events.click();
  assert.equal(clipboard, 'صدای حسین');
  get('voice-clear').events.click();
  assert.equal(vm.runInContext('textValue()', context), '');
  denyPermission = true;
  await vm.runInContext('startRecording()', context);
  assert.equal(get('voice-mic').disabled, false);
  assert(get('voice-status').textContent.includes('اجازه'));
  assert.equal(vm.runInContext('recording', context), false);
  console.log('PASS: microphone lifecycle, pending final, bounded partial queue, subtitle revision, copy/clear, permission denial');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
