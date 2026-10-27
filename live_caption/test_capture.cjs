const assert = require('node:assert/strict');
const { HosseinCapture } = require('./static/hossein-capture.js');
const fs = require('node:fs');
const vm = require('node:vm');

for (const rate of [16000, 44100, 48000]) {
  const packets = [];
  const capture = new HosseinCapture(rate, (p) => packets.push(p));
  const wave = Float32Array.from({ length: rate * 3 }, (_, i) => 0.2 * Math.sin(2 * Math.PI * 220 * i / rate));
  for (let offset = 0; offset < wave.length; offset += 2048) capture.push(wave.subarray(offset, offset + 2048));
  capture.flush();
  assert.equal(packets.length, 2, 'one 2-second partial and one final');
  assert.equal(packets[0].bytes.byteLength, 32000 * 2);
  assert.equal(packets[1].bytes.byteLength, 48000 * 2, 'continuous resampling must not drop block boundaries');
  assert.equal(packets[1].final, true);
  const view = new DataView(packets[1].bytes);
  let max = 0;
  for (let i = 0; i < view.byteLength; i += 2) max = Math.max(max, Math.abs(view.getInt16(i, true)));
  assert(max > 6400 && max < 6600, 'PCM gain and little endian encoding');
}
{
  const packets = [];
  const capture = new HosseinCapture(16000, (p) => packets.push(p));
  capture.push(new Float32Array(16000).fill(0.1));
  capture.push(new Float32Array(40000));
  assert.equal(packets.at(-1).final, true, 'a 2.5-second pause finalizes a phrase');
  assert.equal(packets.at(-1).bytes.byteLength, 56000 * 2, 'quiet audio is preserved');
  capture.push(new Float32Array(16000).fill(0.0005));
  capture.flush();
  assert.equal(packets.at(-1).phrase, 1);
  assert.equal(packets.at(-1).bytes.byteLength, 16000 * 2, 'even very soft speech is retained');
}
{
  const packets = [];
  const capture = new HosseinCapture(16000, (p) => packets.push(p));
  capture.push(new Float32Array(16000 * 61).fill(0.1));
  capture.flush();
  const finals = packets.filter((p) => p.final);
  assert.deepEqual(finals.map((p) => p.bytes.byteLength / 32000), [30, 30, 1]);
  assert.equal(finals.reduce((n, p) => n + p.bytes.byteLength, 0), 61 * 32000, 'no audio is lost at phrase limits');
  assert(finals.every((p) => p.bytes.byteLength <= 40 * 32000));
}
{
  // Stopping between 2048-sample worklet frames must retain the final syllable.
  let Processor;
  const messages = [];
  const sandbox = { AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (m) => messages.push(m) }; } },
    registerProcessor: (_, impl) => { Processor = impl; } };
  vm.runInNewContext(fs.readFileSync(require.resolve('./static/hossein-pcm-worklet.js'), 'utf8'), sandbox);
  const processor = new Processor();
  processor.process([[new Float32Array(2048).fill(0.1)]]);
  processor.process([[new Float32Array(127).fill(0.2)]]);
  processor.port.onmessage({ data: 'flush' });
  assert.equal(messages[0].length, 2048);
  assert.equal(messages[1].length, 127);
  assert.equal(messages[2], 'flushed');
}
console.log('PASS: streaming resampling, PCM encoding, quiet speech, pause finalization, long-speech boundaries');
