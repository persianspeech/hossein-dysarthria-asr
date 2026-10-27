/* Stateful resampling and phrase segmentation, shared by the page and Node tests. */
class HosseinCapture {
  constructor(inputRate, send) {
    this.ratio = inputRate / 16000;
    this.send = send;
    this.resampleSum = 0;
    this.resampleWeight = 0;
    this.position = 0;
    this.samples = [];
    this.phrase = 0;
    this.silence = 0;
    this.voiced = false;
    this.lastPartial = 0;
  }
  push(input) {
    // Integrate each source sample over output-sample boundaries (including 44.1 kHz).
    for (const value of input) {
      let remaining = 1;
      while (remaining > 1e-9) {
        const weight = Math.min(remaining, this.ratio - this.resampleWeight);
        this.resampleSum += value * weight;
        this.resampleWeight += weight;
        remaining -= weight;
        if (this.resampleWeight >= this.ratio - 1e-9) {
          const sample = this.resampleSum / this.ratio;
          this.resampleWeight = 0;
          this.resampleSum = 0;
          this.samples.push(Math.max(-32768, Math.min(32767, Math.round(sample * 32768))));
          this.position++;
          // RMS over 100 ms windows. All audio is retained; this only finds phrase ends.
          this.energy = (this.energy || 0) + sample * sample;
          if (this.position % 1600 === 0) {
            const rms = Math.sqrt(this.energy / 1600);
            this.energy = 0;
            if (rms > 0.004) { this.voiced = true; this.silence = 0; }
            else this.silence += 1600;
            if (this.samples.length >= 30 * 16000 || (this.voiced && this.silence >= 2.5 * 16000)) {
              this.flush();
            } else if (this.samples.length - this.lastPartial >= 2 * 16000) {
              this.emit(false);
            }
          }
        }
      }
    }
  }
  emit(final) {
    if (this.samples.length < 800) return;
    const bytes = new ArrayBuffer(this.samples.length * 2);
    const view = new DataView(bytes);
    this.samples.forEach((v, i) => view.setInt16(i * 2, v, true));
    this.send({ phrase: this.phrase, final, bytes });
    this.lastPartial = this.samples.length;
  }
  flush() {
    this.emit(true);
    this.samples = [];
    this.phrase++;
    this.silence = 0;
    this.voiced = false;
    this.lastPartial = 0;
  }
}
if (typeof module !== 'undefined') module.exports = { HosseinCapture };
