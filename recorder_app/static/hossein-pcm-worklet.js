class HosseinPCM extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        if (this.used) {
          const tail = this.buffer.slice(0, this.used);
          this.port.postMessage(tail, [tail.buffer]);
          this.used = 0;
        }
        this.port.postMessage('flushed');
      }
    };
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    const source = channels[0];
    for (let i = 0; i < source.length; i++) {
      this.buffer[this.used++] = source[i];
      if (this.used === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(2048);
        this.used = 0;
      }
    }
    return true;
  }
}
registerProcessor('hossein-pcm', HosseinPCM);
