const token = sessionStorage.getItem("token");
if (!token) location.replace("/");

const whoEl = document.getElementById("who");
const listEl = document.getElementById("list");
const progressEl = document.getElementById("progress");
const scriptEl = document.getElementById("script");
const uttEl = document.getElementById("utt-id");
const statusEl = document.getElementById("status");
const statePill = document.getElementById("state-pill");
const reference = document.getElementById("reference");
const takeAudio = document.getElementById("take");
const recordBtn = document.getElementById("record");
const recordLabel = document.getElementById("record-label");
const discardBtn = document.getElementById("discard");
const discardModal = document.getElementById("discard-modal");
const discardText = document.getElementById("discard-text");
const discardOk = document.getElementById("discard-ok");
const discardCancel = document.getElementById("discard-cancel");
const confirmBtn = document.getElementById("confirm");
const filterInput = document.getElementById("filter");
const filterButtons = document.querySelectorAll(".seg-btn");

whoEl.textContent = sessionStorage.getItem("username") || "";

let items = [];
let index = 0;
let filterMode = "all";
let mediaRecorder = null;
let chunks = [];
let pendingBlob = null;
let takeUrl = null;
let refUrl = null;
let recording = false;

function authHeaders(extra) {
  return Object.assign({ Authorization: "Bearer " + token }, extra || {});
}
function setStatus(text) { statusEl.textContent = text || ""; }

function setState(kind, label) {
  statePill.className = "state-pill" + (kind ? " " + kind : "");
  statePill.textContent = label;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (res.status === 401) {
    sessionStorage.clear();
    location.replace("/");
    throw new Error("unauthorized");
  }
  return res;
}

function visibleItems() {
  const q = filterInput.value.trim();
  return items.filter((it) => {
    if (filterMode === "done" && !it.recorded) return false;
    if (filterMode === "missing" && it.recorded) return false;
    if (q && !String(it.id).includes(q)) return false;
    return true;
  });
}

function renderList() {
  const done = items.filter((it) => it.recorded).length;
  progressEl.querySelector("span").textContent = done + " / " + items.length;
  const shown = visibleItems();
  const current = items[index];
  listEl.replaceChildren();
  if (!shown.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "چیزی پیدا نشد.";
    listEl.appendChild(empty);
    return;
  }
  for (const it of shown) {
    const btn = document.createElement("button");
    btn.type = "button";
    const active = current && current.id === it.id;
    btn.className = "item" + (it.recorded ? " done" : "") + (active ? " active" : "");
    btn.setAttribute("role", "option");
    if (active) btn.setAttribute("aria-selected", "true");
    const id = document.createElement("span");
    id.className = "id";
    id.textContent = it.id;
    const dot = document.createElement("span");
    dot.className = "dot";
    btn.append(id, dot);
    btn.addEventListener("click", () => openId(it.id));
    listEl.appendChild(btn);
  }
}

function setFilter(mode) {
  filterMode = mode;
  filterButtons.forEach((btn) => {
    btn.classList.toggle("is-on", btn.dataset.mode === mode);
  });
  renderList();
}

function revokeTake() {
  if (takeUrl) URL.revokeObjectURL(takeUrl);
  takeUrl = null;
}

function revokeRef() {
  if (refUrl) URL.revokeObjectURL(refUrl);
  refUrl = null;
}

function setTakeButtons(enabled) {
  discardBtn.disabled = !enabled;
  confirmBtn.disabled = !enabled;
}

async function blobUrl(path) {
  const res = await api(path, { headers: authHeaders() });
  if (!res.ok) throw new Error("audio " + res.status);
  const blob = await res.blob();
  const type = (res.headers.get("Content-Type") || blob.type || "application/octet-stream").split(";")[0];
  return URL.createObjectURL(type ? new Blob([blob], { type }) : blob);
}

async function openIndex(i) {
  if (!items.length || recording) return;
  index = (i + items.length) % items.length;
  const it = items[index];
  sessionStorage.setItem("lastId", it.id);
  uttEl.textContent = it.id;
  scriptEl.textContent = it.text;
  pendingBlob = null;
  revokeTake();
  revokeRef();
  takeAudio.removeAttribute("src");
  takeAudio.load();
  reference.removeAttribute("src");
  reference.load();
  setTakeButtons(false);
  if (it.recorded) {
    setState("done", "ضبط شده");
    setStatus("این فایل قبلاً ضبط شده. می‌توانید بشنوید یا دوباره ضبط کنید.");
  } else {
    setState("pending", "ضبط نشده");
    setStatus("مرجع را بشنوید، جمله را بخوانید، بعد ضبط کنید.");
  }
  try {
    refUrl = await blobUrl("/api/reference?id=" + encodeURIComponent(it.id));
    reference.src = refUrl;
  } catch (err) {
    setStatus("فایل مرجع بارگذاری نشد.");
  }
  if (it.recorded) {
    try {
      takeUrl = await blobUrl("/api/take?id=" + encodeURIComponent(it.id));
      takeAudio.src = takeUrl;
      setTakeButtons(true);
    } catch (err) {
      setStatus("ضبط قبلی بارگذاری نشد.");
    }
  }
  renderList();
  const active = listEl.querySelector(".item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function openId(id) {
  const i = items.findIndex((it) => it.id === id);
  if (i >= 0) openIndex(i);
}

async function loadItems() {
  const res = await api("/api/items", { headers: authHeaders() });
  const data = await res.json();
  items = data.items;
  const last = sessionStorage.getItem("lastId");
  const start = items.findIndex((it) => it.id === last);
  const firstMissing = items.findIndex((it) => !it.recorded);
  await openIndex(start >= 0 ? start : (firstMissing >= 0 ? firstMissing : 0));
}

function setRecMode(live) {
  recordBtn.classList.toggle("live", live);
  recordBtn.setAttribute("aria-pressed", live ? "true" : "false");
  recordLabel.textContent = live ? "توقف" : "ضبط";
}

recordBtn.addEventListener("click", async () => {
  if (recording) {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    return;
  }
  if (!window.isSecureContext) {
    setStatus("ضبط فقط روی localhost یا https کار می‌کند. صفحه را از http://127.0.0.1:8765 باز کنید.");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
    setStatus("این مرورگر ضبط صدا ندارد. Chrome یا Firefox را امتحان کنید.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    const name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      setStatus("اجازه میکروفون داده نشد. قفل کنار آدرس را بزنید و میکروفون را مجاز کنید.");
    } else if (name === "NotFoundError") {
      setStatus("میکروفونی پیدا نشد.");
    } else {
      setStatus("میکروفون باز نشد: " + (name || err));
    }
    return;
  }
  chunks = [];
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  const mime = candidates.find((item) => MediaRecorder.isTypeSupported(item)) || "";
  try {
    mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    setStatus("ضبط در این مرورگر شروع نشد.");
    return;
  }
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.onerror = () => {
    stream.getTracks().forEach((track) => track.stop());
    recording = false;
    setRecMode(false);
    setStatus("ضبط قطع شد.");
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((track) => track.stop());
    recording = false;
    setRecMode(false);
    if (!chunks.length) {
      setStatus("صدایی ضبط نشد. دوباره تلاش کنید.");
      return;
    }
    pendingBlob = new Blob(chunks, { type: mediaRecorder.mimeType || mime || "audio/webm" });
    revokeTake();
    takeUrl = URL.createObjectURL(pendingBlob);
    takeAudio.src = takeUrl;
    setTakeButtons(true);
    setState("pending", "آماده تأیید");
    setStatus("بشنوید. اگر درست بود تأیید کنید، وگرنه پاک کنید.");
  };
  try {
    mediaRecorder.start();
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    setStatus("ضبط شروع نشد.");
    return;
  }
  recording = true;
  setRecMode(true);
  setState("live", "در حال ضبط");
  setStatus("جمله را بخوانید، بعد دوباره همین دکمه را برای توقف بزنید.");
});

function openDiscardModal() {
  const it = items[index];
  if (!it) return;
  discardText.textContent = "صدای ضبط‌شدهٔ فایل " + it.id + " حذف می‌شود و برنمی‌گردد.";
  discardModal.hidden = false;
  discardOk.focus();
}

function closeDiscardModal() {
  discardModal.hidden = true;
  if (!discardBtn.disabled) discardBtn.focus();
}

discardBtn.addEventListener("click", () => {
  if (recording || discardBtn.disabled) return;
  openDiscardModal();
});

discardCancel.addEventListener("click", closeDiscardModal);
discardModal.addEventListener("click", (event) => {
  if (event.target === discardModal) closeDiscardModal();
});

discardOk.addEventListener("click", async () => {
  if (recording) return;
  const it = items[index];
  discardOk.disabled = true;
  pendingBlob = null;
  revokeTake();
  takeAudio.removeAttribute("src");
  takeAudio.load();
  const res = await api("/api/take?id=" + encodeURIComponent(it.id), { method: "DELETE", headers: authHeaders() });
  discardOk.disabled = false;
  closeDiscardModal();
  if (!res.ok) {
    setStatus("پاک نشد.");
    return;
  }
  it.recorded = false;
  setTakeButtons(false);
  setState("pending", "ضبط نشده");
  setStatus("ضبط پاک شد. دوباره ضبط کنید.");
  renderList();
});

confirmBtn.addEventListener("click", async () => {
  if (recording) return;
  const it = items[index];
  if (pendingBlob) {
    setStatus("در حال ذخیره…");
    confirmBtn.disabled = true;
    const res = await api("/api/take?id=" + encodeURIComponent(it.id), {
      method: "POST",
      headers: authHeaders({ "Content-Type": pendingBlob.type || "audio/webm" }),
      body: pendingBlob,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(err.error || "ذخیره نشد");
      confirmBtn.disabled = false;
      return;
    }
    pendingBlob = null;
    it.recorded = true;
  }
  const nextMissing = items.findIndex((item, i) => i > index && !item.recorded);
  await openIndex(nextMissing >= 0 ? nextMissing : index + 1);
});

document.getElementById("prev").addEventListener("click", () => openIndex(index - 1));
document.getElementById("next").addEventListener("click", () => openIndex(index + 1));
document.getElementById("logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST", headers: authHeaders() });
  sessionStorage.clear();
  location.replace("/");
});
filterInput.addEventListener("input", renderList);
filterButtons.forEach((btn) => {
  btn.addEventListener("click", () => setFilter(btn.dataset.mode));
});

document.addEventListener("keydown", (event) => {
  if (!discardModal.hidden) {
    if (event.key === "Escape") { event.preventDefault(); closeDiscardModal(); }
    else if (event.key === "Enter") { event.preventDefault(); discardOk.click(); }
    return;
  }
  const tag = (event.target && event.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); openIndex(index - 1); }
  else if (event.key === "ArrowRight") { event.preventDefault(); openIndex(index + 1); }
  else if (event.code === "Space") {
    event.preventDefault();
    recordBtn.click();
  }
});

loadItems().catch((err) => setStatus(String(err)));
