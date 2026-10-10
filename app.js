const MENU = window.MENU;
const $ = id => document.getElementById(id);
const out = $("out"), ctx = out.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const idleTurn = () => new Promise(resolve => {
  if (window.requestIdleCallback) requestIdleCallback(resolve, { timeout: 50 });
  else setTimeout(resolve, 8);
});
const SURFACE = "#fffdf8";
// Thirty square previews use at most ~60 MiB of bitmap storage. Exports keep
// their original high resolution and do not occupy the preview cache.
const PREVIEW_EDGE = 720;
const LEVELS = Array.from({ length: 30 }, (_, index) => (index + 1) * 10);
const SPRITE_SIZE = 72;
const ATLAS_COLS = 8;
const atlas = document.createElement("canvas");
const palette = [];
const preparedFrames = new Map();
const frameWaiters = new Set();
const transitionCanvas = document.createElement("canvas");
const transitionCtx = transitionCanvas.getContext("2d");

let src = null;
let sourceRevision = 0;
let prepGeneration = 0;
let shuffleN = 0;
let hasFrame = false;
let fadeId = 0;
let displayScheduled = false;
let uploadRevision = 0;
let samplePromise;
let entering = false;
let renderer;
let worker = null;
let workerPromise;
let workerFailed = false;
let workerURL = null;

// Emoji fonts stay on the main thread; only their finished sprite atlas is
// transferred to the worker, so font support is identical in both renderers.
function buildPalette() {
  const foods = Object.keys(MENU);
  atlas.width = ATLAS_COLS * SPRITE_SIZE;
  atlas.height = Math.ceil(foods.length / ATLAS_COLS) * SPRITE_SIZE;
  const g = atlas.getContext("2d", { willReadFrequently: true });
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.font = '60px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
  foods.forEach((e, index) => {
    const x = (index % ATLAS_COLS) * SPRITE_SIZE;
    const y = Math.floor(index / ATLAS_COLS) * SPRITE_SIZE;
    g.fillText(e, x + SPRITE_SIZE / 2, y + SPRITE_SIZE / 2 + 3);
    const d = g.getImageData(x, y, SPRITE_SIZE, SPRITE_SIZE).data;
    let r = 0, green = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] > 160) {
        r += d[i]; green += d[i + 1]; b += d[i + 2]; n++;
      }
    }
    if (n) palette.push({ id: palette.length, e, x, y, r: r / n, g: green / n, b: b / n });
  });
}

function settings(cols = +$("cols").value) {
  return { source: src, sourceRevision, cols, chaos: +$("chaos").value,
    tint: $("tint").checked, shuffle: shuffleN };
}

function gridFor(source, cols) { return renderer.gridFor(source, cols); }
function buildFrame(order, edge) {
  return renderer.render({ ...order, edge }, { yieldTask: nextFrame });
}

function closeFrame(frame) {
  if (typeof frame.canvas.close === "function") frame.canvas.close();
  else frame.canvas.width = frame.canvas.height = 1;
}

function clearPreparedFrames() {
  for (const waiter of frameWaiters) {
    clearTimeout(waiter.timeout);
    waiter.reject(new Error("Recipe changed before the export was prepared"));
  }
  frameWaiters.clear();
  for (const frame of preparedFrames.values()) closeFrame(frame);
  preparedFrames.clear();
}

function updatePreparationProgress() {
  const ready = preparedFrames.size;
  $("prepProgress").value = ready;
  $("prepText").textContent = ready === LEVELS.length ? "จานพร้อมครบ 30 ระดับ · ปรับทีละ 10" : `เชฟเตรียมจาน ${ready}/30 ระดับ`;
}

function acceptFrame(frame, generation) {
  if (generation !== prepGeneration) { closeFrame(frame); return; }
  const previous = preparedFrames.get(frame.cols);
  if (previous) closeFrame(previous);
  preparedFrames.set(frame.cols, frame);
  for (const waiter of frameWaiters) {
    if (waiter.cols === frame.cols && waiter.generation === generation) {
      clearTimeout(waiter.timeout);
      frameWaiters.delete(waiter);
      waiter.resolve(frame);
    }
  }
  updatePreparationProgress();
  if (frame.cols === +$("cols").value) showSelectedFrame();
}

function waitForPreparedFrame(cols, generation) {
  if (preparedFrames.has(cols)) return Promise.resolve(preparedFrames.get(cols));
  return new Promise((resolve, reject) => {
    const waiter = { cols, generation, resolve, reject };
    waiter.timeout = setTimeout(() => {
      frameWaiters.delete(waiter);
      reject(new Error("Export preparation timed out"));
    }, 20000);
    frameWaiters.add(waiter);
  });
}

function failWorker(error) {
  if (workerFailed) return;
  workerFailed = true;
  if (worker) worker.terminate();
  worker = null;
  if (workerURL) { URL.revokeObjectURL(workerURL); workerURL = null; }
  console.warn("Using idle-time kitchen preparation", error);
  beginPreparation();
}

async function getWorker() {
  if (workerFailed || !window.Worker || !window.OffscreenCanvas || !window.createImageBitmap) return null;
  if (!workerPromise) {
    workerPromise = (async () => {
      let bitmap;
      try {
        bitmap = await createImageBitmap(atlas);
        const code = `(${cookingWorker.toString()})(${createFoodRenderer.toString()});`;
        workerURL = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
        worker = new Worker(workerURL, { name: "let-him-cook-prep" });
        worker.onmessage = ({ data }) => {
          if (data.type === "ready") {
            if (workerURL) { URL.revokeObjectURL(workerURL); workerURL = null; }
          } else if (data.type === "frame") acceptFrame(data.frame, data.generation);
          else if (data.type === "error" && data.generation === prepGeneration) failWorker(data.message);
        };
        worker.onerror = event => { event.preventDefault(); failWorker(event.message); };
        worker.postMessage({ type: "init", palette, atlas: bitmap }, [bitmap]);
        return worker;
      } catch (error) {
        if (bitmap) bitmap.close();
        if (worker) worker.terminate();
        worker = null;
        workerFailed = true;
        if (workerURL) { URL.revokeObjectURL(workerURL); workerURL = null; }
        return null;
      }
    })();
  }
  return workerPromise;
}

async function prepareOnIdle(order, generation) {
  $("drop").dataset.engine = "idle";
  while (generation === prepGeneration && preparedFrames.size < LEVELS.length) {
    const selected = +$("cols").value;
    const cols = !preparedFrames.has(selected) ? selected : LEVELS
      .filter(level => !preparedFrames.has(level))
      .sort((a, b) => Math.abs(a - selected) - Math.abs(b - selected))[0];
    await idleTurn();
    const shouldCancel = () => generation !== prepGeneration ||
      (cols !== +$("cols").value && !preparedFrames.has(+$("cols").value));
    const frame = await renderer.render({ ...order, cols, edge: PREVIEW_EDGE }, { yieldTask: idleTurn, shouldCancel });
    if (frame) acceptFrame(frame, generation);
  }
}

async function beginPreparation() {
  if (!src || !palette.length) return;
  const generation = ++prepGeneration;
  const order = settings();
  cancelAnimationFrame(fadeId);
  clearPreparedFrames();
  updatePreparationProgress();
  $("prepProgress").dataset.generation = generation;
  $("drop").setAttribute("aria-busy", "true");
  $("plateState").textContent = `เชฟกำลังเตรียมระดับ ${order.cols}…`;
  if (worker) worker.postMessage({ type: "cancel", generation });
  try {
    const engine = await getWorker();
    if (generation !== prepGeneration) return;
    if (engine && !workerFailed) {
      engine.postMessage({ type: "cancel", generation });
      const source = await createImageBitmap(order.source);
      if (generation !== prepGeneration) { source.close(); return; }
      $("drop").dataset.engine = "worker";
      engine.postMessage({ ...order, type: "prepare", generation, source,
        selected: +$("cols").value, levels: LEVELS, edge: PREVIEW_EDGE }, [source]);
    } else await prepareOnIdle(order, generation);
  } catch (error) {
    if (generation !== prepGeneration) return;
    if (!workerFailed && worker) failWorker(error);
    else {
      console.error("Kitchen preparation failed", error);
      $("drop").setAttribute("aria-busy", "false");
      $("prepText").textContent = "เชฟเตรียมจานไม่สำเร็จ ลองเลือกรูปใหม่";
      $("flash").textContent = "เชฟทำจานหล่น ลองเลือกรูปใหม่อีกครั้ง";
    }
  }
}

function present(frame) {
  cancelAnimationFrame(fadeId);
  const image = frame.canvas;
  const sizeChanged = out.width !== image.width || out.height !== image.height;
  if (!hasFrame || sizeChanged || reducedMotion.matches || $("kitchen").hidden) {
    if (sizeChanged) { out.width = image.width; out.height = image.height; }
    ctx.globalAlpha = 1;
    ctx.drawImage(image, 0, 0);
    hasFrame = true;
    return;
  }
  // Reuse a single transition surface; dragging never allocates a new canvas.
  if (transitionCanvas.width !== out.width || transitionCanvas.height !== out.height) {
    transitionCanvas.width = out.width; transitionCanvas.height = out.height;
  }
  transitionCtx.drawImage(out, 0, 0);
  const start = performance.now();
  const blend = now => {
    const t = Math.min(1, (now - start) / 70);
    ctx.globalAlpha = 1;
    ctx.drawImage(transitionCanvas, 0, 0);
    ctx.globalAlpha = t * t * (3 - 2 * t);
    ctx.drawImage(image, 0, 0);
    ctx.globalAlpha = 1;
    if (t < 1) fadeId = requestAnimationFrame(blend);
  };
  fadeId = requestAnimationFrame(blend);
}

function showSelectedFrame() {
  if (displayScheduled) return;
  displayScheduled = true;
  requestAnimationFrame(() => {
    displayScheduled = false;
    const frame = preparedFrames.get(+$("cols").value);
    if (!frame) return;
    present(frame);
    out.dataset.cols = frame.cols;
    out.dataset.generation = prepGeneration;
    out.dataset.sourceRevision = sourceRevision;
    $("plateState").textContent = `${frame.total.toLocaleString("th-TH")} ชิ้น · เชฟบอกว่าพอดี`;
    $("drop").setAttribute("aria-busy", "false");
    receipt(frame.counts, frame.total);
  });
}

function selectResolution() {
  const cols = +$("cols").value;
  $("colsOut").textContent = cols;
  if (worker && !workerFailed) worker.postMessage({ type: "select", generation: prepGeneration, cols });
  if (preparedFrames.has(cols)) showSelectedFrame();
  else {
    $("drop").setAttribute("aria-busy", "true");
    $("plateState").textContent = `เชฟกำลังเตรียมระดับ ${cols}…`;
  }
}

function receipt(counts, total) {
  const items = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  let price = 0, kcal = 0;
  for (const [e, n] of items) { price += MENU[e][1] * n; kcal += MENU[e][2] * n; }
  const now = new Date();
  const fmt = n => n.toLocaleString("th-TH");
  $("receipt").innerHTML = `
    <h2>let him cook.</h2>
    <div class="sub">ร้านอาหารที่ไม่ควรมีอยู่จริง</div>
    <div class="receipt-label">ใบเสร็จแห่งความสิ้นหวัง</div>
    <div class="sub">โต๊ะ ${String(total % 97).padStart(2, "0")} · ${now.toLocaleDateString("th-TH")} ${now.toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit" })}</div>
    <hr>
    ${items.slice(0, 8).map(([e, n]) => `<div class="row"><span>${e}</span><span>${MENU[e][0]} <span class="q">×${fmt(n)}</span></span><span>${fmt(MENU[e][1] * n)}</span></div>`).join("")}
    ${items.length > 8 ? `<div class="row"><span>…</span><span>อื่น ๆ อีก ${items.length - 8} เมนู</span><span></span></div>` : ""}
    <hr>
    <div class="tot"><span>รวม ${fmt(total)} ชิ้น</span><span>฿${fmt(price)}</span></div>
    <div class="tot"><span>พลังงาน</span><span>${fmt(kcal)} kcal</span></div>
    <div class="tot useless"><span>ประโยชน์ที่ได้รับ</span><span>0.00</span></div>
    <div class="foot">ขอบคุณที่ใช้หน้าเป็นวัตถุดิบ<br>ไม่รับคืน ไม่รับเปลี่ยน ไม่รับประทาน</div>
    <div class="barcode" aria-hidden="true"></div>`;
}

function canvasFromImage(image) {
  const c = document.createElement("canvas");
  const scale = Math.min(1, 1000 / Math.max(image.naturalWidth, image.naturalHeight));
  c.width = Math.max(1, Math.round(image.naturalWidth * scale));
  c.height = Math.max(1, Math.round(image.naturalHeight * scale));
  const g = c.getContext("2d");
  g.fillStyle = SURFACE;
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(image, 0, 0, c.width, c.height);
  return c;
}

function setSource(source) {
  src = source;
  sourceRevision++;
  hasFrame = false;
  cancelAnimationFrame(fadeId);
  beginPreparation();
}

function sampleFace() {
  // Offline fallback if the local meme asset cannot be decoded.
  const c = document.createElement("canvas"); c.width = c.height = 400;
  const g = c.getContext("2d");
  g.fillStyle = "#dce4cf"; g.fillRect(0, 0, 400, 400);
  g.fillStyle = "#3a2414"; g.beginPath(); g.ellipse(200, 175, 135, 145, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#e9b48a"; g.beginPath(); g.ellipse(200, 220, 115, 135, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#fffdf8"; g.fillRect(65, 340, 270, 60);
  g.beginPath(); g.ellipse(200, 90, 140, 60, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#28251f";
  [150, 250].forEach(x => { g.beginPath(); g.arc(x, 210, 10, 0, Math.PI * 2); g.fill(); });
  g.strokeStyle = "#8e2b2b"; g.lineWidth = 8; g.lineCap = "round";
  g.beginPath(); g.arc(200, 260, 40, .15 * Math.PI, .85 * Math.PI); g.stroke();
  return c;
}

function getSample() {
  if (!samplePromise) {
    // File URLs have opaque origins in some browsers. The opening meme still
    // displays normally; a drawn chef keeps the canvas exportable offline.
    if (location.protocol === "file:") return Promise.resolve(sampleFace());
    samplePromise = new Promise(resolve => {
      const image = new Image();
      image.onload = () => resolve(canvasFromImage(image));
      image.onerror = () => resolve(sampleFace());
      image.src = "assets/shef.png";
    });
  }
  return samplePromise;
}

async function useSample() {
  const ticket = ++uploadRevision;
  const source = await getSample();
  if (ticket !== uploadRevision) return;
  setSource(source);
  $("hint").textContent = "จานทดลองของเชฟ · ลากรูปคุณมาวางได้เลย";
  $("flash").textContent = "เชฟยอมใช้หน้าตัวเองเป็นวัตถุดิบแล้ว";
}

function enterKitchen() {
  if (entering || $("welcome").hidden) return;
  entering = true;
  $("welcome").classList.remove("returning");
  $("welcome").classList.add("leaving");
  const finish = () => {
    $("welcome").hidden = true;
    $("welcome").classList.remove("leaving");
    $("kitchen").hidden = false;
    $("kitchen").classList.add("arriving");
    window.scrollTo(0, 0);
    $("back").focus({ preventScroll: true });
    entering = false;
  };
  if (reducedMotion.matches) finish();
  else setTimeout(finish, 240);
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    $("flash").textContent = "อันนี้กินไม่ได้ ลองไฟล์รูป .jpg หรือ .png";
    return;
  }
  const ticket = ++uploadRevision;
  const image = new Image();
  const url = URL.createObjectURL(file);
  image.onload = () => {
    try {
      if (ticket !== uploadRevision) return;
      setSource(canvasFromImage(image));
      $("hint").textContent = "ลากรูปใหม่มาวางเพื่อสั่งจานถัดไป";
      $("flash").textContent = "เสิร์ฟแล้ว! หน้าคุณอร่อยมาก (เชฟบอกเอง)";
    } finally { URL.revokeObjectURL(url); }
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    if (ticket === uploadRevision) $("flash").textContent = "เชฟเปิดรูปนี้ไม่ได้ ลองบันทึกเป็น .jpg แล้วสั่งใหม่";
  };
  image.src = url;
  enterKitchen();
}

$("enter").addEventListener("click", enterKitchen);
$("back").addEventListener("click", () => {
  $("kitchen").hidden = true;
  $("kitchen").classList.remove("arriving");
  $("welcome").hidden = false;
  $("welcome").classList.add("returning");
  window.scrollTo(0, 0);
  $("enter").focus({ preventScroll: true });
});
$("file").addEventListener("change", e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
  e.target.value = ""; // Allow picking the same file again.
});
const drop = $("drop");
let dragDepth = 0;
["dragenter", "dragover"].forEach(type => drop.addEventListener(type, e => {
  e.preventDefault();
  if (type === "dragenter") dragDepth++;
  drop.classList.add("over");
}));
drop.addEventListener("dragleave", e => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) drop.classList.remove("over");
});
drop.addEventListener("drop", e => {
  e.preventDefault(); dragDepth = 0; drop.classList.remove("over");
  loadFile(e.dataTransfer.files[0]);
});
document.addEventListener("paste", e => {
  const file = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/"));
  if (file) { e.preventDefault(); loadFile(file); }
});
$("cols").addEventListener("input", selectResolution);
$("chaos").addEventListener("input", e => { $("chaosOut").textContent = e.target.value; beginPreparation(); });
$("tint").addEventListener("change", beginPreparation);
$("shuffle").addEventListener("click", () => {
  shuffleN++;
  if (+$("chaos").value === 1) { $("chaos").value = 3; $("chaosOut").textContent = 3; }
  beginPreparation();
  $("flash").textContent = "เชฟเปลี่ยนใจ จัดจานใหม่ให้แล้ว ไม่มีอะไรเหมือนเดิมแม้แต่สูตร";
});
$("sample").addEventListener("click", useSample);
$("save").addEventListener("click", async () => {
  if (!src || !palette.length) return;
  const button = $("save");
  const label = button.innerHTML;
  button.disabled = true; button.textContent = "กำลังแพ็กใส่กล่อง…";
  try {
    // Export the exact requested settings, never a partially blended preview.
    const order = settings();
    const prepared = await waitForPreparedFrame(order.cols, prepGeneration);
    const { rows } = gridFor(order.source, order.cols);
    const edge = Math.min(4096, Math.max(1200, Math.max(order.cols, rows) * 12));
    const frame = await buildFrame({ ...order, model: prepared.model }, edge);
    const blob = await new Promise(resolve => frame.canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("PNG encoding failed");
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = `let-him-cook-${Date.now()}.png`;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    $("flash").textContent = "แพ็กกลับบ้านแล้ว เปิดดูได้ แต่ไม่แนะนำให้ชิม";
  } catch (error) {
    console.error("Kitchen export failed", error);
    $("flash").textContent = "กล่องข้าวรั่ว บันทึกไม่สำเร็จ ลองใหม่อีกครั้ง";
  } finally {
    button.disabled = false; button.innerHTML = label;
  }
});

buildPalette();
renderer = createFoodRenderer((width, height) => {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  return canvas;
}, palette, atlas);
useSample().then(() => { if (uploadRevision === 1) $("flash").textContent = ""; });
