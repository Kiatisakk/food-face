const MENU = window.MENU;
const $ = id => document.getElementById(id);
const out = $("out"), ctx = out.getContext("2d");

let src = null;     // current source image (canvas)
let palette = [];   // [{ e, r, g, b }] average color of each food emoji
let shuffleN = 0;
let seed = 1;
const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const rankCache = new Map(); // quantized color -> closest 6 foods
let renderTimer = 0;
const renderSoon = () => { clearTimeout(renderTimer); renderTimer = setTimeout(render, 120); };

// Measure each emoji's average color by drawing it once
function buildPalette() {
  const c = document.createElement("canvas"); c.width = c.height = 48;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.textAlign = "center"; g.textBaseline = "middle"; g.font = "38px serif";
  for (const e of Object.keys(MENU)) {
    g.clearRect(0, 0, 48, 48); g.fillText(e, 24, 26);
    const d = g.getImageData(0, 0, 48, 48).data;
    let r = 0, gg = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i+3] > 160) { r += d[i]; gg += d[i+1]; b += d[i+2]; n++; }
    if (n > 40) palette.push({ e, r: r/n, g: gg/n, b: b/n });
  }
}

// "redmean" approximation of perceived color difference
function dist(p, r, g, b) {
  const rm = (p.r + r) / 2, dr = p.r - r, dg = p.g - g, db = p.b - b;
  return (2 + rm/256) * dr*dr + 4 * dg*dg + (2 + (255-rm)/256) * db*db;
}

function sampleFace() {
  const c = document.createElement("canvas"); c.width = c.height = 400;
  const g = c.getContext("2d");
  const bg = g.createLinearGradient(0, 0, 0, 400); bg.addColorStop(0, "#7ec4e8"); bg.addColorStop(1, "#cfe9a8");
  g.fillStyle = bg; g.fillRect(0, 0, 400, 400);
  g.fillStyle = "#3a2414"; g.beginPath(); g.ellipse(200, 170, 140, 135, 0, Math.PI, 0); g.fill(); // hair
  g.fillStyle = "#e9b48a"; g.beginPath(); g.ellipse(200, 210, 120, 140, 0, 0, Math.PI*2); g.fill(); // face
  g.fillStyle = "#3a2414"; g.beginPath(); g.ellipse(200, 110, 125, 55, 0, Math.PI, Math.PI*2); g.fill(); // fringe
  g.fillStyle = "#c0392b"; g.fillRect(70, 330, 260, 70); // shirt
  g.fillStyle = "#fff"; [150, 250].forEach(x => { g.beginPath(); g.ellipse(x, 200, 26, 18, 0, 0, 7); g.fill(); });
  g.fillStyle = "#2b3a67"; [150, 250].forEach(x => { g.beginPath(); g.arc(x, 202, 11, 0, 7); g.fill(); });
  g.fillStyle = "#f08a7e"; [125, 275].forEach(x => { g.beginPath(); g.ellipse(x, 250, 22, 12, 0, 0, 7); g.fill(); });
  g.strokeStyle = "#8e2b2b"; g.lineWidth = 10; g.lineCap = "round";
  g.beginPath(); g.arc(200, 255, 45, 0.15*Math.PI, 0.85*Math.PI); g.stroke();
  return c;
}

function render() {
  if (!src || !palette.length) return;
  const cols = +$("cols").value, chaos = +$("chaos").value, tint = $("tint").checked;
  const rows = Math.max(1, Math.round(cols * src.height / src.width));
  const cell = Math.max(10, Math.floor(Math.max(1200, cols * 12) / cols));
  out.width = cols * cell; out.height = rows * cell;
  // Downsample: each pixel of the tiny canvas = one cell's average color
  const t = document.createElement("canvas"); t.width = cols; t.height = rows;
  const tg = t.getContext("2d", { willReadFrequently: true });
  tg.imageSmoothingQuality = "high"; tg.drawImage(src, 0, 0, cols, rows);
  const d = tg.getImageData(0, 0, cols, rows).data;
  const surface = getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() || "#fff";
  ctx.fillStyle = surface; ctx.fillRect(0, 0, out.width, out.height);
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = `${cell * 0.92}px serif`;
  const counts = {};
  seed = 1 + (shuffleN * 7919);
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const i = (y * cols + x) * 4, r = d[i], g = d[i+1], b = d[i+2];
    if (tint) { ctx.fillStyle = `rgba(${r},${g},${b},.45)`; ctx.fillRect(x*cell, y*cell, cell, cell); }
    const key = (r >> 2) << 12 | (g >> 2) << 6 | (b >> 2);
    let ranked = rankCache.get(key);
    if (!ranked) {
      ranked = palette.map(p => [dist(p, r, g, b), p]).sort((a, b) => a[0] - b[0]).slice(0, 6).map(a => a[1]);
      rankCache.set(key, ranked);
    }
    const pick = ranked[Math.floor(rand() * chaos)];
    counts[pick.e] = (counts[pick.e] || 0) + 1;
    ctx.fillText(pick.e, x*cell + cell/2, y*cell + cell/2 + cell*0.05);
  }
  receipt(counts, cols * rows);
}

function receipt(counts, total) {
  const items = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const top = items.slice(0, 8);
  let price = 0, kcal = 0;
  for (const [e, n] of items) { price += MENU[e][1] * n; kcal += MENU[e][2] * n; }
  const now = new Date();
  const fmt = n => n.toLocaleString("th-TH");
  $("receipt").innerHTML = `
    <h2>ร้านหน้าฉันกินได้</h2>
    <div class="sub">โต๊ะ ${String(total % 97).padStart(2,"0")} · ${now.toLocaleDateString("th-TH")} ${now.toLocaleTimeString("th-TH",{hour:"2-digit",minute:"2-digit"})}</div>
    <hr>
    ${top.map(([e, n]) => `<div class="row"><span>${e}</span><span>${MENU[e][0]} <span class="q">×${fmt(n)}</span></span><span>${fmt(MENU[e][1]*n)}</span></div>`).join("")}
    ${items.length > 8 ? `<div class="row"><span>…</span><span>อื่น ๆ อีก ${items.length - 8} เมนู</span><span></span></div>` : ""}
    <hr>
    <div class="tot"><span>รวม ${fmt(total)} ชิ้น</span><span>฿${fmt(price)}</span></div>
    <div class="tot"><span>พลังงาน</span><span>${fmt(kcal)} kcal</span></div>
    <div class="tot"><span>ประโยชน์ที่ได้รับ</span><span>0.00</span></div>
    <div class="foot">ขอบคุณที่ใช้หน้าเป็นวัตถุดิบ<br>ไม่รับคืน ไม่รับเปลี่ยน ไม่รับประทาน</div>`;
}

function loadFile(f) {
  if (!f || !f.type.startsWith("image/")) { $("flash").textContent = "ไฟล์นี้ไม่ใช่รูปภาพ ลองไฟล์ .jpg หรือ .png"; return; }
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    const s = Math.min(1, 800 / Math.max(img.width, img.height));
    c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    src = c; URL.revokeObjectURL(img.src);
    $("hint").textContent = "ลากรูปใหม่มาวางเพื่อสั่งจานถัดไป";
    $("flash").textContent = "เสิร์ฟแล้ว! หน้าคุณอร่อยมาก";
    render();
  };
  img.onerror = () => { $("flash").textContent = "เปิดรูปนี้ไม่ได้ ลองบันทึกเป็น .jpg แล้วอัปโหลดใหม่"; };
  img.src = URL.createObjectURL(f);
}

$("file").addEventListener("change", e => loadFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => loadFile(e.dataTransfer.files[0]));
document.addEventListener("paste", e => { const f = [...(e.clipboardData?.files || [])][0]; if (f) loadFile(f); });
$("cols").addEventListener("input", e => { $("colsOut").textContent = e.target.value; renderSoon(); });
$("chaos").addEventListener("input", e => { $("chaosOut").textContent = e.target.value; render(); });
$("tint").addEventListener("change", render);
$("shuffle").addEventListener("click", () => {
  shuffleN++;
  if (+$("chaos").value === 1) { $("chaos").value = 3; $("chaosOut").textContent = 3; }
  render();
  $("flash").textContent = "เชฟเปลี่ยนใจ จัดจานใหม่ให้แล้ว";
});
$("sample").addEventListener("click", () => {
  src = sampleFace();
  $("hint").textContent = "ลากรูปมาวางตรงนี้ได้เลย · ตอนนี้เป็นหน้าตัวอย่าง";
  $("flash").textContent = "";
  render();
});
$("save").addEventListener("click", () => {
  out.toBlob(blob => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `food-face-${Date.now()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

const start = () => { buildPalette(); src = sampleFace(); render(); };
(document.fonts ? document.fonts.ready : Promise.resolve()).then(start);
