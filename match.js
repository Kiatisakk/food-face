/* Lazy, same-origin integration between Food Face Decoder and Food Match. */
(() => {
  const section = document.getElementById("food-match-finale");
  const wrapper = document.getElementById("matchFrameWrap");
  const iframe = document.getElementById("matchFrame");
  const trigger = document.getElementById("matchOpen");
  const status = document.getElementById("matchStatus");
  if (!section || !iframe || !trigger) return;

  let started = false;
  let ready = false;
  let pendingImage = null;
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const setStatus = (message) => { status.textContent = message; };

  function begin() {
    if (started) return;
    started = true;
    wrapper.hidden = false;
    section.setAttribute("aria-busy", "true");
    setStatus("กำลังเปิดครัวสุดท้าย และโหลดลายเซ็นอาหาร 50,000 รูป…");
    iframe.src = "/food-match/index.html?embedded=1";
    trigger.textContent = "เปิดครัวแล้ว";
    trigger.disabled = true;
  }
  function flush() {
    if (!ready || !pendingImage || !iframe.contentWindow) return;
    iframe.contentWindow.postMessage({ type: "foodmatch:send-image", base64: pendingImage }, location.origin);
    pendingImage = null;
    setStatus("ส่งภาพให้ครัวสุดท้ายแล้ว กำลังค้นหาเมนูที่เข้ากัน…");
  }
  function sendDecoded(base64) {
    if (typeof base64 !== "string" || !base64.length) return;
    pendingImage = base64;
    begin();
    section.scrollIntoView({ behavior: motion.matches ? "auto" : "smooth", block: "start" });
    flush();
  }
  window.foodFaceMatch = { sendDecoded, open: begin };
  trigger.addEventListener("click", () => {
    begin();
    wrapper.scrollIntoView({ behavior: motion.matches ? "auto" : "smooth", block: "nearest" });
  });
  window.addEventListener("message", (event) => {
    if (event.origin !== location.origin || event.source !== iframe.contentWindow || !event.data) return;
    const d = event.data;
    if (d.type === "foodmatch:height" && Number.isFinite(d.height)) {
      iframe.style.height = Math.max(500, Math.min(4200, Math.ceil(d.height + 12))) + "px";
    } else if (d.type === "foodmatch:ready") {
      ready = true;
      section.setAttribute("aria-busy", "false");
      setStatus("ครัวสุดท้ายพร้อมแล้ว · ค้นหาอาหารจาก " + Number(d.count).toLocaleString("th-TH") + " รูป" + (d.fake ? " (ข้อมูลทดลอง)" : ""));
      flush();
    } else if (d.type === "foodmatch:received") {
      setStatus("รับภาพที่ Decode แล้ว เริ่มทดลองจัดเรียงพิกเซลเป็นอาหาร");
    } else if (d.type === "foodmatch:error") {
      section.setAttribute("aria-busy", "false");
      setStatus(d.message || "ครัวสุดท้ายมีปัญหา กรุณาลองใหม่");
    }
  });
  iframe.addEventListener("error", () => {
    section.setAttribute("aria-busy", "false");
    setStatus("เปิดครัวสุดท้ายไม่ได้");
  });
  // No dataset / worker downloads on initial page load.
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { begin(); observer.disconnect(); }
    }, { rootMargin: "240px" });
    observer.observe(section);
  }
})();
