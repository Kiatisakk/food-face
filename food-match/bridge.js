/* Food Match bridge: accept a decoded PNG from Food Face without another upload. */
(() => {
  if (window.parent === window) return;
  const notify = (message) => window.parent.postMessage(message, window.location.origin);
  let pending = false;
  let lastHeight = 0;
  function resizeParent() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const h = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      if (h !== lastHeight) {
        lastHeight = h;
        notify({ type: "foodmatch:height", height: h });
      }
    });
  }
  if (window.ResizeObserver) new ResizeObserver(resizeParent).observe(document.querySelector(".wrap"));
  window.addEventListener("resize", resizeParent);
  resizeParent();

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || event.source !== window.parent) return;
    const data = event.data;
    if (!data || data.type !== "foodmatch:send-image" || typeof data.base64 !== "string") return;
    // Prevent huge images / malformed messages from allocating unbounded memory.
    if (data.base64.length > 12 * 1024 * 1024 || !/^[A-Za-z0-9+/]*={0,2}$/.test(data.base64)) {
      notify({ type: "foodmatch:error", message: "ข้อมูลภาพไม่ถูกต้องหรือใหญ่เกินไป" });
      return;
    }
    const image = new Image();
    image.onload = () => {
      if (!image.width || !image.height || image.width * image.height > 2048 * 2048) {
        notify({ type: "foodmatch:error", message: "ภาพใหญ่เกินข้อจำกัด" });
        return;
      }
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 1024 / Math.max(image.width, image.height));
      canvas.width = Math.max(1, Math.round(image.width * scale));
      canvas.height = Math.max(1, Math.round(image.height * scale));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      setImage(canvas);
      resizeParent();
      notify({ type: "foodmatch:received" });
    };
    image.onerror = () => notify({ type: "foodmatch:error", message: "ไม่สามารถเปิดภาพที่ Decode ได้" });
    image.src = "data:image/png;base64," + data.base64;
  });

  ready.p.then(() => {
    notify({ type: "foodmatch:ready", count: meta.count, fake: Boolean(meta.fake) });
    resizeParent();
  }).catch(() => notify({ type: "foodmatch:error", message: "โหลดฐานข้อมูลอาหารไม่ได้ ลองรีเฟรชหรือเช็กอินเทอร์เน็ต" }));
})();
