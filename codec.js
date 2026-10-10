(() => {
  const el = id => document.getElementById("codec" + id);
  const MAX_UPLOAD = 8 * 1024 * 1024;
  const ENCODE_LABEL = 'หั่นรูปเป็นอีโมจิ <span aria-hidden="true">→</span>';
  const DECODE_LABEL = 'ประกอบกลับเป็นรูป <span aria-hidden="true">→</span>';
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let imageFile = null;
  let codecFile = null;
  let previewUrl = null;
  let imageRevision = 0;
  let encodeResult = null;
  let decodeResult = null;
  let modelReady = false;
  let working = false;
  let workingStage = null;
  let copyTimer;

  const notice = (message = "") => { el("Notice").textContent = message; };
  const decodeNotice = (message = "") => { el("DecodeNotice").textContent = message; };

  function updateControls() {
    el("EncodeButton").disabled = working || !modelReady || !imageFile;
    el("DecodeButton").disabled = working || !modelReady ||
      !(codecFile || el("DecodeText").value.trim());
    el("ToDecoder").disabled = working || !encodeResult?.text;
    for (const id of ["ImageInput", "FileInput", "Sample", "ClearFile"]) {
      el(id).disabled = working;
    }
    el("DecodeText").disabled = working || Boolean(codecFile);
  }

  function setWorking(on, stage = null) {
    working = on;
    workingStage = on ? stage : null;
    el("Output").setAttribute("aria-busy", String(on && stage === "encode"));
    el("DecodeOutput").setAttribute("aria-busy", String(on && stage === "decode"));
    el("EncodeButton").innerHTML = on && stage === "encode"
      ? "เชฟกำลังหั่นรูป…" : ENCODE_LABEL;
    el("DecodeButton").innerHTML = on && stage === "decode"
      ? "เชฟกำลังประกอบรูป…" : DECODE_LABEL;
    updateControls();
  }

  async function requestJSON(path, options = {}, timeout = 120000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch { throw new Error("เชื่อมต่อครัวแปลงรูปไม่ได้ ลองเชื่อมต่อใหม่"); }
      if (!response.ok) {
        if (response.status === 503) throw new Error("ครัวนี้ยังไม่มีโมเดลพร้อมใช้งาน");
        if (response.status === 413) throw new Error("วัตถุดิบใหญ่ไป ใช้ไฟล์ไม่เกิน 8 MB และรูปด้านละไม่เกิน 2048 px");
        throw new Error(typeof data.detail === "string"
          ? data.detail : "เชฟแปลงรูปไม่สำเร็จ ลองตรวจสอบไฟล์อีกครั้ง");
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("ครัวใช้เวลานานเกินไป ลองใหม่อีกครั้ง");
      if (error instanceof TypeError) throw new Error("เชื่อมต่อครัวแปลงรูปไม่ได้ ลองเชื่อมต่อใหม่");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  // Two interchangeable engines: the local FastAPI server (npm start) when it is running, otherwise the same
  // model exported to ONNX running in the browser (assets/emocodec/emocodec-web.js), e.g. on GitHub Pages.
  let engine = null; // "api" | "browser"
  const apiPath = path => new URL(path.replace(/^\//, ""), document.baseURI).href;
  async function apiStatus() {
    try {
      const response = await fetch(apiPath("/api/status"), { signal: AbortSignal.timeout(4000) });
      if (!response.ok || !(response.headers.get("content-type") || "").includes("json")) return null;
      const status = await response.json();
      return status.ready ? status : null;
    } catch { return null; }
  }

  async function checkStatus() {
    el("Retry").disabled = true;
    el("StatusLabel").textContent = "กำลังเช็กครัว…";
    try {
      if (location.protocol === "file:") throw new Error("เปิดเว็บผ่านเซิร์ฟเวอร์เพื่อใช้ครัวนี้");
      let status = await apiStatus();
      engine = status ? "api" : null;
      if (!status) {
        if (!window.EmoCodecWeb) throw new Error("ครัวนี้ยังไม่มีโมเดลพร้อมใช้งาน");
        el("StatusLabel").textContent = "กำลังยกโมเดลขึ้นเตาในเบราว์เซอร์…";
        status = await window.EmoCodecWeb.status();
        engine = "browser";
      }
      modelReady = Boolean(status.ready);
      el("StatusLabel").textContent = modelReady
        ? (status.demo_model ? "พร้อมใช้งาน · โมเดลทดลอง" : engine === "browser" ? "ครัวพร้อมแปลงรูป · ทำในเบราว์เซอร์" : "ครัวพร้อมแปลงรูป")
        : "ยังไม่มีโมเดลพร้อมใช้งาน";
    } catch (error) {
      modelReady = false;
      el("StatusLabel").textContent = error.message;
    } finally {
      el("Status").classList.toggle("ready", modelReady);
      el("Retry").hidden = modelReady;
      el("Retry").disabled = false;
      updateControls();
    }
  }

  function resetEncodeResult() {
    encodeResult = null;
    clearTimeout(copyTimer);
    el("Copy").textContent = "คัดลอกสูตร";
    el("TokenBox").textContent = "";
    el("Result").hidden = true;
    el("Placeholder").hidden = false;
    updateControls();
  }

  function resetDecodeResult() {
    decodeResult = null;
    el("DecodeResult").hidden = true;
    el("DecodePlaceholder").hidden = false;
    el("DecodedView").removeAttribute("src");
  }

  async function imageSelected(file) {
    if (!file || working) return false;
    const revision = ++imageRevision;
    notice();
    if (!(file.type ? /^image\/(png|jpeg|webp)$/.test(file.type) : /\.(png|jpe?g|webp)$/i.test(file.name))) {
      notice("ครัวนี้รับรูป PNG, JPEG หรือ WebP เท่านั้น");
      return false;
    }
    if (file.size > MAX_UPLOAD) {
      notice("วัตถุดิบใหญ่ไป เลือกรูปที่ขนาดไม่เกิน 8 MB");
      return false;
    }
    const url = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      if (revision !== imageRevision) { URL.revokeObjectURL(url); return false; }
      if (Math.max(image.naturalWidth, image.naturalHeight) > 2048) {
        throw new Error("รูปใหญ่ไป ย่อให้แต่ละด้านไม่เกิน 2048 px แล้วลองอีกครั้ง");
      }
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      imageFile = file;
      previewUrl = url;
      el("InputPreview").src = url;
      el("InputPreview").hidden = false;
      el("DropArt").hidden = true;
      el("DropTitle").textContent = file.name;
      el("DropHelp").textContent = image.naturalWidth + " × " + image.naturalHeight +
        " px · " + (file.size / 1024).toFixed(1) + " KB";
      resetEncodeResult();
      return true;
    } catch (error) {
      URL.revokeObjectURL(url);
      if (revision === imageRevision) notice(error.name === "EncodingError"
        ? "เชฟเปิดรูปนี้ไม่ได้ ลองเลือกรูปใหม่" : error.message);
      return false;
    }
  }

  // The first kitchen passes over its rendered food-emoji mosaic (not the source photo).
  window.foodFaceCodec = {
    async receiveImage(file) {
      if (working) {
        notice("ครัวนี้กำลังปรุงจานอื่นอยู่ ลองส่งใหม่อีกครั้ง");
        return false;
      }
      const accepted = await imageSelected(file);
      if (accepted) notice(modelReady
        ? "ได้รับจานจากครัวแรกแล้ว กดหั่นรูปเป็นอีโมจิเพื่อปรุงต่อได้เลย"
        : "ได้รับจานจากครัวแรกแล้ว แต่ต้องเชื่อมต่อโมเดลก่อนจึงจะปรุงต่อได้");
      return accepted;
    }
  };

  el("ImageInput").addEventListener("change", event => {
    imageSelected(event.target.files[0]);
    event.target.value = "";
  });

  let dragDepth = 0;
  for (const type of ["dragenter", "dragover"]) {
    el("Drop").addEventListener(type, event => {
      event.preventDefault();
      if (working) return;
      if (type === "dragenter") dragDepth++;
      el("Drop").classList.add("over");
    });
  }
  el("Drop").addEventListener("dragleave", event => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) el("Drop").classList.remove("over");
  });
  el("Drop").addEventListener("drop", event => {
    event.preventDefault();
    dragDepth = 0;
    el("Drop").classList.remove("over");
    imageSelected(event.dataTransfer.files[0]);
  });
  document.getElementById("emoji-codec").addEventListener("paste", event => {
    const file = [...(event.clipboardData?.files || [])].find(item => item.type.startsWith("image/"));
    if (file) { event.preventDefault(); imageSelected(file); }
  });
  el("Sample").addEventListener("click", async () => {
    el("Sample").disabled = true;
    notice();
    try {
      const response = await fetch("assets/shef.png");
      if (!response.ok) throw new Error("ยังหยิบรูปเชฟไม่ได้ ลองเลือกรูปจากเครื่อง");
      await imageSelected(new File([await response.blob()], "shef.png", { type: "image/png" }));
    } catch {
      notice("ยังหยิบรูปเชฟไม่ได้ ลองเลือกรูปจากเครื่อง");
    } finally {
      updateControls();
    }
  });

  function clearCodecFile() {
    codecFile = null;
    el("FileInput").value = "";
    el("SelectedFile").textContent = "";
    el("ClearFile").hidden = true;
    updateControls();
  }
  el("FileInput").addEventListener("change", event => {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = "";
    decodeNotice();
    if (!/\.(emotxt|emoc)$/i.test(file.name)) {
      decodeNotice("เลือกไฟล์ .emotxt หรือ .emoc ที่ได้จาก Encoder");
      return;
    }
    if (file.size > MAX_UPLOAD) {
      decodeNotice("เลือกไฟล์รหัสที่มีขนาดไม่เกิน 8 MB");
      return;
    }
    codecFile = file;
    el("SelectedFile").textContent = "ใช้ไฟล์: " + file.name;
    el("ClearFile").hidden = false;
    resetDecodeResult();
    updateControls();
  });
  el("ClearFile").addEventListener("click", () => {
    clearCodecFile();
    el("DecodeText").focus();
  });
  el("DecodeText").addEventListener("input", () => {
    if (!codecFile) {
      resetDecodeResult();
      decodeNotice();
    }
    updateControls();
  });

  function showEncoded(data) {
    if (!data.text) throw new Error("โมเดลไม่ได้ส่งรหัสอีโมจิกลับมา");
    let payload;
    try { payload = JSON.parse(data.text); }
    catch { throw new Error("สูตรอีโมจิที่ได้ไม่ถูกต้อง"); }
    if (typeof payload.emoji !== "string") throw new Error("สูตรยังไม่มีรหัสอีโมจิ");
    // Do not show the reconstructed image from /api/encode; the decoder owns that step.
    encodeResult = {
      text: data.text,
      text_bytes: data.text_bytes,
      binary_base64: data.binary_base64,
      binary_bytes: data.binary_bytes,
      tokens: data.tokens,
      bpp: data.bpp
    };
    clearTimeout(copyTimer);
    el("Copy").textContent = "คัดลอกสูตร";
    el("Tokens").textContent = data.tokens == null ? "—" : data.tokens.toLocaleString();
    el("BinarySize").textContent = data.binary_bytes == null
      ? "—" : data.binary_bytes.toLocaleString() + " B";
    el("Bpp").textContent = data.bpp == null ? "—" : String(data.bpp);
    el("TokenBox").textContent = payload.emoji;
    el("FileStats").textContent = "สูตรเต็ม " +
      (data.text_bytes == null ? new Blob([data.text]).size : data.text_bytes).toLocaleString() +
      " B · ไฟล์ไบนารี " +
      (data.binary_bytes == null ? "—" : data.binary_bytes.toLocaleString() + " B");
    el("Placeholder").hidden = true;
    el("Result").hidden = false;
    updateControls();
  }

  function showDecoded(data) {
    if (!data.reconstructed_png_base64) throw new Error("โมเดลยังไม่ได้ส่งรูปที่ประกอบกลับมา");
    decodeResult = data;
    el("DecodedView").src = "data:image/png;base64," + data.reconstructed_png_base64;
    el("DecodePlaceholder").hidden = true;
    el("DecodeResult").hidden = false;
  }

  el("EncodeButton").addEventListener("click", async () => {
    if (!imageFile || !modelReady || working) return;
    const body = new FormData();
    body.append("image", imageFile);
    setWorking(true, "encode");
    resetEncodeResult();
    notice("เชฟกำลังแปลงรูปเป็นรหัสอีโมจิ…");
    try {
      const data = engine === "browser" ? await window.EmoCodecWeb.encode(imageFile)
        : await requestJSON(apiPath("/api/encode"), { method: "POST", body });
      showEncoded(data);
      notice("เข้ารหัสสำเร็จ! คัดลอกสูตร หรือส่งลงครัวสุดท้ายได้เลย");
    } catch (error) {
      notice(error.message);
    } finally {
      setWorking(false);
    }
  });

  el("DecodeButton").addEventListener("click", async () => {
    if (!modelReady || working || !(codecFile || el("DecodeText").value.trim())) return;
    const body = new FormData();
    if (codecFile) body.append("upload", codecFile);
    else body.append("text", el("DecodeText").value.trim());
    setWorking(true, "decode");
    resetDecodeResult();
    decodeNotice("เชฟกำลังประกอบรูปจากรหัสอีโมจิ…");
    try {
      const data = engine === "browser"
        ? await window.EmoCodecWeb.decode(codecFile ? { file: codecFile } : { text: el("DecodeText").value.trim() })
        : await requestJSON(apiPath("/api/decode"), { method: "POST", body });
      showDecoded(data);
      decodeNotice("ประกอบรูปกลับแล้ว! รับรูปกลับบ้านได้เลย");
    } catch (error) {
      decodeNotice(error.message);
    } finally {
      setWorking(false);
    }
  });

  function fromBase64(value) {
    return Uint8Array.from(atob(value), character => character.charCodeAt(0));
  }
  function download(bytes, name, type) {
    const url = URL.createObjectURL(new Blob([bytes], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  el("Copy").addEventListener("click", async () => {
    if (!encodeResult?.text) return;
    try {
      // The full .emotxt JSON includes metadata required for successful decoding.
      await navigator.clipboard.writeText(encodeResult.text);
      el("Copy").textContent = "คัดลอกแล้ว ✓";
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => { el("Copy").textContent = "คัดลอกสูตร"; }, 1500);
    } catch {
      notice("คัดลอกไม่ได้ ใช้ปุ่มดาวน์โหลด .emotxt เพื่อเก็บสูตรแทน");
    }
  });
  el("DownloadText").addEventListener("click", () => {
    if (encodeResult?.text) download(encodeResult.text, "let-him-cook.emotxt", "text/plain;charset=utf-8");
  });
  el("DownloadBinary").addEventListener("click", () => {
    if (encodeResult?.binary_base64) download(fromBase64(encodeResult.binary_base64),
      "let-him-cook.emoc", "application/octet-stream");
  });
  el("ToDecoder").addEventListener("click", () => {
    if (!encodeResult?.text || working) return;
    clearCodecFile();
    resetDecodeResult();
    el("DecodeText").value = encodeResult.text;
    updateControls();
    decodeNotice("ได้รับสูตรจากครัวที่สองแล้ว กดประกอบกลับเป็นรูปได้เลย");
    document.getElementById("emoji-decoder").scrollIntoView({
      behavior: prefersReducedMotion.matches ? "auto" : "smooth",
      block: "start"
    });
  });
  el("ToMatch").addEventListener("click", () => {
    if (!decodeResult?.reconstructed_png_base64 || working) return;
    if (window.foodFaceMatch?.sendDecoded) {
      window.foodFaceMatch.sendDecoded(decodeResult.reconstructed_png_base64);
    } else {
      decodeNotice("ยังเปิดครัวสุดท้ายไม่ได้ ลองรีเฟรชเว็บ");
    }
  });

  el("SaveImage").addEventListener("click", () => {
    if (decodeResult?.reconstructed_png_base64) download(fromBase64(decodeResult.reconstructed_png_base64),
      "let-him-cook-reconstructed.png", "image/png");
  });
  el("Retry").addEventListener("click", checkStatus);
  checkStatus();
})();
