/* EmoCodec in the browser: the same VQ-VAE (exported to ONNX) and the same .emotxt / .emoc formats as the
   Python package (emioji-encoder/src/emocodec), so files are interchangeable with the CLI and the local API.
   Used when no local FastAPI server is available (e.g. the static GitHub Pages deploy).

   window.EmoCodecWeb.status()            -> { ready, model_id, model_file, demo_model }
   window.EmoCodecWeb.encode(file)        -> same JSON shape as POST /api/encode (minus the reconstruction)
   window.EmoCodecWeb.decode({file|text}) -> same JSON shape as POST /api/decode
*/
(() => {
  const BASE = new URL(".", document.currentScript.src).href; // assets/emocodec/
  const ORT_VERSION = "1.20.1";
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const TEXT_MAGIC = "EMOTXT1";
  const BIN_MAGIC = [0x45, 0x4d, 0x4f, 0x43, 0x01]; // "EMOC\x01"
  const MAX_UPLOAD = 8 * 1024 * 1024, MAX_DIMENSION = 2048;   // same limits as the API
  const MAX_HEADER = 8192, MAX_PIXELS = 8192 * 8192, MAX_TOKENS = 1024 * 1024;
  const REQUIRED = ["version", "model_id", "alphabet_sha256", "width", "height", "latent_w", "latent_h",
    "stride", "token_count", "payload_sha256"];

  let loading = null; // Promise<{ ort, enc, dec, info, alphabet, reverse }>

  function loadScript(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.async = true; s.crossOrigin = "anonymous";
      s.onload = ok; s.onerror = () => fail(new Error("โหลดตัวรันโมเดลไม่ได้ (เช็กอินเทอร์เน็ต)"));
      document.head.append(s);
    });
  }
  const getJSON = async url => { const r = await fetch(url); if (!r.ok) throw new Error(url + " " + r.status); return r.json(); };

  function load() {
    if (loading) return loading;
    loading = (async () => {
      const [info, alpha] = await Promise.all([getJSON(BASE + "model.json"), getJSON(BASE + "alphabet.json"),
        window.ort ? null : loadScript(ORT_CDN + "ort.wasm.min.js")]);
      const ort = window.ort;
      ort.env.wasm.wasmPaths = ORT_CDN;
      ort.env.wasm.numThreads = 1; // static hosts aren't cross-origin isolated, so no wasm threads
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const [enc, dec] = await Promise.all([ort.InferenceSession.create(BASE + "encoder.onnx", opts),
        ort.InferenceSession.create(BASE + "decoder.onnx", opts)]);
      const alphabet = [...alpha.emoji];
      if (alpha.version !== 1 || alphabet.length !== 256 || new Set(alphabet).size !== 256) throw new Error("Invalid emoji alphabet");
      if (await sha256Hex(new TextEncoder().encode(alphabet.join(""))) !== info.alphabet_sha256) throw new Error("Emoji alphabet does not match the model");
      return { ort, enc, dec, info, alphabet, reverse: new Map(alphabet.map((e, i) => [e, i])) };
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  async function sha256Hex(bytes) {
    const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    return [...d].map(b => b.toString(16).padStart(2, "0")).join("");
  }
  const isInt = v => typeof v === "number" && Number.isInteger(v);

  // ---------- formats (port of codec/formats.py) ----------
  async function validate(meta, ids, M, expectedModelId) {
    if (!meta || typeof meta !== "object" || !REQUIRED.every(k => k in meta)) throw new Error("Invalid/missing codec metadata fields");
    if (!isInt(meta.version) || meta.version !== 1 || meta.alphabet_sha256 !== M.info.alphabet_sha256) throw new Error("Unsupported format/alphabet version");
    if (typeof meta.model_id !== "string" || !/^[0-9a-f]{64}$/.test(meta.model_id)) throw new Error("Invalid model_id (SHA-256 expected)");
    if (expectedModelId && meta.model_id !== expectedModelId) throw new Error("Wrong model checkpoint: image cannot be decoded with a different model");
    for (const k of ["width", "height", "latent_w", "latent_h", "stride", "token_count"])
      if (!isInt(meta[k]) || meta[k] <= 0) throw new Error("Invalid " + k);
    if (meta.width * meta.height > MAX_PIXELS || Math.max(meta.width, meta.height) > 8192) throw new Error("Image dimensions exceed safety limits");
    if (meta.stride !== 16) throw new Error("Unsupported encoder stride");
    if (meta.latent_w !== Math.floor((meta.width + 15) / 16) || meta.latent_h !== Math.floor((meta.height + 15) / 16)) throw new Error("Latent grid does not match image dimensions");
    if (meta.token_count !== meta.latent_w * meta.latent_h || meta.token_count > MAX_TOKENS) throw new Error("Invalid token count or excessive allocation");
    if (ids.length !== meta.token_count || ids.some(x => !(x >= 0 && x <= 255))) throw new Error("Token length/range mismatch");
    if (meta.payload_sha256 !== await sha256Hex(Uint8Array.from(ids))) throw new Error("Token payload checksum mismatch");
  }

  function encodeText(ids, meta, M) {
    // json.dumps({**meta, "magic", "emoji"}, ensure_ascii=False, separators=(",", ":")) + "\n"
    return JSON.stringify({ ...meta, magic: TEXT_MAGIC, emoji: ids.map(i => M.alphabet[i]).join("") }) + "\n";
  }
  function encodeBinary(ids, meta) {
    // header = json.dumps(meta, separators=(",", ":"), sort_keys=True); all values are ASCII
    const sorted = Object.fromEntries(Object.keys(meta).sort().map(k => [k, meta[k]]));
    const header = new TextEncoder().encode(JSON.stringify(sorted));
    if (header.length > MAX_HEADER) throw new Error("Codec header too large");
    const out = new Uint8Array(9 + header.length + ids.length);
    out.set(BIN_MAGIC, 0);
    new DataView(out.buffer).setUint32(5, header.length, false);
    out.set(header, 9); out.set(ids, 9 + header.length);
    return out;
  }
  async function deserialize(bytes, M) {
    const expected = M.info.model_id;
    if (bytes.length >= 5 && BIN_MAGIC.every((b, i) => bytes[i] === b)) {
      if (bytes.length < 9) throw new Error("Invalid binary codec magic");
      const hlen = new DataView(bytes.buffer, bytes.byteOffset).getUint32(5, false);
      if (!(hlen > 0 && hlen <= MAX_HEADER) || bytes.length < 9 + hlen) throw new Error("Truncated/oversized binary header");
      let meta;
      try { meta = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(9, 9 + hlen))); }
      catch { throw new Error("Malformed binary header"); }
      const ids = [...bytes.subarray(9 + hlen)];
      await validate(meta, ids, M, expected);
      return { meta, ids };
    }
    let content;
    try { content = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
    catch { throw new Error("Unknown codec file format"); }
    if (bytes.length > 5_000_000) throw new Error("Emoji file is too large");
    let obj;
    try { obj = JSON.parse(content); } catch { throw new Error("Malformed emoji JSON"); }
    if (!obj || typeof obj !== "object" || obj.magic !== TEXT_MAGIC || typeof obj.emoji !== "string") throw new Error("Invalid text codec magic or emoji payload");
    const ids = [];
    [...obj.emoji].forEach((ch, pos) => {
      const id = M.reverse.get(ch);
      if (id === undefined) throw new Error(`Unknown or modified emoji at token position ${pos}: ${ch}`);
      ids.push(id);
    });
    const { magic, emoji, ...meta } = obj;
    await validate(meta, ids, M, expected);
    return { meta, ids };
  }

  // ---------- images ----------
  async function readRGB(file) {
    if (file.size > MAX_UPLOAD) throw new Error("วัตถุดิบใหญ่ไป ใช้ไฟล์ไม่เกิน 8 MB และรูปด้านละไม่เกิน 2048 px");
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); }
    catch { throw new Error("Not a valid image file"); }
    const { width: w, height: h } = bmp;
    if (Math.max(w, h) > MAX_DIMENSION) { bmp.close(); throw new Error("Demo image max size is 2048×2048"); }
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(bmp, 0, 0); bmp.close();
    return { w, h, data: g.getImageData(0, 0, w, h).data };
  }
  // (1, 3, H, W) float tensor in [0, 1], padded to a multiple of 16 by repeating the edge (F.pad replicate)
  function toTensor({ w, h, data }) {
    const W = Math.ceil(w / 16) * 16, H = Math.ceil(h / 16) * 16, out = new Float32Array(3 * W * H);
    for (let y = 0; y < H; y++) {
      const sy = Math.min(y, h - 1);
      for (let x = 0; x < W; x++) {
        const s = (sy * w + Math.min(x, w - 1)) * 4, d = y * W + x;
        out[d] = data[s] / 255; out[W * H + d] = data[s + 1] / 255; out[2 * W * H + d] = data[s + 2] / 255;
      }
    }
    return { W, H, out };
  }
  function toPngBase64(img, W, H, w, h) {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d"), px = g.createImageData(w, h), p = px.data, A = W * H;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const s = y * W + x, d = (y * w + x) * 4;
      for (let k = 0; k < 3; k++) p[d + k] = Math.round(Math.min(1, Math.max(0, img[k * A + s])) * 255);
      p[d + 3] = 255;
    }
    g.putImageData(px, 0, 0);
    return c.toDataURL("image/png").split(",")[1];
  }
  const b64 = bytes => { let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };

  // ---------- public API ----------
  async function status() {
    const M = await load();
    return { ready: true, model_id: M.info.model_id, model_file: M.info.model_file, demo_model: M.info.model_file === "demo.pt", engine: "browser" };
  }

  async function encode(file) {
    const M = await load();
    const rgb = await readRGB(file);
    const { W, H, out } = toTensor(rgb);
    const res = await M.enc.run({ image: new M.ort.Tensor("float32", out, [1, 3, H, W]) });
    const t = res.ids, [, lh, lw] = t.dims;
    const ids = Array.from(t.data, Number);
    const meta = { version: 1, model_id: M.info.model_id, alphabet_sha256: M.info.alphabet_sha256,
      width: rgb.w, height: rgb.h, latent_w: lw, latent_h: lh, stride: 16, token_count: ids.length,
      payload_sha256: await sha256Hex(Uint8Array.from(ids)) };
    await validate(meta, ids, M);
    const text = encodeText(ids, meta, M), binary = encodeBinary(ids, meta);
    return { text, binary_base64: b64(binary), tokens: ids.length, text_bytes: new TextEncoder().encode(text).length,
      binary_bytes: binary.length, original_dimensions: [rgb.w, rgb.h], model_id: meta.model_id,
      bpp: Math.round(8 * binary.length / (rgb.w * rgb.h) * 1e5) / 1e5, psnr: null, ssim: null, engine: "browser" };
  }

  async function decode({ file, text }) {
    const M = await load();
    let bytes;
    if (file) { if (file.size > MAX_UPLOAD) throw new Error("Encoded payload exceeds 8 MB"); bytes = new Uint8Array(await file.arrayBuffer()); }
    else if (text) bytes = new TextEncoder().encode(text);
    else throw new Error("Upload a .emotxt/.emoc file or paste the full .emotxt JSON");
    const { meta, ids } = await deserialize(bytes, M);
    const t = new M.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, meta.latent_h, meta.latent_w]);
    const img = (await M.dec.run({ ids: t })).image;
    const [, , H, W] = img.dims;
    return { reconstructed_png_base64: toPngBase64(img.data, W, H, meta.width, meta.height), tokens: meta.token_count,
      width: meta.width, height: meta.height, model_id: meta.model_id, engine: "browser" };
  }

  window.EmoCodecWeb = { status, encode, decode, load };
})();
