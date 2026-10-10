// Shared drawing code: the worker prepares previews; the browser also uses
// this renderer for PNG exports and for devices without OffscreenCanvas.
function createFoodRenderer(createCanvas, palette, atlas) {
  const spriteSize = 72;
  const surface = "#fffdf8";
  const rankCache = new Map();

  function distance(p, r, g, b) {
    const rm = (p.r + r) / 2, dr = p.r - r, dg = p.g - g, db = p.b - b;
    return (2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db;
  }

  function nearestFoods(r, g, b) {
    const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    if (rankCache.has(key)) return rankCache.get(key);
    r = (r & 248) + 4; g = (g & 248) + 4; b = (b & 248) + 4;
    const best = [];
    for (const food of palette) {
      const score = distance(food, r, g, b);
      let at = 0;
      while (at < best.length && best[at].score <= score) at++;
      if (at < 6) {
        best.splice(at, 0, { score, food });
        if (best.length > 6) best.pop();
      }
    }
    const ranked = best.map(item => item.food);
    rankCache.set(key, ranked);
    return ranked;
  }

  function gridFor(source, cols) {
    const rows = Math.max(1, Math.min(Math.round(cols * source.height / source.width), Math.floor(120000 / cols)));
    return { cols, rows };
  }

  async function render(order, { yieldTask, shouldCancel = () => false }) {
    if (shouldCancel()) return null;
    const { source, cols, chaos, tint, shuffle, edge } = order;
    const model = order.model;
    const { rows } = model || gridFor(source, cols);
    const scale = edge / Math.max(source.width, source.height);
    const frame = createCanvas(Math.max(1, Math.round(source.width * scale)), Math.max(1, Math.round(source.height * scale)));
    const g = frame.getContext("2d");
    const tiny = createCanvas(cols, rows);
    const tg = tiny.getContext("2d", { willReadFrequently: true });
    let d;
    if (model) {
      d = model.colors;
      const pixels = tg.createImageData(cols, rows);
      pixels.data.set(d);
      tg.putImageData(pixels, 0, 0);
    } else {
      tg.fillStyle = surface;
      tg.fillRect(0, 0, cols, rows);
      tg.imageSmoothingQuality = "high";
      tg.drawImage(source, 0, 0, cols, rows);
      d = tg.getImageData(0, 0, cols, rows).data;
    }
    g.fillStyle = surface;
    g.fillRect(0, 0, frame.width, frame.height);
    if (tint) {
      g.imageSmoothingEnabled = false;
      g.globalAlpha = .45;
      g.drawImage(tiny, 0, 0, frame.width, frame.height);
      g.globalAlpha = 1;
      g.imageSmoothingEnabled = true;
    }
    g.imageSmoothingQuality = "high";
    const cw = frame.width / cols, ch = frame.height / rows;
    const size = Math.min(cw, ch);
    const foods = model ? model.foods : new Uint8Array(cols * rows);
    const counts = model ? { ...model.counts } : {};
    let seed = 1 + shuffle * 7919;
    let sliceStart = performance.now();
    for (let index = 0; index < cols * rows; index++) {
      const i = index * 4;
      let pick;
      if (model) pick = palette[foods[index]];
      else {
        const ranked = nearestFoods(d[i], d[i + 1], d[i + 2]);
        seed = (seed * 16807) % 2147483647;
        pick = ranked[Math.floor((seed / 2147483647) * Math.min(chaos, ranked.length))];
        foods[index] = pick.id;
        counts[pick.e] = (counts[pick.e] || 0) + 1;
      }
      const x = (index % cols) * cw + (cw - size) / 2;
      const y = Math.floor(index / cols) * ch + (ch - size) / 2;
      g.drawImage(atlas, pick.x, pick.y, spriteSize, spriteSize, x, y, size, size);
      if ((index & 127) === 0 && performance.now() - sliceStart > 3) {
        await yieldTask();
        if (shouldCancel()) {
          frame.width = frame.height = 1;
          tiny.width = tiny.height = 1;
          return null;
        }
        sliceStart = performance.now();
      }
    }
    tiny.width = tiny.height = 1;
    return { canvas: frame, counts, total: cols * rows, cols, rows,
      model: { cols, rows, colors: d, foods, counts } };
  }

  return { render, gridFor };
}

// Serialized into a Blob worker so the same code works over HTTP and file://.
// This function must be self-contained apart from its injected renderer.
function cookingWorker(createRenderer) {
  let renderer = null;
  let scene = null;
  let running = false;
  let generation = 0;
  const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));

  async function prepare() {
    if (running || !renderer || !scene || scene.generation !== generation) return;
    running = true;
    try {
      while (scene && scene.generation === generation) {
        const order = scene;
        const cols = !order.done.has(order.selected) ? order.selected : order.levels
          .filter(level => !order.done.has(level))
          .sort((a, b) => Math.abs(a - order.selected) - Math.abs(b - order.selected))[0];
        if (cols === undefined) break;
        const shouldCancel = () => order.generation !== generation || scene !== order ||
          (cols !== order.selected && !order.done.has(order.selected));
        const frame = await renderer.render({ ...order, cols }, { yieldTask, shouldCancel });
        if (!frame || shouldCancel()) {
          if (frame) frame.canvas.width = frame.canvas.height = 1;
          continue;
        }
        const bitmap = frame.canvas.transferToImageBitmap();
        frame.canvas.width = frame.canvas.height = 1;
        order.done.add(cols);
        self.postMessage({ type: "frame", generation: order.generation,
          frame: { ...frame, canvas: bitmap }, ready: order.done.size },
          [bitmap, frame.model.colors.buffer, frame.model.foods.buffer]);
        // Deliver input/configuration messages before choosing the next level.
        await yieldTask();
      }
    } catch (error) {
      self.postMessage({ type: "error", generation, message: error.message });
    } finally {
      running = false;
    }
  }

  self.onmessage = event => {
    const message = event.data;
    if (message.type === "init") {
      renderer = createRenderer((w, h) => new OffscreenCanvas(w, h), message.palette, message.atlas);
      self.postMessage({ type: "ready" });
    } else if (message.type === "cancel") {
      generation = message.generation;
      if (scene) scene.source.close();
      scene = null;
    } else if (message.type === "prepare") {
      if (message.generation !== generation) { message.source.close(); return; }
      if (scene) scene.source.close();
      scene = { ...message, done: new Set() };
      prepare();
    } else if (message.type === "select" && scene && message.generation === generation) {
      scene.selected = message.cols;
      prepare();
    }
  };
}
