#!/usr/bin/env node
// Cross-platform helpers for the EmoCodec Python environment (macOS, Linux and Windows).
//   node scripts/py.mjs setup          create emioji-encoder/.venv and install EmoCodec + web deps
//   node scripts/py.mjs <args...>      run the venv's Python with <args> (e.g. -m uvicorn ...)
//   node scripts/py.mjs --cwd=<dir> <args...>   same, from <dir> (relative to the repo root)
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const VENV = join(ROOT, "emioji-encoder", ".venv");
const VENV_PY = process.platform === "win32" ? join(VENV, "Scripts", "python.exe") : join(VENV, "bin", "python");

const run = (cmd, args, cwd = ROOT) => spawnSync(cmd, args, { cwd, stdio: "inherit", shell: false });

function systemPython() {
  // `python3` on macOS/Linux, `py -3` or `python` on Windows; first one that is Python 3.10+ wins
  const candidates = process.platform === "win32" ? [["py", ["-3"]], ["python", []], ["python3", []]] : [["python3", []], ["python", []]];
  for (const [cmd, pre] of candidates) {
    const r = spawnSync(cmd, [...pre, "-c", "import sys; print(sys.version_info >= (3, 10))"], { encoding: "utf8" });
    if (r.status === 0 && r.stdout.trim() === "True") return [cmd, pre];
  }
  console.error("Python 3.10+ not found. Install it from https://www.python.org/downloads/ and try again.");
  process.exit(1);
}

let args = process.argv.slice(2), cwd = ROOT;
if (args[0]?.startsWith("--cwd=")) { cwd = join(ROOT, args[0].slice(6)); args = args.slice(1); }
if (args[0] === "setup") {
  if (!existsSync(VENV_PY)) {
    const [cmd, pre] = systemPython();
    if (run(cmd, [...pre, "-m", "venv", VENV]).status !== 0) process.exit(1);
  }
  process.exit(run(VENV_PY, ["-m", "pip", "install", "-e", "./emioji-encoder[web]"]).status ?? 1);
}
if (!existsSync(VENV_PY)) {
  console.error("Python environment missing. Run `npm run setup:python` first.");
  process.exit(1);
}
process.exit(run(VENV_PY, args, cwd).status ?? 1);
