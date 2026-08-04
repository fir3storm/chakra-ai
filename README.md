# ⚡ Chakra AI

<div align="center">

**Agentic Coding Terminal · Multi-Engine MoE Inference · Made in India 🇮🇳**

[![PyPI](https://img.shields.io/badge/pypi-chakra--ai-blue)](https://pypi.org/project/chakra-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Made in India](https://img.shields.io/badge/made%20in-India-FF9933.svg)](#-made-in-india)

*A terminal where you talk to AI like a pair programmer. Writes code, runs it in a sandbox, audits for security bugs, self-debugs — all 100% offline on 8GB RAM.* ⚡

</div>

---

```
     ┌──────────────────────────────────────────────┐
     │  ▐▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▌  │
     │  ▐▓▌  ⚡  C H A K R A   A I              ▐▓▌  │
     │  ▐▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▌  │
     └──────────────────────────────────────────────┘
  Made by Abhirup Guha · Info Security Solution · insec.in
```

---

## ❇️ Quick Start

**One command. That's it.**

```bash
pip install chakra-ai
chakra
```

On first run, auto-downloads the model (~1 GB) and benchmarks your system.

```powershell
# Full one-liner (Windows)
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/fir3storm/chakra-ai/main/install.ps1 | iex"
```

```bash
# Full one-liner (Linux/macOS)
curl -fsSL https://raw.githubusercontent.com/fir3storm/chakra-ai/main/setup.sh | bash
```

> ⚡ **Speed:** ~300 tokens/sec via llama.cpp GGUF — code generation takes under 1 second. ~1 GB RAM. No GPU. Offline.

---

## ✨ Features

| Category | What It Does |
|----------|-------------|
| ⚡ **Blazing Fast** | 300+ tok/s via llama.cpp GGUF · ~1 GB RAM |
| 🧠 **Multi-Agent Team** | Architect → Coder → Auditor → Supervisor |
| 📋 **Task Planner** | `/plan build a REST API` — breaks into steps, executes each |
| ✏️ **File Editing** | `/edit main.py fix the bug` — reads, edits, shows diff |
| 🔒 **Security Auditor** | 10+ OWASP checks (secrets, SQL injection, eval, weak crypto) |
| 🏖 **Sandbox Execution** | Isolated subprocess with restricted environment |
| 🔄 **Self-Debugging** | Errors feed back to model — auto-fixes with targeted guidance |
| 🎭 **Persona Switcher** | fullstack, infosec, architect, devops |
| 💾 **Session Memory** | `/sessions`, `/resume`, `.chakra_memory` project context |
| 🌲 **Workspace Aware** | Auto-scans files, builds context for better code |
| ⚡ **Streaming Output** | Tokens appear as generated — like a real conversation |
| 💻 **Git Integration** | `/git status`, `/git commit` from the REPL |
| 📊 **System Benchmark** | Auto-measures tokens/sec, configures optimal settings |

---

## 🎮 The Terminal

```
⟡ [SYSTEM] [INFO] Tuesday, August 05, 2026 00:16 | win32
⟡ [ENGINE] [INFO] llama.cpp GGUF backend (fastest)
⟡ [WORKSPACE] [INFO] project: main.py (234L), utils.py (45L)

(fullstack) > make a python calculator

  ⠋ Thinking about: make a python calculator...
  ✔ Thinking... 28 lines (0.9s)
  💾 chakra_output/generated_script.py → 28 lines
  ⚡ Sandbox execution → Exit 0

(fullstack) > /plan build a REST API with Flask
  Step 1/4: app.py with routes       ✔
  Step 2/4: database.py with models  ✔
  Step 3/4: auth.py with JWT         ✔
  Step 4/4: test_api.py              ✔

(fullstack) > /git status
  M app.py  M database.py

(fullstack) > /git commit -m "Add REST API modules"
  ✔ Committed: a1b2c3d
```

---

## 🔧 All Commands

| Command | What it does |
|---------|-------------|
| `<prompt>` | Code generation + sandbox execution + self-debug |
| `/plan <task>` | Multi-step task breakdown & execution |
| `/edit <file> <cmd>` | AI-powered file editing with diff preview |
| `/team <prompt>` | Multi-agent collaboration |
| `/persona [role]` | Switch persona |
| `/audit <file>` | OWASP security audit |
| `/scan-vuln` | Scan all Python files |
| `/git [cmd]` | Run git commands from REPL |
| `/memory [text]` | Save/load project context |
| `/context` / `/tree` | Workspace overview |
| `/status` | Engine, RAM, session info |
| `/sessions` / `/resume` | Session management |
| `/help` | All commands |
| `/exit` | Save and quit |

---

## 🏗 Three Engine Tiers

```
┌──────────────────────────────────────────────────────────────┐
│ ENGINE A            ENGINE B            ENGINE C (default)   │
│ kimi-k3-in-c        PyTorch K3          llama.cpp GGUF       │
│                                                                  
│ Full 2.78T model    Full 2.78T model    Qwen2.5-Coder 1.5B   │
│ 8.24 GB RAM         8-10 GB RAM         1 GB RAM             │
│ 1.56 TB disk        1.56 TB disk        1 GB disk            │
│ Linux only          Windows/Linux       Any OS               │
│ ─────────────────   ─────────────────   ─────────────────    │
│ --trunk <path>      --trunk <path>      pip install chakra-ai│
└──────────────────────────────────────────────────────────────┘
```

---

## 🔒 Security Auditing

10+ OWASP rules: hardcoded credentials, SQL injection, `eval()`/`exec()`, `os.system`, `subprocess(shell=True)`, MD5/SHA1, weak ciphers, unsafe deserialization, dynamic imports.

---

## ⚙️ CLI Flags

```bash
chakra [OPTIONS]
  --preset   {laptop,desktop,workstation,server}
  --engine   {auto,c-backend,pytorch,local}
  --gen      INT     Max tokens (default: 512)
  --prompt   TEXT    Single-shot mode
  --device   {cpu,cuda}
```

---

## 🌏 Made in India 🇮🇳

**Chakra AI is built in India.** "Chakra" (चक्र) means "wheel" — the continuous cycle of code generation, execution, auditing, and refinement.

| Chakra | Component | Role |
|--------|-----------|------|
| 💬 **Vishuddha** | Chat Engine | Understanding intent |
| 🧠 **Ajna** | Architect Agent | Blueprint design |
| ✋ **Manipura** | Coder Agent | Writing code |
| 🛡 **Anahata** | Auditor Agent | Security vigilance |
| 👑 **Sahasrara** | Supervisor | Orchestration |

**Author:** Abhirup Guha · **Organization:** Info Security Solution · **Web:** [insec.in](https://insec.in)

---

## 📖 Architecture

### Speed
- **llama.cpp GGUF** — Q4_K_M quantized, SIMD kernels, 300+ tok/s
- **Float16** — PyTorch fallback at half precision
- **Multi-threaded** — All CPU cores

### Kernel (Engine B — Full Kimi K3)
- **Fused MXFP4 Matmul** — 7.5x less memory traffic
- **Ring Buffer Trunk Streaming** — Pinned prefix + ring slot
- **Direct I/O** — `O_DIRECT` / `FILE_FLAG_NO_BUFFERING`

```
chakra-ai/
├── chakra/               # Core engine (19 files)
│   ├── agent.py          # Sandbox, self-debugging
│   ├── cli.py            # REPL, /plan, /edit, /git, /memory
│   ├── engine_llama.py   # llama.cpp GGUF backend
│   ├── engine_c_backend.py
│   ├── model.py          # PyTorch K3 + MXFP4Linear
│   ├── multi_agent.py    # Multi-agent orchestrator
│   ├── security.py       # InfoSecAuditor (OWASP)
│   └── ui.py             # Modern terminal UI
├── tests/                # Test suite
├── install.ps1           # Windows one-liner
├── setup.sh              # Linux one-liner
└── README.md
```

---

## 🙏 Acknowledgments

Built on the pioneering work of **Fareed Khan** and **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** — a 176 KB C99 binary proving 2.78T parameter inference on 8GB RAM.

---

## ⚖ License

MIT License · © 2026 Abhirup Guha · Info Security Solution · [insec.in](https://insec.in) · Made in India 🇮🇳
