# ⚡ Chakra AI

<div align="center">

**Agentic Coding Terminal · Multi-Engine MoE Inference · Made in India 🇮🇳**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#-quick-start)
[![Made in India](https://img.shields.io/badge/made%20in-India-FF9933.svg)](#-made-in-india)

*A terminal where you talk to AI like a pair programmer. Writes code, runs it in a sandbox, audits for security bugs, self-debugs — all 100% offline on 8GB RAM.* ⚡

</div>

---

```
  ╔═══════════════════════════════════════════════════════════════╗
  ║   ____ _   _    _    _  ___  _       _    ___                ║
  ║  / ___| | | |  / \  | |/ / |  _ \   / \  |_ _|  CHAKRA AI   ║
  ║ | |   | |_| | / _ \ | ' /| |_) | / _ \  | |   Agentic Code  ║
  ║ | |___|  _  |/ ___ \| . \|  _ < / ___ \ | |   Terminal       ║
  ║  \____|_| |_/_/   \_\_|\_\_|_| \_\_/   \_\___|               ║
  ╚═══════════════════════════════════════════════════════════════╝
```

---

## 🎯 What is Chakra AI?

Chakra AI is a **terminal-based AI coding companion** built from scratch in India. Type what you want in plain English, and an AI agent writes code, executes it in an isolated sandbox, audits it for security vulnerabilities, and self-debugs if anything breaks — all **100% offline** on an 8GB RAM laptop.

```
You: "make a folder called Abhi and put a calculator in it"

  ⠋ Thinking about: make a folder called Abhi...
  ✔ Thinking... 28 lines (0.9s)
  💾 chakra_output/generated_script.py → 28 lines
  ⚡ Sandbox execution → Exit 0

Done. Folder 'Abhi' created with calculator.py inside.
```

No API keys. No internet. No GPU. Just your terminal, an AI, and **~1 GB RAM**.

---

## ❇️ Quick Start

**One command to set up everything:**

```bash
# Windows (double-click)
setup.bat

# Linux / macOS
bash setup.sh
```

This installs all dependencies, compiles the fast inference engine, downloads the model (~1 GB), and benchmarks your system. Takes ~5 minutes.

**Launch:**
```bash
start_chakra_ai.bat       # Windows
python -m chakra.cli       # any OS
```

> ⚡ **Speed:** On a standard laptop CPU, Chakra AI generates ~300 tokens/sec using llama.cpp with a GGUF-quantized model — code generation takes under 1 second.

---

## ✨ Features

| Category | What It Does |
|----------|-------------|
| ⚡ **Blazing Fast** | 300+ tok/s via llama.cpp with GGUF quantized models — ~1 GB RAM |
| 🧠 **Multi-Agent Team** | Architect → Coder → Auditor → Supervisor collaborate to build projects |
| 🔒 **Security Auditor** | 10+ OWASP vulnerability checks (hardcoded secrets, SQL injection, eval, weak crypto) |
| 🏖 **Sandbox Execution** | Code runs in an isolated subprocess with restricted environment |
| 🔄 **Self-Debugging Loop** | If code fails, errors feed back to the model to fix automatically |
| 🎭 **Persona Switcher** | Hot-swap between fullstack, infosec, architect, devops |
| 💾 **Persistent Sessions** | Save and resume conversations with `/sessions` and `/resume` |
| 🌲 **Workspace Awareness** | `/context` indexes your files, `/tree` shows your directory |
| ⚡ **Streaming Output** | Tokens appear as generated — like a real conversation |
| 📊 **System Benchmark** | Auto-measures tokens/sec and configures optimal settings |
| 💻 **Built for 8GB RAM** | Runs comfortably on consumer laptops with 1 GB for the model |

---

## 🎮 The Terminal Experience

```
(fullstack) > hi

  ⠋ Thinking...
  Hello! I'm your coding assistant. What would you like to build today?
  ✔ Thinking... 3 chunks (0.3s)

(fullstack) > make a python calculator

  ⠋ Thinking about: make a python calculator...
  ✔ Thinking... 28 lines (0.9s)
  💾 chakra_output/generated_script.py → 28 lines
  ⚡ Sandbox execution → Exit 0

(fullstack) > /persona infosec
(fullstack) > /audit chakra_output/generated_script.py

  🛡 InfoSec Audit Report
  Target: chakra_output/generated_script.py
  ✔ PASS: Score: 100/100

(fullstack) > /status

  Engine:   llama.cpp (Qwen2.5-Coder-1.5B Q4_K_M)
  Threads:  22
  Persona:  [INFOSEC] - InfoSec Expert
  RAM:      ~1 GB
```

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
│ Bit-exact verified  atol 1e-4           300+ tok/s           │
│ ─────────────────   ─────────────────   ─────────────────    │
│ --trunk <path>      --trunk <path>      setup.bat / setup.sh │
└──────────────────────────────────────────────────────────────┘
```

**Engine C** ships with a `setup.bat`/`setup.sh` one-click installer.  
**Engine A** runs the legendary 176 KB `kimi-k3-in-c` binary as a subprocess.  
**Engine B** is the PyTorch implementation for Windows when you have the full checkpoint.

---

## 🔧 All Commands

| Command | What it does |
|---------|-------------|
| `type anything` | AI generates code, executes it, self-debugs |
| `/team <prompt>` | Multi-agent collaboration (Architect→Coder→Auditor) |
| `/persona [role]` | Switch persona: `fullstack`, `infosec`, `architect`, `devops` |
| `/audit <file>` | OWASP security audit a Python file |
| `/scan-vuln` | Scan all Python files for vulnerabilities |
| `/context` | Workspace file index (counts, sizes) |
| `/tree` | Directory tree |
| `/run <file>` | Execute a Python file in sandbox |
| `/diff [file]` | Code diff between versions |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume a previous session |
| `/status` | Engine, threads, RAM, session info |
| `/clear` | Clear terminal |
| `/help` | Show all commands |
| `/exit` | Save session and quit |

---

## 🔒 Security Auditing

Chakra AI scans every generated code file with 10+ OWASP rules:

- 🔑 **Hardcoded credentials** — passwords, API keys, tokens
- 💉 **SQL injection** — dynamic query construction
- ⚠️ **Code injection** — `eval()`, `exec()`, `__import__`
- 🔨 **Command injection** — `os.system`, `subprocess(shell=True)`
- 🔐 **Weak cryptography** — MD5, SHA1, DES
- 📦 **Unsafe deserialization** — `pickle.loads`, `yaml.unsafe_load`

Each finding gets a severity rating, line number, and actionable remediation.

---

## ⚙️ CLI Flags

```bash
python -m chakra.cli [OPTIONS]

  --preset   {laptop,desktop,workstation,server}  Hardware preset
  --engine   {auto,c-backend,pytorch,local}        Inference engine
  --trunk    PATH                                  Trunk weights (Engine A/B)
  --trunk-gb FLOAT                                 Memory budget for streaming
  --gen      INT                                   Max tokens (default: 512)
  --prompt   TEXT                                  Single-shot mode (no REPL)
  --tiny                                          13-layer test model
  --device   {cpu,cuda}                            Compute device
```

---

## 🌏 Made in India 🇮🇳

**Chakra AI is proudly built in India.** The name "Chakra" (चक्र) is a Sanskrit word meaning "wheel" or "cycle" — representing the continuous cycle of code generation, execution, auditing, and refinement.

- **Author:** Abhirup Guha
- **Organization:** Info Security Solution
- **Location:** India

### Why "Chakra"?

In Indian philosophy, chakras are energy centers. Each Chakra AI component maps to one:

| Chakra | Component | Role |
|--------|-----------|------|
| 💬 **Vishuddha** (Throat) | Chat Engine | Understanding your intent |
| 🧠 **Ajna** (Third Eye) | Architect Agent | Seeing the blueprint first |
| ✋ **Manipura** (Solar Plexus) | Coder Agent | Writing real code |
| 🛡 **Anahata** (Heart) | Auditor Agent | Protecting through vigilance |
| 👑 **Sahasrara** (Crown) | Supervisor | Orchestrating the whole system |

---

## 📖 Technical Architecture

### Speed Optimizations
- **llama.cpp GGUF backend** — Q4_K_M quantized model, SIMD-optimized C++ kernels, 300+ tok/s
- **Float16 precision** — PyTorch fallback at half precision, 3 GB RAM instead of 6 GB
- **Multi-threaded** — Uses all CPU cores for matrix operations
- **`torch.inference_mode()`** — Faster than `no_grad()` for generation

### Kernel-Level (Engine B — Full Kimi K3)
- **Fused MXFP4 Matmul** — Packed 4-bit nibbles, 7.5x less memory traffic
- **Ring Buffer Trunk Streaming** — Pinned prefix + ring slot for 1.56 TB checkpoint
- **Direct I/O Reader** — `O_DIRECT` (Linux), `FILE_FLAG_NO_BUFFERING` (Windows)
- **Bit-Exact Verification** — Gate ladder tests at atol=1e-4

### Repository Structure (48 files)

```
chakra-ai/
├── chakra/                     # Core engine (18 files)
│   ├── agent.py                # Sandbox, self-debugging
│   ├── cli.py                  # REPL terminal & commands
│   ├── engine_c_backend.py     # kimi-k3-in-c wrapper
│   ├── engine_llama.py         # llama.cpp GGUF backend ⚡
│   ├── model.py                # PyTorch K3 + MXFP4Linear
│   ├── multi_agent.py          # Multi-agent orchestrator
│   ├── ops.py                  # Fused kernels, SiTU-GLU, KDA
│   ├── security.py             # InfoSecAuditor (OWASP)
│   ├── session.py              # Session persistence
│   ├── persona.py              # Persona management
│   ├── trunk_streamer.py       # Ring buffer streaming
│   ├── st_reader.py            # Safetensors + DirectReader
│   └── ui.py                   # Spinner, tools, chat roles
├── tests/                      # Test suite
├── tools/                      # Benchmark + model downloader
├── setup.bat / setup.sh        # One-click installer
├── start_chakra_ai.bat         # Windows launcher
└── README.md
```

---

## 🙏 Standing on the Shoulders of Giants

Chakra AI builds upon the pioneering work of **Fareed Khan** and his remarkable **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** project — a 176 KB C99 binary proving a 2.78 trillion parameter model runs on 8GB RAM. His innovations in zero-copy streaming, fused MXFP4 matmul, O_DIRECT I/O, and bit-exact verification set the gold standard.

Chakra AI extends that vision with a complete agentic coding system while remaining honest about where it differs. To Fareed Khan: thank you. 🙏

---

## ⚖ License

MIT License · Copyright © 2026 Abhirup Guha · Info Security Solution · Made in India 🇮🇳
