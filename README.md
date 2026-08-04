# ⚡ Chakra AI

<div align="center">

**Agentic Coding Terminal · Multi-Engine MoE Inference · Made in India 🇮🇳**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-174%20passed-brightgreen.svg)](#-testing)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#-quick-start)
[![Made in India](https://img.shields.io/badge/made%20in-India-FF9933.svg)](#-made-in-india)

*A terminal where you talk to AI like a pair programmer. It writes code, runs it in a sandbox, finds security bugs, and fixes errors — all on an 8GB laptop.*

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

Chakra AI is a **terminal-based AI coding companion** built from scratch in India. You type what you want in plain English, and an AI agent writes code, executes it in an isolated sandbox, audits it for security vulnerabilities, and self-debugs if anything breaks — all **100% offline** on an 8GB RAM laptop.

```
You: "make a folder called Abhi and put a calculator in it"

  ⠋ Thinking about: make a folder called Abhi...
  ✔ Thinking... 12 lines (45.2s)
  💾 chakra_output/generated_script.py → 12 lines
  ⚡ Sandbox execution → Exit 0

Done. The folder 'Abhi' exists with calculator.py inside.
```

No API keys. No internet. No GPU. Just your terminal and an AI that builds things with you.

---

## ✨ Features

| Category | What It Does |
|----------|-------------|
| 🧠 **Multi-Agent Team** | Architect → Coder → Auditor → Supervisor collaborate to build complete projects |
| 🔒 **Security Auditor** | 10+ OWASP vulnerability checks (hardcoded secrets, SQL injection, eval, weak crypto) |
| 🏖 **Sandbox Execution** | Code runs in an isolated subprocess, never touches your system |
| 🔄 **Self-Debugging Loop** | If code fails, errors are fed back to the model to fix automatically |
| 🎭 **Persona Switcher** | Hot-swap between fullstack, infosec, architect, devops |
| 💾 **Persistent Sessions** | Save and resume conversations with `/sessions` and `/resume` |
| 🌲 **Workspace Awareness** | `/context` indexes your files, `/tree` shows your directory |
| ⚡ **Streaming Output** | Tokens appear as generated — like a real conversation |
| 📊 **System Benchmark** | Auto-measures tokens/sec and configures optimal settings |

---

## ❇️ Quick Start

```bash
# Clone
git clone https://github.com/fir3storm/chakra-ai.git
cd chakra-ai

# Install
pip install -e .
pip install hf_xet              # for fast model downloads

# Launch
start_chakra_ai.bat             # Windows double-click
python -m chakra.cli             # any OS terminal
```

On first launch, Chakra AI auto-downloads a 1.5B parameter coding model (~1.1 GB). It takes a few minutes with a decent internet connection. After that, everything runs completely offline.

---

## 🎮 The Terminal Experience

```
(fullstack) > hi

  ⠋ Thinking...
  Hello! I'm your coding assistant. What would you like to build today?
  ✔ Thinking... 3 chunks (2.1s)

(fullstack) > make a python calculator

  ⠋ Thinking about: make a python calculator...
  ✔ Thinking... 12 lines (45.2s)
  💾 chakra_output/generated_script.py → 12 lines
  ⚡ Sandbox execution → Exit 0

(fullstack) > /persona infosec
(fullstack) > /audit chakra_output/generated_script.py

  🛡 InfoSec Audit Report
  Target: chakra_output/generated_script.py
  ✔ PASS: No vulnerabilities detected. Score: 100/100

(fullstack) > /status

  Engine:   LocalModelRunner (Qwen2.5-Coder-1.5B)
  Device:   cpu
  Persona:  [INFOSEC] - InfoSec Expert & Security Auditor
  Session:  sess_53161
  RAM:      2847.3 MB
```

---

## 🏗 Three Engine Tiers

Chakra AI adapts to whatever hardware you have:

```
┌──────────────────────────────────────────────────────────┐
│ ENGINE A           ENGINE B           ENGINE C (default) │
│ kimi-k3-in-c       PyTorch K3         Qwen2.5-Coder      │
│ Full 2.78T model   Full 2.78T model   1.5B coding model  │
│ 8.24 GB RAM        8-10 GB RAM        2.5 GB RAM         │
│ 1.56 TB disk       1.56 TB disk       1.1 GB disk        │
│ Linux only         Windows/Linux      Any OS             │
│ Bit-exact verified atol 1e-4          Always works       │
│ ═════════════════  ═════════════════  ═════════════════   │
│ Setup: build_k3.sh --trunk <path>     Auto-downloads     │
└──────────────────────────────────────────────────────────┘
```

---

## 🔧 All Commands

| Command | What it does |
|---------|-------------|
| `type anything` | AI generates code, executes it, self-debugs |
| `/team <prompt>` | Multi-agent collaboration (Architect→Coder→Auditor) |
| `/persona [role]` | Switch persona: `fullstack`, `infosec`, `architect`, `devops` |
| `/audit <file>` | Security audit a single Python file |
| `/scan-vuln` | Scan all Python files in workspace for vulnerabilities |
| `/context` | Show workspace file index (counts, sizes) |
| `/tree` | Render directory tree |
| `/run <file>` | Execute a Python file in sandbox |
| `/diff [file]` | Show code diff between versions |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume a previous session |
| `/save <file>` | Save last generated code |
| `/status` | Show engine, RAM, session info |
| `/clear` | Clear terminal |
| `/help` | Show all commands |
| `/exit` | Save session and quit |

---

## 🔒 Security Auditing

Chakra AI scans every piece of generated code with 10+ OWASP rules:

- 🔑 **Hardcoded credentials** — passwords, API keys, tokens
- 💉 **SQL injection** — dynamic query construction
- ⚠️ **Code injection** — `eval()`, `exec()`, `__import__`
- 🔨 **Command injection** — `os.system`, `subprocess(shell=True)`
- 🔐 **Weak cryptography** — MD5, SHA1, DES
- 📦 **Unsafe deserialization** — `pickle.loads`, `yaml.unsafe_load`

Each finding gets a severity rating, line number, and actionable remediation advice.

---

## ⚙️ CLI Flags

```bash
python -m chakra.cli [OPTIONS]

  --preset   {laptop,desktop,workstation,server}  Hardware preset
  --engine   {auto,c-backend,pytorch,local}        Inference engine
  --trunk    PATH                                  Trunk weights (Engine A/B)
  --trunk-gb FLOAT                                 Memory budget for streaming
  --gen      INT                                   Max tokens (default: 192)
  --prompt   TEXT                                  Single-shot mode (no REPL)
  --tiny                                          13-layer test model
  --device   {cpu,cuda}                            Compute device
```

---

## 🧪 Testing

```bash
python -m pytest tests/ --ignore=tests/unit -v     # 174 tests

# Gate ladder — kernel correctness
python -m pytest tests/test_gate_ladder.py -v

# Numerical — bit-exact verification
python -m pytest tests/test_numerical.py -v

# Pipeline — multi-agent integration
python -m pytest tests/test_pipeline.py -v
```

**174 tests** covering gate ladder verification, numerical correctness, multi-agent pipeline, and CLI integration.

---

## 🌏 Made in India 🇮🇳

**Chakra AI is proudly built in India.** The name "Chakra" (चक्र) is a Sanskrit word meaning "wheel" or "cycle" — representing the continuous cycle of code generation, execution, auditing, and refinement that powers every interaction.

- **Author:** Abhirup Guha
- **Organization:** Info Security Solution
- **Location:** India

Built from scratch during the rise of open-source AI tooling. Zero external dependencies for the agentic pipeline. Every line of the agent system, multi-agent orchestrator, sandbox runner, and security auditor was written by hand. The AI model itself runs locally on your machine — your code never leaves your laptop.

### Why "Chakra"?

In Indian philosophy, chakras are energy centers that power the body. In Chakra AI, each component is a "chakra" powering the agent:

| Chakra | Component | Energy |
|--------|-----------|--------|
| 💬 **Vishuddha** (Throat) | Chat & Communication | Natural language understanding |
| 🧠 **Ajna** (Third Eye) | Architect Agent | Seeing the blueprint before building |
| ✋ **Manipura** (Solar Plexus) | Coder Agent | Taking action — writing real code |
| 🛡 **Anahata** (Heart) | Auditor Agent | Protecting through vigilance |
| 👑 **Sahasrara** (Crown) | Supervisor | Orchestrating the whole system |

---

## 📖 Technical Architecture

### Kernel-Level Optimizations

- **Fused MXFP4 Matmul** — Operates directly on packed 4-bit nibbles (17.55 MB/expert), 7.5x less memory traffic than dequantizing to float32
- **Ring Buffer Trunk Streaming** — Pinned prefix layers + ring slot for streaming remaining layers
- **Direct I/O Reader** — Linux `O_DIRECT` bypasses page cache; Windows `FILE_FLAG_NO_BUFFERING` via ctypes
- **Bit-Exact Verification** — Gate ladder tests comparing against PyTorch reference with atol=1e-4

### Repository Structure

```
chakra-ai/
├── chakra/                    # Core engine
│   ├── agent.py               # Sandbox, self-debugging loop
│   ├── cli.py                 # REPL terminal & commands
│   ├── engine_c_backend.py    # kimi-k3-in-c subprocess wrapper
│   ├── model.py               # PyTorch K3 + MXFP4Linear
│   ├── multi_agent.py         # Multi-agent orchestrator
│   ├── ops.py                 # Fused kernels, SiTU-GLU, KDA
│   ├── security.py            # InfoSecAuditor (OWASP)
│   ├── session.py             # Session persistence
│   ├── persona.py             # Persona management
│   ├── trunk_streamer.py      # Ring buffer streaming
│   ├── st_reader.py           # Safetensors + DirectReader
│   └── ui.py                  # Terminal UI (spinner, tools)
├── tests/                     # 174 tests
├── tools/
│   ├── benchmark.py           # System benchmark
│   └── download_model.py      # Model downloader
├── scripts/build_k3.sh        # Build kimi-k3-in-c
├── start_chakra_ai.bat        # Windows launcher
└── README.md
```

---

## 🙏 Standing on the Shoulders of Giants

Chakra AI builds upon the foundational work of **Fareed Khan** and his exceptional [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) project — a 176 KB C99 binary that proved a 2.78 trillion parameter model could run on 8GB of RAM. His innovations in zero-copy streaming, fused MXFP4 matmul, O_DIRECT I/O, and rigorous bit-exact verification set the gold standard for memory-efficient MoE inference.

Chakra AI extends that vision into a complete agentic coding system with multi-agent collaboration, security auditing, and an interactive terminal experience — while remaining honest about where it differs and improving where it can.

To Fareed Khan: thank you for showing what's possible. 🙏

---

## ⚖ License

MIT License · Copyright © 2026 Abhirup Guha · Info Security Solution · Made in India 🇮🇳
