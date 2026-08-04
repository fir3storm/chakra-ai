# Chakra AI

> **Agentic Coding Terminal with Multi-Engine Kimi K3 Support**  
> **Author & Creator:** Abhirup Guha  
> **Organization:** Info Security Solution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-174%20passed-brightgreen.svg)](#testing)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#installation)

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

## What is Chakra AI?

Chakra AI is an **agentic coding terminal** that combines AI-powered code generation with multi-agent collaboration, security auditing, and sandboxed execution. It runs on standard consumer hardware with **8GB RAM**.

**Key capabilities:**
- Multi-agent team collaboration (Architect, Coder, Auditor, Supervisor)
- Automatic code generation, sandbox execution, and self-debugging
- InfoSec static security analysis (10+ OWASP rules)
- Persistent sessions and persona switching
- Three inference engine tiers for different hardware

---

## Three Engine Tiers

### Engine A: kimi-k3-in-c (Linux, Full Kimi K3)
- **Model**: Full 2.78 trillion parameter Kimi K3
- **RAM**: 8.24 GB peak (measured)
- **Storage**: 1.56 TB checkpoint + 109 GB packed trunk
- **Speed**: ~32 s/token (disk I/O bound)
- **Verification**: Bit-exact against PyTorch reference
- **Setup**: `scripts/build_k3.sh`, then `--trunk <path> --engine c-backend`

### Engine B: PyTorch (Windows, Full Kimi K3)
- **Model**: Same Kimi K3, PyTorch implementation
- **RAM**: ~8-10 GB (trunk streaming with ring buffer)
- **Storage**: 1.56 TB checkpoint
- **Features**: Fused MXFP4 matmul (7.5x less memory traffic), ring buffer streaming, direct I/O
- **Setup**: `--trunk <path> --engine pytorch`

### Engine C: Qwen2.5-Coder (Any OS, Lightweight) — Default
- **Model**: Qwen2.5-Coder-1.5B-Instruct (~1.1 GB)
- **RAM**: ~2.5 GB total
- **Speed**: ~4 tokens/sec on CPU
- **Features**: Fluent code generation, chat, 100% offline
- **Setup**: Auto-downloads on first launch

---

## Quick Start

```bash
# Clone
git clone https://github.com/fir3storm/chakra-ai.git
cd chakra-ai

# Install
pip install -e .

# Launch (Windows)
start_chakra_ai.bat

# Launch (any OS)
python -m kimipy.cli
```

---

## Agentic Features

### Multi-Agent Pipeline
```
User: "make a folder called Abhi and put a calculator in it"
  ↓
Architect → Designs blueprint (config → engine → main)
  ↓
Coder → Generates Python code for each module
  ↓
Auditor → Security audit + sandbox execution
  ↓
Result: Code saved to file, executed, output shown
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `<prompt>` | Natural language code generation + execution |
| `/team <prompt>` | Multi-agent team collaboration |
| `/persona [role]` | Switch persona (fullstack, infosec, architect, devops) |
| `/audit <file>` | Security audit a single file |
| `/scan-vuln` | Scan workspace for vulnerabilities |
| `/context` | Show workspace file index |
| `/tree` | Show directory tree |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume a saved session |
| `/status` | Show engine info, RAM usage |
| `/help` | Show all commands |

### Personas

| Persona | Focus |
|---------|-------|
| **fullstack** | General software engineering |
| **infosec** | Security auditing, OWASP vulnerabilities |
| **architect** | System design, clean architecture |
| **devops** | Deployment, automation, infrastructure |

---

## Security Auditing

Chakra AI includes `InfoSecAuditor` with 10+ OWASP security rules:

- Hardcoded credentials/secrets
- SQL injection via dynamic queries
- `eval()`/`exec()` code injection
- `os.system` / `subprocess(shell=True)` command injection
- Weak cryptography (MD5, SHA1)
- Unsafe deserialization (pickle, yaml.unsafe_load)

```bash
(fullstack) > /audit kimipy/agent.py
(fullstack) > /scan-vuln
```

---

## Kernel-Level Optimizations

### Fused MXFP4 Matmul
- Operates on packed nibbles (17.55 MB/expert) instead of dequantizing to float32 (132 MB)
- 7.5x less memory traffic per expert
- Group-by-group accumulation with shared scale

### Ring Buffer Trunk Streaming
- Pinned prefix layers (first N layers always in RAM)
- Ring slot for streaming remaining layers
- Configurable via `--trunk-gb`

### Direct I/O Reader
- Linux: `O_DIRECT` bypasses page cache (5.9 GB/s measured)
- Windows: `FILE_FLAG_NO_BUFFERING` via ctypes
- Graceful fallback to mmap

### Bit-Exact Verification
- Gate ladder tests: tokenizer parity, config validation, MXFP4 dequantization, model forward pass
- Numerical equivalence: atol=1e-4 against dequant-first path

---

## One-Time System Benchmark

On first launch, Chakra AI can measure your hardware's tokens/sec and auto-configure optimal settings:

```bash
python tools/benchmark.py
```

Results are cached in `.chakra_benchmark.json` and used automatically.

---

## CLI Flags

```
python -m kimipy.cli [OPTIONS]

Options:
  --preset {laptop,desktop,workstation,server}  Hardware preset (default: laptop)
  --engine {auto,c-backend,pytorch,local}       Inference engine (default: auto)
  --trunk PATH                                  Path to trunk weights (Engine A/B)
  --trunk-gb FLOAT                              Trunk memory budget for ring buffer
  --local-model PATH                            Path to local model directory
  --gen INT                                     Max tokens to generate (default: 192)
  --prompt TEXT                                 Single prompt mode (no REPL)
  --tiny                                        Use 13-layer test model
  --device {cpu,cuda}                           Compute device
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ --ignore=tests/unit -v

# Run specific test suites
python -m pytest tests/test_gate_ladder.py -v    # Kernel verification
python -m pytest tests/test_numerical.py -v       # Bit-exact tests
python -m pytest tests/test_pipeline.py -v        # Multi-agent pipeline
python -m pytest tests/test_agent.py -v           # Agent & sandbox
```

174 tests covering:
- Gate ladder verification (tokenizer, config, MXFP4, model forward, sandbox)
- Numerical equivalence (RMSNorm, SiTU-GLU, KDA decay, router bias)
- Multi-agent pipeline (architect blueprints, auditor, orchestrator)
- CLI consolidation (sessions, personas, engine selection)

---

## Repository Structure

```
chakra-ai/
├── kimipy/                    # Core engine package
│   ├── agent.py               # Agent, sandbox, self-debugging loop
│   ├── cli.py                 # REPL terminal & commands
│   ├── engine_c_backend.py    # kimi-k3-in-c subprocess wrapper
│   ├── model.py               # PyTorch Kimi K3 architecture + MXFP4Linear
│   ├── multi_agent.py         # Multi-agent orchestrator
│   ├── ops.py                 # Fused MXFP4 matmul, RMSNorm, SiTU-GLU, KDA
│   ├── security.py            # InfoSecAuditor (OWASP rules)
│   ├── session.py             # Session persistence
│   ├── persona.py             # Persona management
│   ├── trunk_streamer.py      # Ring buffer trunk streaming
│   ├── st_reader.py           # Safetensors reader + DirectReader
│   └── ui.py                  # Terminal UI
├── tests/                     # Test suite (174 tests)
├── tools/
│   ├── benchmark.py           # One-time system benchmark
│   └── download_model.py      # Model downloader
├── scripts/
│   └── build_k3.sh            # Build kimi-k3-in-c
├── start_chakra_ai.bat        # Windows launcher
└── README.md
```

---

## Acknowledgments

Chakra AI builds upon the pioneering work of **Fareed Khan** and his [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) project.

Fareed Khan's C implementation demonstrated that the 2.78 trillion parameter Kimi K3 model could run on a single CPU with 8GB RAM through zero-copy weight streaming and layer-by-layer execution. His work on fused MXFP4 matmul, O_DIRECT I/O, and bit-exact verification set the standard for memory-efficient MoE inference.

Chakra AI extends this vision with:
- Cross-platform Python implementation (Windows, Linux, macOS)
- Agentic coding pipeline (multi-agent collaboration)
- Security auditing (OWASP static analysis)
- Session management and persona switching

We are grateful for Fareed Khan's foundational contribution to the open-source AI community.

**Original project:** [github.com/FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)

---

## License

MIT License - see [LICENSE](LICENSE)

**Author:** Abhirup Guha  
**Organization:** Info Security Solution
