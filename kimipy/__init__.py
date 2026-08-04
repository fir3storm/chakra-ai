"""
kimipy: Python Engine & Streaming Runtime for Kimi K3 on Windows.

Provides fast Safetensors parsing, single-layer trunk streaming, MXFP4 dequantization,
and MoE expert LRU caching designed to run Kimi K3 efficiently on consumer Windows hardware with 8GB RAM constraints.
"""

from .agent import KimiAgent, LocalModelRunner, apply_diff, execute_sandbox, generate_diff, run_in_sandbox
from .config import KimiConfig, load_config
from .dequant import E2M1_TABLE, dequantize_mxfp4_numpy, dequantize_mxfp4_torch
from .expert_cache import ExpertLRUCache
from .multi_agent import ArchitectAgent, AuditorAgent, CoderAgent, MultiAgentOrchestrator
from .persona import PersonaManager
from .security import InfoSecAuditor
from .session import SessionManager
from .st_reader import SafetensorsReader, TensorInfo
from .tokenizer import KimiTokenizer
from .trunk_streamer import TrunkStreamer
from .ui import clear_screen, print_agent_step, print_banner, print_code_box, print_step
from .workspace import WorkspaceIndexer

__version__ = "0.1.0"

__all__ = [
    "KimiAgent",
    "LocalModelRunner",
    "ArchitectAgent",
    "CoderAgent",
    "AuditorAgent",
    "MultiAgentOrchestrator",
    "WorkspaceIndexer",
    "SessionManager",
    "InfoSecAuditor",
    "PersonaManager",
    "generate_diff",
    "apply_diff",
    "run_in_sandbox",
    "execute_sandbox",
    "KimiConfig",
    "load_config",
    "SafetensorsReader",
    "TensorInfo",
    "TrunkStreamer",
    "ExpertLRUCache",
    "E2M1_TABLE",
    "dequantize_mxfp4_numpy",
    "dequantize_mxfp4_torch",
    "KimiTokenizer",
    "clear_screen",
    "print_agent_step",
    "print_banner",
    "print_code_box",
    "print_step",
]

