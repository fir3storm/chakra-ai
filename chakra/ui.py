# chakra/ui.py
"""
chakra.ui - Terminal UI formatting and styling components for KimiPy.

Author & Creator: Abhirup Guha (Info Security Solution)
"""
from __future__ import annotations

import os
import sys

# Enable ANSI escape sequence processing and UTF-8 console output on Windows
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _safe_print(text: str) -> None:
    """Safely prints text handling potential encoding limitations on legacy Windows terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
            sys.stdout.buffer.write(encoded + b"\n")
            sys.stdout.buffer.flush()
        except Exception:
            # Fallback ascii strip
            print(text.encode("ascii", errors="replace").decode("ascii"))


class Colors:
    """ANSI Escape Codes for Terminal Styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def clear_screen() -> None:
    """Screen clearing helper using cls on Windows (or clear on Unix)."""
    os.system("cls" if os.name == "nt" else "clear")


def print_banner() -> None:
    """Render ASCII banner with branding for Chakra AI."""
    banner_art = r"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║   ____ _   _    _    _  ___  _       _    ___                             ║
  ║  / ___| | | |  / \  | |/ / |  _ \   / \  |_ _|   CHAKRA AI               ║
  ║ | |   | |_| | / _ \ | ' /| |_) | / _ \  | |    Agentic Code Terminal     ║
  ║ | |___|  _  |/ ___ \| . \|  _ < / ___ \ | |    Multi-Engine MoE          ║
  ║  \____|_| |_/_/   \_\_|\_\_|_| \_\_/   \_\___|                            ║
  ║                                                                           ║
  ║  Author: Abhirup Guha (Info Security Solution)                            ║
  ║  github.com/fir3storm/chakra-ai                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    c = Colors
    _safe_print(f"{c.BRIGHT_CYAN}{c.BOLD}{banner_art}{c.RESET}")


def print_step(
    tag: str,
    message: str = "",
    status: str | None = None,
) -> None:
    """
    Print formatted step indicators like [✦ Planning], [⚡ Generating],
    [⚙ Executing Sandbox], [✔ Verified], [❌ Failed].

    Args:
        tag: Step indicator or tag name (e.g. "✦ Planning", "[⚡ Generating]", "AGENT").
        message: Informational message text.
        status: Optional status indicator ("INFO", "SUCCESS", "FAIL", "WARN", "WAIT", etc.).
    """
    c = Colors

    if status is not None:
        # Legacy/CLI mode with explicit status badge
        status_map = {
            "INFO": f"{c.BRIGHT_BLUE}[INFO]{c.RESET}",
            "SUCCESS": f"{c.BRIGHT_GREEN}[SUCCESS]{c.RESET}",
            "FAIL": f"{c.BRIGHT_RED}[FAIL]{c.RESET}",
            "WARN": f"{c.BRIGHT_YELLOW}[WARN]{c.RESET}",
            "WAIT": f"{c.BRIGHT_MAGENTA}[RUNNING]{c.RESET}",
        }
        badge = status_map.get(status.upper(), f"[{status}]")
        prefix = f"{c.BRIGHT_CYAN}{c.BOLD}⟡ [{tag}]{c.RESET}"

        if message:
            _safe_print(f"{prefix} {badge} {c.WHITE}{message}{c.RESET}")
        else:
            _safe_print(f"{prefix} {badge}")
    else:
        # Standard step indicator mode (e.g. [✦ Planning], [⚡ Generating], etc.)
        formatted_tag = tag if (tag.startswith("[") and tag.endswith("]")) else f"[{tag}]"

        # Determine color based on icon or keyword in tag
        tag_upper = tag.upper()
        if "✦" in tag or "PLANNING" in tag_upper:
            color = f"{c.BRIGHT_CYAN}{c.BOLD}"
        elif "⚡" in tag or "GENERATING" in tag_upper:
            color = f"{c.BRIGHT_YELLOW}{c.BOLD}"
        elif "⚙" in tag or "EXECUTING" in tag_upper or "SANDBOX" in tag_upper:
            color = f"{c.BRIGHT_MAGENTA}{c.BOLD}"
        elif "✔" in tag or "VERIFIED" in tag_upper or "SUCCESS" in tag_upper:
            color = f"{c.BRIGHT_GREEN}{c.BOLD}"
        elif "❌" in tag or "FAILED" in tag_upper or "ERROR" in tag_upper or "FAIL" in tag_upper:
            color = f"{c.BRIGHT_RED}{c.BOLD}"
        else:
            color = f"{c.BRIGHT_BLUE}{c.BOLD}"

        if message:
            _safe_print(f"{color}{formatted_tag}{c.RESET} {c.WHITE}{message}{c.RESET}")
        else:
            _safe_print(f"{color}{formatted_tag}{c.RESET}")


def print_agent_step(role: str, message: str = "") -> None:
    """
    Print formatted status badges for team sub-agents (Architect, Coder, Auditor, Supervisor).

    Args:
        role: Team agent role name ('Architect', 'Coder', 'Auditor', 'Supervisor').
        message: Informational status message text.
    """
    c = Colors
    role_norm = role.strip().capitalize()

    role_badges = {
        "Architect": f"{c.BRIGHT_CYAN}{c.BOLD}[🏛 Architect]{c.RESET}",
        "Coder": f"{c.BRIGHT_YELLOW}{c.BOLD}[💻 Coder]{c.RESET}",
        "Auditor": f"{c.BRIGHT_MAGENTA}{c.BOLD}[🔍 Auditor]{c.RESET}",
        "Supervisor": f"{c.BRIGHT_GREEN}{c.BOLD}[👑 Supervisor]{c.RESET}",
    }

    badge = role_badges.get(role_norm, f"{c.BRIGHT_BLUE}{c.BOLD}[🤖 {role_norm}]{c.RESET}")

    if message:
        _safe_print(f"{badge} {c.WHITE}{message}{c.RESET}")
    else:
        _safe_print(f"{badge}")



def print_code_box(code_text: str, title: str = "Generated Code") -> None:
    """
    Draw box-bordered panels around code blocks (┌─── Title ───┐ ... └───────────┘).

    Args:
        code_text: Code string to present in the box panel.
        title: Title to display in the top border header.
    """
    c = Colors
    lines = code_text.splitlines() if code_text else ["# No code generated"]
    max_len = max((len(line) for line in lines), default=30)

    header_text = f"─── {title} ───"
    box_width = max(max_len + 4, len(header_text) + 4, 50)
    padding_dashes = box_width - len(header_text) - 2

    top_border = f"┌{header_text}" + "─" * max(0, padding_dashes) + "┐"
    bottom_border = "└" + "─" * (box_width - 2) + "┘"

    _safe_print(f"{c.BRIGHT_YELLOW}{c.BOLD}{top_border}{c.RESET}")
    for line in lines:
        padded_line = line.ljust(box_width - 4)
        _safe_print(f"{c.BRIGHT_BLACK}│{c.RESET} {c.BRIGHT_GREEN}{padded_line}{c.RESET} {c.BRIGHT_BLACK}│{c.RESET}")
    _safe_print(f"{c.BRIGHT_YELLOW}{c.BOLD}{bottom_border}{c.RESET}")


def print_diff_box(diff_text: str, title: str = "Code Diff Preview") -> None:
    """
    Draw formatted box-bordered panel for unified git/code diffs with syntax coloring.

    Args:
        diff_text: Unified diff text string.
        title: Header title for the panel.
    """
    c = Colors
    lines = diff_text.splitlines() if diff_text and diff_text.strip() else ["# No changes / empty diff"]
    max_len = max((len(line) for line in lines), default=30)

    header_text = f"─── {title} ───"
    box_width = max(max_len + 4, len(header_text) + 4, 60)
    padding_dashes = box_width - len(header_text) - 2

    top_border = f"┌{header_text}" + "─" * max(0, padding_dashes) + "┐"
    bottom_border = "└" + "─" * (box_width - 2) + "┘"

    _safe_print(f"{c.BRIGHT_CYAN}{c.BOLD}{top_border}{c.RESET}")
    for line in lines:
        padded_line = line.ljust(box_width - 4)
        if line.startswith("+") and not line.startswith("+++"):
            line_str = f"{c.BRIGHT_GREEN}{padded_line}{c.RESET}"
        elif line.startswith("-") and not line.startswith("---"):
            line_str = f"{c.BRIGHT_RED}{padded_line}{c.RESET}"
        elif line.startswith("@"):
            line_str = f"{c.BRIGHT_MAGENTA}{c.BOLD}{padded_line}{c.RESET}"
        elif line.startswith("---") or line.startswith("+++"):
            line_str = f"{c.BRIGHT_YELLOW}{c.BOLD}{padded_line}{c.RESET}"
        else:
            line_str = f"{c.WHITE}{padded_line}{c.RESET}"

        _safe_print(f"{c.BRIGHT_BLACK}│{c.RESET} {line_str} {c.BRIGHT_BLACK}│{c.RESET}")
    _safe_print(f"{c.BRIGHT_CYAN}{c.BOLD}{bottom_border}{c.RESET}")


def print_vuln_report(filepath: str, findings: List[Dict[str, str]]) -> None:
    """
    Render formatted InfoSec vulnerability and static code audit report.

    Args:
        filepath: Target file or scope analyzed.
        findings: List of vulnerability dictionaries with keys: line, severity, category, title, description.
    """
    c = Colors
    _safe_print(f"\n{c.BRIGHT_MAGENTA}{c.BOLD}═════════════════════════════════════════════════════════════════════════════{c.RESET}")
    _safe_print(f"{c.BRIGHT_MAGENTA}{c.BOLD} 🛡 InfoSec Security Vulnerability Audit Report{c.RESET}")
    _safe_print(f"{c.BRIGHT_CYAN} Target: {c.WHITE}{filepath}{c.RESET}")
    _safe_print(f"{c.BRIGHT_MAGENTA}{c.BOLD}═════════════════════════════════════════════════════════════════════════════{c.RESET}")

    if not findings:
        _safe_print(f"{c.BRIGHT_GREEN}{c.BOLD} [✔] PASS: No security vulnerabilities or risk patterns detected.{c.RESET}\n")
        return

    sev_colors = {
        "CRITICAL": f"{c.BRIGHT_RED}{c.BOLD}[CRITICAL]{c.RESET}",
        "HIGH": f"{c.RED}{c.BOLD}[HIGH]{c.RESET}",
        "MEDIUM": f"{c.BRIGHT_YELLOW}{c.BOLD}[MEDIUM]{c.RESET}",
        "LOW": f"{c.BRIGHT_BLUE}[LOW]{c.RESET}",
        "INFO": f"{c.BRIGHT_CYAN}[INFO]{c.RESET}",
    }

    for idx, f in enumerate(findings, 1):
        sev = f.get("severity", "MEDIUM").upper()
        badge = sev_colors.get(sev, f"[{sev}]")
        line_info = f" (Line {f['line']})" if f.get("line") else ""
        _safe_print(f" {idx}. {badge} {c.BOLD}{f.get('title', 'Security Finding')}{c.RESET}{line_info}")
        _safe_print(f"    Category: {c.BRIGHT_YELLOW}{f.get('category', 'Code Security')}{c.RESET}")
        _safe_print(f"    Details : {c.WHITE}{f.get('description', '')}{c.RESET}")
        if f.get("remediation"):
            _safe_print(f"    Fix     : {c.BRIGHT_GREEN}{f.get('remediation')}{c.RESET}")
        _safe_print(f"{c.BRIGHT_BLACK} ─────────────────────────────────────────────────────────────────────────────{c.RESET}")
    _safe_print("")


def print_sessions_list(sessions: List[Dict[str, str]], active_session_id: Optional[str] = None) -> None:
    """
    Render formatted list of saved REPL chat sessions.

    Args:
        sessions: List of session metadata dicts (id, timestamp, prompt_count, title).
        active_session_id: Currently active session ID string if any.
    """
    c = Colors
    _safe_print(f"\n{c.BRIGHT_CYAN}{c.BOLD}┌── Saved REPL Sessions List ─────────────────────────────────────────────┐{c.RESET}")
    if not sessions:
        _safe_print(f"{c.BRIGHT_BLACK}│  No saved sessions found.                                              │{c.RESET}")
    else:
        for s in sessions:
            sid = s.get("id", "unknown")
            is_active = "*" if sid == active_session_id else " "
            title = s.get("title", "Untitled Session")[:35]
            ts = s.get("timestamp", "")
            msgs = s.get("message_count", 0)
            status_tag = f"{c.BRIGHT_GREEN}[ACTIVE]{c.RESET}" if sid == active_session_id else f"{c.BRIGHT_BLACK}[SAVED]{c.RESET}"
            _safe_print(f"{c.BRIGHT_BLACK}│{c.RESET} {is_active} {c.BRIGHT_YELLOW}{sid:<8}{c.RESET} | {status_tag} | {ts:<16} | {msgs:>2} msgs | {c.WHITE}{title}{c.RESET}")
    _safe_print(f"{c.BRIGHT_CYAN}{c.BOLD}└─────────────────────────────────────────────────────────────────────────┘{c.RESET}\n")

