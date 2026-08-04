"""
Workspace Indexer for scanning project source files, extracting signatures, building context trees, and formatting LLM context summaries.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class WorkspaceIndexer:
    """
    Scans a workspace directory for source files (.py, .json, .yaml, .md, .sql),
    extracts function and class signatures, file trees, and builds an in-memory context index.
    """

    DEFAULT_EXTENSIONS = (".py", ".json", ".yaml", ".yml", ".md", ".sql")
    DEFAULT_EXCLUDE_DIRS = (
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".gemini",
        "brain",
        "tmp",
        ".idea",
        ".vscode",
    )

    def __init__(
        self,
        root_dir: Union[str, Path] = ".",
        exclude_dirs: Optional[Union[List[str], Tuple[str, ...]]] = None,
        extensions: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.exclude_dirs = set(exclude_dirs if exclude_dirs is not None else self.DEFAULT_EXCLUDE_DIRS)
        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions if extensions is not None else self.DEFAULT_EXTENSIONS)
        )
        self.index: Dict[str, Any] = {}
        self.scanned: bool = False

    def scan_workspace(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        extensions: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ) -> Dict[str, Any]:
        """
        Scans the current project directory for source files, extracts structural metadata
        (function signatures, AST definitions, file stats, documentation headers),
        and returns an in-memory context index.
        """
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()

        if extensions is not None:
            self.extensions = tuple(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in extensions
            )

        file_index: Dict[str, Any] = {}
        dir_tree: Dict[str, Any] = {}

        if not self.root_dir.exists():
            self.index = {
                "root_dir": str(self.root_dir),
                "files": {},
                "file_count": 0,
                "tree": {},
            }
            self.scanned = True
            return self.index

        for entry in self._walk_dir(self.root_dir):
            rel_path = str(entry.relative_to(self.root_dir)).replace("\\", "/")
            ext = entry.suffix.lower()

            try:
                stat = entry.stat()
                file_info: Dict[str, Any] = {
                    "rel_path": rel_path,
                    "abs_path": str(entry),
                    "size_bytes": stat.st_size,
                    "ext": ext,
                }

                if ext == ".py":
                    file_info["python_ast"] = self._extract_python_signatures(entry)
                elif ext in (".md", ".markdown"):
                    file_info["markdown_headers"] = self._extract_markdown_headers(entry)
                elif ext in (".json", ".yaml", ".yml"):
                    file_info["config_summary"] = self._extract_config_summary(entry)
                elif ext == ".sql":
                    file_info["sql_summary"] = self._extract_sql_summary(entry)

                file_index[rel_path] = file_info
                self._add_to_tree_dict(dir_tree, rel_path)
            except Exception as e:
                file_index[rel_path] = {
                    "rel_path": rel_path,
                    "abs_path": str(entry),
                    "error": str(e),
                }

        self.index = {
            "root_dir": str(self.root_dir),
            "files": file_index,
            "file_count": len(file_index),
            "tree": dir_tree,
        }
        self.scanned = True
        return self.index

    def get_tree(self, max_depth: int = 5) -> str:
        """
        Returns an ASCII tree representation of the scanned workspace.
        """
        if not self.scanned:
            self.scan_workspace()

        lines: List[str] = [f"{self.root_dir.name}/"]
        self._build_tree_str(self.index.get("tree", {}), lines, prefix="", depth=1, max_depth=max_depth)
        return "\n".join(lines)

    def get_context_summary(self, max_tokens_approx: int = 2000) -> str:
        """
        Builds and returns a formatted text summary of the indexed workspace
        suitable for LLM context injection.
        """
        if not self.scanned:
            self.scan_workspace()

        lines: List[str] = []
        lines.append(f"# Workspace Index: {self.root_dir.name}")
        lines.append(f"Total Source Files: {self.index.get('file_count', 0)}")
        lines.append("\n## File Tree")
        lines.append("```")
        lines.append(self.get_tree(max_depth=4))
        lines.append("```")

        lines.append("\n## Key Python Signatures & Structures")
        files_dict = self.index.get("files", {})

        py_files = [f for f, data in files_dict.items() if data.get("ext") == ".py"]
        for rel_path in py_files:
            data = files_dict[rel_path]
            ast_data = data.get("python_ast", {})
            funcs = ast_data.get("functions", [])
            classes = ast_data.get("classes", [])

            if not funcs and not classes:
                continue

            lines.append(f"\n### `{rel_path}`")
            for cl in classes:
                bases = f"({', '.join(cl['bases'])})" if cl.get("bases") else ""
                lines.append(f"- `class {cl['name']}{bases}`")
                for method in cl.get("methods", []):
                    lines.append(f"  - `{method['signature']}`")
            for fn in funcs:
                lines.append(f"- `{fn['signature']}`")

        other_files = [
            f for f, data in files_dict.items()
            if data.get("ext") in (".json", ".yaml", ".yml", ".md", ".sql")
        ]
        if other_files:
            lines.append("\n## Configurations & Documentation")
            for rel_path in other_files:
                data = files_dict[rel_path]
                ext = data.get("ext")
                if ext in (".md", ".markdown"):
                    headers = data.get("markdown_headers", [])
                    if headers:
                        lines.append(f"- `{rel_path}` headers: {', '.join(headers[:5])}")
                    else:
                        lines.append(f"- `{rel_path}` (markdown doc)")
                elif ext in (".json", ".yaml", ".yml"):
                    cfg = data.get("config_summary", {})
                    keys = cfg.get("top_keys", [])
                    lines.append(f"- `{rel_path}` config keys: {', '.join(keys[:8])}")
                elif ext == ".sql":
                    sql_sum = data.get("sql_summary", {})
                    queries = sql_sum.get("tables_or_views", [])
                    lines.append(f"- `{rel_path}` tables/views: {', '.join(queries[:5])}")

        result = "\n".join(lines)
        max_chars = max_tokens_approx * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated for context limit)"
        return result

    def _walk_dir(self, directory: Path):
        for entry in directory.iterdir():
            if entry.is_dir():
                if entry.name in self.exclude_dirs or entry.name.startswith("."):
                    continue
                yield from self._walk_dir(entry)
            elif entry.is_file():
                if entry.suffix.lower() in self.extensions:
                    yield entry

    def _add_to_tree_dict(self, tree: Dict[str, Any], rel_path: str):
        parts = rel_path.split("/")
        curr = tree
        for part in parts[:-1]:
            curr = curr.setdefault(part + "/", {})
        curr[parts[-1]] = None

    def _build_tree_str(
        self,
        tree_node: Dict[str, Any],
        lines: List[str],
        prefix: str = "",
        depth: int = 1,
        max_depth: int = 5,
    ):
        if depth > max_depth or not tree_node:
            return

        items = sorted(tree_node.items(), key=lambda x: (not x[0].endswith("/"), x[0]))
        count = len(items)

        for i, (name, subtree) in enumerate(items):
            is_last = (i == count - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")

            if name.endswith("/") and isinstance(subtree, dict):
                next_prefix = prefix + ("    " if is_last else "│   ")
                self._build_tree_str(subtree, lines, prefix=next_prefix, depth=depth + 1, max_depth=max_depth)

    def _extract_python_signatures(self, file_path: Path) -> Dict[str, Any]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            parsed = ast.parse(content, filename=str(file_path))
        except Exception as e:
            return {"error": str(e), "functions": [], "classes": []}

        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []

        for node in ast.iter_child_nodes(parsed):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({
                    "name": node.name,
                    "signature": self._format_ast_func(node),
                    "doc": ast.get_docstring(node),
                })
            elif isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if hasattr(ast, "unparse"):
                        bases.append(ast.unparse(b))
                    elif isinstance(b, ast.Name):
                        bases.append(b.id)
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": item.name,
                            "signature": self._format_ast_func(item),
                            "doc": ast.get_docstring(item),
                        })
                classes.append({
                    "name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "doc": ast.get_docstring(node),
                })

        return {"functions": functions, "classes": classes}

    def _format_ast_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        args_list = []
        for arg in node.args.args:
            ann = ""
            if arg.annotation and hasattr(ast, "unparse"):
                ann = f": {ast.unparse(arg.annotation)}"
            args_list.append(f"{arg.arg}{ann}")

        if node.args.vararg:
            args_list.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args_list.append(f"**{node.args.kwarg.arg}")

        ret_ann = ""
        if node.returns and hasattr(ast, "unparse"):
            ret_ann = f" -> {ast.unparse(node.returns)}"

        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({', '.join(args_list)}){ret_ann}"

    def _extract_markdown_headers(self, file_path: Path) -> List[str]:
        headers: List[str] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    headers.append(stripped)
        except Exception:
            pass
        return headers

    def _extract_config_summary(self, file_path: Path) -> Dict[str, Any]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if file_path.suffix.lower() == ".json":
                data = json.loads(content)
                if isinstance(data, dict):
                    return {"top_keys": list(data.keys())}
                elif isinstance(data, list):
                    return {"item_count": len(data)}
            else:
                top_keys = []
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and ":" in stripped:
                        key = stripped.split(":", 1)[0].strip()
                        if key and not line.startswith(" "):
                            top_keys.append(key)
                return {"top_keys": top_keys[:10]}
        except Exception:
            pass
        return {}

    def _extract_sql_summary(self, file_path: Path) -> Dict[str, Any]:
        tables: List[str] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                stripped = line.strip().upper()
                if "CREATE TABLE" in stripped or "CREATE VIEW" in stripped:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        tables.append(parts[2].strip("`\"[]();"))
        except Exception:
            pass
        return {"tables_or_views": tables}
