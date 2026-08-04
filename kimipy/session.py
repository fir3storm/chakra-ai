"""
Session Manager for local persistence of chat history, code artifacts, and multi-agent trace logs.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


class SessionManager:
    """
    Manages loading, saving, listing, and organizing local session data
    including chat history, generated code artifacts, and multi-agent trace logs.
    Saved to ~/.chakra_ai/sessions/ by default.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        if storage_dir is None:
            self.storage_dir = Path.home() / ".chakra_ai" / "sessions"
        else:
            self.storage_dir = Path(storage_dir).resolve()

        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        trace_logs: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves chat history, code artifacts, multi-agent trace logs, and metadata locally.
        If session_id is not provided, a unique timestamped session_id is generated.
        Returns the session_id.
        """
        self._ensure_storage_dir()

        now_str = datetime.now().isoformat()

        if not session_id:
            time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
            rand_part = uuid.uuid4().hex[:6]
            session_id = f"session_{time_part}_{rand_part}"

        file_path = self.storage_dir / f"{session_id}.json"

        created_at = now_str
        existing_data = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    created_at = existing_data.get("created_at", now_str)
            except Exception:
                pass

        session_payload = {
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": now_str,
            "history": history if history is not None else existing_data.get("history", []),
            "artifacts": artifacts if artifacts is not None else existing_data.get("artifacts", []),
            "trace_logs": trace_logs if trace_logs is not None else existing_data.get("trace_logs", []),
            "metadata": metadata if metadata is not None else existing_data.get("metadata", {}),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_payload, f, indent=2, ensure_ascii=False, default=str)

        return session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists all saved session records in local storage sorted by last modified time (newest first).
        Returns a list of summary dictionaries containing session metadata.
        """
        self._ensure_storage_dir()

        sessions: List[Dict[str, Any]] = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                stat = file_path.stat()
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                history = data.get("history", [])
                artifacts = data.get("artifacts", [])
                trace_logs = data.get("trace_logs", [])

                artifacts_count = len(artifacts) if isinstance(artifacts, list) else (1 if artifacts else 0)

                sessions.append({
                    "session_id": data.get("session_id", file_path.stem),
                    "file_path": str(file_path),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "mtime": stat.st_mtime,
                    "history_count": len(history) if isinstance(history, list) else 0,
                    "artifacts_count": artifacts_count,
                    "trace_logs_count": len(trace_logs) if isinstance(trace_logs, list) else 0,
                    "metadata": data.get("metadata", {}),
                })
            except Exception:
                continue

        sessions.sort(key=lambda s: s.get("mtime", 0), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Loads and returns session data (history, artifacts, trace_logs, metadata) by session_id.
        Raises FileNotFoundError if the session does not exist.
        """
        self._ensure_storage_dir()

        clean_id = session_id[:-5] if session_id.endswith(".json") else session_id
        file_path = self.storage_dir / f"{clean_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Session '{clean_id}' not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a saved session file by session_id.
        Returns True if deleted, False if not found.
        """
        clean_id = session_id[:-5] if session_id.endswith(".json") else session_id
        file_path = self.storage_dir / f"{clean_id}.json"

        if file_path.exists():
            file_path.unlink()
            return True
        return False
