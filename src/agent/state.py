from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class PipelineState:
    algorithm: str = "svm"
    mode: str = "single"
    preprocess: Dict[str, int] = field(default_factory=lambda: {"low": 8, "high": 30})
    current_step: str = "EEG采集"
    uploaded_files: List[Dict[str, str]] = field(default_factory=list)
    last_run: Optional[Dict[str, Any]] = None


class TaskStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, task_type: str, payload: Optional[Dict[str, Any]] = None) -> str:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        self._tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "running",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "payload": payload or {},
            "result": None,
            "error": None,
        }
        return task_id

    def success(self, task_id: str, result: Dict[str, Any]) -> None:
        self._tasks[task_id]["status"] = "success"
        self._tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
        self._tasks[task_id]["result"] = result

    def fail(self, task_id: str, error: str) -> None:
        self._tasks[task_id]["status"] = "failed"
        self._tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
        self._tasks[task_id]["error"] = error

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_id)
