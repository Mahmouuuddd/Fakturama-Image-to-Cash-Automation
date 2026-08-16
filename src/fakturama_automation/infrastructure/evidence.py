from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EvidenceRecorder:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_directory / "events.jsonl"

    def record(self, event: str, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **data,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def screenshot_path(self, name: str) -> Path:
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        directory = self.run_directory / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe_name}.png"
