from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


_GLOBAL_LOCK = threading.RLock()
_GLOBAL_COLLECTOR: Optional["MetricsCollector"] = None


@dataclass
class MetricEvent:
    timestamp: float
    component: str
    event: str
    data: Dict[str, Any]


class MetricsCollector:
    """Thread-safe JSONL metrics collector.

    Writes one JSON object per line to a target file. Safe to use across
    threads and processes (append-only). Lightweight and optional: if disabled,
    all calls are no-ops.
    """

    def __init__(self, output_path: str = "logs/metrics.jsonl", enabled: bool = False) -> None:
        self.enabled = enabled
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        if self.enabled:
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If we cannot create the directory, disable gracefully
                self.logger.warning("Failed to create logs directory; disabling metrics")
                self.enabled = False

    def track(self, component: str, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        metric = MetricEvent(timestamp=time.time(), component=component, event=event, data=data)
        payload = asdict(metric)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            try:
                with self.output_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                # Non-fatal: log once per error type to avoid noise
                try:
                    self.logger.debug(f"Failed to write metrics: {e}")
                except Exception:
                    pass

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            old = self.enabled
            self.enabled = enabled
            if enabled and not old:
                try:
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    self.enabled = False


def get_metrics() -> MetricsCollector:
    """Return the global metrics collector singleton."""
    global _GLOBAL_COLLECTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_COLLECTOR is None:
            # Enable via env var RAG_ENABLE_METRICS=1 by default; CLI may toggle
            enabled = os.getenv("RAG_ENABLE_METRICS", "0") == "1"
            output = os.getenv("RAG_METRICS_PATH", "logs/metrics.jsonl")
            _GLOBAL_COLLECTOR = MetricsCollector(output_path=output, enabled=enabled)
        return _GLOBAL_COLLECTOR


def enable_metrics(enabled: bool = True, output_path: Optional[str] = None) -> None:
    """Configure global metrics collector at runtime."""
    mc = get_metrics()
    if output_path:
        with mc._lock:
            mc.output_path = Path(output_path)
            if enabled:
                try:
                    mc.output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    enabled = False
    mc.set_enabled(enabled)

