"""Observability and metrics tracking."""
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RunMetrics:
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    state: Optional[str] = None
    tool_calls: int = 0
    errors: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    steps: List[Dict[str, Any]] = field(default_factory=list)

class ObservabilityManager:
    def __init__(self):
        self.runs: Dict[str, RunMetrics] = {}
        self.metrics = defaultdict(list)
        self.current_run: Optional[RunMetrics] = None
        self.event_log: List[Dict[str, Any]] = []

    def start_run(self, user_input: str) -> str:
        run_id = f"run_{int(time.time() * 1000)}"
        self.current_run = RunMetrics(run_id=run_id, start_time=time.time())
        self.runs[run_id] = self.current_run
        return run_id

    def record_run(self, run_id: str, duration: float, state: str):
        if run_id in self.runs:
            run = self.runs[run_id]
            run.duration = duration; run.state = state; run.end_time = time.time()

    def record_tool_call(self, tool_name: str, arguments: Dict, result: Dict):
        if self.current_run:
            self.current_run.tool_calls += 1
            self.current_run.steps.append({"type": "tool_call", "tool": tool_name, "success": "error" not in result, "timestamp": time.time()})

    def record_error(self, error: str, context: Optional[Dict] = None):
        if self.current_run: self.current_run.errors += 1

    def get_metrics(self) -> Dict[str, Any]:
        total_runs = len(self.runs)
        completed_runs = [r for r in self.runs.values() if r.duration is not None]
        if completed_runs:
            avg_duration = sum(r.duration for r in completed_runs) / len(completed_runs)
            total_errors = sum(r.errors for r in completed_runs)
            total_tool_calls = sum(r.tool_calls for r in completed_runs)
        else: avg_duration, total_errors, total_tool_calls = 0, 0, 0
        return {"total_runs": total_runs, "completed_runs": len(completed_runs), "average_duration": round(avg_duration, 2), "total_errors": total_errors, "total_tool_calls": total_tool_calls, "error_rate": round(total_errors / total_tool_calls, 2) if total_tool_calls > 0 else 0}

    def export_logs(self, filepath: str):
        with open(filepath, 'w') as f: json.dump(self.event_log, f, indent=2)
