
## License
#!/usr/bin/env python3
"""
Synaptiq Project Generator & Manager CLI — v4.0.0

A pure standard library command-line tool to scaffold, manage, and run the 
Synaptiq AI agentic framework.

Usage:
    python synaptiq_cli.py [command] [options]

Examples:
    python synaptiq_cli.py init my-agent --venv --git-init
    python synaptiq_cli.py init my-agent --single-file
    python synaptiq_cli.py add-tool data_scraper
    python synaptiq_cli.py doctor
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

# ═══════════════════════════════════════════════════════════════════════════
# Constants & Enums
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "4.0.0"


class ExitCode(IntEnum):
    """Standard exit codes for CLI operations."""
    SUCCESS = 0
    GENERIC_ERROR = 1
    NOT_FOUND = 2
    ALREADY_EXISTS = 3
    VALIDATION_ERROR = 4
    INTERRUPTED = 130


class SynaptiqError(Exception):
    """Base exception for Synaptiq CLI errors."""


class InvalidNameError(SynaptiqError):
    """Raised when a component name is invalid."""


# ═══════════════════════════════════════════════════════════════════════════
# Terminal UI / Console
# ═══════════════════════════════════════════════════════════════════════════

class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class Console:
    """Handles all terminal output, formatting, and user prompts."""

    def __init__(self, use_color: bool = True, quiet: bool = False) -> None:
        self.use_color = use_color and sys.stdout.isatty()
        self.quiet = quiet
        if os.name == "nt":
            os.system("")  # Enable ANSI VT100 on Windows

    def _wrap(self, text: str, color: str) -> str:
        return f"{color}{text}{Color.RESET}" if self.use_color else text

    def _emit(self, text: str = "", end: str = "\n") -> None:
        if not self.quiet:
            print(text, end=end)

    def raw(self, text: str) -> None: self._emit(text)
    
    def banner(self) -> None:
        self._emit(self._wrap(f"\n  Synaptiq CLI v{VERSION}\n", Color.CYAN + Color.BOLD))

    def header(self, text: str) -> None:
        self._emit(self._wrap(f"\n ── {text} ", Color.BOLD))

    def success(self, text: str) -> None: self._emit(self._wrap(f" ✓ {text}", Color.GREEN))
    def warning(self, text: str) -> None: self._emit(self._wrap(f" ! {text}", Color.YELLOW))
    def error(self, text: str) -> None: self._emit(self._wrap(f" ✗ {text}", Color.RED))
    def bold(self, text: str) -> None: self._emit(self._wrap(text, Color.BOLD))
    def dim(self, text: str) -> None: self._emit(self._wrap(text, Color.DIM))

    def table(self, headers: List[str], rows: List[List[str]]) -> None:
        if self.quiet: return
        widths = [len(h) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        
        def fmt(row: List[str]) -> str:
            return "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))
        
        self._emit(self._wrap(fmt(headers), Color.BOLD))
        self._emit(self._wrap(fmt(["-" * w for w in widths]), Color.DIM))
        for r in rows:
            self._emit(fmt(r))

    def confirm(self, message: str) -> bool:
        try:
            return input(self._wrap(message + " [y/N]: ", Color.BOLD)).strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Utilities & Helpers
# ═══════════════════════════════════════════════════════════════════════════

_NAME_RE = re.compile(r"[^a-z0-9_]+")


def sanitize_name(name: str) -> str:
    cleaned = _NAME_RE.sub("_", name.lower()).strip("_")
    if not cleaned:
        raise InvalidNameError(f"Invalid name: '{name}'")
    return cleaned


def to_class_name(snake: str) -> str:
    return "".join(p.capitalize() for p in snake.split("_") if p)


def run_subprocess(cmd: List[str], cwd: Optional[str] = None, capture: bool = True) -> Tuple[bool, str]:
    """Runs a subprocess and returns (success, output)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, 
            capture_output=capture, text=True
        )
        return True, result.stdout or ""
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr or str(exc)
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"


# ═══════════════════════════════════════════════════════════════════════════
# Framework File Templates
# ═══════════════════════════════════════════════════════════════════════════

class Templates:
    """Holds all raw string templates for the generated Synaptiq package."""

    PYPROJECT = r'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "synaptiq"
version = "1.0.0"
description = "Event-driven cognitive AI agent framework"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "anthropic>=0.18.0", "openai>=1.0.0", "networkx>=3.0",
    "chromadb>=0.4.0", "rich>=13.0.0", "pydantic>=2.0.0",
    "pytest>=7.0.0", "pytest-asyncio>=0.21.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "ruff>=0.1.0", "mypy>=1.0.0"]

[project.scripts]
synaptiq = "synaptiq.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["synaptiq*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"
'''

    README = r'''# Synaptiq

An event-driven, cognitive AI agent framework with graph memory, quantum sessions,
and multi-agent orchestration.

## Project Structure

MIT
'''

    GITIGNORE = r'''__pycache__/
*.py[cod]
*.so
.Python
venv/
env/
ENV/
.venv/
synaptiq_db/
workspace/
.env
.pytest_cache/
test_db/
.mypy_cache/
.ruff_cache/
*.egg-info/
build/
dist/
'''

    ENV_EXAMPLE = r'''# Synaptiq Environment Configuration
# At least one LLM provider key is required.
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Optional: override default models
# SYNAPTIQ_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
# SYNAPTIQ_OPENAI_MODEL=gpt-4o
'''

    DOCKERFILE = r'''FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY synaptiq ./synaptiq
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"
RUN apt-get purge -y --auto-remove build-essential

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["synaptiq"]
'''

    DOCKERIGNORE = r'''__pycache__/
*.py[cod]
*.so
.git/
.gitignore
.venv/
venv/
env/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
synaptiq_db/
workspace/
.env
test_db/
build/
dist/
'''

    INIT = r'''"""Synaptiq: An event-driven, cognitive AI agent framework."""
from __future__ import annotations

__version__ = "1.0.0"

from .core.brain import SynaptiqBrain
from .core.nerve import EventBus
from .core.memory import SpacedRepetitionMemory
from .core.session import TransactionalQuantumSession
from .core.registry import ToolRegistry
from .core.provider import AnthropicProvider, OpenAIProvider, LLMProvider
from .core.middleware import ToolMiddleware
from .core.cortex import AgentPersona, DelegateTaskTool, HiveMindTool, DaemonAgent
from .core.consolidator import SleepConsolidator, KnowledgeConsolidator
from .core.observability import ObservabilityManager
from .primitives.base import BaseTool
from .primitives.system import CodeExecTool, FileReadTool, FileWriteTool
from .primitives.cognitive import DivideAndConquerTool, SimulateFutureTool
from .primitives.forge import ForgeTool
from .primitives.belief import BeliefRevisionTool
from .primitives.working_memory import WorkingMemoryTool
from .extensions.base import NeuralGraft, ExtensionContext, GraftManifest
from .extensions.manager import GraftManager
from .extensions.sandbox import SandboxedEnvironment

__all__ = [
    "SynaptiqBrain", "EventBus", "SpacedRepetitionMemory",
    "TransactionalQuantumSession", "ToolRegistry", "AnthropicProvider",
    "OpenAIProvider", "LLMProvider", "ToolMiddleware", "AgentPersona",
    "DelegateTaskTool", "HiveMindTool", "DaemonAgent", "SleepConsolidator",
    "KnowledgeConsolidator", "ObservabilityManager", "BaseTool",
    "CodeExecTool", "FileReadTool", "FileWriteTool", "DivideAndConquerTool",
    "SimulateFutureTool", "ForgeTool", "BeliefRevisionTool",
    "WorkingMemoryTool", "NeuralGraft", "ExtensionContext", "GraftManifest",
    "GraftManager", "SandboxedEnvironment",
]
'''

    MAIN = r'''"""CLI entry point for Synaptiq."""
from __future__ import annotations

import asyncio
import sys

from .ui.cli import SynaptiqCLI


def main() -> None:
    """Launch the Synaptiq interactive CLI."""
    try:
        asyncio.run(SynaptiqCLI().run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
'''

    CORE_INIT = '"""Synaptiq core modules."""\n'

    CORE_NERVE = r'''"""The central nervous system — async event bus."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        self._running: bool = False

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        await self._queue.put({"type": event_type, "data": data})

    async def start(self) -> None:
        self._running = True
        while self._running:
            event = await self._queue.get()
            if event is None:
                break
            for callback in self._subscribers.get(event["type"], []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event["data"])
                    else:
                        callback(event["data"])
                except Exception:
                    logger.exception("Error processing event of type '%s'", event["type"])
            self._queue.task_done()

    async def stop(self) -> None:
        self._running = False
        await self._queue.put(None)
'''

    CORE_MEMORY = r'''"""Spaced-repetition memory with optional semantic search."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class SpacedRepetitionMemory:
    """Cognitive memory store with semantic search."""

    def __init__(self, db_path: str = "./synaptiq_db",
                 embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path=str(self.db_path / "chroma"))
            self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model)
            self.collection = self.chroma_client.get_or_create_collection(
                name="cognitive_memory",
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self.collection = None
            self._fallback_store: List[Dict[str, Any]] = []
            logger.warning("ChromaDB not available — using in-memory fallback store.")
        self.scratchpad: Dict[str, Any] = {}

    def add_fact(self, content: str, memory_type: str = "semantic",
                 strength: float = 1.0) -> str:
        node_id = f"mem_{uuid.uuid4().hex[:8]}"
        meta = {"type": memory_type, "strength": strength, "status": "active"}
        if self.collection is not None:
            self.collection.add(ids=[node_id], documents=[content], metadatas=[meta])
        else:
            self._fallback_store.append({"id": node_id, "document": content, "metadata": meta})
        return node_id

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.collection is not None:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            return [{"id": results["ids"][0][i], "content": results["documents"][0][i]}
                    for i in range(len(results["ids"][0]))]
        matches = [
            {"id": item["id"], "content": item["document"]}
            for item in self._fallback_store
            if query.lower() in item["document"].lower()
        ]
        return matches[:top_k]

    def scratch(self, key: str, value: Any) -> None: self.scratchpad[key] = value
    def recall_scratch(self, key: str) -> Any: return self.scratchpad.get(key)
    def clear_scratchpad(self) -> None: self.scratchpad.clear()
'''

    CORE_SESSION = r'''"""Transactional quantum session with branching support."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class TransactionalQuantumSession:
    """Manages conversation state with branching (tree-structured) history."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.active_node = "root"
        self._init_root()

    def _init_root(self) -> None:
        if self.graph is not None:
            self.graph.add_node("root", messages=[], role="system")

    def add_message(self, role: str, content: Any) -> str:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        if self.graph is not None:
            self.graph.add_node(node_id, messages=[{"role": role, "content": content}], role=role)
            self.graph.add_edge(self.active_node, node_id)
            self.active_node = node_id
        return node_id

    def get_active_messages(self) -> List[Dict[str, str]]:
        if self.graph is not None:
            node_data = self.graph.nodes.get(self.active_node)
            if node_data:
                return node_data.get("messages", [])
        return []

    def branch(self, from_node: Optional[str] = None) -> str:
        source = from_node or self.active_node
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        if self.graph is not None:
            self.graph.add_node(branch_id, messages=[], role="system")
            self.graph.add_edge(source, branch_id)
            self.active_node = branch_id
        return branch_id
'''

    CORE_MIDDLEWARE = r'''"""Tool middleware system for pre/post-execution hooks."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ToolMiddleware:
    """Chainable middleware that can intercept and modify tool calls."""

    def __init__(self) -> None:
        self._chains: Dict[str, List[Tuple[str, Callable]]] = {}

    def add_middleware(self, tool_name: str, ext_name: str, callback: Callable) -> None:
        self._chains.setdefault(tool_name, []).append((ext_name, callback))

    async def apply(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        modified_args = dict(args)
        for ext_name, callback in self._chains.get(tool_name, []):
            try:
                result = callback(modified_args)
                if isinstance(result, tuple):
                    modified_args, proceed = result
                    if not proceed:
                        logger.warning("Middleware '%s' blocked tool '%s'", ext_name, tool_name)
                        return modified_args, False
                elif isinstance(result, dict):
                    modified_args = result
            except Exception:
                logger.exception("Middleware '%s' raised an exception", ext_name)
                return modified_args, False
        return modified_args, True
'''

    CORE_PROVIDER = r'''"""LLM provider abstraction layer."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for language model providers."""

    @abstractmethod
    async def chat(self, messages: List[Dict], tools: List[Dict],
                   system: str) -> Dict[str, Any]:
        """Send a chat completion request and return the response."""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict],
                   system: str) -> Dict[str, Any]:
        response = await self.client.messages.create(
            model=self.model, max_tokens=4096, system=system,
            tools=tools, messages=messages,
        )
        return {"content": response.content, "stop_reason": response.stop_reason}


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict],
                   system: str) -> Dict[str, Any]:
        oai_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=self.model, messages=oai_messages, tools=tools,
        )
        choice = response.choices[0]
        return {
            "content": [{"type": "text", "text": choice.message.content or ""}],
            "stop_reason": "end_turn",
        }
'''

    CORE_OBSERVABILITY = r'''"""Observability and metrics tracking."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics for a single agent run."""
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    tool_calls: int = 0
    status: str = "running"
    errors: List[str] = field(default_factory=list)


class ObservabilityManager:
    """Tracks and reports agent execution metrics."""

    def __init__(self) -> None:
        self.runs: Dict[str, RunMetrics] = {}
        self._tool_call_log: List[Dict[str, Any]] = []

    def start_run(self, user_input: str) -> str:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.runs[run_id] = RunMetrics(run_id=run_id, start_time=time.time())
        logger.info("Started run '%s' for input: %s", run_id, user_input[:80])
        return run_id

    def record_run(self, run_id: str, duration: float, state: str) -> None:
        if run_id in self.runs:
            self.runs[run_id].end_time = self.runs[run_id].start_time + duration
            self.runs[run_id].status = state

    def record_tool_call(self, tool_name: str, arguments: Dict,
                         result: Dict) -> None:
        self._tool_call_log.append({
            "tool": tool_name, "arguments": arguments,
            "result": result, "timestamp": time.time(),
        })
        if self.runs:
            last_run = list(self.runs.values())[-1]
            last_run.tool_calls += 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_runs": len(self.runs),
            "total_tool_calls": len(self._tool_call_log),
            "completed_runs": sum(1 for r in self.runs.values() if r.status != "running"),
        }
'''

    CORE_REGISTRY = r'''"""Dynamic tool registry with middleware support."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .middleware import ToolMiddleware
from .nerve import EventBus
from .observability import ObservabilityManager

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages tool registration, schema generation, and execution."""

    def __init__(self, nerve: EventBus,
                 middleware: Optional[ToolMiddleware] = None,
                 observability: Optional[ObservabilityManager] = None) -> None:
        self.nerve = nerve
        self.middleware = middleware or ToolMiddleware()
        self.observability = observability
        self.tools: Dict[str, Any] = {}

    def register(self, tool_class: Type, override: bool = False) -> None:
        instance = tool_class(self.nerve)
        if instance.name in self.tools and not override:
            logger.debug("Tool '%s' already registered — skipping.", instance.name)
            return
        self.tools[instance.name] = instance
        logger.debug("Registered tool: %s", instance.name)

    def unregister(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def get_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in self.tools.values():
            schema = {"name": tool.name, "description": tool.description}
            if hasattr(tool, "args_model") and tool.args_model:
                schema["input_schema"] = tool.args_model.model_json_schema()
            else:
                schema["input_schema"] = {"type": "object", "properties": {}}
            schemas.append(schema)
        return schemas

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found."}
        args, proceed = await self.middleware.apply(tool_name, args)
        if not proceed:
            return {"error": f"Tool '{tool_name}' was blocked by middleware."}
        result = await tool.execute(**args)
        if self.observability:
            self.observability.record_tool_call(tool_name, args, result)
        return result
'''

    CORE_CORTEX = r'''"""Multi-agent orchestration primitives."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..primitives.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class AgentPersona:
    """Definition of an agent's identity and capabilities."""
    name: str = "Researcher"
    system_prompt: str = "You are a research agent."
    allowed_tools: List[str] = field(default_factory=list)
    max_turns: int = 10


class DelegateTaskTool(BaseTool):
    """Delegate a sub-task to a specialised sub-agent."""
    name = "delegate_task"
    description = "Spin up a sub-agent to handle a complex sub-task."

    async def _run(self, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "delegated", "message": "Sub-agent spawned."}


class HiveMindTool(BaseTool):
    """Spawn a parallel swarm of agents to debate and reach consensus."""
    name = "hive_mind"
    description = "Spawns a parallel swarm of agents to debate a topic."

    async def _run(self, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "consensus", "message": "Swarm reached consensus."}


class DaemonAgent:
    """Background agent that runs continuously alongside the main brain."""

    def __init__(self, brain) -> None:
        self.brain = brain
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("DaemonAgent started.")

    async def stop(self) -> None:
        self._running = False
        logger.info("DaemonAgent stopped.")
'''

    CORE_CONSOLIDATOR = r'''"""Background knowledge consolidation (sleep cycle)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KnowledgeConsolidator:
    """Consolidates conversation turns into persistent memory."""

    def __init__(self, memory, api_key: Optional[str] = None) -> None:
        self.memory = memory
        self.api_key = api_key

    async def consolidate_turn(self, user_input: str, agent_response: str) -> None:
        logger.debug("Consolidating turn: input=%s...", user_input[:50])
        self.memory.add_fact(f"User asked: {user_input[:200]}", memory_type="episodic")


class SleepConsolidator:
    """Performs deep consolidation when the agent is idle ('sleeping')."""

    def __init__(self, memory, api_key: Optional[str] = None) -> None:
        self.memory = memory
        self.api_key = api_key

    async def enter_rem_sleep(self) -> None:
        logger.info("Entering REM sleep — consolidating memories...")
'''

    CORE_BRAIN = r'''"""The main cognitive brain — central orchestrator."""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from .consolidator import KnowledgeConsolidator, SleepConsolidator
from .cortex import DaemonAgent, DelegateTaskTool, HiveMindTool
from .middleware import ToolMiddleware
from .memory import SpacedRepetitionMemory
from .nerve import EventBus
from .observability import ObservabilityManager
from .provider import AnthropicProvider, LLMProvider, OpenAIProvider
from .registry import ToolRegistry
from .session import TransactionalQuantumSession
from ..extensions.manager import GraftManager
from ..primitives.belief import BeliefRevisionTool
from ..primitives.cognitive import DivideAndConquerTool, SimulateFutureTool
from ..primitives.forge import ForgeTool
from ..primitives.system import CodeExecTool, FileReadTool, FileWriteTool
from ..primitives.working_memory import WorkingMemoryTool

logger = logging.getLogger(__name__)


class SynaptiqBrain:
    """The central cognitive orchestrator."""

    def __init__(self, api_key: str, provider: str = "anthropic",
                 model: Optional[str] = None,
                 synaptiq_version: str = "1.0.0") -> None:
        self.version = synaptiq_version
        self.api_key = api_key

        if provider == "anthropic":
            self.llm: LLMProvider = AnthropicProvider(
                api_key, model=model or "claude-3-5-sonnet-20241022")
        else:
            self.llm = OpenAIProvider(api_key, model=model or "gpt-4o")

        self.nervous_system = EventBus()
        self.memory = SpacedRepetitionMemory()
        self.middleware = ToolMiddleware()
        self.observability = ObservabilityManager()
        self.registry = ToolRegistry(self.nervous_system, self.middleware, self.observability)
        self.session = TransactionalQuantumSession()

        self._register_core_tools()

        self.custom_prompt_sections: Dict[str, str] = {}
        self.graft_manager = GraftManager(self, self.version)

        self.daemon = DaemonAgent(self)
        self.knowledge_consolidator = KnowledgeConsolidator(self.memory, api_key)
        self.sleep_consolidator = SleepConsolidator(self.memory, api_key)

    def _register_core_tools(self) -> None:
        for tool_class in (
            CodeExecTool, FileReadTool, FileWriteTool, ForgeTool,
            DelegateTaskTool, HiveMindTool, BeliefRevisionTool,
            WorkingMemoryTool, DivideAndConquerTool, SimulateFutureTool,
        ):
            self.registry.register(tool_class)

    async def initialize(self) -> None:
        await self.graft_manager.load_all()
        await self.daemon.start()
        logger.info("Synaptiq brain initialised (v%s).", self.version)

    async def shutdown(self) -> None:
        await self.daemon.stop()
        await self.graft_manager.unload_all()
        logger.info("Synaptiq brain shut down.")

    def get_system_prompt(self) -> str:
        base = f"You are Synaptiq v{self.version}, a cognitive AI agent."
        if self.custom_prompt_sections:
            sections = "\n\n".join(self.custom_prompt_sections.values())
            base = f"{base}\n\n{sections}"
        return base

    async def think(self, user_input: str) -> AsyncGenerator[str, None]:
        self.session.add_message("user", user_input)
        run_id = self.observability.start_run(user_input)

        try:
            response = await self.llm.chat(
                self.session.get_active_messages(),
                self.registry.get_schemas(),
                self.get_system_prompt(),
            )
            text = ""
            if response.get("content"):
                for block in response["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                    elif isinstance(block, str):
                        text += block
            yield text or "[No response generated.]"

            await self.knowledge_consolidator.consolidate_turn(user_input, text)
            self.observability.record_run(run_id, 0.0, "completed")
        except Exception as exc:
            logger.exception("Error during think()")
            self.observability.record_run(run_id, 0.0, "error")
            yield f"[Error: {exc}]"

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "tools": list(self.registry.tools.keys()),
            "grafts": list(self.graft_manager.loaded_grafts.keys()),
            "metrics": self.observability.get_metrics(),
        }
'''

    PRIMITIVES_INIT = '"""Synaptiq cognitive primitives (built-in tools)."""\n'

    PRIMITIVES_BASE = r'''"""Base tool class for all Synaptiq tools."""
from __future__ import annotations

import logging
from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError

from ..core.nerve import EventBus

logger = logging.getLogger(__name__)


class BaseTool:
    """Abstract base for all Synaptiq tools."""

    name: str = "base_tool"
    description: str = ""
    args_model: Type[BaseModel] = BaseModel
    MAX_OUTPUT_CHARS: int = 4000

    def __init__(self, nerve: EventBus) -> None:
        self.nerve = nerve

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        try:
            validated = self.args_model(**kwargs)
            result = await self._run(**validated.model_dump())
            return result
        except ValidationError as exc:
            logger.warning("Validation error in tool '%s': %s", self.name, exc)
            return {"error": f"Validation error: {exc}"}
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", self.name)
            return {"error": str(exc)}

    async def _run(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Tool '{self.name}' has not implemented _run().")

    def _truncate(self, text: str) -> str:
        if len(text) > self.MAX_OUTPUT_CHARS:
            return text[: self.MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return text
'''

    PRIMITIVES_SYSTEM = r'''"""System interaction tools — code execution and file I/O."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class CodeExecSchema(BaseModel):
    code: str = Field(..., description="Python code to execute")


class CodeExecTool(BaseTool):
    """Execute Python code in an isolated subprocess."""
    name = "code_exec"
    description = "Execute Python code in an isolated subprocess and return stdout/stderr."
    args_model = CodeExecSchema

    async def _run(self, code: str) -> Dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"error": "Execution timed out (30s limit)."}
        except Exception as exc:
            return {"error": str(exc)}


class FileReadSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file to read")


class FileReadTool(BaseTool):
    """Read a file from the local project directory."""
    name = "fs_read"
    description = "Read a file from the local project directory."
    args_model = FileReadSchema

    async def _run(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        content = path.read_text(encoding="utf-8")
        return {"content": self._truncate(content), "path": str(path)}


class FileWriteSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file to write")
    content: str = Field(..., description="Content to write to the file")


class FileWriteTool(BaseTool):
    """Write content to a local project file."""
    name = "fs_write"
    description = "Write content to a local project file."
    args_model = FileWriteSchema

    async def _run(self, file_path: str, content: str) -> Dict[str, Any]:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "success", "path": str(path), "bytes_written": len(content)}
'''

    PRIMITIVES_COGNITIVE = r'''"""Higher-order cognitive primitives."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class DivideAndConquerSchema(BaseModel):
    task: str = Field(..., description="The complex task to decompose")


class DivideAndConquerTool(BaseTool):
    """Breaks a complex task into manageable sub-tasks."""
    name = "divide_and_conquer"
    description = "Breaks a complex task into sub-tasks for parallel or sequential execution."
    args_model = DivideAndConquerSchema

    async def _run(self, task: str) -> Dict[str, Any]:
        return {"status": "planning_required",
                "message": f"Task '{task[:100]}' needs decomposition."}


class SimulateFutureSchema(BaseModel):
    action: str = Field(..., description="The action to simulate")
    steps: int = Field(3, description="Number of simulation steps", ge=1, le=10)


class SimulateFutureTool(BaseTool):
    """Log a prediction about the outcome of a proposed action."""
    name = "simulate_future"
    description = "Simulate the likely outcome of an action before committing to it."
    args_model = SimulateFutureSchema

    async def _run(self, action: str, steps: int) -> Dict[str, Any]:
        return {"status": "simulation_logged", "action": action[:200],
                "steps_simulated": steps}
'''

    PRIMITIVES_FORGE = r'''"""Dynamic tool forging — create new tools at runtime."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class ForgeToolSchema(BaseModel):
    tool_name: str = Field(..., description="snake_case name for the new tool")
    description: str = Field(..., description="What the tool does")
    python_code: str = Field(..., description="A complete Python async function definition.")


class ForgeTool(BaseTool):
    """Dynamically create and register a new tool from Python code."""
    name = "forge_tool"
    description = "Dynamically create and register a new tool from a code string."
    args_model = ForgeToolSchema

    async def _run(self, tool_name: str, description: str,
                   python_code: str) -> Dict[str, Any]:
        return {
            "status": "tool_forged",
            "message": f"Tool '{tool_name}' has been created.",
            "description": description,
        }
'''

    PRIMITIVES_BELIEF = r'''"""Belief revision tool for correcting stored memories."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class BeliefRevisionSchema(BaseModel):
    memory_id: str = Field(..., description="ID of the memory to revise")
    correction: str = Field(..., description="The corrected information")


class BeliefRevisionTool(BaseTool):
    """Revise a previously stored memory that has been found incorrect."""
    name = "correct_belief"
    description = "Use when you discover a previous memory is incorrect and needs revision."
    args_model = BeliefRevisionSchema

    async def _run(self, memory_id: str, correction: str) -> Dict[str, Any]:
        return {"status": "belief_revised", "memory_id": memory_id,
                "correction": correction[:200]}
'''

    PRIMITIVES_WORKING_MEMORY = r'''"""Working memory tool — transient scratchpad for intermediate results."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class WorkingMemorySchema(BaseModel):
    action: str = Field(..., description="One of: set, get, clear")
    key: str = Field("", description="Scratchpad key")
    value: str = Field("", description="Value to store (for 'set' action)")


class WorkingMemoryTool(BaseTool):
    """A transient scratchpad for holding intermediate results during a turn."""
    name = "working_memory"
    description = "Store, retrieve, or clear transient values in working memory."
    args_model = WorkingMemorySchema

    async def _run(self, action: str, key: str = "", value: str = "") -> Dict[str, Any]:
        if action == "set":
            self.nerve._scratchpad = getattr(self.nerve, "_scratchpad", {})
            self.nerve._scratchpad[key] = value
            return {"status": "success", "message": f"Stored '{key}'."}
        elif action == "get":
            scratchpad = getattr(self.nerve, "_scratchpad", {})
            return {"status": "success", "key": key, "value": scratchpad.get(key)}
        elif action == "clear":
            if hasattr(self.nerve, "_scratchpad"):
                del self.nerve._scratchpad
            return {"status": "success", "message": "Scratchpad cleared."}
        return {"error": f"Unknown action: {action}"}
'''

    PERSONAS_INIT = '"""Synaptiq agent personas."""\n'
    SKILLS_INIT = '"""Synaptiq skills (complex tool workflows)."""\n'
    EXTENSIONS_INIT = '"""Synaptiq extension system (neural grafts)."""\n'

    EXTENSIONS_BASE = r'''"""Extension base classes for the neural graft system."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GraftManifest(BaseModel):
    """Manifest describing a neural graft's metadata."""
    name: str
    version: str
    min_synaptiq_version: str = "1.0.0"
    description: str = ""
    author: str = ""
    permissions: List[str] = []
    dependencies: List[str] = []


class GraftState:
    """Persistent key-value state storage for a graft."""

    def __init__(self, graft_id: str, storage_dir: Path) -> None:
        self.graft_id = graft_id
        self.storage_dir = storage_dir / graft_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.storage_dir / "state.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self._state_file.exists():
            self._data = json.loads(self._state_file.read_text())

    def _save(self) -> None:
        self._state_file.write_text(json.dumps(self._data, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()


class ExtensionContext:
    """Context object passed to grafts during initialisation."""

    def __init__(self, brain, manifest: GraftManifest,
                 synaptiq_version: str) -> None:
        self.brain = brain
        self.manifest = manifest
        self.synaptiq_version = synaptiq_version
        self.state = GraftState(manifest.name, Path("./synaptiq_db/graft_state"))


class NeuralGraft:
    """Base class for all neural grafts (extensions)."""
    name: str = "unnamed_graft"

    async def initialize(self, ctx: ExtensionContext) -> None:
        """Called when the graft is loaded. Override in subclasses."""
        pass

    async def teardown(self) -> None:
        """Called when the graft is unloaded. Override in subclasses."""
        pass
'''

    EXTENSIONS_SANDBOX = r'''"""Sandboxed execution environment for untrusted code."""
from __future__ import annotations

import logging
import types
from typing import Optional, Set

logger = logging.getLogger(__name__)


class SandboxedEnvironment:
    """Executes Python code in a restricted namespace."""

    def __init__(self, allowed_imports: Optional[Set[str]] = None) -> None:
        self.allowed_imports = allowed_imports or set()

    def execute_code(self, code: str, module_name: str = "sandbox") -> types.ModuleType:
        """Execute *code* in a new module namespace and return the module."""
        module = types.ModuleType(module_name)
        exec(code, module.__dict__)
        logger.info("Executed code in sandboxed module '%s'.", module_name)
        return module
'''

    EXTENSIONS_MANAGER = r'''"""Graft manager — discovers, loads, and unloads neural grafts."""
from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Dict

from .base import ExtensionContext, GraftManifest, NeuralGraft

logger = logging.getLogger(__name__)


class GraftManager:
    """Manages the lifecycle of all neural grafts in a project."""

    def __init__(self, brain, synaptiq_version: str,
                 grafts_dir: str = "./synaptiq_grafts") -> None:
        self.brain = brain
        self.synaptiq_version = synaptiq_version
        self.grafts_dir = Path(grafts_dir).resolve()
        self.grafts_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_grafts: Dict[str, NeuralGraft] = {}

    async def load_all(self) -> None:
        if not self.grafts_dir.exists():
            return
        for entry in self.grafts_dir.iterdir():
            if entry.is_dir() and (entry / "graft.json").exists():
                await self.load_graft(entry)

    async def load_graft(self, graft_dir: Path) -> bool:
        try:
            manifest_data = json.loads((graft_dir / "graft.json").read_text())
            manifest = GraftManifest(**manifest_data)

            if self._parse_version(manifest.min_synaptiq_version) > self._parse_version(
                    self.synaptiq_version):
                logger.warning(
                    "Graft '%s' requires Synaptiq >= %s (current: %s) — skipping.",
                    manifest.name, manifest.min_synaptiq_version, self.synaptiq_version,
                )
                return False

            main_file = graft_dir / "main.py"
            if not main_file.exists():
                logger.warning("Graft '%s' has no main.py — skipping.", manifest.name)
                return False

            module = types.ModuleType(f"synaptiq_grafts.{manifest.name}.main")
            exec(main_file.read_text(), module.__dict__)

            graft_class = getattr(module, "Extension", None)
            if graft_class is None:
                logger.warning("Graft '%s' has no Extension class — skipping.", manifest.name)
                return False

            ctx = ExtensionContext(self.brain, manifest, self.synaptiq_version)
            instance = graft_class()
            instance.name = manifest.name
            await instance.initialize(ctx)
            self.loaded_grafts[manifest.name] = instance
            logger.info("Loaded graft: %s v%s", manifest.name, manifest.version)
            return True
        except Exception:
            logger.exception("Failed to load graft from %s", graft_dir)
            return False

    async def unload_all(self) -> None:
        for name, graft in list(self.loaded_grafts.items()):
            try:
                await graft.teardown()
            except Exception:
                logger.exception("Error tearing down graft '%s'", name)
            del self.loaded_grafts[name]

    @staticmethod
    def _parse_version(v: str) -> tuple:
        return tuple(int(x) for x in v.split(".") if x.isdigit())
'''

    UI_INIT = '"""Synaptiq UI components."""\n'

    UI_CLI = r'''"""Rich CLI interface for Synaptiq."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ..core.brain import SynaptiqBrain

BANNER = "Synaptiq AI"


class SynaptiqCLI:
    """Interactive Rich-based CLI for the Synaptiq agent."""

    def __init__(self) -> None:
        self.console = Console()
        self.brain: Optional[SynaptiqBrain] = None

    async def run(self) -> None:
        """Start the interactive REPL."""
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.console.print(
                "[bold red]Error:[/bold red] Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.")
            sys.exit(1)

        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
        self.brain = SynaptiqBrain(api_key, provider=provider)
        await self.brain.initialize()

        self.console.print(Panel(
            Text(BANNER, style="bold cyan"),
            border_style="blue",
            subtitle=f"v{self.brain.version}"))
        self.console.print("[dim]Type 'exit' to quit. Type 'help' for commands.[/dim]\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask("[bold magenta]You >[/bold magenta]"))
                stripped = user_input.strip()
                if not stripped:
                    continue
                if stripped.lower() in ("exit", "quit"):
                    await self.brain.shutdown()
                    break
                if stripped.lower() == "help":
                    self._show_help()
                    continue
                if stripped.lower() == "status":
                    self._show_status()
                    continue

                async for response in self.brain.think(stripped):
                    self.console.print(Markdown(response))

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted.[/yellow]")
                continue
            except Exception as exc:
                self.console.print(f"\n[bold red]Error:[/bold red] {exc}")

    def _show_help(self) -> None:
        table = Table(title="Synaptiq Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")
        table.add_row("help", "Show this help message")
        table.add_row("status", "Show brain status and metrics")
        table.add_row("exit", "Exit the CLI")
        self.console.print(table)

    def _show_status(self) -> None:
        if self.brain is None:
            return
        status = self.brain.get_status()
        table = Table(title="Synaptiq Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for key, value in status.items():
            if isinstance(value, list):
                value = ", ".join(value) if value else "(none)"
            elif isinstance(value, dict):
                value = json.dumps(value)
            table.add_row(key, str(value))
        self.console.print(table)
'''

    TEST_CORE = r'''"""Basic tests for Synaptiq core components."""
import asyncio

import pytest

from synaptiq.core.memory import SpacedRepetitionMemory
from synaptiq.core.nerve import EventBus


def test_memory_add_fact():
    """Adding a fact should return a non-empty node ID."""
    mem = SpacedRepetitionMemory(db_path="./test_db")
    node_id = mem.add_fact("Python is a programming language", memory_type="semantic")
    assert node_id is not None
    assert node_id.startswith("mem_")


def test_memory_scratchpad():
    """Scratchpad should store and retrieve values correctly."""
    mem = SpacedRepetitionMemory(db_path="./test_db")
    mem.scratch("test_key", "test_value")
    assert mem.recall_scratch("test_key") == "test_value"
    mem.clear_scratchpad()
    assert mem.recall_scratch("test_key") is None


@pytest.mark.asyncio
async def test_event_bus_basic():
    """EventBus should deliver events to subscribers."""
    bus = EventBus()
    received = []

    async def handler(data):
        received.append(data)

    bus.subscribe("test_event", handler)
    await bus.publish("test_event", {"message": "hello"})

    task = asyncio.create_task(bus.start())
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(received) == 1
    assert received[0]["message"] == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''


# ═══════════════════════════════════════════════════════════════════════════
# Single-File Synaptiq Template
# ═══════════════════════════════════════════════════════════════════════════

SINGLE_FILE_TEMPLATE = r'''#!/usr/bin/env python3
"""Synaptiq single-file framework — runtime agent in one module."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("synaptiq")

__version__ = "1.0.0"


# ── EventBus ────────────────────────────────────────────────────────────────
class EventBus:
    def __init__(self) -> None:
        self._scratchpad: Dict[str, Any] = {}


# ── Memory ────────────────────────────────────────────────────────────────
class SpacedRepetitionMemory:
    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def add_fact(self, content: str, memory_type: str = "semantic") -> str:
        node_id = f"mem_{uuid.uuid4().hex[:8]}"
        self._store[node_id] = content
        return node_id

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return [{"id": k, "content": v} for k, v in list(self._store.items())[:top_k]]


# ── Provider ──────────────────────────────────────────────────────────────
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, tools, system) -> Dict[str, Any]: ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages, tools, system) -> Dict[str, Any]:
        r = await self.client.messages.create(
            model=self.model, max_tokens=4096, system=system,
            tools=tools, messages=messages)
        return {"content": r.content, "stop_reason": r.stop_reason}


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages, tools, system) -> Dict[str, Any]:
        oai = [{"role": "system", "content": system}] + messages
        r = await self.client.chat.completions.create(
            model=self.model, messages=oai, tools=tools)
        return {"content": [{"type": "text",
                             "text": r.choices[0].message.content or ""}],
                "stop_reason": "end_turn"}


# ── Tools ─────────────────────────────────────────────────────────────────
class BaseTool:
    name: str = "base"
    description: str = ""

    def __init__(self, nerve: EventBus) -> None:
        self.nerve = nerve

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        try:
            return await self._run(**kwargs)
        except Exception as exc:
            return {"error": str(exc)}

    async def _run(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code in a subprocess."

    async def _run(self, code: str = "") -> Dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {"stdout": out.decode(), "stderr": err.decode(),
                    "exit_code": proc.returncode}
        except asyncio.TimeoutError:
            return {"error": "timed out"}


class FileReadTool(BaseTool):
    name = "fs_read"
    description = "Read a file."

    async def _run(self, file_path: str = "") -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return {"content": f.read()[:4000], "path": file_path}
        except Exception as exc:
            return {"error": str(exc)}


class FileWriteTool(BaseTool):
    name = "fs_write"
    description = "Write a file."

    async def _run(self, file_path: str = "", content: str = "") -> Dict[str, Any]:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "bytes_written": len(content)}
        except Exception as exc:
            return {"error": str(exc)}


# ── Registry ──────────────────────────────────────────────────────────────
class ToolRegistry:
    def __init__(self, nerve: EventBus) -> None:
        self.nerve = nerve
        self.tools: Dict[str, BaseTool] = {}

    def register(self, cls) -> None:
        inst = cls(self.nerve)
        self.tools[inst.name] = inst

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": {"type": "object", "properties": {}}}
                for t in self.tools.values()]

    async def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        return await tool.execute(**args)


# ── Brain ─────────────────────────────────────────────────────────────────
class SynaptiqBrain:
    def __init__(self, api_key: str, provider: str = "anthropic",
                 model: Optional[str] = None) -> None:
        self.version = __version__
        if provider == "anthropic":
            self.llm = AnthropicProvider(api_key, model or "claude-3-5-sonnet-20241022")
        else:
            self.llm = OpenAIProvider(api_key, model or "gpt-4o")
        self.nerve = EventBus()
        self.memory = SpacedRepetitionMemory()
        self.registry = ToolRegistry(self.nerve)
        self.messages: List[Dict[str, str]] = []
        for t in (CodeExecTool, FileReadTool, FileWriteTool):
            self.registry.register(t)

    def get_system_prompt(self) -> str:
        return f"You are Synaptiq v{self.version}, a cognitive AI agent."

    async def think(self, user_input: str) -> AsyncGenerator[str, None]:
        self.messages.append({"role": "user", "content": user_input})
        try:
            r = await self.llm.chat(self.messages, self.registry.get_schemas(),
                                     self.get_system_prompt())
            text = ""
            for b in r.get("content", []):
                if isinstance(b, dict) and b.get("type") == "text":
                    text += b.get("text", "")
                elif isinstance(b, str):
                    text += b
            self.messages.append({"role": "assistant", "content": text})
            self.memory.add_fact(f"User asked: {user_input[:200]}")
            yield text or "[No response]"
        except Exception as exc:
            yield f"[Error: {exc}]"


# ── REPL ──────────────────────────────────────────────────────────────────
async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.")
        sys.exit(1)
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    brain = SynaptiqBrain(api_key, provider=provider)
    print(f"\n  Synaptiq AI v{brain.version}  (Ctrl+C to exit)\n")
    while True:
        try:
            user_input = input("\033[95mYou >\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            async for response in brain.think(user_input):
                print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
'''


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold Registry
# ═══════════════════════════════════════════════════════════════════════════

class Scaffold:
    """Metadata and file mapping for project initialisation."""
    DIRECTORIES: List[str] = [
        "synaptiq/core",
        "synaptiq/primitives",
        "synaptiq/personas",
        "synaptiq/skills",
        "synaptiq/extensions",
        "synaptiq/ui",
        "tests",
    ]

    FILES: Dict[str, str] = {
        "pyproject.toml": Templates.PYPROJECT,
        "README.md": Templates.README,
        ".gitignore": Templates.GITIGNORE,
        ".env.example": Templates.ENV_EXAMPLE,
        "synaptiq/__init__.py": Templates.INIT,
        "synaptiq/__main__.py": Templates.MAIN,
        "synaptiq/core/__init__.py": Templates.CORE_INIT,
        "synaptiq/core/nerve.py": Templates.CORE_NERVE,
        "synaptiq/core/memory.py": Templates.CORE_MEMORY,
        "synaptiq/core/session.py": Templates.CORE_SESSION,
        "synaptiq/core/middleware.py": Templates.CORE_MIDDLEWARE,
        "synaptiq/core/provider.py": Templates.CORE_PROVIDER,
        "synaptiq/core/registry.py": Templates.CORE_REGISTRY,
        "synaptiq/core/cortex.py": Templates.CORE_CORTEX,
        "synaptiq/core/consolidator.py": Templates.CORE_CONSOLIDATOR,
        "synaptiq/core/observability.py": Templates.CORE_OBSERVABILITY,
        "synaptiq/core/brain.py": Templates.CORE_BRAIN,
        "synaptiq/primitives/__init__.py": Templates.PRIMITIVES_INIT,
        "synaptiq/primitives/base.py": Templates.PRIMITIVES_BASE,
        "synaptiq/primitives/system.py": Templates.PRIMITIVES_SYSTEM,
        "synaptiq/primitives/cognitive.py": Templates.PRIMITIVES_COGNITIVE,
        "synaptiq/primitives/forge.py": Templates.PRIMITIVES_FORGE,
        "synaptiq/primitives/belief.py": Templates.PRIMITIVES_BELIEF,
        "synaptiq/primitives/working_memory.py": Templates.PRIMITIVES_WORKING_MEMORY,
        "synaptiq/personas/__init__.py": Templates.PERSONAS_INIT,
        "synaptiq/skills/__init__.py": Templates.SKILLS_INIT,
        "synaptiq/extensions/__init__.py": Templates.EXTENSIONS_INIT,
        "synaptiq/extensions/base.py": Templates.EXTENSIONS_BASE,
        "synaptiq/extensions/sandbox.py": Templates.EXTENSIONS_SANDBOX,
        "synaptiq/extensions/manager.py": Templates.EXTENSIONS_MANAGER,
        "synaptiq/ui/__init__.py": Templates.UI_INIT,
        "synaptiq/ui/cli.py": Templates.UI_CLI,
        "tests/test_core.py": Templates.TEST_CORE,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI Command Handlers
# ═══════════════════════════════════════════════════════════════════════════

class CommandHandler:
    """Encapsulates all CLI command logic, reducing global namespace bloat."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def _resolve_target(self, args: argparse.Namespace) -> Path:
        return Path(args.dir).resolve()

    def _write_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _scan_components(self, target: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
        tools_dir = target / "synaptiq" / "primitives"
        skills_dir = target / "synaptiq" / "skills"
        agents_dir = target / "synaptiq" / "personas"
        grafts_dir = target / "synaptiq_grafts"
        
        tools = ([f.stem for f in tools_dir.glob("*.py")
                  if f.name not in ("__init__.py", "base.py")]
                 if tools_dir.exists() else [])
        skills = ([f.stem for f in skills_dir.glob("*.py") if f.name != "__init__.py"]
                  if skills_dir.exists() else [])
        agents = ([f.stem for f in agents_dir.glob("*.py") if f.name != "__init__.py"]
                  if agents_dir.exists() else [])
        grafts = ([d.name for d in grafts_dir.iterdir()
                   if d.is_dir() and (d / "graft.json").exists()]
                  if grafts_dir.exists() else [])
        return tools, skills, agents, grafts

    # ── Project Lifecycle ────────────────────────────────────────────────

    def init(self, args: argparse.Namespace) -> int:
        parent = self._resolve_target(args)
        target = parent / args.name

        if args.single_file:
            return self._init_single_file(target, args)

        return self._init_full_project(target, args)

    def _init_single_file(self, target: Path, args: argparse.Namespace) -> int:
        target.mkdir(parents=True, exist_ok=True)
        target_file = target / "synaptiq_agent.py"
        if target_file.exists() and not args.force:
            self.console.error(f"{target_file} already exists. Use --force to overwrite.")
            return ExitCode.ALREADY_EXISTS

        self._write_file(target_file, SINGLE_FILE_TEMPLATE)
        self.console.success(f"Created single-file framework: {target_file}")
        self.console.bold("\n Next steps:")
        self.console.dim(f"  cd {target.name}")
        self.console.dim("  pip install anthropic openai  # rich optional")
        self.console.dim("  export ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY")
        self.console.dim(f"  python {target_file.name}\n")
        return ExitCode.SUCCESS

    def _init_full_project(self, target: Path, args: argparse.Namespace) -> int:
        files_to_write = dict(Scaffold.FILES)
        if args.no_tests:
            files_to_write = {k: v for k, v in files_to_write.items()
                              if not k.startswith("tests/")}

        if target.exists() and any(target.iterdir()) and not args.force:
            self.console.error(f"Directory '{target}' is not empty.")
            self.console.dim(" Use --force to overwrite existing files.")
            return ExitCode.ALREADY_EXISTS

        if args.dry_run:
            self.console.header("[DRY RUN] Preview")
            self.console.dim(f" Target: {target}\n")
            for d in Scaffold.DIRECTORIES:
                self.console.dim(f"  [dir]  {d}/")
            for f in sorted(files_to_write):
                self.console.dim(f"  [file] {f}")
            return ExitCode.SUCCESS

        target.mkdir(parents=True, exist_ok=True)
        self.console.bold(f"\n Creating project at: {target}\n")

        for d in Scaffold.DIRECTORIES:
            (target / d).mkdir(parents=True, exist_ok=True)
            if args.verbose:
                self.console.dim(f"  [+] mkdir {d}/")

        file_count, total_bytes = 0, 0
        for rel_path, content in sorted(files_to_write.items()):
            full_path = target / rel_path
            try:
                self._write_file(full_path, content)
                file_count += 1
                total_bytes += len(content)
                if args.verbose:
                    self.console.success(f"write {rel_path} ({len(content):,} bytes)")
            except OSError as exc:
                self.console.error(f"Failed to write {rel_path}: {exc}")
                return ExitCode.GENERIC_ERROR

        self.console.success(f"Wrote {file_count} files ({total_bytes:,} bytes).")

        if args.git_init:
            self.console.header("Git Repository")
            ok, msg = run_subprocess(["git", "init"], cwd=str(target))
            if ok:
                run_subprocess(["git", "add", "."], cwd=str(target))
                run_subprocess(["git", "commit", "-m", "Initial commit: Synaptiq project scaffold"], cwd=str(target))
                self.console.success("Git repository initialised.")
            else:
                self.console.warning(f"git init failed: {msg}")

        if args.venv:
            self.console.header("Virtual Environment")
            venv_path = target / ".venv"
            ok, msg = run_subprocess([sys.executable, "-m", "venv", str(venv_path)])
            if ok:
                self.console.success(f"Created virtual environment at: {venv_path}")
                pip_path = str(venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip")
                self.console.dim(" Installing dependencies (this may take a moment)...")
                ok, msg = run_subprocess([pip_path, "install", "-e", ".[dev]"], cwd=str(target))
                if ok:
                    self.console.success("Dependencies installed.")
                else:
                    self.console.warning(f"pip install failed. Run manually: {pip_path} install -e .[dev]")
            else:
                self.console.warning(f"venv creation failed: {msg}")

        self.console.header("Next Steps")
        print(f"\n  cd {target.name}")
        if not args.venv:
            activate = ".venv\\Scripts\\activate" if os.name == "nt" else "source .venv/bin/activate"
            self.console.dim("  python -m venv .venv")
            self.console.dim(f"  {activate}")
            self.console.dim("  pip install -e .[dev]")
        self.console.dim("  cp .env.example .env   # Add your API keys")
        self.console.dim("  synaptiq               # Launch the interactive CLI\n")
        return ExitCode.SUCCESS

    def run(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        if not (target / "synaptiq" / "__main__.py").exists():
            self.console.error("Not a Synaptiq project (missing synaptiq/__main__.py).")
            return ExitCode.NOT_FOUND
        self.console.header("Launching Synaptiq CLI")
        try:
            subprocess.run([sys.executable, "-m", "synaptiq"], cwd=str(target))
            return ExitCode.SUCCESS
        except KeyboardInterrupt:
            self.console.warning("Synaptiq CLI stopped.")
            return ExitCode.SUCCESS
        except Exception as exc:
            self.console.error(f"Failed to run CLI: {exc}")
            return ExitCode.GENERIC_ERROR

    def test(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        if not (target / "tests").exists():
            self.console.error("No 'tests/' directory found.")
            return ExitCode.NOT_FOUND
        self.console.header("Running pytest")
        try:
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                           cwd=str(target), check=True)
            return ExitCode.SUCCESS
        except subprocess.CalledProcessError:
            return ExitCode.GENERIC_ERROR

    def install(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        if not (target / "pyproject.toml").exists():
            self.console.error("pyproject.toml not found. Are you in a project root?")
            return ExitCode.NOT_FOUND
        self.console.header("Installing Dependencies")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
                           cwd=str(target), check=True)
            self.console.success("Dependencies installed.")
            return ExitCode.SUCCESS
        except subprocess.CalledProcessError as exc:
            self.console.error(f"pip install failed: {exc}")
            return ExitCode.GENERIC_ERROR

    # ── Component Scaffolding ────────────────────────────────────────────

    def add_tool(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        primitives_dir = target / "synaptiq" / "primitives"
        if not primitives_dir.exists():
            self.console.error("'synaptiq/primitives' directory not found. Is this a Synaptiq project?")
            return ExitCode.NOT_FOUND
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        class_name = to_class_name(safe_name) + "Tool"
        file_path = primitives_dir / f"{safe_name}.py"
        if file_path.exists():
            self.console.error(f"Tool '{safe_name}.py' already exists.")
            return ExitCode.ALREADY_EXISTS

        boilerplate = f'''"""Custom tool: {class_name}."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .base import BaseTool


class {class_name}Schema(BaseModel):
    input_param: str = Field(..., description="Example input parameter for {safe_name}")


class {class_name}Tool(BaseTool):
    """TODO: Describe what {safe_name} does."""

    name = "{safe_name}"
    description = "Description of what {safe_name} does."
    args_model = {class_name}Schema

    async def _run(self, input_param: str) -> Dict[str, Any]:
        # TODO: Implement tool logic
        return {{"status": "success", "input_received": input_param}}
'''
        self._write_file(file_path, boilerplate)
        self.console.success(f"Created tool file: {file_path}")
        self.console.bold("\n Next steps:")
        self.console.dim("  1. Implement the logic in _run")
        self.console.dim("  2. Register it in synaptiq/core/brain.py inside _register_core_tools:")
        self.console.dim(f"     self.registry.register({class_name}Tool)\n")
        return ExitCode.SUCCESS

    def add_skill(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        skills_dir = target / "synaptiq" / "skills"
        if not skills_dir.exists():
            self.console.error("'synaptiq/skills' directory not found.")
            return ExitCode.NOT_FOUND
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        class_name = to_class_name(safe_name) + "Skill"
        file_path = skills_dir / f"{safe_name}.py"
        if file_path.exists():
            self.console.error(f"Skill '{safe_name}.py' already exists.")
            return ExitCode.ALREADY_EXISTS

        boilerplate = f'''"""Custom Skill: {class_name}."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from synaptiq.primitives.base import BaseTool


class {class_name}Schema(BaseModel):
    task_description: str = Field(..., description="The complex task to execute")


class {class_name}Skill(BaseTool):
    """Executes the {safe_name} skill workflow."""

    name = "{safe_name}_skill"
    description = "Executes the {safe_name} skill workflow."
    args_model = {class_name}Schema

    async def _run(self, task_description: str) -> Dict[str, Any]:
        # TODO: Implement the multi-step skill workflow
        return {{"status": "skill_executed", "summary": "Skill logic not yet implemented."}}
'''
        self._write_file(file_path, boilerplate)
        self.console.success(f"Created skill file: {file_path}")
        self.console.bold("\n Next steps:")
        self.console.dim("  1. Implement the workflow logic in _run")
        self.console.dim("  2. Register it in synaptiq/core/brain.py:")
        self.console.dim(f"     self.registry.register({class_name}Skill)\n")
        return ExitCode.SUCCESS

    def add_agent(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        personas_dir = target / "synaptiq" / "personas"
        if not personas_dir.exists():
            self.console.error("'synaptiq/personas' directory not found.")
            return ExitCode.NOT_FOUND
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        class_name = to_class_name(safe_name)
        file_path = personas_dir / f"{safe_name}.py"
        if file_path.exists():
            self.console.error(f"Agent persona '{safe_name}.py' already exists.")
            return ExitCode.ALREADY_EXISTS

        boilerplate = f'''"""Agent Persona: {class_name}."""
from __future__ import annotations

from synaptiq.core.cortex import AgentPersona


def get_persona() -> AgentPersona:
    """Return the {class_name} agent persona configuration."""
    return AgentPersona(
        name="{class_name}",
        system_prompt="You are a {safe_name} agent. Focus on your specific task and return concise results.",
        allowed_tools=[
            "fs_read",
            "code_exec",
        ],
    )
'''
        self._write_file(file_path, boilerplate)
        self.console.success(f"Created agent persona: {file_path}")
        self.console.bold("\n Next steps:")
        self.console.dim("  1. Customize the system_prompt and allowed_tools")
        self.console.dim("  2. Import and use it:")
        self.console.dim(f"     from synaptiq.personas.{safe_name} import get_persona\n")
        return ExitCode.SUCCESS

    def add_graft(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        grafts_dir = target / "synaptiq_grafts"
        grafts_dir.mkdir(parents=True, exist_ok=True)
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        graft_dir = grafts_dir / safe_name
        if graft_dir.exists():
            self.console.error(f"Graft '{safe_name}' already exists.")
            return ExitCode.ALREADY_EXISTS

        graft_dir.mkdir()
        manifest = json.dumps({
            "name": safe_name, "version": "0.1.0", "min_synaptiq_version": "1.0.0",
            "description": "A custom neural graft.", "author": "",
            "permissions": [], "dependencies": [],
        }, indent=4)

        main_py = f'''"""Neural Graft: {safe_name}."""
from __future__ import annotations

import logging

from synaptiq.extensions.base import ExtensionContext, NeuralGraft

logger = logging.getLogger(__name__)


class Extension(NeuralGraft):
    name = "{safe_name}"

    async def initialize(self, ctx: ExtensionContext) -> None:
        logger.info("Graft '{safe_name}' initialised.")

    async def teardown(self) -> None:
        logger.info("Graft '{safe_name}' torn down.")
'''
        self._write_file(graft_dir / "graft.json", manifest)
        self._write_file(graft_dir / "main.py", main_py)
        self.console.success(f"Created graft directory: {graft_dir}")
        self.console.dim("  - graft.json (Manifest)")
        self.console.dim("  - main.py (Entry point)")
        self.console.dim("\n  The GraftManager will hot-reload this when you run synaptiq.\n")
        return ExitCode.SUCCESS

    def remove(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        type_map = {
            "tool": target / "synaptiq" / "primitives" / f"{safe_name}.py",
            "skill": target / "synaptiq" / "skills" / f"{safe_name}.py",
            "agent": target / "synaptiq" / "personas" / f"{safe_name}.py",
            "graft": target / "synaptiq_grafts" / safe_name,
        }
        target_path = type_map.get(args.type)
        if not target_path or not target_path.exists():
            self.console.error(f"{args.type.capitalize()} '{safe_name}' not found.")
            return ExitCode.NOT_FOUND

        if not args.yes and not self.console.confirm(f"You are about to delete: {target_path}"):
            self.console.warning("Aborted.")
            return ExitCode.SUCCESS

        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            self.console.success(f"Removed {args.type} '{safe_name}'.")
            if args.type in ("tool", "skill"):
                self.console.dim("  Reminder: remove the registration from synaptiq/core/brain.py\n")
            return ExitCode.SUCCESS
        except OSError as exc:
            self.console.error(f"Failed to remove: {exc}")
            return ExitCode.GENERIC_ERROR

    # ── Listing & Management ─────────────────────────────────────────────

    def list_agents(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        personas_dir = target / "synaptiq" / "personas"
        if not personas_dir.exists():
            self.console.warning("No synaptiq/personas directory found.")
            return ExitCode.SUCCESS

        personas: List[Tuple[str, str, List[str]]] = []
        for py_file in sorted(personas_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
                prompt_match = re.search(r'system_prompt\s*=\s*"([^"]+)"', content)
                tools_match = re.search(r"allowed_tools\s*=\s*\[(.*?)\]", content, re.DOTALL)
                name = name_match.group(1) if name_match else py_file.stem
                prompt = prompt_match.group(1) if prompt_match else "(no prompt)"
                tools = re.findall(r'"([^"]+)"', tools_match.group(1)) if tools_match else []
                personas.append((name, prompt, tools))
            except Exception:
                continue

        if not personas:
            self.console.warning("No valid agent personas found.")
            return ExitCode.SUCCESS

        if getattr(args, "json", False):
            print(json.dumps([{"name": n, "prompt": p, "tools": t}
                              for n, p, t in personas], indent=2))
            return ExitCode.SUCCESS

        self.console.success(f"Found {len(personas)} personas:\n")
        for name, prompt, tools in personas:
            self.console.bold(f"  ┌─ {name} ({len(tools)} tools)")
            self.console.dim(f"  │ Prompt: {prompt}")
            self.console.dim(f"  │ Tools: {', '.join(tools) if tools else 'None'}")
            self.console.dim("  └──────────────────────────────────────────────────────\n")
        return ExitCode.SUCCESS

    def update_agent(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        file_path = target / "synaptiq" / "personas" / f"{safe_name}.py"
        if not file_path.exists():
            self.console.error(f"Agent persona '{safe_name}' not found.")
            return ExitCode.NOT_FOUND

        content = file_path.read_text(encoding="utf-8")
        modified = False

        if args.set_prompt:
            encoded = json.dumps(args.set_prompt)
            content = re.sub(r'system_prompt\s*=\s*".*?"',
                             f"system_prompt={encoded}", content, count=1, flags=re.DOTALL)
            modified = True
            self.console.success(f"Updated system prompt for '{safe_name}'.")

        if args.add_tool:
            for tool in args.add_tool:
                if f'"{tool}"' not in content:
                    content = re.sub(
                        r"(allowed_tools\s*=\s*\[)([^\]]*?)(\])",
                        lambda m: (f'{m.group(1)}{m.group(2)}{" " if m.group(2).strip() else ""}"{tool}",\n{m.group(3)}'
                                   if m.group(2).strip()
                                   else f'{m.group(1)}\n    "{tool}",\n{m.group(3)}'),
                        content, count=1, flags=re.DOTALL,
                    )
                    modified = True
                    self.console.success(f"Added tool '{tool}' to '{safe_name}'.")
                else:
                    self.console.warning(f"Tool '{tool}' already exists in '{safe_name}'.")

        if args.remove_tool:
            for tool in args.remove_tool:
                pattern = r'\s*"' + re.escape(tool) + r'",?'
                new_content = re.sub(pattern, "", content)
                if new_content != content:
                    content = new_content
                    modified = True
                    self.console.success(f"Removed tool '{tool}' from '{safe_name}'.")
                else:
                    self.console.warning(f"Tool '{tool}' not found in '{safe_name}'.")

        if modified:
            self._write_file(file_path, content)
            self.console.success(f"Agent '{safe_name}' updated.\n")
        else:
            self.console.warning(f"No changes requested for '{safe_name}'.\n")
        return ExitCode.SUCCESS

    def test_agent(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        try:
            safe_name = sanitize_name(args.name)
        except InvalidNameError as exc:
            self.console.error(str(exc))
            return ExitCode.VALIDATION_ERROR

        file_path = target / "synaptiq" / "personas" / f"{safe_name}.py"
        if not file_path.exists():
            self.console.error(f"Agent persona '{safe_name}' not found.")
            return ExitCode.NOT_FOUND

        self.console.header(f"Testing Agent: {safe_name}")
        self.console.dim(f"  Simulated prompt: {args.prompt}\n")

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"personas.{safe_name}", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_persona"):
                    persona = module.get_persona()
                    self.console.bold("\n[System Prompt]")
                    self.console.dim(f"  {persona.system_prompt}")
                    self.console.bold("\n[Allowed Tools]")
                    if persona.allowed_tools:
                        for t in persona.allowed_tools:
                            self.console.dim(f"  - {t}")
                    else:
                        self.console.dim("  (none)")
                    self.console.bold("\n[User Message]")
                    self.console.dim(f"  {args.prompt}")
                    self.console.success("\nSimulation complete. In a live environment the LLM would process this request.\n")
                    return ExitCode.SUCCESS
        except Exception as exc:
            self.console.warning(f"Could not import module ({exc}). Falling back to regex.\n")

        content = file_path.read_text(encoding="utf-8")
        prompt_match = re.search(r'system_prompt\s*=\s*"([^"]+)"', content)
        tools_match = re.search(r"allowed_tools\s*=\s*\[(.*?)\]", content, re.DOTALL)
        self.console.bold("\n[System Prompt]")
        self.console.dim(f"  {prompt_match.group(1) if prompt_match else '(not found)'}")
        self.console.bold("\n[Allowed Tools]")
        if tools_match:
            tools = re.findall(r'"([^"]+)"', tools_match.group(1))
            for t in tools:
                self.console.dim(f"  - {t}")
        else:
            self.console.dim("  (none)")
        self.console.bold("\n[User Message]")
        self.console.dim(f"  {args.prompt}")
        self.console.success("\nSimulation complete.\n")
        return ExitCode.SUCCESS

    def list_tools(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        self.console.dim("\n Scanning for Synaptiq tools...\n")
        tools_dir = target / "synaptiq" / "primitives"
        skills_dir = target / "synaptiq" / "skills"
        grafts_dir = target / "synaptiq_grafts"
        all_tools: List[Tuple[str, str, str]] = []

        if tools_dir.exists():
            for py_file in sorted(tools_dir.glob("*.py")):
                if py_file.name in ("__init__.py", "base.py"):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    for match in re.findall(r"class\s+(\w+Tool)\b", content):
                        all_tools.append(("Primitive", py_file.stem, match))
                except Exception:
                    pass
        if skills_dir.exists():
            for py_file in sorted(skills_dir.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    for match in re.findall(r"class\s+(\w+Skill)\b", content):
                        all_tools.append(("Skill", py_file.stem, match))
                except Exception:
                    pass
        if grafts_dir.exists():
            for graft_dir in sorted(grafts_dir.iterdir()):
                if graft_dir.is_dir() and (graft_dir / "main.py").exists():
                    try:
                        content = (graft_dir / "main.py").read_text(encoding="utf-8")
                        for match in re.findall(r"class\s+(\w+Tool)\b", content):
                            all_tools.append(("Graft", graft_dir.name, match))
                    except Exception:
                        pass

        if not all_tools:
            self.console.warning("No tools found.")
            return ExitCode.SUCCESS

        if getattr(args, "json", False):
            print(json.dumps([{"type": t, "module": m, "class": c}
                              for t, m, c in all_tools], indent=2))
            return ExitCode.SUCCESS

        self.console.success(f"Found {len(all_tools)} tools:\n")
        self.console.table(["Type", "Module", "Class Name"],
                           [[t, m, c] for t, m, c in all_tools])
        print()
        return ExitCode.SUCCESS

    def list_grafts(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        self.console.dim("\n Scanning for Neural Grafts...\n")
        grafts_dir = target / "synaptiq_grafts"
        if not grafts_dir.exists():
            self.console.warning("No synaptiq_grafts directory found.")
            return ExitCode.SUCCESS

        found: List[List[str]] = []
        for graft_dir in sorted(grafts_dir.iterdir()):
            manifest_file = graft_dir / "graft.json"
            if graft_dir.is_dir() and manifest_file.exists():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    found.append([
                        data.get("name", "unknown"),
                        data.get("version", "0.0.0"),
                        data.get("description", "").replace("\n", " ")[:50],
                    ])
                except Exception:
                    pass

        if not found:
            self.console.warning("No valid grafts found.")
            return ExitCode.SUCCESS

        if getattr(args, "json", False):
            print(json.dumps([{"name": r[0], "version": r[1], "description": r[2]}
                              for r in found], indent=2))
            return ExitCode.SUCCESS

        self.console.success(f"Found {len(found)} grafts:\n")
        self.console.table(["Name", "Version", "Description"], found)
        print()
        return ExitCode.SUCCESS

    def info(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        self.console.dim("\n Scanning project directory...\n")
        synaptiq_dir = target / "synaptiq"
        if not synaptiq_dir.exists():
            self.console.error("Not a Synaptiq project (missing 'synaptiq/' directory).")
            return ExitCode.NOT_FOUND

        version = "Unknown"
        pyproject = target / "pyproject.toml"
        if pyproject.exists():
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version ="):
                    version = line.split("=", 1)[1].strip().strip("\"'")
                    break

        tools, skills, agents, grafts = self._scan_components(target)
        db_dir = target / "synaptiq_db"
        db_size_kb = (sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file()) / 1024
                      if db_dir.exists() else 0)

        if getattr(args, "json", False):
            print(json.dumps({
                "version": version, "directory": str(target),
                "tools": tools, "skills": skills, "agents": agents, "grafts": grafts,
                "memory_db_size_kb": round(db_size_kb, 2),
            }, indent=2))
            return ExitCode.SUCCESS

        self.console.header("Synaptiq Project Info")
        self.console.success(f"Version   : {version}")
        self.console.dim(f"Directory : {target}")
        self.console.success(f"\nTools ({len(tools)}):")
        for t in tools: self.console.dim(f"  - {t}")
        self.console.success(f"\nSkills ({len(skills)}):")
        for s in skills: self.console.dim(f"  - {s}")
        self.console.success(f"\nAgents ({len(agents)}):")
        for a in agents: self.console.dim(f"  - {a}")
        self.console.success(f"\nGrafts ({len(grafts)}):")
        for g in grafts: self.console.dim(f"  - {g}")
        if db_size_kb > 0:
            self.console.warning(f"\nMemory DB : {db_size_kb:.2f} KB")
        else:
            self.console.dim("\nMemory DB : Not initialised")
        print()
        return ExitCode.SUCCESS

    def memory(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        db_dir = target / "synaptiq_db"
        workspace_dir = target / "workspace"

        if args.action == "stats":
            self.console.header("Memory Statistics")
            if not db_dir.exists():
                self.console.warning("Memory database not found.")
                self.console.dim("  It will be created automatically when the agent first stores a memory.\n")
                return ExitCode.SUCCESS
            db_size = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())
            file_count = sum(1 for f in db_dir.rglob("*") if f.is_file())
            self.console.dim(f"  Path   : {db_dir}")
            self.console.success(f"  Size   : {db_size / 1024:.2f} KB")
            self.console.success(f"  Files  : {file_count}")
            if workspace_dir.exists():
                ws_size = sum(f.stat().st_size for f in workspace_dir.rglob("*") if f.is_file())
                self.console.dim(f"\n  Workspace : {workspace_dir}")
                self.console.success(f"  WS Size   : {ws_size / 1024:.2f} KB")
            print()
            return ExitCode.SUCCESS

        if args.action == "clear":
            dirs_to_clear = [db_dir, workspace_dir, target / "test_db"]
            if not any(d.exists() for d in dirs_to_clear):
                self.console.warning("No memory databases or workspaces found to clear.\n")
                return ExitCode.SUCCESS
            self.console.warning("The following will be permanently deleted:")
            for d in dirs_to_clear:
                if d.exists():
                    self.console.dim(f"  - {d}")
            if not args.yes and not self.console.confirm("\n Are you sure?"):
                self.console.warning("Aborted.")
                return ExitCode.SUCCESS
            self.console.dim("\n Clearing memory and workspace...")
            for d in dirs_to_clear:
                if d.exists() and d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    self.console.success(f"Removed {d.name}")
            self.console.success("Memory cleared.\n")
            return ExitCode.SUCCESS
        return ExitCode.SUCCESS

    def clear_cache(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        exact_dirs = ["__pycache__", ".pytest_cache", "build", "dist", ".mypy_cache", ".ruff_cache"]
        glob_patterns = ["*.egg-info"]

        self.console.warning("The following will be removed:")
        for name in exact_dirs:
            self.console.dim(f"  - {name}")
        for pat in glob_patterns:
            self.console.dim(f"  - {pat}")

        if not args.yes and not self.console.confirm("\n Are you sure?"):
            self.console.warning("Aborted.")
            return ExitCode.SUCCESS

        self.console.dim("\n Cleaning up...")
        for name in exact_dirs:
            path = target / name
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                self.console.success(f"Removed {name}")
        for pattern in glob_patterns:
            for path in target.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    self.console.success(f"Removed {path.name}")
        for pycache in target.rglob("__pycache__"):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)
        self.console.success("Cleanup complete.\n")
        return ExitCode.SUCCESS

    def docs(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        self.console.header("Generating ARCHITECTURE.md")
        synaptiq_dir = target / "synaptiq"
        if not synaptiq_dir.exists():
            self.console.error("Not a Synaptiq project (missing 'synaptiq/' directory).")
            return ExitCode.NOT_FOUND

        tools, skills, agents, grafts = self._scan_components(target)

        lines = [
            "# Synaptiq Architecture", "",
            "This document provides an overview of the Synaptiq project structure.", "",
            "## Core Components",
            "- **Brain** (`synaptiq/core/brain.py`): The main orchestrator.",
            "- **Memory** (`synaptiq/core/memory.py`): Spaced repetition and semantic search.",
            "- **Session** (`synaptiq/core/session.py`): Transactional quantum session with branching.",
            "- **Registry** (`synaptiq/core/registry.py`): Tool management.",
            "- **Cortex** (`synaptiq/core/cortex.py`): Multi-agent orchestration.", "",
            f"## Tools ({len(tools)})",
        ]
        lines.extend(f"- `{t}`" for t in tools)
        lines.extend(["", f"## Skills ({len(skills)})"])
        lines.extend(f"- `{s}`" for s in skills)
        lines.extend(["", f"## Agent Personas ({len(agents)})"])
        lines.extend(f"- `{a}`" for a in agents)
        lines.extend(["", f"## Neural Grafts ({len(grafts)})"])
        lines.extend(f"- `{g}`" for g in grafts)
        lines.append("")

        docs_path = target / "ARCHITECTURE.md"
        self._write_file(docs_path, "\n".join(lines))
        self.console.success(f"Generated {docs_path}\n")
        return ExitCode.SUCCESS

    def docker(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        dockerfile_path = target / "Dockerfile"
        dockerignore_path = target / ".dockerignore"
        if dockerfile_path.exists() or dockerignore_path.exists():
            self.console.error("Dockerfile or .dockerignore already exists.")
            return ExitCode.ALREADY_EXISTS
        self._write_file(dockerfile_path, Templates.DOCKERFILE)
        self._write_file(dockerignore_path, Templates.DOCKERIGNORE)
        self.console.success("Created Dockerfile")
        self.console.success("Created .dockerignore")
        self.console.bold("\n To build and run:")
        self.console.dim("  docker build -t synaptiq-agent .")
        self.console.dim("  docker run -it --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY synaptiq-agent\n")
        return ExitCode.SUCCESS

    def config(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        env_path = target / ".env"
        env_example = target / ".env.example"

        if not env_path.exists() and env_example.exists():
            shutil.copy(env_example, env_path)
            self.console.success("Created .env from .env.example")
        if not env_path.exists():
            self._write_file(env_path, "# Synaptiq Environment Configuration\n")
            self.console.success("Created empty .env file")

        if args.action == "show":
            self.console.bold(f"\n Contents of {env_path}:\n")
            self.console.dim(env_path.read_text(encoding="utf-8"))
            return ExitCode.SUCCESS

        if args.action == "set":
            if not args.key or not args.value:
                self.console.error("Both key and value must be provided for 'set'.")
                return ExitCode.VALIDATION_ERROR
            lines = env_path.read_text(encoding="utf-8").splitlines()
            new_lines: List[str] = []
            found = False
            for line in lines:
                if line.strip().startswith(f"{args.key}="):
                    new_lines.append(f"{args.key}={args.value}")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"{args.key}={args.value}")
            self._write_file(env_path, "\n".join(new_lines) + "\n")
            self.console.success(f"Set {args.key} in .env")
            return ExitCode.SUCCESS
        return ExitCode.SUCCESS

    def doctor(self, args: argparse.Namespace) -> int:
        target = self._resolve_target(args)
        self.console.header("Synaptiq Doctor")

        checks: List[Tuple[str, bool, str]] = []
        checks.append(("Python >= 3.10", sys.version_info >= (3, 10),
                       f"Python {sys.version.split()[0]}"))
        checks.append(("pyproject.toml exists", (target / "pyproject.toml").exists(),
                       str(target / "pyproject.toml")))
        checks.append(("synaptiq/__init__.py", (target / "synaptiq" / "__init__.py").exists(),
                       str(target / "synaptiq" / "__init__.py")))
        checks.append(("synaptiq/__main__.py", (target / "synaptiq" / "__main__.py").exists(),
                       str(target / "synaptiq" / "__main__.py")))
        checks.append(("synaptiq/core/brain.py",
                       (target / "synaptiq" / "core" / "brain.py").exists(),
                       str(target / "synaptiq" / "core" / "brain.py")))
        env = target / ".env"
        env_example = target / ".env.example"
        checks.append((".env file", env.exists() or env_example.exists(), str(env)))
        checks.append(("tests/ directory", (target / "tests").exists(), str(target / "tests")))
        pycache_count = sum(1 for _ in target.rglob("__pycache__"))
        checks.append(("No __pycache__ dirs", pycache_count == 0, f"{pycache_count} found"))

        all_passed = True
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            color = Color.GREEN if passed else Color.RED
            self.console.raw(f"  {self.console._wrap(f'[{status}]', color)} {name:<25} "
                             f"{self.console._wrap(detail, Color.DIM)}")
            if not passed:
                all_passed = False

        if all_passed:
            self.console.success("\nAll checks passed — project looks healthy!\n")
        else:
            self.console.warning("\nSome checks failed — review the issues above.\n")
        return ExitCode.SUCCESS if all_passed else ExitCode.VALIDATION_ERROR

    def list_scaffold(self, args: argparse.Namespace) -> int:
        self.console.bold("\n Base Init Scaffold Contents:\n")
        self.console.bold(" Directories:")
        for d in Scaffold.DIRECTORIES:
            self.console.dim(f"  [dir]  {d}/")
        self.console.bold(f"\n Files ({len(Scaffold.FILES)} total):")
        total = 0
        for f in sorted(Scaffold.FILES):
            size = len(Scaffold.FILES[f])
            total += size
            self.console.dim(f"  [file] {f:<45} {size:>6,} bytes")
        self.console.dim(f"\n  {'Total':<52} {total:>6,} bytes\n")
        return ExitCode.SUCCESS


# ═══════════════════════════════════════════════════════════════════════════
# CLI Parser Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-d", "--dir", default=".", help="Project root directory")
    common.add_argument("--no-color", action="store_true", help="Disable coloured output")
    common.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    confirm = argparse.ArgumentParser(add_help=False)
    confirm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    json_out = argparse.ArgumentParser(add_help=False)
    json_out.add_argument("--json", action="store_true", help="Output as JSON")

    parser = argparse.ArgumentParser(
        prog="synaptiq-cli",
        description="Generate and manage Synaptiq AI agent framework projects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'synaptiq-cli <command> --help' for command-specific options.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"synaptiq-cli {VERSION}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", metavar="<command>")

    p = subparsers.add_parser("init", parents=[common], help="Initialise a new Synaptiq project")
    p.add_argument("name", nargs="?", default="synaptiq-project", help="Project directory name")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing files")
    p.add_argument("-v", "--verbose", action="store_true", help="Print each file as it is created")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    p.add_argument("--no-tests", action="store_true", help="Skip generating test files")
    p.add_argument("--git-init", action="store_true", help="Initialise a git repository")
    p.add_argument("--venv", action="store_true", help="Create a venv and install dependencies")
    p.add_argument("--single-file", action="store_true",
                   help="Generate a single synaptiq_agent.py file instead of a full package")

    p = subparsers.add_parser("add-tool", parents=[common], help="Scaffold a new custom tool")
    p.add_argument("name", help="Name of the tool (snake_case)")

    p = subparsers.add_parser("add-skill", parents=[common], help="Scaffold a new skill")
    p.add_argument("name", help="Name of the skill")

    p = subparsers.add_parser("add-agent", parents=[common], help="Scaffold a new agent persona")
    p.add_argument("name", help="Name of the agent persona")

    subparsers.add_parser("list-agents", parents=[common, json_out],
                          help="List all configured agent personas")

    p = subparsers.add_parser("update-agent", parents=[common],
                              help="Modify an agent's prompt or tools")
    p.add_argument("name", help="Agent persona name to update")
    p.add_argument("-p", "--set-prompt", help="New system prompt")
    p.add_argument("--add-tool", action="append", default=[], help="Add a tool to allowed_tools")
    p.add_argument("--remove-tool", action="append", default=[], help="Remove a tool from allowed_tools")

    p = subparsers.add_parser("test-agent", parents=[common],
                              help="Simulate a prompt with an agent persona")
    p.add_argument("name", help="Agent persona name to test")
    p.add_argument("prompt", help="Simulated user prompt")

    p = subparsers.add_parser("add-graft", parents=[common],
                              help="Scaffold a new neural graft (extension)")
    p.add_argument("name", help="Name of the graft")

    p = subparsers.add_parser("remove", parents=[common, confirm],
                              help="Remove a tool, skill, agent, or graft")
    p.add_argument("type", choices=["tool", "skill", "agent", "graft"], help="Component type")
    p.add_argument("name", help="Name of the component")

    subparsers.add_parser("docker", parents=[common],
                          help="Generate Dockerfile and .dockerignore")
    subparsers.add_parser("install", parents=[common], help="Install project dependencies")

    p = subparsers.add_parser("config", parents=[common],
                              help="Manage environment variables (.env)")
    p.add_argument("action", choices=["show", "set"], help="Action to perform")
    p.add_argument("key", nargs="?", help="Environment variable key (for 'set')")
    p.add_argument("value", nargs="?", help="Environment variable value (for 'set')")

    subparsers.add_parser("info", parents=[common, json_out], help="Show project information")
    subparsers.add_parser("run", parents=[common], help="Launch the Synaptiq interactive CLI")
    subparsers.add_parser("test", parents=[common], help="Run the pytest suite")
    subparsers.add_parser("list-tools", parents=[common, json_out], help="List all available tools")
    subparsers.add_parser("list-grafts", parents=[common, json_out],
                          help="List all installed Neural Grafts")

    p = subparsers.add_parser("memory", parents=[common, confirm], help="Manage cognitive memory")
    p.add_argument("action", choices=["stats", "clear"], help="Action to perform")

    subparsers.add_parser("clear-cache", parents=[common, confirm],
                          help="Clean up generated artifacts")
    subparsers.add_parser("docs", parents=[common], help="Generate ARCHITECTURE.md")
    subparsers.add_parser("doctor", parents=[common], help="Diagnose project health")
    subparsers.add_parser("list-scaffold", parents=[common],
                          help="List files in the base init scaffold")

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Backward compatibility: assume 'init' if just a project name is passed
    known_commands = [
        "init", "add-tool", "add-skill", "add-agent", "list-agents", "update-agent", 
        "test-agent", "add-graft", "remove", "docker", "install", "config", "info", 
        "run", "test", "list-tools", "list-grafts", "memory", "clear-cache", "docs", 
        "doctor", "list-scaffold", "-h", "--help", "--version", "-V"
    ]
    if argv and not argv[0].startswith('-') and argv[0] not in known_commands:
        argv = ["init"] + list(argv)
    elif not argv:
        argv = ["init"]

    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console(
        use_color=not getattr(args, "no_color", False),
        quiet=getattr(args, "quiet", False),
    )
    console.banner()

    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return ExitCode.SUCCESS

    handler = CommandHandler(console)
    cmd_func = getattr(handler, command.replace("-", "_"), None)
    
    if cmd_func is None:
        parser.print_help()
        return ExitCode.SUCCESS

    try:
        return int(cmd_func(args))
    except KeyboardInterrupt:
        console.warning("\nInterrupted by user.")
        return ExitCode.INTERRUPTED
    except SynaptiqError as exc:
        console.error(str(exc))
        return ExitCode.GENERIC_ERROR
    except Exception as exc:
        console.error(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        return ExitCode.GENERIC_ERROR


if __name__ == "__main__":
    sys.exit(main())