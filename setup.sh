#!/bin/bash

# Synaptiq Project Generator
# This script generates the complete Synaptiq AI agentic framework.

mkdir -p synaptiq-project/synaptiq/core
mkdir -p synaptiq-project/synaptiq/primitives
mkdir -p synaptiq-project/synaptiq/extensions
mkdir -p synaptiq-project/synaptiq/ui
mkdir -p synaptiq-project/tests

cd synaptiq-project

# --- pyproject.toml ---
cat << '__EOF_SYNAPTIQ__' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "synaptiq"
version = "1.0.0"
description = "An event-driven, cognitive AI agent framework with graph memory, quantum sessions, and multi-agent orchestration"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Synaptiq Core Team"}
]
keywords = ["ai", "agent", "llm", "framework", "cognitive", "multi-agent"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "anthropic>=0.18.0",
    "openai>=1.0.0",
    "networkx>=3.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "rich>=13.0.0",
    "prompt-toolkit>=3.0.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "watchfiles>=0.20.0",
    "semver>=3.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
synaptiq = "synaptiq.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["synaptiq*"]
__EOF_SYNAPTIQ__

# --- README.md ---
cat << '__EOF_SYNAPTIQ__' > README.md
# Synaptiq

An event-driven, cognitive AI agent framework with graph memory, quantum sessions, and multi-agent orchestration.

## Features

- **Cognitive Memory**: Spaced repetition, belief revision, semantic search, working memory
- **Quantum Sessions**: Branching context trees, transactional filesystem, episodic crystallization
- **Multi-Agent**: Sub-agent delegation, hive mind consensus, daemon watchers
- **Neural Grafts**: Sandboxed extensions with hot-reloading, middleware, UI panels
- **Self-Evolution**: Dynamic tool forging, belief correction, auto-compaction
- **Multi-Provider**: Anthropic, OpenAI, local LLM support
__EOF_SYNAPTIQ__

# --- synaptiq/__init__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/__init__.py
"""Synaptiq: An event-driven, cognitive AI agent framework."""

__version__ = "1.0.0"
__author__ = "Synaptiq Core Team"

from .core.brain import SynaptiqBrain
from .core.nerve import EventBus
from .core.memory import SpacedRepetitionMemory
from .core.session import TransactionalQuantumSession
from .core.registry import ToolRegistry
from .core.provider import AnthropicProvider, OpenAIProvider, LLMProvider
from .core.middleware import ToolMiddleware
from .core.cortex import SubAgent, HiveMindTool, DelegateTaskTool, DaemonAgent
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
    "SynaptiqBrain", "EventBus", "SpacedRepetitionMemory", "TransactionalQuantumSession",
    "ToolRegistry", "AnthropicProvider", "OpenAIProvider", "LLMProvider", "ToolMiddleware",
    "SubAgent", "HiveMindTool", "DelegateTaskTool", "DaemonAgent", "SleepConsolidator",
    "KnowledgeConsolidator", "ObservabilityManager", "BaseTool", "CodeExecTool", "FileReadTool",
    "FileWriteTool", "DivideAndConquerTool", "SimulateFutureTool", "ForgeTool", "BeliefRevisionTool",
    "WorkingMemoryTool", "NeuralGraft", "ExtensionContext", "GraftManifest", "GraftManager", "SandboxedEnvironment",
]
__EOF_SYNAPTIQ__

# --- synaptiq/__main__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/__main__.py
"""CLI entry point for Synaptiq."""
import asyncio
import sys
import os
from pathlib import Path

from .ui.cli import SynaptiqCLI

def main():
    """Main CLI entry point."""
    cli = SynaptiqCLI()
    asyncio.run(cli.run())

if __name__ == "__main__":
    main()
__EOF_SYNAPTIQ__

# --- synaptiq/core/__init__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/__init__.py
"""Synaptiq core modules."""
__EOF_SYNAPTIQ__

# --- synaptiq/core/nerve.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/nerve.py
"""The central nervous system - async event bus for inter-component communication."""
import asyncio
from typing import Any, Dict, List, Callable, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """The central nervous system of Synaptiq."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def subscribe(self, event_type: str, callback: Callable):
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self._subscribers:
            try: self._subscribers[event_type].remove(callback)
            except ValueError: pass

    async def publish(self, event_type: str, data: Dict[str, Any]):
        await self._queue.put({"type": event_type, "data": data})

    async def start(self):
        self._running = True
        while self._running:
            event = await self._queue.get()
            if event is None: break
            event_type = event["type"]
            data = event["data"]
            for callback in self._subscribers.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(callback): await callback(data)
                    else: callback(data)
                except Exception as e:
                    logger.error(f"Error processing event {event_type}: {e}")
            self._queue.task_done()

    async def stop(self):
        self._running = False
        await self._queue.put(None)
        if self._task: await self._task

    async def wait_until_empty(self):
        await self._queue.join()
__EOF_SYNAPTIQ__

# --- synaptiq/core/memory.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/memory.py
"""Spaced repetition memory with semantic search, graph relationships, and working memory."""
import time
import uuid
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

class SpacedRepetitionMemory:
    def __init__(self, db_path: str = "./synaptiq_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path=str(self.db_path / "chroma"))
            self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
            self.collection = self.chroma_client.get_or_create_collection(name="cognitive_memory", embedding_function=self.embedder, metadata={"hnsw:space": "cosine"})
        else:
            self.collection = None
            self._fallback_store: List[Dict[str, Any]] = []
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.scratchpad: Dict[str, Any] = {}

    def add_fact(self, content: str, memory_type: str = "semantic", strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        node_id = f"mem_{uuid.uuid4().hex[:8]}"
        current_time = time.time()
        meta = {"type": memory_type, "strength": strength, "created_at": current_time, "last_accessed": current_time, "access_count": 0, "sm2_interval": 1, "sm2_repetitions": 0, "sm2_ease_factor": 2.5, "status": "active"}
        if metadata: meta.update(metadata)
        if self.collection:
            self.collection.add(ids=[node_id], documents=[content], metadatas=[meta])
        else:
            self._fallback_store.append({"id": node_id, "document": content, "metadata": meta})
        if self.graph: self.graph.add_node(node_id, content=content, type=memory_type, **meta)
        return node_id

    def semantic_search(self, query: str, top_k: int = 3, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.collection:
            where_filter = {"status": "active"}
            if memory_type: where_filter["type"] = memory_type
            results = self.collection.query(query_texts=[query], n_results=top_k * 2, where=where_filter)
            formatted = []
            for i, doc_id in enumerate(results['ids'][0]):
                meta = results['metadatas'][0][i]
                is_due = meta['last_accessed'] + (meta['sm2_interval'] * 86400) < time.time()
                review_boost = 1.2 if is_due else 1.0
                similarity = 1.0 - results['distances'][0][i]
                final_score = similarity * review_boost
                formatted.append({"id": doc_id, "content": results['documents'][0][i], "score": float(final_score), "metadata": meta})
            formatted.sort(key=lambda x: x['score'], reverse=True)
            for item in formatted[:top_k]: self._update_access(item['id'])
            return formatted[:top_k]
        else:
            results = []
            query_lower = query.lower()
            for item in self._fallback_store:
                if item['metadata'].get('status') != 'active': continue
                if memory_type and item['metadata'].get('type') != memory_type: continue
                content_lower = item['document'].lower()
                if query_lower in content_lower:
                    score = content_lower.count(query_lower) / len(content_lower.split())
                    results.append({"id": item['id'], "content": item['document'], "score": score, "metadata": item['metadata']})
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

    def _update_access(self, node_id: str):
        if self.collection:
            data = self.collection.get(ids=[node_id])
            if data['metadatas']:
                meta = data['metadatas'][0]
                meta['last_accessed'] = time.time()
                meta['access_count'] = meta.get('access_count', 0) + 1
                self.collection.update(ids=[node_id], metadatas=[meta])

    def update_memory_srs(self, node_id: str, quality: int):
        if self.collection:
            data = self.collection.get(ids=[node_id])
            if not data['metadatas']: return
            meta = data['metadatas'][0]
        else:
            meta = None
            for item in self._fallback_store:
                if item['id'] == node_id: meta = item['metadata']; break
            if not meta: return
        if quality < 3:
            meta['sm2_repetitions'] = 0; meta['sm2_interval'] = 1
        else:
            meta['sm2_repetitions'] += 1
            if meta['sm2_repetitions'] == 1: meta['sm2_interval'] = 1
            elif meta['sm2_repetitions'] == 2: meta['sm2_interval'] = 3
            else: meta['sm2_interval'] = round(meta['sm2_interval'] * meta['sm2_ease_factor'])
            meta['sm2_ease_factor'] = max(1.3, meta['sm2_ease_factor'] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        meta['last_accessed'] = time.time()
        if self.collection: self.collection.update(ids=[node_id], metadatas=[meta])

    def supersede_fact(self, old_id: str, new_id: str, reason: str):
        if self.collection:
            data = self.collection.get(ids=[old_id])
            if data['metadatas']:
                meta = data['metadatas'][0]; meta['status'] = 'superseded'
                self.collection.update(ids=[old_id], metadatas=[meta])
        else:
            for item in self._fallback_store:
                if item['id'] == old_id: item['metadata']['status'] = 'superseded'; break
        if self.graph: self.graph.add_edge(old_id, new_id, relationship=f"superseded_by: {reason}")

    def connect_facts(self, source_id: str, target_id: str, relationship: str):
        if self.graph:
            if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                self.graph.add_edge(source_id, target_id, relationship=relationship)

    def find_path(self, source_concept: str, target_concept: str) -> List[str]:
        if not self.graph: return ["Graph not available"]
        src_nodes = [n for n in self.graph.nodes if source_concept.lower() in str(n).lower()]
        tgt_nodes = [n for n in self.graph.nodes if target_concept.lower() in str(n).lower()]
        if not src_nodes or not tgt_nodes: return ["Could not find matching nodes in memory."]
        try:
            path = nx.shortest_path(self.graph, source=src_nodes[0], target=tgt_nodes[0])
            path_narrative = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.graph.edges[u, v]
                path_narrative.append(f"{u} --[{edge_data['relationship']}]--> {v}")
            return path_narrative
        except nx.NetworkXNoPath: return ["No cognitive path found between these concepts."]
        except Exception as e: return [f"Pathfinding error: {e}"]

    def consolidate(self) -> int:
        forgotten = 0; current_time = time.time()
        if self.collection:
            all_data = self.collection.get(where={"status": "active"})
            for i, meta in enumerate(all_data['metadatas']):
                time_since_access = current_time - meta['last_accessed']
                decay = math.exp(-time_since_access / (60 * 60 * 24 * 30))
                meta['strength'] = meta.get('strength', 1.0) * decay
                if meta['strength'] < 0.1 and meta.get('access_count', 0) == 0:
                    meta['status'] = 'archived'; forgotten += 1
            if forgotten > 0: self.collection.update(ids=all_data['ids'], metadatas=all_data['metadatas'])
        return forgotten

    def get_stats(self) -> Dict[str, Any]:
        if self.collection:
            all_data = self.collection.get()
            total = len(all_data['ids'])
            active = sum(1 for m in all_data['metadatas'] if m.get('status') == 'active')
            superseded = sum(1 for m in all_data['metadatas'] if m.get('status') == 'superseded')
            archived = sum(1 for m in all_data['metadatas'] if m.get('status') == 'archived')
        else:
            total = len(self._fallback_store)
            active = sum(1 for i in self._fallback_store if i['metadata'].get('status') == 'active')
            superseded = sum(1 for i in self._fallback_store if i['metadata'].get('status') == 'superseded')
            archived = sum(1 for i in self._fallback_store if i['metadata'].get('status') == 'archived')
        return {"total_memories": total, "active": active, "superseded": superseded, "archived": archived, "graph_nodes": self.graph.number_of_nodes() if self.graph else 0, "graph_edges": self.graph.number_of_edges() if self.graph else 0, "scratchpad_items": len(self.scratchpad)}

    def scratch(self, key: str, value: Any): self.scratchpad[key] = value
    def recall_scratch(self, key: str) -> Any: return self.scratchpad.get(key)
    def clear_scratchpad(self): self.scratchpad.clear()

    def export_graph(self) -> str:
        if not self.graph: return "{}"
        try:
            from networkx.readwrite import json_graph
            return json.dumps(json_graph.node_link_data(self.graph), indent=2)
        except Exception: return "{}"
__EOF_SYNAPTIQ__

# --- synaptiq/core/session.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/session.py
"""Transactional quantum session with branching, crystallization, and filesystem snapshots."""
import time
import uuid
import json
import shutil
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)

class TransactionalQuantumSession:
    def __init__(self, fast_llm=None, workspace_dir: str = "./workspace", max_tokens: int = 80000):
        self.fast_llm = fast_llm
        self.max_tokens = max_tokens
        self.workspace = Path(workspace_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.workspace / ".synaptiq_snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        if NETWORKX_AVAILABLE: self.graph = nx.DiGraph()
        else: self.graph = None; self._fallback_nodes: Dict[str, Dict] = {}
        self.root_id = "root"
        self.active_node = self.root_id
        self._node_counter = 0
        self._init_root()

    def _init_root(self):
        if self.graph:
            self.graph.add_node(self.root_id, messages=[], role="system", is_crystallized=False, snapshot_path=None, timestamp=time.time())
        else:
            self._fallback_nodes[self.root_id] = {"messages": [], "role": "system", "is_crystallized": False, "snapshot_path": None, "timestamp": time.time(), "children": []}

    def _get_node(self, node_id: str) -> Optional[Dict]:
        if self.graph: return self.graph.nodes.get(node_id)
        else: return self._fallback_nodes.get(node_id)

    def _add_node(self, node_id: str, parent_id: str, data: Dict):
        if self.graph:
            self.graph.add_node(node_id, **data)
            self.graph.add_edge(parent_id, node_id)
        else:
            data["children"] = []
            self._fallback_nodes[node_id] = data
            if parent_id in self._fallback_nodes: self._fallback_nodes[parent_id]["children"].append(node_id)

    def add_message(self, role: str, content: Any, fs_mutated: bool = False, parent_id: Optional[str] = None) -> str:
        parent = parent_id or self.active_node
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"
        parent_data = self._get_node(parent)
        if parent_data is None: parent_data = {"messages": []}
        parent_messages = parent_data.get("messages", [])
        current_messages = parent_messages + [{"role": role, "content": content}]
        snapshot_path = None
        if fs_mutated:
            snapshot_path = self.snapshot_dir / node_id
            snapshot_path.mkdir(exist_ok=True)
            try:
                for item in self.workspace.iterdir():
                    if item.name != ".synaptiq_snapshots":
                        if item.is_dir(): shutil.copytree(item, snapshot_path / item.name, dirs_exist_ok=True)
                        else: shutil.copy2(item, snapshot_path / item.name)
            except Exception as e:
                logger.warning(f"Failed to create snapshot: {e}")
                snapshot_path = None
        node_data = {"messages": current_messages, "role": role, "is_crystallized": False, "snapshot_path": str(snapshot_path) if snapshot_path else None, "timestamp": time.time()}
        self._add_node(node_id, parent, node_data)
        self.active_node = node_id
        if self._estimate_tokens(current_messages) > self.max_tokens: asyncio.create_task(self.crystallize_branch())
        return node_id

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    async def crystallize_branch(self):
        if not self.fast_llm: return
        active_data = self._get_node(self.active_node)
        if not active_data: return
        active_msgs = active_data.get("messages", [])
        prompt = "Summarize the following conversation history into a dense, factual episode log:\n\n"
        for m in active_msgs:
            content = m.get("content", "")
            if isinstance(content, str): prompt += f"{m['role']}: {content[:500]}\n"
            else: prompt += f"{m['role']}: [complex content]\n"
        try:
            response = self.fast_llm.messages.create(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}])
            summary = response.content[0].text
            self._node_counter += 1
            crystallized_id = f"crystal_{self._node_counter}"
            node_data = {"messages": [{"role": "system", "content": f"[Crystallized Episode Summary]:\n{summary}"}], "role": "system", "is_crystallized": True, "snapshot_path": active_data.get("snapshot_path"), "timestamp": time.time()}
            self._add_node(crystallized_id, self.active_node, node_data)
            self.active_node = crystallized_id
        except Exception as e: logger.error(f"Crystallization failed: {e}")

    def rollback(self, to_node: str) -> bool:
        target_data = self._get_node(to_node)
        if target_data is None: return False
        snapshot_path_str = target_data.get("snapshot_path")
        if snapshot_path_str and snapshot_path_str != "None":
            snapshot_path = Path(snapshot_path_str)
            if snapshot_path.exists():
                try:
                    for item in self.workspace.iterdir():
                        if item.name != ".synaptiq_snapshots":
                            if item.is_dir(): shutil.rmtree(item)
                            else: item.unlink()
                    for item in snapshot_path.iterdir():
                        if item.is_dir(): shutil.copytree(item, self.workspace / item.name, dirs_exist_ok=True)
                        else: shutil.copy2(item, self.workspace / item.name)
                except Exception as e:
                    logger.error(f"Filesystem rollback failed: {e}")
                    return False
        self.active_node = to_node
        return True

    def branch(self, from_node: Optional[str] = None) -> str:
        branch_parent = from_node or self.active_node
        return self.add_message("system", "[Branching Context]", parent_id=branch_parent)

    def get_active_messages(self) -> List[Dict[str, str]]:
        data = self._get_node(self.active_node)
        if data is None: return []
        return data.get("messages", [])

    def prune_branch(self, node_id: str):
        if node_id == self.root_id: return
        if self.graph:
            descendants = list(nx.descendants(self.graph, node_id)) if NETWORKX_AVAILABLE else []
            for n in [node_id] + descendants:
                if n in self.graph:
                    node_data = self.graph.nodes[n]
                    snapshot_path = node_data.get("snapshot_path")
                    if snapshot_path and Path(snapshot_path).exists(): shutil.rmtree(snapshot_path, ignore_errors=True)
                    self.graph.remove_node(n)
        else:
            if node_id in self._fallback_nodes: del self._fallback_nodes[node_id]

    def get_session_tree(self) -> Dict[str, Any]:
        if self.graph:
            nodes = []
            for node_id in self.graph.nodes:
                data = self.graph.nodes[node_id]
                nodes.append({"id": node_id, "role": data.get("role", ""), "is_crystallized": data.get("is_crystallized", False), "timestamp": data.get("timestamp", 0), "message_count": len(data.get("messages", [])), "is_active": node_id == self.active_node})
            edges = [{"from": u, "to": v} for u, v in self.graph.edges]
            return {"nodes": nodes, "edges": edges, "active_node": self.active_node}
        else:
            nodes = []
            for node_id, data in self._fallback_nodes.items():
                nodes.append({"id": node_id, "role": data.get("role", ""), "is_crystallized": data.get("is_crystallized", False), "timestamp": data.get("timestamp", 0), "message_count": len(data.get("messages", [])), "is_active": node_id == self.active_node})
            return {"nodes": nodes, "edges": [], "active_node": self.active_node}
__EOF_SYNAPTIQ__

# --- synaptiq/core/middleware.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/middleware.py
"""Tool middleware system for interception and modification of tool calls."""
from typing import Dict, Any, List, Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ToolMiddleware:
    def __init__(self):
        self._chains: Dict[str, List[Tuple[str, Callable]]] = {}

    def add_middleware(self, tool_name: str, ext_name: str, callback: Callable):
        if tool_name not in self._chains: self._chains[tool_name] = []
        self._chains[tool_name].append((ext_name, callback))

    def remove_middleware(self, tool_name: str, ext_name: str):
        if tool_name in self._chains:
            self._chains[tool_name] = [(name, cb) for name, cb in self._chains[tool_name] if name != ext_name]

    async def apply(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        current_args = args
        for ext_name, callback in self._chains.get(tool_name, []):
            try:
                modified = await callback(current_args)
                if modified is not None: current_args = modified
            except Exception as e:
                logger.warning(f"Middleware {ext_name} blocked {tool_name}: {e}")
                return current_args, False
        return current_args, True
__EOF_SYNAPTIQ__

# --- synaptiq/core/provider.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/provider.py
"""LLM provider abstraction for multi-model support."""
import json
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]: pass
    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]: pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError: raise ImportError("anthropic package not installed.")
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            anthropic_tools = [{"name": t["name"], "description": t["description"], "input_schema": t.get("input_schema", t.get("parameters", {}))} for t in tools]
            response = await self.client.messages.create(model=self.model, max_tokens=4096, system=system, tools=anthropic_tools, messages=messages)
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"content": [{"type": "text", "text": f"API Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        return {"content": response.content, "stop_reason": response.stop_reason}

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError: raise ImportError("openai package not installed.")
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            oai_messages = [{"role": "system", "content": system}] + messages
            oai_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t.get("input_schema", t.get("parameters", {}))}} for t in tools]
            response = await self.client.chat.completions.create(model=self.model, messages=oai_messages, tools=oai_tools if oai_tools else None)
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"content": [{"type": "text", "text": f"API Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        choice = response.choices[0]
        message = choice.message
        content = []
        if message.content: content.append({"type": "text", "text": message.content})
        stop_reason = "end_turn"
        if message.tool_calls:
            stop_reason = "tool_use"
            for tc in message.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments)})
        return {"content": content, "stop_reason": stop_reason}

class LocalProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            import aiohttp
            oai_messages = [{"role": "system", "content": system}] + messages
            async with aiohttp.ClientSession() as session:
                payload = {"model": self.model, "messages": oai_messages, "stream": False}
                async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"content": [{"type": "text", "text": data.get("message", {}).get("content", "")}], "stop_reason": "end_turn"}
                    else:
                        return {"content": [{"type": "text", "text": f"Local LLM error: {await response.text()}"}], "stop_reason": "error"}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Local LLM Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        return response
__EOF_SYNAPTIQ__

# --- synaptiq/core/registry.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/registry.py
"""Dynamic tool registry with middleware support."""
from typing import Dict, Any, Type, Optional, List
import logging

from .nerve import EventBus
from .middleware import ToolMiddleware

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, nerve: EventBus, fast_llm_client=None, middleware: Optional[ToolMiddleware] = None):
        self.nerve = nerve
        self.fast_llm = fast_llm_client
        self.middleware = middleware or ToolMiddleware()
        self.tools: Dict[str, Any] = {}

    def register(self, tool_class: Type, override: bool = False):
        from ..primitives.base import BaseTool
        instance = tool_class(self.nerve, self.fast_llm)
        if instance.name in self.tools and not override: return
        self.tools[instance.name] = instance

    def unregister(self, tool_name: str):
        if tool_name in self.tools: del self.tools[tool_name]

    def get_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for tool in self.tools.values():
            try:
                schema = tool.args_model.model_json_schema()
                if "required" not in schema: schema["required"] = []
                schemas.append({"name": tool.name, "description": tool.description, "input_schema": schema})
            except Exception:
                schemas.append({"name": tool.name, "description": tool.description, "input_schema": {"type": "object", "properties": {}, "required": []}})
        return schemas

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_name)
        if not tool: return {"error": f"Tool '{tool_name}' not found."}
        modified_args, should_execute = await self.middleware.apply(tool_name, args)
        if not should_execute: return {"error": "Execution blocked by middleware."}
        try: return await tool.execute(**modified_args)
        except Exception as e: return {"error": str(e)}

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self.tools.items()}
__EOF_SYNAPTIQ__

# --- synaptiq/core/cortex.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/cortex.py
"""Multi-agent orchestration: sub-agents, hive mind, and daemon watchers."""
import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .nerve import EventBus
from ..primitives.base import BaseTool

logger = logging.getLogger(__name__)

@dataclass
class AgentPersona:
    name: str = "Researcher"
    system_prompt: str = "You are a focused research agent. Find the answer and return it concisely."
    allowed_tools: List[str] = None
    def __post_init__(self):
        if self.allowed_tools is None: self.allowed_tools = []

class SubAgent:
    def __init__(self, brain, persona: AgentPersona):
        self.brain = brain
        self.persona = persona
        self.session: List[Dict] = []

    async def execute_task(self, task: str, timeout: float = 120.0) -> str:
        self.session.append({"role": "user", "content": task})
        all_schemas = self.brain.registry.get_schemas()
        allowed_schemas = [s for s in all_schemas if s["name"] in self.persona.allowed_tools]
        max_iterations = 20
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            try: response = await asyncio.wait_for(self.brain.llm.chat(self.session, allowed_schemas, system_prompt=self.persona.system_prompt), timeout=timeout)
            except asyncio.TimeoutError: return f"Sub-agent timed out after {timeout} seconds."
            if response["stop_reason"] == "tool_use":
                self.session.append({"role": "assistant", "content": response["content"]})
                for block in response["content"]:
                    block_dict = block if isinstance(block, dict) else {"type": getattr(block, 'type', ''), "id": getattr(block, 'id', ''), "name": getattr(block, 'name', ''), "input": getattr(block, 'input', {})}
                    if block_dict.get("type") == "tool_use":
                        result = await self.brain.registry.execute(block_dict.get("name", "unknown"), block_dict.get("input", {}))
                        self.session.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block_dict.get("id", ""), "content": json.dumps(result)}]})
            else:
                if response["content"]:
                    first_item = response["content"][0]
                    if hasattr(first_item, 'text'): return first_item.text
                    elif isinstance(first_item, dict): return first_item.get("text", str(first_item))
                return str(response["content"])
        return "Sub-agent reached maximum iterations."

class DelegateTaskSchema(BaseModel):
    task_description: str = Field(..., description="The complex task to delegate to a sub-agent")
    persona_name: str = Field(default="Researcher", description="The role of the sub-agent")

class DelegateTaskTool(BaseTool):
    name = "delegate_task"
    description = "Spin up a sub-agent to handle an exhausting or complex sub-task."
    args_model = DelegateTaskSchema

    async def _run(self, task_description: str, persona_name: str) -> Dict[str, Any]:
        personas = {
            "Researcher": AgentPersona(name="Researcher", allowed_tools=["fs_read", "code_exec"], system_prompt="You are a researcher."),
            "Coder": AgentPersona(name="Coder", allowed_tools=["fs_read", "fs_write", "code_exec"], system_prompt="You are a coder."),
            "Analyst": AgentPersona(name="Analyst", allowed_tools=["fs_read", "code_exec"], system_prompt="You are an analyst.")
        }
        persona = personas.get(persona_name, personas["Researcher"])
        worker = SubAgent(self.brain, persona)
        try:
            result = await asyncio.wait_for(worker.execute_task(task_description), timeout=120.0)
            return {"status": "success", "sub_agent_result": result}
        except asyncio.TimeoutError: return {"error": "Sub-agent timed out."}
        except Exception as e: return {"error": f"Sub-agent failed: {str(e)}"}

class HiveMindSchema(BaseModel):
    task: str = Field(..., description="The complex architectural or coding problem to debate.")
    personas: List[str] = Field(default=["Architect", "Security", "Optimizer"], description="The diverse perspectives to include.")

class HiveMindTool(BaseTool):
    name = "hive_mind"
    description = "Spawns a parallel swarm of specialized agents to debate a complex problem."
    args_model = HiveMindSchema

    async def _run(self, task: str, personas: List[str]) -> Dict[str, Any]:
        persona_prompts = {
            "Architect": "You are a Software Architect. Focus on maintainability and design patterns.",
            "Security": "You are a Security Auditor. Focus on vulnerabilities.",
            "Optimizer": "You are a Performance Optimizer. Focus on Big-O and speed.",
            "UX": "You are a UX Expert. Focus on user experience."
        }
        async def run_persona(persona_name: str) -> str:
            persona = AgentPersona(name=persona_name, system_prompt=persona_prompts.get(persona_name, "You are a helpful assistant."), allowed_tools=["fs_read", "code_exec"])
            worker = SubAgent(self.brain, persona)
            return await worker.execute_task(f"Analyze this task from your perspective:\n{task}")
        tasks = [run_persona(p) for p in personas]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, str)]
        if not valid_results: return {"error": "All hive mind agents failed."}
        synthesis_prompt = f"Synthesize these proposals into one optimal solution:\nTask: {task}\nProposals:\n" + "\n".join(valid_results)
        try:
            if self.fast_llm:
                import anthropic
                synthesis_response = self.fast_llm.messages.create(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": synthesis_prompt}])
                consensus = synthesis_response.content[0].text
            else: consensus = "Synthesis unavailable. Raw:\n" + "\n".join(valid_results)
            return {"status": "consensus_reached", "consensus": consensus, "individual_proposals": valid_results}
        except Exception as e: return {"status": "partial_success", "error": str(e), "individual_proposals": valid_results}

class DaemonAgent:
    def __init__(self, brain):
        self.brain = brain
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.tool_calls: Dict[str, int] = defaultdict(int)
        self._active = False
        self.brain.nervous_system.subscribe("tool_end", self.monitor_errors)
        self.brain.nervous_system.subscribe("tool_start", self.monitor_patterns)

    async def start(self): self._active = True
    async def stop(self): self._active = False

    async def monitor_errors(self, event_data: Dict):
        if not self._active: return
        tool_name = event_data.get("tool", "unknown")
        success = event_data.get("success", False)
        if not success:
            self.error_counts[tool_name] += 1
            if self.error_counts[tool_name] >= 3:
                await self.inject_interrupt(f"🚨 [Daemon Watcher] You have failed using '{tool_name}' 3 times. Stop using it and try a different approach.")
                self.error_counts[tool_name] = 0
        else: self.error_counts[tool_name] = 0

    async def monitor_patterns(self, event_data: Dict):
        if not self._active: return
        tool_name = event_data.get("tool", "unknown")
        self.tool_calls[tool_name] += 1
        if self.tool_calls[tool_name] == 10:
            await self.inject_interrupt(f"⚠️ [Daemon Watcher] You have called '{tool_name}' 10 times. Consider a more efficient approach.")

    async def inject_interrupt(self, message: str):
        if hasattr(self.brain, 'session'): self.brain.session.add_message("system", message)
__EOF_SYNAPTIQ__

# --- synaptiq/core/consolidator.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/consolidator.py
"""Background knowledge consolidation and 'sleep' processing."""
import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class KnowledgeConsolidator:
    def __init__(self, memory, api_key: Optional[str] = None):
        self.memory = memory
        self.api_key = api_key

    async def consolidate_turn(self, user_input: str, agent_response: str):
        if not self.api_key: return
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            prompt = f"""Analyze this interaction and extract knowledge for long-term memory.
Return ONLY JSON: {{"facts": [{{"content": "A fact learned"}}], "lessons": [{{"content": "A mistake corrected"}}]}}
User: {user_input}
Agent: {agent_response}"""
            response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
            data = json.loads(response.content[0].text)
            fact_ids = []
            for fact in data.get("facts", []): fact_ids.append(self.memory.add_fact(fact["content"], memory_type="semantic"))
            for lesson in data.get("lessons", []): self.memory.add_fact(lesson["content"], memory_type="reflection", strength=2.0)
            if len(fact_ids) > 1:
                for i in range(len(fact_ids) - 1): self.memory.connect_facts(fact_ids[i], fact_ids[i + 1], "related_context")
        except Exception: pass

class SleepConsolidator:
    def __init__(self, memory, api_key: Optional[str] = None):
        self.memory = memory
        self.api_key = api_key

    async def enter_rem_sleep(self):
        if not self.api_key: return
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            if self.memory.collection: all_mems = self.memory.collection.get(where={"type": "episodic", "status": "active"})
            else:
                all_mems = {"ids": [], "documents": []}
                for item in self.memory._fallback_store:
                    if item['metadata'].get('type') == 'episodic' and item['metadata'].get('status') == 'active': all_mems["ids"].append(item['id']); all_mems["documents"].append(item['document'])
            if len(all_mems['ids']) < 5: return
            ids_to_compress = all_mems['ids'][:10]
            docs_to_compress = all_mems['documents'][:10]
            prompt = f"Synthesize these raw memories into 1-2 high-level rules:\n{json.dumps(docs_to_compress)}\nReturn JSON: {{\"rules\": [\"rule1\"]}}"
            response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
            rules = json.loads(response.content[0].text).get("rules", [])
            for rule in rules: self.memory.add_fact(rule, memory_type="semantic_principle", strength=2.0)
            if self.memory.collection:
                for mid in ids_to_compress:
                    meta = self.memory.collection.get(ids=[mid])['metadatas'][0]
                    meta['status'] = 'archived'
                    self.memory.collection.update(ids=[mid], metadatas=[meta])
            else:
                for item in self.memory._fallback_store:
                    if item['id'] in ids_to_compress: item['metadata']['status'] = 'archived'
        except Exception: pass
__EOF_SYNAPTIQ__

# --- synaptiq/core/observability.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/observability.py
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
__EOF_SYNAPTIQ__

# --- synaptiq/core/brain.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/core/brain.py
"""The main cognitive brain - orchestrates all Synaptiq components."""
import asyncio
import json
import time
import logging
from typing import AsyncGenerator, Dict, Any, Optional

from .nerve import EventBus
from .memory import SpacedRepetitionMemory
from .session import TransactionalQuantumSession
from .registry import ToolRegistry
from .middleware import ToolMiddleware
from .provider import AnthropicProvider, OpenAIProvider, LLMProvider
from .cortex import DaemonAgent, DelegateTaskTool, HiveMindTool
from .consolidator import KnowledgeConsolidator, SleepConsolidator
from .observability import ObservabilityManager

from ..primitives.system import CodeExecTool, FileReadTool, FileWriteTool
from ..primitives.cognitive import DivideAndConquerTool, SimulateFutureTool
from ..primitives.forge import ForgeTool
from ..primitives.belief import BeliefRevisionTool
from ..primitives.working_memory import WorkingMemoryTool

from ..extensions.manager import GraftManager

logger = logging.getLogger(__name__)

class SynaptiqBrain:
    def __init__(self, api_key: str, provider: str = "anthropic", model: Optional[str] = None, synaptiq_version: str = "1.0.0"):
        self.version = synaptiq_version
        self.api_key = api_key
        if provider == "anthropic": self.llm = AnthropicProvider(api_key, model or "claude-3-5-sonnet-20241022")
        elif provider == "openai": self.llm = OpenAIProvider(api_key, model or "gpt-4o")
        else: raise ValueError(f"Unknown provider: {provider}")
        
        self.nervous_system = EventBus()
        self.memory = SpacedRepetitionMemory()
        self.middleware = ToolMiddleware()
        
        try:
            import anthropic
            self.fast_llm_client = anthropic.Anthropic(api_key=api_key)
        except ImportError: self.fast_llm_client = None

        self.registry = ToolRegistry(self.nervous_system, self.fast_llm_client, self.middleware)
        self.session = TransactionalQuantumSession(fast_llm=self.fast_llm_client, workspace_dir="./workspace")
        self.observability = ObservabilityManager()
        
        self._register_core_tools()
        
        self.custom_prompt_sections: Dict[str, str] = {}
        self.ui_panels: Dict[str, Any] = {}
        self.graft_manager = GraftManager(self, self.version)
        
        self.daemon = DaemonAgent(self)
        self.knowledge_consolidator = KnowledgeConsolidator(self.memory, api_key)
        self.sleep_consolidator = SleepConsolidator(self.memory, api_key)
        
        self._nerve_task: Optional[asyncio.Task] = None
        self._initialized = False

    def _register_core_tools(self):
        self.registry.register(CodeExecTool)
        self.registry.register(FileReadTool)
        self.registry.register(FileWriteTool)
        self.registry.register(ForgeTool)
        self.registry.register(DelegateTaskTool)
        self.registry.register(HiveMindTool)
        self.registry.register(DivideAndConquerTool)
        self.registry.register(SimulateFutureTool)
        self.registry.register(BeliefRevisionTool)
        self.registry.register(WorkingMemoryTool)
        self.registry.tools["forge_tool"].registry = self.registry
        self.registry.tools["delegate_task"].brain = self
        self.registry.tools["hive_mind"].brain = self

    async def initialize(self):
        if self._initialized: return
        self._nerve_task = asyncio.create_task(self.nervous_system.start())
        await self.daemon.start()
        await self.graft_manager.load_all()
        self._initialized = True

    async def shutdown(self):
        await self.graft_manager.unload_all()
        await self.daemon.stop()
        await self.nervous_system.stop()
        if self._nerve_task: self._nerve_task.cancel()

    def get_system_prompt(self) -> str:
        prompt = f"""You are Synaptiq v{self.version}, an advanced cognitive agent.
## Core Abilities
- Read files with `fs_read` tool
- Write files with `fs_write` tool
- Execute code with `code_exec` tool
- Delegate tasks with `delegate_task` tool
- Spawn hive mind with `hive_mind` tool
- Forge new tools with `forge_tool` tool
- Revise beliefs with `correct_belief` tool
- Use working memory with `working_memory` tool

## Guidelines
- Use `delegate_task` for complex tasks to preserve context
- Use `hive_mind` for architectural decisions
- Use `forge_tool` to create custom tools
- If you make a mistake, acknowledge it and try a different approach
- A background Daemon watches for loops and will interrupt if you get stuck

## Memory
You have access to long-term memory that persists across sessions."""
        for section_name, section_text in self.custom_prompt_sections.items():
            prompt += f"\n\n## [{section_name} Extension]\n{section_text}"
        return prompt

    async def think(self, user_input: str) -> AsyncGenerator[str, None]:
        if not self._initialized: await self.initialize()
        run_id = self.observability.start_run(user_input)
        start_time = time.time()
        
        related_memories = self.memory.semantic_search(user_input, top_k=3)
        memory_context = ""
        if related_memories:
            memory_context = "\n\n[Recalled Memories]\n"
            for mem in related_memories: memory_context += f"- {mem['content']}\n"
            
        system_prompt = self.get_system_prompt()
        if memory_context: system_prompt += memory_context
        
        self.session.add_message("user", user_input)
        rollback_point = self.session.active_node
        
        max_iterations = 30
        iteration = 0
        error_count = 0
        final_response_text = ""
        
        while iteration < max_iterations:
            iteration += 1
            current_messages = self.session.get_active_messages()
            response = await self.llm.chat(current_messages, self.registry.get_schemas(), system_prompt)
            
            if response["stop_reason"] == "tool_use":
                self.session.add_message("assistant", response["content"])
                for block in response["content"]:
                    block_dict = block if isinstance(block, dict) else {"type": getattr(block, 'type', ''), "id": getattr(block, 'id', ''), "name": getattr(block, 'name', ''), "input": getattr(block, 'input', {})}
                    if block_dict.get("type") == "tool_use":
                        tool_name = block_dict.get("name", "unknown")
                        tool_input = block_dict.get("input", {})
                        result = await self.registry.execute(tool_name, tool_input)
                        self.observability.record_tool_call(tool_name, tool_input, result)
                        fs_mutated = tool_name in ["fs_write", "forge_tool", "code_exec"]
                        self.session.add_message("user", json.dumps(result), fs_mutated=fs_mutated)
                        if "error" in result:
                            error_count += 1
                            if error_count >= 3:
                                yield "[System: Auto-rolling back due to repeated tool errors.]"
                                self.session.rollback(rollback_point)
                                error_count = 0
                                break
                        else: error_count = 0
            else:
                if response["content"]:
                    first_item = response["content"][0]
                    if hasattr(first_item, 'text'): final_response_text = first_item.text
                    elif isinstance(first_item, dict): final_response_text = first_item.get("text", str(first_item))
                    else: final_response_text = str(first_item)
                self.session.add_message("assistant", final_response_text)
                break
                
        if not final_response_text: final_response_text = "[Synaptiq reached maximum iterations.]"
        self.memory.clear_scratchpad()
        duration = time.time() - start_time
        self.observability.record_run(run_id, duration, "completed")
        if self.api_key: asyncio.create_task(self.knowledge_consolidator.consolidate_turn(user_input, final_response_text))
        stats = self.memory.get_stats()
        if stats["total_memories"] > 0 and stats["total_memories"] % 20 == 0: asyncio.create_task(self.sleep_consolidator.enter_rem_sleep())
        yield final_response_text

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "tools": list(self.registry.tools.keys()),
            "memory_stats": self.memory.get_stats(),
            "session_info": {"active_node": self.session.active_node, "tree_size": len(self.session.get_session_tree().get("nodes", []))},
            "extensions": list(self.graft_manager.loaded_grafts.keys()),
            "observability": self.observability.get_metrics()
        }
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/__init__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/__init__.py
"""Synaptiq cognitive primitives (tools)."""
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/base.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/base.py
"""Base tool class with auto-summarization and event emission."""
import asyncio
import time
import logging
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from ..core.nerve import EventBus

logger = logging.getLogger(__name__)

class BaseTool:
    name: str = "base_tool"
    description: str = ""
    args_model: Type[BaseModel] = BaseModel
    MAX_OUTPUT_CHARS = 4000

    def __init__(self, nerve: EventBus, fast_llm_client=None):
        self.nerve = nerve
        self.fast_llm = fast_llm_client

    async def execute(self, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        await self.nerve.publish("tool_start", {"tool": self.name, "args": kwargs})
        try:
            validated_args = self.args_model(**kwargs)
            result = await self._run(**validated_args.model_dump())
            result_str = str(result)
            if len(result_str) > self.MAX_OUTPUT_CHARS: result = await self._summarize_output(result_str)
            await self.nerve.publish("tool_end", {"tool": self.name, "duration": time.time() - start_time, "success": "error" not in result})
            return result
        except ValidationError as e:
            await self.nerve.publish("tool_error", {"tool": self.name, "error": str(e)})
            return {"error": str(e)}
        except Exception as e:
            await self.nerve.publish("tool_error", {"tool": self.name, "error": str(e)})
            return {"error": str(e)}

    async def _summarize_output(self, output: str) -> Dict[str, Any]:
        if not self.fast_llm: return {"warning": "Output truncated.", "truncated_output": output[:self.MAX_OUTPUT_CHARS]}
        try:
            import anthropic
            response = self.fast_llm.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": f"Summarize concisely:\n\n{output[:15000]}"}])
            return {"warning": "Auto-summarized.", "summary": response.content[0].text}
        except Exception: return {"warning": "Truncated.", "truncated_output": output[:self.MAX_OUTPUT_CHARS]}

    async def _run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/system.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/system.py
"""System interaction tools: code execution and filesystem operations."""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

from pydantic import BaseModel, Field
from .base import BaseTool

logger = logging.getLogger(__name__)

class CodeExecSchema(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=10, description="Max execution time in seconds")

class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code in an isolated subprocess."
    args_model = CodeExecSchema

    async def _run(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code); temp_path = f.name
        try:
            process = await asyncio.create_subprocess_exec('python', temp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env={"PATH": os.environ.get("PATH", "")})
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return {"stdout": stdout.decode('utf-8')[:2000], "stderr": stderr.decode('utf-8')[:2000], "exit_code": process.returncode}
        except asyncio.TimeoutError: return {"error": f"Timed out after {timeout}s."}
        except Exception as e: return {"error": str(e)}
        finally:
            try: os.unlink(temp_path)
            except Exception: pass

class FileReadSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file")

class FileReadTool(BaseTool):
    name = "fs_read"
    description = "Read a file from the local project directory."
    args_model = FileReadSchema

    async def _run(self, file_path: str) -> Dict[str, Any]:
        root = Path(os.getcwd()).resolve()
        target = (root / file_path).resolve()
        if not str(target).startswith(str(root)): return {"error": "Access denied: Path traversal detected."}
        if not target.exists(): return {"error": "File not found."}
        try:
            content = target.read_text(encoding='utf-8')
            return {"file_path": file_path, "content": content}
        except Exception as e: return {"error": str(e)}

class FileWriteSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file")
    content: str = Field(..., description="Content to write")

class FileWriteTool(BaseTool):
    name = "fs_write"
    description = "Write content to a local project file."
    args_model = FileWriteSchema

    async def _run(self, file_path: str, content: str) -> Dict[str, Any]:
        root = Path(os.getcwd()).resolve()
        target = (root / file_path).resolve()
        if not str(target).startswith(str(root)): return {"error": "Access denied: Path traversal detected."}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return {"status": "success", "message": f"Wrote {len(content)} chars to {file_path}"}
        except Exception as e: return {"error": str(e)}
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/cognitive.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/cognitive.py
"""Cognitive primitives: divide and conquer, simulate future."""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from .base import BaseTool

class DivideAndConquerSchema(BaseModel):
    task: str = Field(..., description="The complex task to break down")
    constraints: List[str] = Field(default=[], description="Any constraints to consider")

class DivideAndConquerTool(BaseTool):
    name = "divide_and_conquer"
    description = "Breaks a complex task into a structured JSON DAG of subtasks."
    args_model = DivideAndConquerSchema

    async def _run(self, task: str, constraints: List[str]) -> Dict[str, Any]:
        return {"status": "planning_required", "template": {"subtasks": [{"id": "1", "task": "...", "depends_on": []}]}, "constraints": constraints}

class SimulateFutureSchema(BaseModel):
    action_description: str = Field(..., description="What you are about to do")
    predicted_outcome: str = Field(..., description="What you THINK will happen")

class SimulateFutureTool(BaseTool):
    name = "simulate_future"
    description = "Log a prediction about the outcome of an action BEFORE taking it."
    args_model = SimulateFutureSchema

    async def _run(self, action_description: str, predicted_outcome: str) -> Dict[str, Any]:
        return {"status": "simulation_logged", "action": action_description, "prediction": predicted_outcome}
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/forge.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/forge.py
"""Dynamic tool forging - allows the agent to create new tools on the fly."""
import types
from typing import Dict, Any
from pydantic import BaseModel, Field
from .base import BaseTool

class ForgeToolSchema(BaseModel):
    tool_name: str = Field(..., description="snake_case name for the new tool")
    description: str = Field(..., description="What the tool does")
    python_code: str = Field(..., description="A complete Python async function. Must start with 'async def {tool_name}(**kwargs):' and return a dict.")

class ForgeTool(BaseTool):
    name = "forge_tool"
    description = "Dynamically create and register a new tool by writing Python code."
    args_model = ForgeToolSchema

    async def _run(self, tool_name: str, description: str, python_code: str) -> Dict[str, Any]:
        try:
            dangerous = ["import os", "import subprocess", "import sys", "__import__"]
            for d in dangerous:
                if d in python_code: return {"error": f"Security violation: '{d}' not allowed."}
            module = types.ModuleType("dynamic_tool")
            exec(python_code, {"__builtins__": {"print": print, "len": len, "str": str, "int": int, "float": float, "dict": dict, "list": list, "range": range, "enumerate": enumerate}})
            new_func = getattr(module, tool_name, None)
            if not callable(new_func): return {"error": f"Function {tool_name} not found."}
            
            class DynamicSchema(BaseModel): pass
            class DynamicTool(BaseTool):
                name = tool_name
                description = description
                args_model = DynamicSchema
                async def _run(self, **kwargs): return await new_func(**kwargs)
                
            if hasattr(self, 'registry') and self.registry:
                self.registry.register(DynamicTool, override=True)
                return {"status": "tool_forged", "message": f"Tool '{tool_name}' added."}
            return {"error": "Registry not available."}
        except Exception as e: return {"error": f"Failed to compile: {str(e)}"}
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/belief.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/belief.py
"""Belief revision tool - allows the agent to correct outdated memories."""
from typing import Dict, Any
from pydantic import BaseModel, Field
from .base import BaseTool

class BeliefRevisionSchema(BaseModel):
    old_memory_id: str = Field(..., description="The ID of the memory that is now incorrect")
    new_content: str = Field(..., description="The corrected, updated information")
    reason: str = Field(..., description="Why the old memory was wrong")

class BeliefRevisionTool(BaseTool):
    name = "correct_belief"
    description = "Use when you discover a previous memory or assumption is incorrect."
    args_model = BeliefRevisionSchema

    async def _run(self, old_memory_id: str, new_content: str, reason: str) -> Dict[str, Any]:
        if not hasattr(self, 'memory') or self.memory is None: return {"error": "Memory system not available."}
        new_id = self.memory.add_fact(new_content, memory_type="semantic")
        self.memory.supersede_fact(old_memory_id, new_id, reason)
        return {"status": "belief_revised", "new_memory_id": new_id}
__EOF_SYNAPTIQ__

# --- synaptiq/primitives/working_memory.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/primitives/working_memory.py
"""Working memory tool - transient scratchpad for complex multi-step tasks."""
from typing import Dict, Any
from pydantic import BaseModel, Field
from .base import BaseTool

class ScratchpadSchema(BaseModel):
    action: str = Field(..., description="Either 'write', 'read', or 'clear'")
    key: str = Field(..., description="The variable name to store or retrieve")
    value: str = Field(default="", description="The value to store (if action is 'write')")

class WorkingMemoryTool(BaseTool):
    name = "working_memory"
    description = "A transient scratchpad for holding intermediate results."
    args_model = ScratchpadSchema

    async def _run(self, action: str, key: str, value: str = "") -> Dict[str, Any]:
        if not hasattr(self, 'memory') or self.memory is None: return {"error": "Memory system not available."}
        if action == "write":
            self.memory.scratch(key, value)
            return {"status": "success"}
        elif action == "read":
            val = self.memory.recall_scratch(key)
            return {"status": "success", "value": val if val else "Key not found."}
        elif action == "clear":
            self.memory.clear_scratchpad()
            return {"status": "success"}
        return {"error": "Invalid action."}
__EOF_SYNAPTIQ__

# --- synaptiq/extensions/__init__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/extensions/__init__.py
"""Synaptiq extension system (Neural Grafts)."""
__EOF_SYNAPTIQ__

# --- synaptiq/extensions/base.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/extensions/base.py
"""Extension base classes and context."""
import json
from pathlib import Path
from typing import Dict, Any, List, Type, Callable, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class GraftManifest(BaseModel):
    name: str
    version: str
    min_synaptiq_version: str = "1.0.0"
    description: str = ""
    author: str = ""
    permissions: List[str] = []
    dependencies: List[str] = []
    overrides: List[str] = []

class GraftState:
    def __init__(self, graft_id: str, storage_dir: Path):
        self.file_path = storage_dir / f"{graft_id}_state.json"
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try: return json.loads(self.file_path.read_text())
            except Exception: return {}
        return {}

    def save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(json.dumps(self._data, indent=2))
        except Exception: pass

    def get(self, key: str, default: Any = None) -> Any: return self._data.get(key, default)
    def set(self, key: str, value: Any): self._data[key] = value; self.save()

class ExtensionContext:
    def __init__(self, brain, manifest: GraftManifest, synaptiq_version: str):
        self.brain = brain
        self.manifest = manifest
        self.synaptiq_version = synaptiq_version
        self.state = GraftState(manifest.name, Path("./synaptiq_grafts/.state"))

    def register_tool(self, tool_class: Type, override: bool = False):
        self.brain.registry.register(tool_class, override=override)

    def register_middleware(self, tool_name: str, callback: Callable):
        self.brain.middleware.add_middleware(tool_name, self.manifest.name, callback)

    def add_system_prompt(self, text: str):
        self.brain.custom_prompt_sections[self.manifest.name] = text

    def register_ui_panel(self, panel_id: str, render_callback: Callable):
        self.brain.ui_panels[panel_id] = render_callback

    def subscribe_event(self, event_type: str, callback: Callable):
        self.brain.nervous_system.subscribe(event_type, callback)

class NeuralGraft:
    name: str = "unnamed_graft"
    async def initialize(self, ctx: ExtensionContext): pass
    async def teardown(self): pass
__EOF_SYNAPTIQ__

# --- synaptiq/extensions/sandbox.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/extensions/sandbox.py
"""Sandboxed execution environment for untrusted extension code."""
import types
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SandboxedEnvironment:
    def __init__(self, allowed_imports: Optional[List[str]] = None):
        self.allowed_imports = allowed_imports or ["math", "json", "re", "datetime", "collections"]

    def _safe_import(self, name, *args, **kwargs):
        if name in self.allowed_imports: return __import__(name, *args, **kwargs)
        raise ImportError(f"Module '{name}' is not allowed.")

    def execute_code(self, code: str, module_name: str) -> types.ModuleType:
        safe_builtins = {"print": print, "len": len, "str": str, "int": int, "float": float, "dict": dict, "list": list, "tuple": tuple, "set": set, "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "sorted": sorted, "reversed": reversed, "any": any, "all": all, "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "TypeError": TypeError, "ValueError": ValueError, "Exception": Exception, "StopIteration": StopIteration, "__build_class__": __build_class__}
        restricted_globals = {"__builtins__": safe_builtins, "__name__": module_name, "__import__": self._safe_import}
        module = types.ModuleType(module_name)
        module.__dict__.update(restricted_globals)
        try:
            exec(compile(code, f"<{module_name}>", "exec"), module.__dict__)
            return module
        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            raise
__EOF_SYNAPTIQ__

# --- synaptiq/extensions/manager.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/extensions/manager.py
"""Graft manager - discovers, loads, and hot-reloads Neural Grafts."""
import json
import os
import importlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import semver; SEMVER_AVAILABLE = True
except ImportError: SEMVER_AVAILABLE = False
try:
    from watchfiles import awatch; WATCHFILES_AVAILABLE = True
except ImportError: WATCHFILES_AVAILABLE = False

from .base import GraftManifest, ExtensionContext, NeuralGraft

logger = logging.getLogger(__name__)

class GraftManager:
    def __init__(self, brain, synaptiq_version: str, grafts_dir: str = "./synaptiq_grafts"):
        self.brain = brain
        self.synaptiq_version = synaptiq_version
        self.grafts_dir = Path(grafts_dir).resolve()
        self.grafts_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_grafts: Dict[str, NeuralGraft] = {}
        self._watcher_task: Optional[asyncio.Task] = None

    async def load_all(self):
        for dirpath in self.grafts_dir.iterdir():
            if dirpath.is_dir() and (dirpath / "graft.json").exists(): await self.load_graft(dirpath)
        if WATCHFILES_AVAILABLE: self._watcher_task = asyncio.create_task(self._watch_extensions())

    async def load_graft(self, graft_dir: Path) -> bool:
        try:
            manifest_data = json.loads((graft_dir / "graft.json").read_text())
            manifest = GraftManifest(**manifest_data)
            if manifest.name in self.loaded_grafts: await self.unload_graft(manifest.name)
            main_file = graft_dir / "main.py"
            if not main_file.exists(): return False
            
            import sys
            graft_path = str(graft_dir.parent)
            if graft_path not in sys.path: sys.path.insert(0, graft_path)
            
            module_name = f"synaptiq_grafts.{manifest.name}.main"
            try:
                module = importlib.import_module(module_name)
                module = importlib.reload(module)
            except ImportError:
                import types
                module = types.ModuleType(module_name)
                exec(main_file.read_text(), module.__dict__)
                
            graft_class = getattr(module, "Extension", None)
            if graft_class is None: return False
            
            ctx = ExtensionContext(self.brain, manifest, self.synaptiq_version)
            instance = graft_class()
            await instance.initialize(ctx)
            self.loaded_grafts[manifest.name] = instance
            return True
        except Exception as e: return False

    async def unload_graft(self, name: str):
        if name in self.loaded_grafts:
            graft = self.loaded_grafts[name]
            try: await graft.teardown()
            except Exception: pass
            del self.loaded_grafts[name]
            if name in self.brain.custom_prompt_sections: del self.brain.custom_prompt_sections[name]

    async def unload_all(self):
        for name in list(self.loaded_grafts.keys()): await self.unload_graft(name)

    async def install_from_git(self, git_url: str) -> bool:
        import subprocess, uuid
        temp_name = f"temp_{uuid.uuid4().hex[:4]}"
        target_dir = self.grafts_dir / temp_name
        try: subprocess.run(["git", "clone", git_url, str(target_dir)], check=True, capture_output=True)
        except Exception: return False
        manifest_file = target_dir / "graft.json"
        if not manifest_file.exists(): subprocess.run(["rm", "-rf", str(target_dir)]); return False
        manifest_data = json.loads(manifest_file.read_text())
        real_name = manifest_data.get("name", temp_name)
        final_dir = self.grafts_dir / real_name
        if final_dir.exists(): subprocess.run(["rm", "-rf", str(final_dir)])
        target_dir.rename(final_dir)
        return await self.load_graft(final_dir)

    async def _watch_extensions(self):
        try:
            async for changes in awatch(str(self.grafts_dir)):
                for change_type, path in changes:
                    path = Path(path)
                    if path.name in ["main.py", "graft.json"]: await self.load_graft(path.parent)
        except Exception: pass
__EOF_SYNAPTIQ__

# --- synaptiq/ui/__init__.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/ui/__init__.py
"""Synaptiq UI components."""
__EOF_SYNAPTIQ__

# --- synaptiq/ui/cli.py ---
cat << '__EOF_SYNAPTIQ__' > synaptiq/ui/cli.py
"""Rich CLI interface for Synaptiq."""
import asyncio
import sys
import os
import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from ..core.brain import SynaptiqBrain

logger = logging.getLogger(__name__)

BANNER = """
███████╗██╗   ██╗███╗   ██╗ █████╗ ███████╗████████╗███████╗██╗
██╔════╝██║   ██║████╗  ██║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║
███████╗██║   ██║██╔██╗ ██║███████║███████╗   ██║   █████╗  ██║
╚════██║██║   ██║██║╚███╔╝██╔══██║╚════██║   ██║   ██╔══╝  ██║
███████║╚██████╔╝██║ ╚████║╚█████╔╝███████║   ██║   ███████╗███████╗
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚════╝ ╚══════╝   ╚═╝   ╚══════╝╚══════╝
"""

class SynaptiqCLI:
    def __init__(self):
        self.console = Console()
        self.brain: Optional[SynaptiqBrain] = None

    async def run(self):
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.console.print("[bold red]Error: API_KEY environment variable not set.[/bold red]")
            sys.exit(1)
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

        if len(sys.argv) > 1 and sys.argv[1] == "install":
            if len(sys.argv) < 3: self.console.print("[red]Usage: synaptiq install <git-url>[/red]"); return
            self.brain = SynaptiqBrain(api_key, provider=provider)
            await self.brain.graft_manager.install_from_git(sys.argv[2]); return

        self.brain = SynaptiqBrain(api_key, provider=provider)
        await self.brain.initialize()

        self.console.print(Panel(Text(BANNER, style="bold cyan"), border_style="blue"))
        self.console.print(f"[bold green]Synaptiq Cognitive Engine v{self.brain.version} Initialized[/bold green]")
        self.console.print("[dim]Type 'exit' to quit. Type 'help' for commands.[/dim]\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: Prompt.ask("[bold magenta]You ❯[/bold magenta]"))
                if user_input.lower() in ['exit', 'quit']:
                    await self.brain.shutdown(); break
                if user_input.lower() == 'help': self._show_help(); continue
                if user_input.lower() == 'status': self._show_status(); continue
                if user_input.lower() == 'memory': self._show_memory(); continue
                if user_input.lower() == 'tools': self._show_tools(); continue
                if user_input.lower() == 'extensions': self._show_extensions(); continue
                if user_input.lower() == 'session': self._show_session(); continue
                if not user_input.strip(): continue

                self.console.print("[bold cyan]Synaptiq ❯[/bold cyan] [dim]Thinking...[/dim]", end="\r")
                async for response in self.brain.think(user_input):
                    self.console.print(f"[bold cyan]Synaptiq ❯[/bold cyan] ")
                    self.console.print(Markdown(response))
            except KeyboardInterrupt: self.console.print("\n[bold yellow]Interrupted[/bold yellow]"); continue
            except Exception as e: self.console.print(f"\n[bold red]Error: {e}[/bold red]")

    def _show_help(self):
        table = Table(title="Synaptiq Commands")
        table.add_column("Command", style="cyan"); table.add_column("Description", style="green")
        table.add_row("help", "Show this help"); table.add_row("status", "Show brain status")
        table.add_row("memory", "Show memory stats"); table.add_row("tools", "List tools")
        table.add_row("extensions", "List grafts"); table.add_row("session", "Show session tree")
        self.console.print(table)

    def _show_status(self):
        status = self.brain.get_status()
        table = Table(title="Synaptiq Status")
        table.add_column("Property", style="cyan"); table.add_column("Value", style="green")
        for k, v in status.items(): table.add_row(k, str(v))
        self.console.print(table)

    def _show_memory(self):
        stats = self.brain.memory.get_stats()
        table = Table(title="Memory Statistics")
        table.add_column("Metric", style="cyan"); table.add_column("Value", style="green")
        for k, v in stats.items(): table.add_row(k, str(v))
        self.console.print(table)

    def _show_tools(self):
        tools = self.brain.registry.list_tools()
        table = Table(title="Registered Tools")
        table.add_column("Name", style="cyan"); table.add_column("Description", style="green")
        for n, d in tools.items(): table.add_row(n, d[:80])
        self.console.print(table)

    def _show_extensions(self):
        if not self.brain.graft_manager.loaded_grafts: self.console.print("[yellow]No extensions loaded.[/yellow]"); return
        table = Table(title="Loaded Grafts")
        table.add_column("Name", style="cyan"); table.add_column("Version", style="green")
        for n, g in self.brain.graft_manager.loaded_grafts.items(): table.add_row(n, getattr(g, 'version', 'unknown'))
        self.console.print(table)

    def _show_session(self):
        tree = self.brain.session.get_session_tree()
        table = Table(title="Session Tree")
        table.add_column("Node ID", style="cyan"); table.add_column("Role", style="green")
        table.add_column("Messages", style="yellow"); table.add_column("Active", style="magenta")
        for n in tree["nodes"]: table.add_row(n["id"], n["role"], str(n["message_count"]), "→" if n["is_active"] else "")
        self.console.print(table)
__EOF_SYNAPTIQ__

# --- tests/test_core.py ---
cat << '__EOF_SYNAPTIQ__' > tests/test_core.py
"""Basic tests for Synaptiq core components."""
import pytest
import asyncio
from synaptiq.core.nerve import EventBus
from synaptiq.core.memory import SpacedRepetitionMemory
from synaptiq.core.middleware import ToolMiddleware

@pytest.mark.asyncio
async def test_event_bus():
    bus = EventBus()
    received = []
    async def callback(data): received.append(data)
    bus.subscribe("test_event", callback)
    task = asyncio.create_task(bus.start())
    await bus.publish("test_event", {"message": "hello"})
    await asyncio.sleep(0.1)
    await bus.stop()
    assert len(received) == 1
    assert received[0]["message"] == "hello"

def test_memory_basic():
    mem = SpacedRepetitionMemory(db_path="./test_db")
    node_id = mem.add_fact("Python is a programming language", memory_type="semantic")
    results = mem.semantic_search("programming")
    assert len(results) > 0
    assert "Python" in results[0]["content"]
    mem.scratch("test_key", "test_value")
    assert mem.recall_scratch("test_key") == "test_value"
    mem.clear_scratchpad()
    assert mem.recall_scratch("test_key") is None

def test_memory_supersede():
    mem = SpacedRepetitionMemory(db_path="./test_db")
    old_id = mem.add_fact("The sky is green", memory_type="semantic")
    new_id = mem.add_fact("The sky is blue", memory_type="semantic")
    mem.supersede_fact(old_id, new_id, "Corrected color")
    results = mem.semantic_search("sky color")
    contents = [r["content"] for r in results]
    assert "The sky is blue" in contents
    assert "The sky is green" not in contents

@pytest.mark.asyncio
async def test_middleware():
    mw = ToolMiddleware()
    async def blocking_middleware(args): raise Exception("Blocked!")
    mw.add_middleware("test_tool", "test_ext", blocking_middleware)
    args, should_execute = await mw.apply("test_tool", {"key": "value"})
    assert should_execute is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
__EOF_SYNAPTIQ__

echo "Synaptiq project generated successfully in synaptiq-project/"