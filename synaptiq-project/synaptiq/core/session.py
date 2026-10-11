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
