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
