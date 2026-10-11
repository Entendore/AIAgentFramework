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
