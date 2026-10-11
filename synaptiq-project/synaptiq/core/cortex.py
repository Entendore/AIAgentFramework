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
