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
