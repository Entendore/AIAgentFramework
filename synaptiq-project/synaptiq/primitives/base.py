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
