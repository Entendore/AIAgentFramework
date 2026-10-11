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
