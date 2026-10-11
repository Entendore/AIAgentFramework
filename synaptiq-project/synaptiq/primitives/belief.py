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
