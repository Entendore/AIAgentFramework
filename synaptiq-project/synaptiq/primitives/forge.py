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
