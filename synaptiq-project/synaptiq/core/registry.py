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
