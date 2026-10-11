"""Tool middleware system for interception and modification of tool calls."""
from typing import Dict, Any, List, Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ToolMiddleware:
    def __init__(self):
        self._chains: Dict[str, List[Tuple[str, Callable]]] = {}

    def add_middleware(self, tool_name: str, ext_name: str, callback: Callable):
        if tool_name not in self._chains: self._chains[tool_name] = []
        self._chains[tool_name].append((ext_name, callback))

    def remove_middleware(self, tool_name: str, ext_name: str):
        if tool_name in self._chains:
            self._chains[tool_name] = [(name, cb) for name, cb in self._chains[tool_name] if name != ext_name]

    async def apply(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        current_args = args
        for ext_name, callback in self._chains.get(tool_name, []):
            try:
                modified = await callback(current_args)
                if modified is not None: current_args = modified
            except Exception as e:
                logger.warning(f"Middleware {ext_name} blocked {tool_name}: {e}")
                return current_args, False
        return current_args, True
