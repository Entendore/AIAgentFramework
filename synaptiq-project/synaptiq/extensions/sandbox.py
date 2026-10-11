"""Sandboxed execution environment for untrusted extension code."""
import types
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SandboxedEnvironment:
    def __init__(self, allowed_imports: Optional[List[str]] = None):
        self.allowed_imports = allowed_imports or ["math", "json", "re", "datetime", "collections"]

    def _safe_import(self, name, *args, **kwargs):
        if name in self.allowed_imports: return __import__(name, *args, **kwargs)
        raise ImportError(f"Module '{name}' is not allowed.")

    def execute_code(self, code: str, module_name: str) -> types.ModuleType:
        safe_builtins = {"print": print, "len": len, "str": str, "int": int, "float": float, "dict": dict, "list": list, "tuple": tuple, "set": set, "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "sorted": sorted, "reversed": reversed, "any": any, "all": all, "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "TypeError": TypeError, "ValueError": ValueError, "Exception": Exception, "StopIteration": StopIteration, "__build_class__": __build_class__}
        restricted_globals = {"__builtins__": safe_builtins, "__name__": module_name, "__import__": self._safe_import}
        module = types.ModuleType(module_name)
        module.__dict__.update(restricted_globals)
        try:
            exec(compile(code, f"<{module_name}>", "exec"), module.__dict__)
            return module
        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            raise
