"""System interaction tools: code execution and filesystem operations."""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

from pydantic import BaseModel, Field
from .base import BaseTool

logger = logging.getLogger(__name__)

class CodeExecSchema(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=10, description="Max execution time in seconds")

class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code in an isolated subprocess."
    args_model = CodeExecSchema

    async def _run(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code); temp_path = f.name
        try:
            process = await asyncio.create_subprocess_exec('python', temp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env={"PATH": os.environ.get("PATH", "")})
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return {"stdout": stdout.decode('utf-8')[:2000], "stderr": stderr.decode('utf-8')[:2000], "exit_code": process.returncode}
        except asyncio.TimeoutError: return {"error": f"Timed out after {timeout}s."}
        except Exception as e: return {"error": str(e)}
        finally:
            try: os.unlink(temp_path)
            except Exception: pass

class FileReadSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file")

class FileReadTool(BaseTool):
    name = "fs_read"
    description = "Read a file from the local project directory."
    args_model = FileReadSchema

    async def _run(self, file_path: str) -> Dict[str, Any]:
        root = Path(os.getcwd()).resolve()
        target = (root / file_path).resolve()
        if not str(target).startswith(str(root)): return {"error": "Access denied: Path traversal detected."}
        if not target.exists(): return {"error": "File not found."}
        try:
            content = target.read_text(encoding='utf-8')
            return {"file_path": file_path, "content": content}
        except Exception as e: return {"error": str(e)}

class FileWriteSchema(BaseModel):
    file_path: str = Field(..., description="Relative path to the file")
    content: str = Field(..., description="Content to write")

class FileWriteTool(BaseTool):
    name = "fs_write"
    description = "Write content to a local project file."
    args_model = FileWriteSchema

    async def _run(self, file_path: str, content: str) -> Dict[str, Any]:
        root = Path(os.getcwd()).resolve()
        target = (root / file_path).resolve()
        if not str(target).startswith(str(root)): return {"error": "Access denied: Path traversal detected."}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return {"status": "success", "message": f"Wrote {len(content)} chars to {file_path}"}
        except Exception as e: return {"error": str(e)}
