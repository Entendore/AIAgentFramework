"""Graft manager - discovers, loads, and hot-reloads Neural Grafts."""
import json
import os
import importlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import semver; SEMVER_AVAILABLE = True
except ImportError: SEMVER_AVAILABLE = False
try:
    from watchfiles import awatch; WATCHFILES_AVAILABLE = True
except ImportError: WATCHFILES_AVAILABLE = False

from .base import GraftManifest, ExtensionContext, NeuralGraft

logger = logging.getLogger(__name__)

class GraftManager:
    def __init__(self, brain, synaptiq_version: str, grafts_dir: str = "./synaptiq_grafts"):
        self.brain = brain
        self.synaptiq_version = synaptiq_version
        self.grafts_dir = Path(grafts_dir).resolve()
        self.grafts_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_grafts: Dict[str, NeuralGraft] = {}
        self._watcher_task: Optional[asyncio.Task] = None

    async def load_all(self):
        for dirpath in self.grafts_dir.iterdir():
            if dirpath.is_dir() and (dirpath / "graft.json").exists(): await self.load_graft(dirpath)
        if WATCHFILES_AVAILABLE: self._watcher_task = asyncio.create_task(self._watch_extensions())

    async def load_graft(self, graft_dir: Path) -> bool:
        try:
            manifest_data = json.loads((graft_dir / "graft.json").read_text())
            manifest = GraftManifest(**manifest_data)
            if manifest.name in self.loaded_grafts: await self.unload_graft(manifest.name)
            main_file = graft_dir / "main.py"
            if not main_file.exists(): return False
            
            import sys
            graft_path = str(graft_dir.parent)
            if graft_path not in sys.path: sys.path.insert(0, graft_path)
            
            module_name = f"synaptiq_grafts.{manifest.name}.main"
            try:
                module = importlib.import_module(module_name)
                module = importlib.reload(module)
            except ImportError:
                import types
                module = types.ModuleType(module_name)
                exec(main_file.read_text(), module.__dict__)
                
            graft_class = getattr(module, "Extension", None)
            if graft_class is None: return False
            
            ctx = ExtensionContext(self.brain, manifest, self.synaptiq_version)
            instance = graft_class()
            await instance.initialize(ctx)
            self.loaded_grafts[manifest.name] = instance
            return True
        except Exception as e: return False

    async def unload_graft(self, name: str):
        if name in self.loaded_grafts:
            graft = self.loaded_grafts[name]
            try: await graft.teardown()
            except Exception: pass
            del self.loaded_grafts[name]
            if name in self.brain.custom_prompt_sections: del self.brain.custom_prompt_sections[name]

    async def unload_all(self):
        for name in list(self.loaded_grafts.keys()): await self.unload_graft(name)

    async def install_from_git(self, git_url: str) -> bool:
        import subprocess, uuid
        temp_name = f"temp_{uuid.uuid4().hex[:4]}"
        target_dir = self.grafts_dir / temp_name
        try: subprocess.run(["git", "clone", git_url, str(target_dir)], check=True, capture_output=True)
        except Exception: return False
        manifest_file = target_dir / "graft.json"
        if not manifest_file.exists(): subprocess.run(["rm", "-rf", str(target_dir)]); return False
        manifest_data = json.loads(manifest_file.read_text())
        real_name = manifest_data.get("name", temp_name)
        final_dir = self.grafts_dir / real_name
        if final_dir.exists(): subprocess.run(["rm", "-rf", str(final_dir)])
        target_dir.rename(final_dir)
        return await self.load_graft(final_dir)

    async def _watch_extensions(self):
        try:
            async for changes in awatch(str(self.grafts_dir)):
                for change_type, path in changes:
                    path = Path(path)
                    if path.name in ["main.py", "graft.json"]: await self.load_graft(path.parent)
        except Exception: pass
