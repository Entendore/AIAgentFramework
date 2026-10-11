"""Extension base classes and context."""
import json
from pathlib import Path
from typing import Dict, Any, List, Type, Callable, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class GraftManifest(BaseModel):
    name: str
    version: str
    min_synaptiq_version: str = "1.0.0"
    description: str = ""
    author: str = ""
    permissions: List[str] = []
    dependencies: List[str] = []
    overrides: List[str] = []

class GraftState:
    def __init__(self, graft_id: str, storage_dir: Path):
        self.file_path = storage_dir / f"{graft_id}_state.json"
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try: return json.loads(self.file_path.read_text())
            except Exception: return {}
        return {}

    def save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(json.dumps(self._data, indent=2))
        except Exception: pass

    def get(self, key: str, default: Any = None) -> Any: return self._data.get(key, default)
    def set(self, key: str, value: Any): self._data[key] = value; self.save()

class ExtensionContext:
    def __init__(self, brain, manifest: GraftManifest, synaptiq_version: str):
        self.brain = brain
        self.manifest = manifest
        self.synaptiq_version = synaptiq_version
        self.state = GraftState(manifest.name, Path("./synaptiq_grafts/.state"))

    def register_tool(self, tool_class: Type, override: bool = False):
        self.brain.registry.register(tool_class, override=override)

    def register_middleware(self, tool_name: str, callback: Callable):
        self.brain.middleware.add_middleware(tool_name, self.manifest.name, callback)

    def add_system_prompt(self, text: str):
        self.brain.custom_prompt_sections[self.manifest.name] = text

    def register_ui_panel(self, panel_id: str, render_callback: Callable):
        self.brain.ui_panels[panel_id] = render_callback

    def subscribe_event(self, event_type: str, callback: Callable):
        self.brain.nervous_system.subscribe(event_type, callback)

class NeuralGraft:
    name: str = "unnamed_graft"
    async def initialize(self, ctx: ExtensionContext): pass
    async def teardown(self): pass
