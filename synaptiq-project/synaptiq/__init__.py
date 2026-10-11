"""Synaptiq: An event-driven, cognitive AI agent framework."""

__version__ = "1.0.0"
__author__ = "Synaptiq Core Team"

from .core.brain import SynaptiqBrain
from .core.nerve import EventBus
from .core.memory import SpacedRepetitionMemory
from .core.session import TransactionalQuantumSession
from .core.registry import ToolRegistry
from .core.provider import AnthropicProvider, OpenAIProvider, LLMProvider
from .core.middleware import ToolMiddleware
from .core.cortex import SubAgent, HiveMindTool, DelegateTaskTool, DaemonAgent
from .core.consolidator import SleepConsolidator, KnowledgeConsolidator
from .core.observability import ObservabilityManager

from .primitives.base import BaseTool
from .primitives.system import CodeExecTool, FileReadTool, FileWriteTool
from .primitives.cognitive import DivideAndConquerTool, SimulateFutureTool
from .primitives.forge import ForgeTool
from .primitives.belief import BeliefRevisionTool
from .primitives.working_memory import WorkingMemoryTool

from .extensions.base import NeuralGraft, ExtensionContext, GraftManifest
from .extensions.manager import GraftManager
from .extensions.sandbox import SandboxedEnvironment

__all__ = [
    "SynaptiqBrain", "EventBus", "SpacedRepetitionMemory", "TransactionalQuantumSession",
    "ToolRegistry", "AnthropicProvider", "OpenAIProvider", "LLMProvider", "ToolMiddleware",
    "SubAgent", "HiveMindTool", "DelegateTaskTool", "DaemonAgent", "SleepConsolidator",
    "KnowledgeConsolidator", "ObservabilityManager", "BaseTool", "CodeExecTool", "FileReadTool",
    "FileWriteTool", "DivideAndConquerTool", "SimulateFutureTool", "ForgeTool", "BeliefRevisionTool",
    "WorkingMemoryTool", "NeuralGraft", "ExtensionContext", "GraftManifest", "GraftManager", "SandboxedEnvironment",
]
