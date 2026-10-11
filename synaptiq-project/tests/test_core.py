"""Basic tests for Synaptiq core components."""
import pytest
import asyncio
from synaptiq.core.nerve import EventBus
from synaptiq.core.memory import SpacedRepetitionMemory
from synaptiq.core.middleware import ToolMiddleware

@pytest.mark.asyncio
async def test_event_bus():
    bus = EventBus()
    received = []
    async def callback(data): received.append(data)
    bus.subscribe("test_event", callback)
    task = asyncio.create_task(bus.start())
    await bus.publish("test_event", {"message": "hello"})
    await asyncio.sleep(0.1)
    await bus.stop()
    assert len(received) == 1
    assert received[0]["message"] == "hello"

def test_memory_basic():
    mem = SpacedRepetitionMemory(db_path="./test_db")
    node_id = mem.add_fact("Python is a programming language", memory_type="semantic")
    results = mem.semantic_search("programming")
    assert len(results) > 0
    assert "Python" in results[0]["content"]
    mem.scratch("test_key", "test_value")
    assert mem.recall_scratch("test_key") == "test_value"
    mem.clear_scratchpad()
    assert mem.recall_scratch("test_key") is None

def test_memory_supersede():
    mem = SpacedRepetitionMemory(db_path="./test_db")
    old_id = mem.add_fact("The sky is green", memory_type="semantic")
    new_id = mem.add_fact("The sky is blue", memory_type="semantic")
    mem.supersede_fact(old_id, new_id, "Corrected color")
    results = mem.semantic_search("sky color")
    contents = [r["content"] for r in results]
    assert "The sky is blue" in contents
    assert "The sky is green" not in contents

@pytest.mark.asyncio
async def test_middleware():
    mw = ToolMiddleware()
    async def blocking_middleware(args): raise Exception("Blocked!")
    mw.add_middleware("test_tool", "test_ext", blocking_middleware)
    args, should_execute = await mw.apply("test_tool", {"key": "value"})
    assert should_execute is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
