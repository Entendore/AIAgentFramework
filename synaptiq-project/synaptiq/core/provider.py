"""LLM provider abstraction for multi-model support."""
import json
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]: pass
    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]: pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError: raise ImportError("anthropic package not installed.")
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            anthropic_tools = [{"name": t["name"], "description": t["description"], "input_schema": t.get("input_schema", t.get("parameters", {}))} for t in tools]
            response = await self.client.messages.create(model=self.model, max_tokens=4096, system=system, tools=anthropic_tools, messages=messages)
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"content": [{"type": "text", "text": f"API Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        return {"content": response.content, "stop_reason": response.stop_reason}

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError: raise ImportError("openai package not installed.")
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            oai_messages = [{"role": "system", "content": system}] + messages
            oai_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t.get("input_schema", t.get("parameters", {}))}} for t in tools]
            response = await self.client.chat.completions.create(model=self.model, messages=oai_messages, tools=oai_tools if oai_tools else None)
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"content": [{"type": "text", "text": f"API Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        choice = response.choices[0]
        message = choice.message
        content = []
        if message.content: content.append({"type": "text", "text": message.content})
        stop_reason = "end_turn"
        if message.tool_calls:
            stop_reason = "tool_use"
            for tc in message.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments)})
        return {"content": content, "stop_reason": stop_reason}

class LocalProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: List[Dict], tools: List[Dict], system: str) -> Dict[str, Any]:
        try:
            import aiohttp
            oai_messages = [{"role": "system", "content": system}] + messages
            async with aiohttp.ClientSession() as session:
                payload = {"model": self.model, "messages": oai_messages, "stream": False}
                async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"content": [{"type": "text", "text": data.get("message", {}).get("content", "")}], "stop_reason": "end_turn"}
                    else:
                        return {"content": [{"type": "text", "text": f"Local LLM error: {await response.text()}"}], "stop_reason": "error"}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Local LLM Error: {e}"}], "stop_reason": "error"}

    def parse_response(self, response: Any) -> Dict[str, Any]:
        return response
