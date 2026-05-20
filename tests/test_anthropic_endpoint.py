import json
import pytest

from schemas.anthropic_request import AnthropicMessagesRequest
from app.endpoints import anthropic as anthropic_module


def test_convert_anthropic_text_only_request():
    req = AnthropicMessagesRequest(
        model="gemini-3.5-flash-extended",
        messages=[
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=1024,
    )
    result = anthropic_module.convert_anthropic_to_openai(req)
    assert result["model"] == "gemini-3.5-flash-extended"
    assert result["stream"] is False
    assert len(result["messages"]) == 1
    assert result["messages"][0] == {"role": "user", "content": "Hello"}


def test_convert_anthropic_system_message():
    req = AnthropicMessagesRequest(
        model="gemini-3.5-flash-extended",
        system="You are helpful.",
        messages=[
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=1024,
    )
    result = anthropic_module.convert_anthropic_to_openai(req)
    assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert result["messages"][1] == {"role": "user", "content": "Hi"}


def test_convert_anthropic_tool_use_message():
    req = AnthropicMessagesRequest(
        model="gemini-3.5-flash-extended",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Beijing"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "Sunny 25C"},
                ],
            },
        ],
        max_tokens=1024,
    )
    result = anthropic_module.convert_anthropic_to_openai(req)
    # assistant message should have tool_calls
    assistant_msg = result["messages"][0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Let me check."
    assert len(assistant_msg["tool_calls"]) == 1
    tc = assistant_msg["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}
    # tool result
    tool_msg = result["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_1"
    assert tool_msg["content"] == "Sunny 25C"


def test_convert_anthropic_tools_definition():
    req = AnthropicMessagesRequest(
        model="gemini-3.5-flash-extended",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        tools=[
            {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
        ],
    )
    result = anthropic_module.convert_anthropic_to_openai(req)
    assert len(result["tools"]) == 1
    t = result["tools"][0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "get_weather"
    assert t["function"]["parameters"]["type"] == "object"


def test_convert_openai_response_to_anthropic():
    openai_resp = {
        "id": "chatcmpl-123",
        "model": "gemini-3.5-flash-extended",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = anthropic_module.convert_openai_to_anthropic(openai_resp)
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hello!"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5


def test_convert_openai_tool_calls_response_to_anthropic():
    openai_resp = {
        "id": "chatcmpl-456",
        "model": "gemini-3.5-flash-extended",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    result = anthropic_module.convert_openai_to_anthropic(openai_resp)
    assert result["stop_reason"] == "tool_use"
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "tool_use"
    assert result["content"][0]["id"] == "call_1"
    assert result["content"][0]["name"] == "get_weather"
    assert result["content"][0]["input"] == {"city": "Beijing"}


def test_anthropic_model_not_found_returns_text():
    result = anthropic_module.model_not_found_response_anthropic("missing-model")
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert "missing-model" in result["content"][0]["text"]
    assert result["stop_reason"] == "end_turn"
