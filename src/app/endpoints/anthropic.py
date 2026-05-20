# src/app/endpoints/anthropic.py
import json
import time
import uuid
from typing import Optional, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.logger import logger
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from models.gemini import ModelNotFoundError, get_model_registry, refresh_models_if_needed
from schemas.anthropic_request import AnthropicMessagesRequest

router = APIRouter()


def convert_anthropic_to_openai(req: AnthropicMessagesRequest) -> dict[str, Any]:
    """Convert an Anthropic Messages request to OpenAI Chat Completions format."""
    openai_messages = []

    # System message
    if req.system:
        if isinstance(req.system, str):
            openai_messages.append({"role": "system", "content": req.system})
        elif isinstance(req.system, list):
            # Anthropic system can be a list of content blocks
            text_parts = []
            for block in req.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, dict):
                    text_parts.append(json.dumps(block))
            openai_messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Messages
    for msg in req.messages:
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Check if there are tool_use / tool_result blocks
            text_parts = []
            tool_calls = []
            tool_result = None

            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    })
                elif block.type == "tool_result":
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id or "",
                        "content": block.content if isinstance(block.content, str) else json.dumps(block.content or ""),
                    }

            if msg.role == "assistant":
                openai_msg = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                openai_messages.append(openai_msg)
            elif tool_result:
                openai_messages.append(tool_result)
            else:
                openai_messages.append({"role": msg.role, "content": "\n".join(text_parts)})

    # Tools
    openai_tools = None
    if req.tools:
        openai_tools = []
        for tool in req.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema or {},
                },
            })

    result = {
        "model": req.model,
        "messages": openai_messages,
        "stream": req.stream or False,
    }
    if openai_tools:
        result["tools"] = openai_tools
    return result


def convert_openai_to_anthropic(openai_resp: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_resp["choices"][0]
    message = choice["message"]
    stop_reason = "end_turn"
    content = []

    # Text content
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    # Tool calls
    if message.get("tool_calls"):
        stop_reason = "tool_use"
        for tc in message["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": args,
            })

    if not content:
        content.append({"type": "text", "text": ""})

    # Map finish_reason
    if choice.get("finish_reason") == "tool_calls":
        stop_reason = "tool_use"

    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "role": "assistant",
        "content": content,
        "model": openai_resp.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_resp.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_resp.get("usage", {}).get("completion_tokens", 0),
        },
    }


def model_not_found_response_anthropic(model: str) -> dict[str, Any]:
    """Return an Anthropic-format message for unknown models."""
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "role": "assistant",
        "content": [{"type": "text", "text": f"模型不存在：{model}"}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _build_anthropic_sse_events(openai_resp: dict[str, Any], model: str) -> list[str]:
    """Convert an OpenAI Chat Completions response to Anthropic SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    choice = openai_resp["choices"][0]
    message = choice["message"]
    stop_reason = "end_turn"
    if choice.get("finish_reason") == "tool_calls":
        stop_reason = "tool_use"

    events = []

    # message_start
    events.append(f"event: message_start\ndata: {json.dumps({
        'type': 'message_start',
        'message': {
            'type': 'message',
            'id': msg_id,
            'role': 'assistant',
            'content': [],
            'model': model,
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {'input_tokens': openai_resp.get('usage', {}).get('prompt_tokens', 0), 'output_tokens': 0},
        }
    })}\n\n")

    # Text content
    if message.get("content"):
        events.append(f"event: content_block_start\ndata: {json.dumps({
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''},
        })}\n\n")
        events.append(f"event: content_block_delta\ndata: {json.dumps({
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'text_delta', 'text': message['content']},
        })}\n\n")
        events.append(f"event: content_block_stop\ndata: {json.dumps({
            'type': 'content_block_stop',
            'index': 0,
        })}\n\n")

    # Tool calls
    if message.get("tool_calls"):
        idx = 1 if message.get("content") else 0
        for tc in message["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            events.append(f"event: content_block_start\ndata: {json.dumps({
                'type': 'content_block_start',
                'index': idx,
                'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['function']['name'], 'input': {}},
            })}\n\n")
            events.append(f"event: content_block_delta\ndata: {json.dumps({
                'type': 'content_block_delta',
                'index': idx,
                'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(args)},
            })}\n\n")
            events.append(f"event: content_block_stop\ndata: {json.dumps({
                'type': 'content_block_stop',
                'index': idx,
            })}\n\n")
            idx += 1

    # message_delta
    events.append(f"event: message_delta\ndata: {json.dumps({
        'type': 'message_delta',
        'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
        'usage': {'output_tokens': openai_resp.get('usage', {}).get('completion_tokens', 0)},
    })}\n\n")

    # message_stop
    events.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")

    return events


@router.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest):
    refresh_models_if_needed()

    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Model validation
    try:
        get_model_registry().resolve(request.model)
    except ModelNotFoundError:
        return model_not_found_response_anthropic(request.model)

    # Convert to OpenAI format
    from app.endpoints.chat import chat_completions
    from schemas.request import OpenAIChatRequest

    openai_dict = convert_anthropic_to_openai(request)
    openai_req = OpenAIChatRequest(
        model=openai_dict["model"],
        messages=openai_dict["messages"],
        stream=request.stream,
        tools=openai_dict.get("tools"),
    )

    try:
        response = await chat_completions(openai_req)
    except ModelNotFoundError:
        return model_not_found_response_anthropic(request.model)

    # Handle streaming
    if request.stream:
        # For streaming, we need to collect the OpenAI response and convert
        # Since our current streaming returns a single chunk, we handle it directly
        if isinstance(response, StreamingResponse):
            # Collect the stream body and convert
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                chunks.append(chunk)

            # Parse the OpenAI SSE data
            openai_data = None
            for chunk in chunks:
                for line in chunk.split("\n"):
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            openai_data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass

            if openai_data:
                events = _build_anthropic_sse_events(openai_data, request.model)
            else:
                events = _build_anthropic_sse_events(
                    {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "model": request.model},
                    request.model,
                )

            async def anthropic_stream():
                for event in events:
                    yield event

            return StreamingResponse(anthropic_stream(), media_type="text/event-stream")

    # Non-streaming: response is a dict from convert_to_openai_format
    if isinstance(response, dict):
        return convert_openai_to_anthropic(response)

    # Fallback
    return model_not_found_response_anthropic(request.model)
