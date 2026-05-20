# src/app/endpoints/chat.py
import json
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from app.logger import logger
from schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.session_manager import get_translate_session_manager
from models.gemini import ModelNotFoundError, get_model_registry

router = APIRouter()

@router.get("/v1/gems")
async def list_gems():
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        gems = await gemini_client.fetch_gems()
        return {
            "gems": [
                {
                    "id": gem.id,
                    "name": gem.name,
                    "description": gem.description,
                    "predefined": gem.predefined,
                }
                for gem in gems
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching gems: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching gems: {str(e)}")


@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        response = await session_manager.get_response(request.model, request.message, request.files, request.gem)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


def _build_tools_prompt(tools: list) -> str:
    """Convert OpenAI tool definitions to a system prompt for Gemini."""
    declarations = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            declarations.append(t["function"])
    if not declarations:
        return ""
    lines = [
        "You have access to the following tools. When you want to call a tool, respond with "
        "ONLY a JSON object in this exact format, with no other text before or after:\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<arguments>}}}\n',
        "Available tools:",
    ]
    for fn in declarations:
        lines.append(f"- {fn['name']}: {fn.get('description', '')}")
        if fn.get("parameters"):
            lines.append(f"  Parameters: {json.dumps(fn['parameters'])}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> Optional[dict]:
    """Extract a tool_call JSON object from model response text."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and "tool_call" in obj:
                    return obj["tool_call"]
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def convert_to_openai_format(response_text: str, model: str, stream: bool = False, tool_call: Optional[dict] = None):
    ts = int(time.time())
    choice_key = "delta" if stream else "message"
    
    if tool_call:
        args = tool_call.get("arguments", {})
        content = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{ts}",
                "type": "function",
                "function": {
                    "name": tool_call.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                },
            }],
        }
        return {
            "id": f"chatcmpl-{ts}",
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "created": ts,
            "model": model,
            "choices": [{
                "index": 0,
                choice_key: content,
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return {
        "id": f"chatcmpl-{ts}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": ts,
        "model": model,
        "choices": [{
            "index": 0,
            choice_key: {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def model_not_found_response(model: str, stream: bool = False):
    return convert_to_openai_format(f"模型不存在：{model}", model, stream)


@router.get("/v1/models")
async def list_models():
    ts = int(time.time())
    return {
        "object": "list",
        "data": get_model_registry().openai_models(created=ts),
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    is_stream = request.stream if request.stream is not None else False

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    if not request.model:
        raise HTTPException(status_code=400, detail="Model not specified in the request.")

    # Build tools prompt from OpenAI tool definitions
    tools_prompt = _build_tools_prompt(request.tools) if request.tools else ""

    # Separate system message, conversation turns, and latest user message
    system_content = ""
    history_turns = []  # list of (role, content) tuples for Gemini session
    latest_user_msg = None

    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""

        if role == "system":
            system_content = content
        elif role == "user":
            latest_user_msg = content
            history_turns.append(("user", content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    history_turns.append(("assistant", f"[Called tool {fn.get('name')} with args {fn.get('arguments', '')}]"))
            elif content:
                history_turns.append(("assistant", content))
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            history_turns.append(("tool", f"[Tool result {tool_call_id}]: {content}"))

    if not latest_user_msg:
        raise HTTPException(status_code=400, detail="No user message found.")

    # Prepend system + tools to the first user message for context
    context_prefix = ""
    if system_content:
        context_prefix += f"[System Instructions]\n{system_content}\n\n"
    if tools_prompt:
        context_prefix += f"[Available Tools]\n{tools_prompt}\n\n"
    if context_prefix:
        context_prefix += "[End of System Instructions]\n\n"

    try:
        # Use ChatSession for structured multi-turn conversation
        # This lets Gemini understand the dialogue structure instead of
        # treating the whole history as a flat text blob.
        chat_session = gemini_client.start_chat(model=request.model, gem=request.gem)

        # Feed conversation history as structured turns
        # Only send up to (but not including) the last user message as history,
        # then send the last message separately to get the response.
        history = history_turns[:-1]  # everything before the last user message

        # If there's history, send it as a single context-setting message first
        if history:
            history_lines = []
            for role, text in history:
                if role == "user":
                    history_lines.append(f"User: {text}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {text}")
                elif role == "tool":
                    history_lines.append(text)
            history_prompt = "Previous conversation context:\n\n" + "\n".join(history_lines)
            await chat_session.send_message(prompt=history_prompt, temporary=True)

        # Send the latest user message with system/tools prefix
        final_prompt = context_prefix + latest_user_msg if context_prefix else latest_user_msg
        response = await chat_session.send_message(prompt=final_prompt)

        # Clean up: delete the temporary chat from Gemini history
        try:
            await gemini_client.client.delete_chat(chat_session.cid)
        except Exception as cleanup_err:
            logger.debug(f"Failed to delete temporary chat: {cleanup_err}")

        logger.debug(f"Gemini raw response: {response.text!r}")
        tool_call = _parse_tool_call(response.text) if request.tools else None
        logger.debug(f"Parsed tool_call: {tool_call}")
        
        openai_response = convert_to_openai_format(response.text, request.model, is_stream, tool_call)
        
        if is_stream:
            from fastapi.responses import StreamingResponse
            async def sse_stream():
                yield f"data: {json.dumps(openai_response)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_stream(), media_type="text/event-stream")
            
        return openai_response
    except ModelNotFoundError:
        return model_not_found_response(request.model)
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
