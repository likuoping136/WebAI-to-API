# src/schemas/anthropic_request.py
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class AnthropicContentBlock(BaseModel):
    type: str  # "text", "tool_use", "tool_result"
    text: Optional[str] = None
    # tool_use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result fields
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None


class AnthropicMessage(BaseModel):
    role: str  # "user", "assistant"
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = Field(default=None, alias="input_schema")


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

    model_config = {"populate_by_name": True}
