import json
from pathlib import Path

import pytest

from schemas.request import OpenAIChatRequest
from models.gemini import ModelNotFoundError, ModelRegistry, MyGeminiClient
from app.endpoints import chat as chat_module


class FakeInnerClient:
    def __init__(self):
        self.started = []

    def start_chat(self, model=None, gem=None):
        self.started.append(model)
        return object()


def test_registry_loads_configured_models_and_builds_extended_header(tmp_path):
    config_path = tmp_path / "config.models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gemini-flash-extended",
                        "displayName": "3.5 Flash",
                        "modeId": "fbb127bbb056c959",
                        "thinkingLevel": 2,
                        "familyCode": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = ModelRegistry(config_path)

    assert registry.model_ids() == ["gemini-flash-extended"]
    assert registry.openai_models(created=123) == [
        {
            "id": "gemini-flash-extended",
            "object": "model",
            "created": 123,
            "owned_by": "google",
        }
    ]
    assert registry.resolve("gemini-flash-extended") == {
        "model_name": "gemini-flash-extended",
        "model_header": {
            "x-goog-ext-525001261-jspb": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4],null,null,2]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0]",
        },
    }


def test_unknown_model_raises_model_not_found_and_does_not_start_chat(monkeypatch):
    fake_inner = FakeInnerClient()
    client = MyGeminiClient("psid", "psidts")
    client.client = fake_inner

    with pytest.raises(ModelNotFoundError) as exc:
        client.start_chat(model="missing-model")

    assert "missing-model" in str(exc.value)
    assert fake_inner.started == []


def test_model_not_found_error_response_is_openai_compatible():
    response = chat_module.model_not_found_response("missing-model")

    assert response.status_code == 404
    body = json.loads(response.body)
    assert body == {
        "error": {
            "message": "Model not found: missing-model",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_chat_completion_returns_model_not_found_without_sending_to_gemini(monkeypatch):
    class RejectingClient:
        def __init__(self):
            self.started = []

        def start_chat(self, model, gem=None):
            self.started.append(model)
            raise ModelNotFoundError(f"Model not found: {model}")

    async def run():
        fake_client = RejectingClient()
        monkeypatch.setattr(chat_module, "get_gemini_client", lambda: fake_client)

        response = await chat_module.chat_completions(
            OpenAIChatRequest(
                model="missing-model",
                messages=[{"role": "user", "content": "hello"}],
            )
        )

        assert response.status_code == 404
        assert json.loads(response.body)["error"]["code"] == "model_not_found"
        assert fake_client.started == ["missing-model"]

    import asyncio

    asyncio.run(run())
