import json
import subprocess
from pathlib import Path

import pytest

from schemas.request import OpenAIChatRequest
from models.gemini import DailyModelDiscovery, ModelNotFoundError, ModelRegistry, MyGeminiClient
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
                        "id": "gemini-3.5-flash-extended",
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

    assert registry.model_ids() == ["gemini-3.5-flash-extended"]
    assert registry.openai_models(created=123) == [
        {
            "id": "gemini-3.5-flash-extended",
            "object": "model",
            "created": 123,
            "owned_by": "google",
        }
    ]
    assert registry.resolve("gemini-3.5-flash-extended") == {
        "model_name": "gemini-3.5-flash-extended",
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


def test_model_not_found_response_is_chat_completion_text_not_api_error():
    response = chat_module.model_not_found_response("missing-model", stream=False)

    assert response["choices"][0]["message"]["content"] == "模型不存在：missing-model"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["model"] == "missing-model"


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

        assert response["choices"][0]["message"]["content"] == "模型不存在：missing-model"
        assert fake_client.started == ["missing-model"]

    import asyncio

    asyncio.run(run())


def test_daily_discovery_runs_once_per_day(tmp_path):
    calls = []

    discovery = DailyModelDiscovery(
        state_path=tmp_path / "state.json",
        today=lambda: "2026-05-20",
        runner=lambda: calls.append("run"),
    )

    discovery.refresh_if_needed()
    discovery.refresh_if_needed()

    assert calls == ["run"]
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["lastSuccessDate"] == "2026-05-20"


def test_daily_discovery_failure_does_not_block_and_allows_retry(tmp_path):
    calls = []

    def failing_runner():
        calls.append("run")
        raise RuntimeError("cdp unavailable")

    discovery = DailyModelDiscovery(
        state_path=tmp_path / "state.json",
        today=lambda: "2026-05-20",
        runner=failing_runner,
    )

    discovery.refresh_if_needed()
    discovery.refresh_if_needed()

    assert calls == ["run", "run"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["lastSuccessDate"] is None
    assert "cdp unavailable" in state["lastError"]


def test_default_daily_discovery_uses_webai_cdp_port_9223(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd, env, check, timeout, capture_output, text, encoding):
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.chdir(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "discover-gemini-web-models.mjs").write_text("", encoding="utf-8")
    monkeypatch.delenv("WEBAI_CDP_URL", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    discovery = DailyModelDiscovery(state_path=tmp_path / "state.json")
    discovery._run_discovery_script()

    assert captured["env"]["WEBAI_CDP_URL"] == "http://127.0.0.1:9223"
