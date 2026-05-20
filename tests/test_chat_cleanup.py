import asyncio

from app.endpoints import chat as chat_module
from schemas.request import OpenAIChatRequest


class FakeResponse:
    text = "OK"


class FakeChatSession:
    cid = "cid-123"

    async def send_message(self, prompt, **kwargs):
        return FakeResponse()


class FakeInnerClient:
    def __init__(self):
        self.deleted = []

    async def delete_chat(self, cid):
        self.deleted.append(cid)


class FakeGeminiClient:
    def __init__(self):
        self.client = FakeInnerClient()

    def start_chat(self, model, gem=None):
        return FakeChatSession()


def test_chat_completion_deletes_temporary_chat_via_underlying_client(monkeypatch):
    async def run():
        fake_client = FakeGeminiClient()
        monkeypatch.setattr(chat_module, "get_gemini_client", lambda: fake_client)

        request = OpenAIChatRequest(
            model="gemini-3-pro",
            messages=[{"role": "user", "content": "hello"}],
        )

        await chat_module.chat_completions(request)

        assert fake_client.client.deleted == ["cid-123"]

    asyncio.run(run())
