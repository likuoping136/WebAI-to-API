import configparser
import json
import logging
import os
from typing import Optional, List, Union, Any
from pathlib import Path
from gemini_webapi import GeminiClient as WebGeminiClient
from app.config import CONFIG

logger = logging.getLogger("app")

MODEL_CONFIG_PATH = Path("config.models.json")
MODEL_HEADER_KEY = "x-goog-ext-525001261-jspb"


class ModelNotFoundError(ValueError):
    """Raised when a requested model is not registered by WebAI."""


def _build_custom_model(model_id: str, mode_id: str, thinking_level: int) -> dict[str, Any]:
    return {
        "model_name": model_id,
        "model_header": {
            MODEL_HEADER_KEY: f'[1,null,null,null,"{mode_id}",null,null,0,[4],null,null,{thinking_level}]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0]",
        },
    }


DEFAULT_MODEL_CONFIG = {
    "models": [
        {
            "id": "gemini-flash-extended",
            "displayName": "3.5 Flash",
            "modeId": "fbb127bbb056c959",
            "thinkingLevel": 2,
            "familyCode": 1,
        },
        {
            "id": "gemini-pro-extended",
            "displayName": "3.1 Pro",
            "modeId": "9d8ca3786ebdfbea",
            "thinkingLevel": 2,
            "familyCode": 3,
        },
        {
            "id": "gemini-flash-lite-extended",
            "displayName": "3.1 Flash-Lite",
            "modeId": "cf41b0e0dd7d53e5",
            "thinkingLevel": 2,
            "familyCode": 6,
        },
    ]
}


class ModelRegistry:
    """Strict WebAI model registry backed by config.models.json with built-in fallback."""

    def __init__(self, config_path: Path = MODEL_CONFIG_PATH):
        self.config_path = Path(config_path)
        self._models = self._load_models()

    def _load_models(self) -> list[dict[str, Any]]:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = DEFAULT_MODEL_CONFIG

        models = data.get("models") or []
        normalized = []
        for item in models:
            if not all(k in item for k in ("id", "modeId", "thinkingLevel")):
                raise ValueError(f"Invalid model config entry: {item}")
            normalized.append(item)
        return normalized

    def model_ids(self) -> list[str]:
        return [m["id"] for m in self._models]

    def openai_models(self, created: int) -> list[dict[str, Any]]:
        return [
            {
                "id": m["id"],
                "object": "model",
                "created": created,
                "owned_by": "google",
            }
            for m in self._models
        ]

    def resolve(self, model: str) -> dict[str, Any]:
        for item in self._models:
            if item["id"] == model:
                return _build_custom_model(
                    model_id=item["id"],
                    mode_id=item["modeId"],
                    thinking_level=int(item["thinkingLevel"]),
                )
        raise ModelNotFoundError(f"Model not found: {model}")


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


def resolve_model_name(model: str) -> dict[str, Any]:
    """Resolve a model name to a custom gemini-webapi model dict; unknown names are rejected."""
    return get_model_registry().resolve(model)

class MyGeminiClient:
    """
    Wrapper for the Gemini Web API client.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: str | None = None) -> None:
        self.client = WebGeminiClient(secure_1psid, secure_1psidts, proxy)
        self._gems_cache = None

    async def init(self) -> None:
        """Initialize the Gemini client and persist any rotated cookies."""
        await self.client.init()
        await self._persist_cookies()

    async def _persist_cookies(self) -> None:
        """Persist rotated cookies back to config.conf to survive restarts."""
        config_path = "config.conf"
        if not os.path.exists(config_path):
            return
        try:
            cookies = self.client.cookies
            psid = cookies.get("__Secure-1PSID")
            psidts = cookies.get("__Secure-1PSIDTS")
            if not psid:
                return
            cfg = configparser.ConfigParser()
            cfg.read(config_path, encoding="utf-8")
            if "Cookies" not in cfg:
                cfg["Cookies"] = {}
            cfg["Cookies"]["gemini_cookie_1psid"] = psid
            if psidts:
                cfg["Cookies"]["gemini_cookie_1psidts"] = psidts
            with open(config_path, "w", encoding="utf-8") as f:
                cfg.write(f)
            logger.info("Cookies persisted to config.conf after rotation.")
        except Exception as e:
            logger.warning(f"Failed to persist cookies: {e}")

    async def generate_content(
        self,
        message: str,
        model: str,
        files: Optional[List[Union[str, Path]]] = None,
        gem: Optional[str] = None,
    ):
        """
        Generate content using the Gemini client.
        """
        resolved_model = resolve_model_name(model)
        resolved_gem = await self._resolve_gem(gem) if gem else None
        return await self.client.generate_content(message, model=resolved_model, files=files, gem=resolved_gem)

    async def fetch_gems(self):
        """Fetch available gems and cache them."""
        self._gems_cache = await self.client.fetch_gems()
        return self._gems_cache

    async def _resolve_gem(self, gem_id_or_name: str):
        """Resolve a gem by ID or name."""
        if self._gems_cache is None:
            await self.fetch_gems()
        for gem in self._gems_cache:
            if gem.id == gem_id_or_name or gem.name.lower() == gem_id_or_name.lower():
                return gem
        return gem_id_or_name

    async def close(self) -> None:
        """Close the Gemini client."""
        await self.client.close()

    def start_chat(self, model: str, gem: Optional[str] = None):
        """
        Start a chat session with the given model.
        """
        resolved_model = resolve_model_name(model)
        # Note: Gem resolution might need to be async if we want to support name resolution here
        # For now, we'll assume gem is passed as ID or already resolved if possible
        # but the underlying library might expect a Gem object.
        return self.client.start_chat(model=resolved_model, gem=gem)
