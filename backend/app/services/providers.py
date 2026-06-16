import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.database import db_session, utc_now


PROVIDERS: dict[str, dict[str, str]] = {
    "local": {
        "label": "Local fallback",
        "env": "",
        "default_model": "deterministic-local",
        "kind": "local",
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
        "kind": "openai-responses",
    },
    "anthropic": {
        "label": "Anthropic",
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-latest",
        "kind": "anthropic-messages",
    },
    "ollama": {
        "label": "Ollama",
        "env": "",
        "default_model": "",
        "kind": "ollama",
    },
}

BLOCKED_PROVIDER_STATUSES = {"auth_failed", "model_unavailable", "quota_exhausted", "rate_limited", "missing_key"}


@dataclass(frozen=True)
class ProviderKey:
    value: str
    source: str


def list_provider_settings() -> list[dict]:
    ensure_provider_rows()
    return [get_provider_settings(provider) for provider in PROVIDERS]


def get_provider_settings(provider: str) -> dict:
    ensure_provider_rows()
    config = provider_config(provider)
    row = get_provider_row(provider)
    key = resolve_provider_key(provider, row)
    settings = {
        **row,
        "label": config["label"],
        "kind": config["kind"],
        "env_var": config["env"],
        "configured": provider in {"local", "ollama"} or bool(key.value),
        "key_source": key.source,
        "api_key": "",
        "masked_key": mask_key(key.value),
    }
    if provider == "ollama":
        settings["installed_models"] = installed_ollama_models()
    return settings


def save_provider_settings(provider: str, enabled: bool, api_key: str, model: str) -> dict:
    ensure_provider_rows()
    config = provider_config(provider)
    model = model.strip() or config["default_model"]
    key_to_store = api_key.strip()
    with db_session() as conn:
        current = conn.execute("SELECT * FROM provider_settings WHERE provider=?", (provider,)).fetchone()
        stored_key = current["api_key"]
        status = current["last_status"]
        message = current["last_message"]
        if key_to_store:
            stored_key = key_to_store
            status = "configured"
            message = f"{config['label']} key saved locally. Test connection to verify it."
        elif model != current["model"]:
            status = "configured"
            message = f"{config['label']} model saved locally. Test connection to verify it."
        conn.execute(
            """
            UPDATE provider_settings
            SET enabled=?, api_key=?, model=?, last_status=?, last_message=?, updated_at=?
            WHERE provider=?
            """,
            (1 if enabled else 0, stored_key, model, status, message, utc_now(), provider),
        )
    return get_provider_settings(provider)


def test_provider_connection(provider: str) -> dict:
    settings = get_provider_settings(provider)
    kind = settings["kind"]
    model = settings.get("model") or provider_config(provider)["default_model"]

    if provider == "local":
        result = {"connected": True, "status": "connected", "message": "Local fallback is always available.", "source": "local"}
        update_provider_status(provider, result["status"], result["message"])
        return {**get_provider_settings(provider), **result}

    if kind == "ollama":
        try:
            if not model.strip():
                result = {
                    "connected": False,
                    "status": "model_unavailable",
                    "message": "Choose an installed Ollama model name, then test again.",
                    "source": "local_service",
                }
                update_provider_status(provider, result["status"], result["message"])
                return {**get_provider_settings(provider), **result}
            response = httpx.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model, "prompt": "Reply with OK.", "stream": False, "options": {"num_predict": 8}},
                timeout=30,
            )
            connected = response.status_code == 200
            status = "connected" if connected else classify_local_provider_error(response)
            result = {
                "connected": connected,
                "status": status,
                "message": (
                    f"Ollama model `{model}` verified."
                    if connected
                    else provider_error_message("Ollama", model, response)
                ),
                "source": "local_service",
            }
        except Exception as exc:
            result = {"connected": False, "status": "failed", "message": str(exc), "source": "local_service"}
        update_provider_status(provider, result["status"], result["message"])
        return {**get_provider_settings(provider), **result}

    key = resolve_provider_key(provider, get_provider_row(provider))
    if not key.value:
        result = {"connected": False, "status": "missing_key", "message": f"No {settings['label']} API key is configured.", "source": "none"}
        update_provider_status(provider, result["status"], result["message"])
        return {**settings, **result}

    try:
        if provider == "openai":
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {key.value}", "Content-Type": "application/json"},
                json={"model": model, "input": "Reply with OK.", "max_output_tokens": 16},
                timeout=20,
            )
        elif provider == "anthropic":
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key.value, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with OK."}]},
                timeout=20,
            )
        else:
            response = httpx.Response(400)
        if response.status_code == 200:
            result = {
                "connected": True,
                "status": "connected",
                "message": f"{settings['label']} key and model `{model}` verified.",
                "source": key.source,
            }
        else:
            status = classify_provider_error(response)
            result = {
                "connected": False,
                "status": status,
                "message": provider_error_message(settings["label"], model, response),
                "source": key.source,
            }
    except Exception as exc:
        result = {"connected": False, "status": "failed", "message": str(exc), "source": key.source}
    update_provider_status(provider, result["status"], result["message"])
    return {**get_provider_settings(provider), **result}


def call_configured_model(prompt: str) -> str | None:
    ensure_provider_rows()
    for settings in list_provider_settings():
        if not settings["enabled"] or settings["provider"] == "local":
            continue
        if settings.get("last_status") in BLOCKED_PROVIDER_STATUSES:
            continue
        text = call_provider(settings["provider"], prompt)
        if text:
            return text
    return None


def call_provider(provider: str, prompt: str) -> str | None:
    settings = get_provider_settings(provider)
    kind = settings["kind"]
    key = resolve_provider_key(provider, get_provider_row(provider))
    model = settings.get("model") or provider_config(provider)["default_model"]

    if kind == "ollama":
        try:
            response = httpx.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            if response.status_code == 200:
                update_provider_status(provider, "connected", "Ollama synthesis succeeded.")
                return response.json().get("response")
            update_provider_status(provider, classify_local_provider_error(response), provider_error_message(settings["label"], model, response))
        except Exception as exc:
            update_provider_status(provider, "failed", str(exc))
        return None

    if not key.value:
        return None

    try:
        if kind == "openai-responses":
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {key.value}", "Content-Type": "application/json"},
                json={"model": model, "input": prompt},
                timeout=60,
            )
            text = parse_openai_response(response)
        elif kind == "anthropic-messages":
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key.value, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            text = parse_anthropic_response(response)
        else:
            text = None
        if text:
            update_provider_status(provider, "connected", f"{settings['label']} synthesis succeeded.")
            return text
        if "response" in locals() and response.status_code >= 400:
            update_provider_status(provider, classify_provider_error(response), provider_error_message(settings["label"], model, response))
    except Exception as exc:
        update_provider_status(provider, "failed", str(exc))
    return None


def parse_openai_response(response: httpx.Response) -> str | None:
    if response.status_code >= 400:
        return None
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("text"):
                return content["text"]
    return None


def parse_anthropic_response(response: httpx.Response) -> str | None:
    if response.status_code >= 400:
        return None
    data = response.json()
    parts = [part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"]
    return "\n".join(part for part in parts if part).strip() or None


def model_is_available(model: str, model_names: list[str]) -> bool:
    if model in model_names:
        return True
    return any(name.split(":", 1)[0] == model for name in model_names)


def installed_ollama_models() -> list[str]:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if response.status_code != 200:
            return []
        return sorted(item.get("name", "") for item in response.json().get("models", []) if item.get("name"))
    except Exception:
        return []


def provider_error_message(label: str, model: str, response: httpx.Response) -> str:
    detail = provider_error_detail(response)
    status = classify_provider_error(response)
    if status == "quota_exhausted":
        base = f"{label} key appears exhausted or billing-limited for model `{model}`."
    elif status == "rate_limited":
        base = f"{label} is rate-limiting requests for model `{model}`."
    elif status == "auth_failed":
        base = f"{label} rejected the API key for model `{model}`."
    elif status == "model_unavailable":
        base = f"{label} rejected or cannot access model `{model}`."
    else:
        base = f"{label} rejected key or model `{model}` with HTTP {response.status_code}."
    return f"{base} {detail}".strip()


def provider_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "")
        if isinstance(error, str):
            return error
    except Exception:
        return response.text[:180]
    return ""


def classify_provider_error(response: httpx.Response) -> str:
    detail = provider_error_detail(response).lower()
    if response.status_code in {401, 403}:
        return "auth_failed"
    if response.status_code == 404:
        return "model_unavailable"
    if response.status_code == 429:
        if any(term in detail for term in ("quota", "credit", "credits", "billing", "balance", "exceeded your current quota")):
            return "quota_exhausted"
        return "rate_limited"
    if response.status_code == 402 or any(term in detail for term in ("insufficient_quota", "quota", "credit balance", "billing hard limit", "billing")):
        return "quota_exhausted"
    if any(term in detail for term in ("model", "does not exist", "not found", "not have access", "unsupported")):
        return "model_unavailable"
    return "failed"


def classify_local_provider_error(response: httpx.Response) -> str:
    detail = provider_error_detail(response).lower()
    if response.status_code == 404 or any(term in detail for term in ("model", "not found", "pull", "not installed")):
        return "model_unavailable"
    return "failed"


def ensure_provider_rows() -> None:
    now = utc_now()
    with db_session() as conn:
        placeholders = ",".join("?" for _ in PROVIDERS)
        conn.execute(f"DELETE FROM provider_settings WHERE provider NOT IN ({placeholders})", tuple(PROVIDERS))
        for provider, config in PROVIDERS.items():
            enabled = 1 if provider == "local" else 0
            conn.execute(
                """
                INSERT OR IGNORE INTO provider_settings
                (provider, enabled, api_key, model, last_status, last_message, created_at, updated_at)
                VALUES (?, ?, '', ?, 'untested', '', ?, ?)
                """,
                (provider, enabled, config["default_model"], now, now),
            )


def get_provider_row(provider: str) -> dict[str, Any]:
    provider_config(provider)
    with db_session() as conn:
        row = conn.execute("SELECT * FROM provider_settings WHERE provider=?", (provider,)).fetchone()
    if not row:
        ensure_provider_rows()
        with db_session() as conn:
            row = conn.execute("SELECT * FROM provider_settings WHERE provider=?", (provider,)).fetchone()
    return row


def resolve_provider_key(provider: str, row: dict[str, Any]) -> ProviderKey:
    env_name = provider_config(provider)["env"]
    if env_name:
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return ProviderKey(env_key, "environment")
    stored_key = row.get("api_key", "").strip()
    if stored_key:
        return ProviderKey(stored_key, "local_settings")
    return ProviderKey("", "none")


def update_provider_status(provider: str, status: str, message: str) -> None:
    ensure_provider_rows()
    with db_session() as conn:
        conn.execute(
            """
            UPDATE provider_settings
            SET last_status=?, last_message=?, last_checked_at=?, updated_at=?
            WHERE provider=?
            """,
            (status, message, utc_now(), utc_now(), provider),
        )


def provider_config(provider: str) -> dict[str, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDERS[provider]


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "********"
    return f"{value[:4]}...{value[-4:]}"
