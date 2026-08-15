"""Configuration and network lock.

dashAI is local-first: its API has **no authentication** of any kind (checked
endpoint by endpoint). That is reasonable for something running on localhost,
but it turns the base URL into a security control: pointing this server at a
remote host hands a model an ML backend with no door on it.

So the default is strict, and stepping outside it has to be deliberate.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Hosts we treat as "the same machine".
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})

TIMEOUT_SECONDS = float(os.getenv("DASHAI_TIMEOUT", "30"))


class ConfigError(RuntimeError):
    """Configuration prevents operating. The message is aimed at whoever reads it."""


def base_url() -> str:
    """dashAI backend URL, without a trailing slash."""
    return os.getenv("DASHAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def allow_remote() -> bool:
    return os.getenv("DASHAI_ALLOW_REMOTE", "").lower() in ("1", "true", "yes")


def api_url(path: str) -> str:
    """Builds the full URL of an API v1 endpoint, validating the target."""
    url = base_url()
    host = (urlparse(url).hostname or "").lower()

    if host not in LOCAL_HOSTS and not allow_remote():
        raise ConfigError(
            f"DASHAI_BASE_URL points to '{host}', which is not local, and dashAI's "
            "API has no authentication: exposing it to the network leaves the "
            "backend open to anyone who can reach it.\n"
            "If the target is one you trust and is protected some other way (SSH "
            "tunnel, private network), enable it explicitly with "
            "DASHAI_ALLOW_REMOTE=1."
        )

    return f"{url}{API_PREFIX}/{path.lstrip('/')}"
