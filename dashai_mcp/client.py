"""Shared HTTP client and error translation.

Every error comes back as actionable text instead of a traceback: whoever reads
this is a model that has to decide the next step, and "Connection refused" does
not tell it that dashAI needs to be started.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import TIMEOUT_SECONDS, ConfigError, api_url, base_url


class DashAIError(RuntimeError):
    """An error already translated into something an agent can be shown."""


def _detalle_http(exc: httpx.HTTPStatusError) -> str:
    """Pulls FastAPI's `detail`, which usually explains exactly what was missing."""
    try:
        cuerpo = exc.response.json()
    except ValueError:
        return exc.response.text[:300]
    if isinstance(cuerpo, dict) and "detail" in cuerpo:
        return str(cuerpo["detail"])[:500]
    return str(cuerpo)[:300]


async def request(method: str, path: str, **kwargs: Any) -> Any:
    """Calls dashAI's v1 API and returns the decoded JSON.

    Raises DashAIError with an actionable message on any failure.
    """
    try:
        url = api_url(path)
    except ConfigError as e:
        raise DashAIError(str(e)) from e

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            respuesta = await client.request(method, url, **kwargs)
            respuesta.raise_for_status()
            if not respuesta.content:
                return None
            return respuesta.json()

    except httpx.ConnectError as e:
        raise DashAIError(
            f"No response from dashAI at {base_url()}.\n"
            "Check that it is running: run `dashai` in a terminal (it starts the "
            "backend on port 8000) or open the desktop app.\n"
            "If you have it on another port, adjust DASHAI_BASE_URL."
        ) from e

    except httpx.TimeoutException as e:
        raise DashAIError(
            f"dashAI did not respond within {TIMEOUT_SECONDS:.0f} s. Long operations "
            "(training, explaining, predicting) are NOT waited on over HTTP: they are "
            "enqueued as a job and polled with dashai_job_status. If what timed out "
            "was a listing, raise DASHAI_TIMEOUT."
        ) from e

    except httpx.HTTPStatusError as e:
        codigo = e.response.status_code
        detalle = _detalle_http(e)

        if codigo == 404:
            raise DashAIError(
                f"dashAI could not find the resource ({detalle}). Verify the id with "
                "the matching listing tool before retrying."
            ) from e
        if codigo == 409:
            raise DashAIError(f"Conflict: {detalle}") from e
        if codigo == 422:
            raise DashAIError(
                f"dashAI rejected the parameters: {detalle}\n"
                "Model, metric and task names must be passed exactly as "
                "dashai_list_components returns them — they are case-sensitive."
            ) from e
        raise DashAIError(f"dashAI responded {codigo}: {detalle}") from e


async def get(path: str, **kwargs: Any) -> Any:
    return await request("GET", path, **kwargs)


async def post(path: str, **kwargs: Any) -> Any:
    return await request("POST", path, **kwargs)
