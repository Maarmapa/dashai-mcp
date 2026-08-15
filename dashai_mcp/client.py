"""Cliente HTTP compartido y traducción de errores.

Todos los errores se devuelven como texto accionable en vez de una traza: el
que lee esto es un modelo que tiene que decidir el siguiente paso, y "Connection
refused" no le dice que falta levantar dashAI.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import TIMEOUT_SECONDS, ConfigError, api_url, base_url


class DashAIError(RuntimeError):
    """Error ya traducido a algo que se le puede mostrar a un agente."""


def _detalle_http(exc: httpx.HTTPStatusError) -> str:
    """Extrae el `detail` de FastAPI, que suele explicar exactamente qué faltó."""
    try:
        cuerpo = exc.response.json()
    except ValueError:
        return exc.response.text[:300]
    if isinstance(cuerpo, dict) and "detail" in cuerpo:
        return str(cuerpo["detail"])[:500]
    return str(cuerpo)[:300]


async def request(method: str, path: str, **kwargs: Any) -> Any:
    """Llama a la API v1 de dashAI y devuelve el JSON ya decodificado.

    Lanza DashAIError con un mensaje accionable ante cualquier fallo.
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
            f"No hay respuesta de dashAI en {base_url()}.\n"
            "Comprueba que esté corriendo: ejecuta `dashai` en una terminal (levanta "
            "el backend en el puerto 8000) o abre la aplicación de escritorio.\n"
            "Si lo tienes en otro puerto, ajusta DASHAI_BASE_URL."
        ) from e

    except httpx.TimeoutException as e:
        raise DashAIError(
            f"dashAI no respondió en {TIMEOUT_SECONDS:.0f} s. Las operaciones largas "
            "(entrenar, explicar, predecir) NO se esperan por HTTP: se encolan como "
            "job y se consultan con dashai_job_status. Si el que expiró fue un listado, "
            "sube DASHAI_TIMEOUT."
        ) from e

    except httpx.HTTPStatusError as e:
        codigo = e.response.status_code
        detalle = _detalle_http(e)

        if codigo == 404:
            raise DashAIError(
                f"dashAI no encontró el recurso ({detalle}). Verifica el id con la "
                "herramienta de listado correspondiente antes de reintentar."
            ) from e
        if codigo == 409:
            raise DashAIError(f"Conflicto: {detalle}") from e
        if codigo == 422:
            raise DashAIError(
                f"dashAI rechazó los parámetros: {detalle}\n"
                "Los nombres de modelos, métricas y tareas deben venir tal cual los "
                "devuelve dashai_list_components — son sensibles a mayúsculas."
            ) from e
        raise DashAIError(f"dashAI respondió {codigo}: {detalle}") from e


async def get(path: str, **kwargs: Any) -> Any:
    return await request("GET", path, **kwargs)


async def post(path: str, **kwargs: Any) -> Any:
    return await request("POST", path, **kwargs)
