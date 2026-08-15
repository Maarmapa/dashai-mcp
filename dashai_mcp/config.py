"""Configuración y candado de red.

dashAI es local-first: su API **no tiene autenticación** de ningún tipo (se
revisó endpoint por endpoint). Eso es razonable para algo que corre en
localhost, pero convierte la URL base en un control de seguridad: apuntar este
servidor a un host remoto es entregarle a un modelo un backend de ML sin puerta.

Por eso el default es estricto y hay que salirse de él a propósito.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Hosts que consideramos "la misma máquina".
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})

TIMEOUT_SECONDS = float(os.getenv("DASHAI_TIMEOUT", "30"))


class ConfigError(RuntimeError):
    """La configuración impide operar. El mensaje va dirigido a quien lo lea."""


def base_url() -> str:
    """URL del backend de dashAI, sin barra final."""
    return os.getenv("DASHAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def allow_remote() -> bool:
    return os.getenv("DASHAI_ALLOW_REMOTE", "").lower() in ("1", "true", "yes")


def api_url(path: str) -> str:
    """Arma la URL completa de un endpoint de la API v1, validando el destino."""
    url = base_url()
    host = (urlparse(url).hostname or "").lower()

    if host not in LOCAL_HOSTS and not allow_remote():
        raise ConfigError(
            f"DASHAI_BASE_URL apunta a '{host}', que no es local, y la API de dashAI "
            "no tiene autenticación: exponerla a la red deja el backend abierto a "
            "cualquiera que lo alcance.\n"
            "Si el destino es de tu confianza y está protegido por otra vía (túnel "
            "SSH, red privada), habilítalo explícitamente con DASHAI_ALLOW_REMOTE=1."
        )

    return f"{url}{API_PREFIX}/{path.lstrip('/')}"
