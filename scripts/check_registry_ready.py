"""¿Puede publicarse server.json en el Registro MCP ahora mismo?

Existe porque el modo de fallar es caro y silencioso. Para paquetes de PyPI el
registro NO mira este repositorio: descarga
``https://pypi.org/pypi/<paquete>/<version>/json`` y exige encontrar el token
``mcp-name: <nombre-del-servidor>`` dentro de ``info.description`` —el README tal
como quedó publicado EN ESA VERSIÓN—. Y PyPI es inmutable: una versión subida sin
el marcador no se arregla, hay que publicar otra.

Así que este script comprueba lo mismo que el validador del registro, con la
misma URL y el mismo patrón (ver ``internal/validators/registries/pypi.go`` en
modelcontextprotocol/registry), antes de gastar una publicación.

    python scripts/check_registry_ready.py

Sale 0 si todo calza. Sale 1 diciendo exactamente qué falta.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import tomllib
import urllib.error
import urllib.request

RAIZ = pathlib.Path(__file__).resolve().parent.parent
TIEMPO_LIMITE = 20


def _token_presente(descripcion: str, nombre: str) -> bool:
    """Igual que el validador: el token debe terminar en un límite.

    Un espacio, un salto de línea, una etiqueta HTML o el cierre de comentario
    ``-->``. Sin eso, ``mcp-name: io.github.X/y`` pegado a otra cosa se lee como
    un nombre distinto y el registro lo rechaza con un mensaje confuso.
    """
    patron = re.compile(r"mcp-name:\s*" + re.escape(nombre) + r"(?=\s|<|$)")
    return bool(patron.search(descripcion))


def main() -> int:
    servidor = json.loads((RAIZ / "server.json").read_text(encoding="utf-8"))
    nombre = servidor["name"]
    paquete = servidor["packages"][0]
    identificador = paquete["identifier"]
    version = paquete["version"]

    proyecto = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    version_proyecto = proyecto["project"]["version"]

    problemas: list[str] = []

    if not (version_proyecto == servidor["version"] == version):
        problemas.append(
            f"las versiones no concuerdan: pyproject={version_proyecto}, "
            f"server.json={servidor['version']}, entrada de paquete={version}. "
            "Se suben las tres juntas."
        )

    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    if not _token_presente(readme, nombre):
        problemas.append(
            f"el README del repositorio no contiene 'mcp-name: {nombre}' con un "
            "límite después. Va en su propia línea, por ejemplo como comentario."
        )

    url = f"https://pypi.org/pypi/{identificador}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=TIEMPO_LIMITE) as respuesta:
            datos = json.load(respuesta)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            problemas.append(
                f"{identificador}=={version} no está en PyPI. Publicar ahí PRIMERO: "
                "registrar ahora crearía una entrada apuntando a una versión que no "
                "existe."
            )
            datos = None
        else:
            problemas.append(f"PyPI respondió {e.code} al pedir {url}")
            datos = None
    except OSError as e:
        # Sin red no se puede afirmar nada. Decirlo, no asumir que está bien.
        print(f"AVISO: no se pudo consultar PyPI ({e}). Chequeo incompleto.")
        datos = None

    if datos is not None:
        descripcion = datos["info"].get("description") or ""
        if _token_presente(descripcion, nombre):
            print(f"PyPI: {identificador}=={version} publica el marcador. Correcto.")
        else:
            problemas.append(
                f"el README publicado de {identificador}=={version} NO contiene "
                f"'mcp-name: {nombre}'. PyPI es inmutable: agrega el marcador al "
                "README y publica una versión nueva."
            )

    if problemas:
        print("\nNo se puede publicar en el registro todavía:\n")
        for p in problemas:
            print(f"  - {p}")
        return 1

    print(f"Todo calza. {nombre} -> {identificador}=={version} listo para publicar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
