"""Los cuatro lugares donde vive la versión, y el marcador de propiedad.

Sin red: solo comprueba que los archivos del repositorio concuerden entre sí.
El chequeo contra PyPI en vivo es `scripts/check_registry_ready.py`, que corre
en el workflow de publicación.

Existe porque el modo de fallar es caro: el registro valida la propiedad
buscando `mcp-name: <nombre>` dentro del README **de la versión publicada en
PyPI**, y PyPI es inmutable. Una versión subida sin el marcador no se corrige;
hay que quemar un número de versión. Un test barato acá evita eso.
"""

from __future__ import annotations

import json
import pathlib
import re
import tomllib

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def servidor() -> dict:
    return json.loads((RAIZ / "server.json").read_text(encoding="utf-8"))


def test_las_tres_versiones_concuerdan(servidor):
    """server.json (dos veces) y pyproject. Si se separan, el registro publica
    una entrada que apunta a una versión distinta de la que se subió."""
    proyecto = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    assert (
        proyecto["project"]["version"]
        == servidor["version"]
        == servidor["packages"][0]["version"]
    )


def test_la_version_del_paquete_es_la_del_modulo(servidor):
    """__version__ es lo que el servidor MCP le declara al cliente."""
    init = (RAIZ / "dashai_mcp" / "__init__.py").read_text(encoding="utf-8")
    (declarada,) = re.findall(r'__version__\s*=\s*"([^"]+)"', init)
    assert declarada == servidor["version"]


def test_el_readme_lleva_el_marcador_de_propiedad(servidor):
    """Y terminado en un límite, que es lo que el validador exige.

    El token debe ir seguido de un espacio, un salto de línea, una etiqueta HTML
    o el cierre de comentario. Pegado a otra cosa, el registro lo lee como un
    nombre distinto y rechaza la publicación con un mensaje confuso.
    """
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    patron = re.compile(r"mcp-name:\s*" + re.escape(servidor["name"]) + r"(?=\s|<|$)")
    assert patron.search(readme), (
        f"falta 'mcp-name: {servidor['name']}' en el README, en su propia línea"
    )


def test_el_nombre_usa_el_namespace_que_el_oidc_puede_probar(servidor):
    """La publicación se autentica con el OIDC de este repositorio, que solo
    acredita el namespace io.github.<owner>. Cualquier otro nombre se rechaza."""
    assert servidor["name"].startswith("io.github.Maarmapa/")


def test_el_paquete_es_de_pypi_y_habla_por_stdio(servidor):
    paquete = servidor["packages"][0]
    assert paquete["registryType"] == "pypi"
    assert paquete["identifier"] == "dashai-mcp"
    assert paquete["transport"]["type"] == "stdio"


def test_la_guarda_de_red_esta_documentada_como_paso_del_workflow():
    """El script tiene que seguir siendo lo que el workflow ejecuta.

    Si alguien lo renombra y no toca el YAML, la publicación deja de verificarse
    y volvemos al modo de fallar caro.
    """
    workflow = (RAIZ / ".github/workflows/publish-mcp-registry.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/check_registry_ready.py" in workflow
    assert (RAIZ / "scripts/check_registry_ready.py").exists()
