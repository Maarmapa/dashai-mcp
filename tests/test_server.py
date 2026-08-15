"""Tests deterministas del servidor MCP.

No necesitan una instancia de dashAI: las respuestas HTTP se simulan con respx.
Lo que se prueba es el contrato — qué llamadas hace, en qué orden, con qué
cuerpo, y qué le dice al agente cuando algo sale mal.
"""

import json

import httpx
import pytest
import respx

from dashai_mcp import client, config
from dashai_mcp.server import (
    ObtenerRun,
    dashai_get_run,
    dashai_list_components,
    DescribirDataset,
    EntrenarModelo,
    EstadoJob,
    ListarComponentes,
    SinArgumentos,
    dashai_describe_dataset,
    dashai_job_status,
    dashai_server_info,
    dashai_train_model,
)

API = "http://localhost:8000/api/v1"


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Cada test parte de un entorno conocido."""
    monkeypatch.delenv("DASHAI_BASE_URL", raising=False)
    monkeypatch.delenv("DASHAI_ALLOW_REMOTE", raising=False)


# --- El candado de red -------------------------------------------------------

def test_url_local_pasa():
    assert config.api_url("dataset/") == f"{API}/dataset/"


def test_url_remota_se_bloquea(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "http://ml.ejemplo.com:8000")
    with pytest.raises(config.ConfigError) as e:
        config.api_url("dataset/")
    # El mensaje tiene que explicar POR QUÉ, no solo negarse.
    assert "no tiene autenticación" in str(e.value)
    assert "DASHAI_ALLOW_REMOTE" in str(e.value)


def test_url_remota_con_opt_in_explicito(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "http://ml.ejemplo.com:8000")
    monkeypatch.setenv("DASHAI_ALLOW_REMOTE", "1")
    assert config.api_url("dataset/").startswith("http://ml.ejemplo.com:8000/api/v1")


async def test_el_bloqueo_llega_al_agente_como_error(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "https://ml.ejemplo.com")
    salida = await dashai_server_info(SinArgumentos())
    assert salida.startswith("Error:")
    assert "DASHAI_ALLOW_REMOTE" in salida


# --- Errores accionables -----------------------------------------------------

@respx.mock
async def test_dashai_apagado_dice_como_levantarlo():
    respx.get(f"{API}/dataset/").mock(side_effect=httpx.ConnectError("refused"))
    salida = await dashai_server_info(SinArgumentos())
    assert "`dashai`" in salida
    assert "DASHAI_BASE_URL" in salida


@respx.mock
async def test_422_apunta_a_list_components():
    respx.post(f"{API}/model-session/").mock(
        return_value=httpx.Response(422, json={"detail": "Unknown task 'Clasificacion'"})
    )
    salida = await dashai_train_model(
        EntrenarModelo(
            dataset_id=1, task_name="Clasificacion", model_name="RandomForest",
            input_columns=["a"], output_columns=["b"], metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )
    assert "Unknown task" in salida
    assert "dashai_list_components" in salida


# --- Validación de entrada ---------------------------------------------------

def test_splits_deben_sumar_uno():
    with pytest.raises(ValueError, match="suman"):
        EntrenarModelo(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy",
            splits={"train": 0.8, "validation": 0.3, "test": 0.1},
        )


def test_splits_incompletos_se_rechazan():
    with pytest.raises(ValueError, match="faltan"):
        EntrenarModelo(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy", splits={"train": 1.0},
        )


def test_tipo_de_componente_invalido_se_rechaza():
    with pytest.raises(ValueError, match="no reconocidos"):
        ListarComponentes(types=["Modelo"])  # en español, no existe


async def test_goal_metric_debe_estar_en_metrics():
    salida = await dashai_train_model(
        EntrenarModelo(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="F1",
        )
    )
    assert salida.startswith("Error:")
    assert "goal_metric" in salida


# --- El recorrido de entrenamiento ------------------------------------------

@respx.mock
async def test_entrenar_encadena_las_tres_llamadas():
    sesion = respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(201, json={"id": 7}))
    run = respx.post(f"{API}/run/").mock(return_value=httpx.Response(201, json={"id": 42}))
    job = respx.post(f"{API}/job/").mock(return_value=httpx.Response(201, json={"id": "huey-abc"}))

    salida = json.loads(
        await dashai_train_model(
            EntrenarModelo(
                dataset_id=3, task_name="TabularClassificationTask", model_name="RandomForestClassifier",
                input_columns=["largo", "ancho"], output_columns=["especie"],
                metrics=["Accuracy", "F1"], goal_metric="F1", parameters={"n_estimators": 100},
            )
        )
    )

    assert sesion.called and run.called and job.called
    assert salida == {
        "job_id": "huey-abc", "run_id": 42, "model_session_id": 7,
        "status": "enqueued", "next_step": salida["next_step"],
    }
    assert "dashai_job_status" in salida["next_step"]

    # splits viaja como STRING JSON: dashAI lo declara str, aunque su doc lo pinte objeto.
    cuerpo_sesion = json.loads(sesion.calls[0].request.content)
    assert isinstance(cuerpo_sesion["splits"], str)
    assert json.loads(cuerpo_sesion["splits"])["train"] == 0.7
    # Las métricas se aplican a los tres splits.
    assert cuerpo_sesion["train_metrics"] == cuerpo_sesion["test_metrics"] == ["Accuracy", "F1"]

    # El run enlaza con la sesión recién creada y manda los plot_path obligatorios.
    cuerpo_run = json.loads(run.calls[0].request.content)
    assert cuerpo_run["model_session_id"] == 7
    assert cuerpo_run["plot_history_path"] == ""
    assert cuerpo_run["parameters"] == {"n_estimators": 100}

    # El job apunta al run, no a la sesión — y viaja como form data (ver
    # test_el_job_se_encola_como_form_data para el porqué).
    cuerpo_job = dict(par.split("=", 1) for par in job.calls[0].request.content.decode().split("&"))
    assert cuerpo_job["job_type"] == "ModelJob"
    assert "42" in cuerpo_job["kwargs"]


@respx.mock
async def test_si_falla_la_sesion_no_se_encola_nada():
    respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(404, json={"detail": "Dataset no existe"}))
    ruta_job = respx.post(f"{API}/job/")

    salida = await dashai_train_model(
        EntrenarModelo(
            dataset_id=999, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )
    assert salida.startswith("Error:")
    assert not ruta_job.called, "no se debe encolar un job si el experimento no se creó"


# --- Estado de jobs ----------------------------------------------------------

@respx.mock
@pytest.mark.parametrize(
    "estado,terminado,fallido",
    [("not_started", False, False), ("started", False, False), ("finished", True, False), ("error", False, True)],
)
async def test_estados_de_job(estado, terminado, fallido):
    respx.get(f"{API}/job/status/j1").mock(return_value=httpx.Response(200, json={"status": estado}))
    salida = json.loads(await dashai_job_status(EstadoJob(job_id="j1")))
    assert salida["status"] == estado
    assert salida["finished"] is terminado
    assert salida["failed"] is fallido


# --- Degradación -------------------------------------------------------------

@respx.mock
async def test_describir_dataset_degrada_si_falta_una_parte():
    """Un dataset recién creado no tiene tipos ni muestra: no es motivo de error."""
    respx.get(f"{API}/dataset/5").mock(return_value=httpx.Response(200, json={"id": 5, "name": "iris"}))
    respx.get(f"{API}/dataset/5/info").mock(return_value=httpx.Response(200, json={"filas": 150}))
    respx.get(f"{API}/dataset/5/types").mock(return_value=httpx.Response(404, json={"detail": "sin tipos"}))
    respx.get(f"{API}/dataset/5/sample").mock(return_value=httpx.Response(500, text="boom"))

    salida = json.loads(await dashai_describe_dataset(DescribirDataset(dataset_id=5)))
    assert salida["dataset"]["name"] == "iris"
    assert salida["info"] == {"filas": 150}
    assert salida["column_types"] is None
    assert salida["sample"] is None


@respx.mock
async def test_si_el_dataset_no_existe_si_es_error():
    respx.get(f"{API}/dataset/999").mock(return_value=httpx.Response(404, json={"detail": "no existe"}))
    salida = await dashai_describe_dataset(DescribirDataset(dataset_id=999))
    assert salida.startswith("Error:")


# --- Superficie del servidor -------------------------------------------------

async def test_ninguna_herramienta_borra():
    """Invariante de diseño: dashAI no tiene autenticación ni deshacer."""
    from dashai_mcp.server import mcp

    nombres = [t.name for t in await mcp.list_tools()]
    assert nombres, "el servidor debe exponer herramientas"
    for prohibido in ("delete", "remove", "drop", "borrar"):
        assert not any(prohibido in n for n in nombres), f"apareció una herramienta con '{prohibido}'"


async def test_las_herramientas_declaran_anotaciones():
    from dashai_mcp.server import mcp

    for tool in await mcp.list_tools():
        assert tool.annotations is not None, f"{tool.name} sin anotaciones"
        assert tool.description, f"{tool.name} sin descripción"


# --- Regresiones halladas corriendo contra dashAI 0.9.7.post1 de verdad -------
# Los cuatro bugs de abajo pasaron los tests con dobles y fallaron contra la
# instancia real. Cada uno tiene su test para que no vuelvan.

@respx.mock
async def test_select_types_va_como_parametros_repetidos():
    """La doc de dashAI muestra ?select_types=["Model","Metric"] y NO funciona.

    El backend lo recibe como UN tipo llamado literalmente '["Model","Metric"]'
    y responde 422. Deben ser parámetros repetidos.
    """
    ruta = respx.get(f"{API}/component/").mock(return_value=httpx.Response(200, json=[]))
    await dashai_list_components(ListarComponentes(types=["Task", "Model"]))

    consulta = str(ruta.calls[0].request.url.params)
    assert "select_types=Task" in consulta and "select_types=Model" in consulta
    assert "%5B" not in consulta, "se está mandando un array JSON, que dashAI rechaza"


def test_acepta_los_13_tipos_del_registro():
    """El registro real tiene 13 tipos, no 4: validar de menos rechazaba entradas buenas."""
    from dashai_mcp.server import TIPOS_COMPONENTE

    for t in ("Converter", "GlobalExplainer", "Explorer", "DatasetSource", "GenerativeModel"):
        assert t in TIPOS_COMPONENTE
        ListarComponentes(types=[t])  # no debe lanzar


@respx.mock
async def test_el_job_se_encola_como_form_data():
    """POST /job/ no acepta JSON: espera form data con kwargs como string JSON.

    Su propio openapi.json no declara requestBody para esta ruta porque el
    endpoint parsea `request` a mano. Mandar json= devuelve
    422 "Missing job_type or kwargs".
    """
    respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(201, json={"id": 1}))
    respx.post(f"{API}/run/").mock(return_value=httpx.Response(201, json={"id": 1}))
    job = respx.post(f"{API}/job/").mock(return_value=httpx.Response(201, json={"id": "h1"}))

    await dashai_train_model(
        EntrenarModelo(
            dataset_id=1, task_name="TabularClassificationTask", model_name="RandomForestClassifier",
            input_columns=["a"], output_columns=["b"], metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )

    peticion = job.calls[0].request
    assert "application/x-www-form-urlencoded" in peticion.headers["content-type"]
    cuerpo = dict(p.split("=", 1) for p in peticion.content.decode().split("&"))
    assert cuerpo["job_type"] == "ModelJob"
    # kwargs viaja serializado, no como campos sueltos
    assert "run_id" in cuerpo["kwargs"]


@respx.mock
async def test_get_run_poda_los_indices_y_traduce_el_estado():
    """split_indexes son ~59 KB en un dataset de 10k filas: el 99% de la respuesta."""
    indices = {"train_indexes": list(range(7000)), "test_indexes": list(range(1500)),
               "val_indexes": list(range(1500))}
    respx.get(f"{API}/run/9").mock(
        return_value=httpx.Response(200, json={
            "id": 9, "status": 3, "split_indexes": json.dumps(indices),
            "test_metrics": {"Accuracy": 0.91},
        })
    )

    salida = await dashai_get_run(ObtenerRun(run_id=9))
    datos = json.loads(salida)

    assert "split_indexes" not in datos, "los índices crudos no deben llegar al agente"
    assert datos["split_sizes"] == {"train_indexes": 7000, "test_indexes": 1500, "val_indexes": 1500}
    assert datos["status_name"] == "FINISHED"
    assert datos["status"] == 3, "el valor original se conserva"
    assert datos["test_metrics"] == {"Accuracy": 0.91}
    assert len(salida) < 1500, f"la respuesta sigue siendo enorme: {len(salida)} chars"


@respx.mock
async def test_queue_empty_se_desanida():
    """/job/is_empty devuelve {"is_empty": bool}, no un bool pelado."""
    respx.get(f"{API}/dataset/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/run/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/job/is_empty").mock(return_value=httpx.Response(200, json={"is_empty": True}))

    datos = json.loads(await dashai_server_info(SinArgumentos()))
    assert datos["queue_empty"] is True
