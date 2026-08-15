#!/usr/bin/env python3
"""Servidor MCP para dashAI — workbench de ML open source (MIT, DashAISoftware).

dashAI expone 142 endpoints REST. Este servidor NO los envuelve todos: envolver
una API endpoint por endpoint produce un catálogo que el modelo no sabe navegar
y en el que elige peor. Acá hay nueve herramientas que cubren el recorrido real
de trabajo — mirar datos, ver qué modelos hay, entrenar, seguir el job, leer
resultados, predecir.

La pieza central es `dashai_train_model`. En la API cruda, entrenar son TRES
llamadas encadenadas (model-session → run → job) con campos que la interfaz
gráfica rellena sola y que nadie documenta. Acá es una sola llamada.

Qué NO hay, a propósito
-----------------------
Ninguna herramienta borra nada. dashAI no tiene autenticación ni deshacer, y un
`DELETE /dataset/{id}` disparado por un modelo que malinterpretó una frase es
irreversible. Borrar se hace desde la interfaz, mirando lo que se borra.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__, client
from .client import DashAIError
from .config import base_url

mcp = MCPServer(name="dashai_mcp", version=__version__)

# Tipos de componente que dashAI registra y que sirven para configurar un entrenamiento.
# Los 13 tipos del registro, verificados contra una instancia real de dashAI
# 0.9.7 (el mensaje de error del propio backend los enumera).
# RunStatus llega como entero. Nombres tomados de
# DashAI/back/core/enums/status.py (dashAI 0.9.7.post1).
ESTADO_RUN = {
    0: "NOT_STARTED",
    1: "DELIVERED",
    2: "STARTED",
    3: "FINISHED",
    4: "ERROR",
}

TIPOS_COMPONENTE = (
    "Task", "GenerativeTask", "Model", "GenerativeModel", "DataLoader",
    "DatasetSource", "Metric", "Optimizer", "Job", "LocalExplainer",
    "GlobalExplainer", "Explorer", "Converter",
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades compartidas
# ─────────────────────────────────────────────────────────────────────────────

def _json(dato: Any) -> str:
    return json.dumps(dato, indent=2, ensure_ascii=False, default=str)


def _error(e: DashAIError) -> str:
    return f"Error: {e}"


async def _encolar_job(job_type: str, kwargs: Dict[str, Any]) -> Any:
    """Encola un job en dashAI.

    OJO: `POST /job/` NO acepta JSON. Espera **form data** con `job_type` y con
    `kwargs` serializado como string JSON. Verificado contra dashAI 0.9.7.post1:
    el endpoint parsea `request` a mano, y por eso su propio `openapi.json` no
    declara ningún requestBody para esta ruta. Mandar `json=` devuelve
    422 "Missing job_type or kwargs".
    """
    return await client.post(
        "job/",
        data={"job_type": job_type, "kwargs": json.dumps(kwargs)},
    )


def _resumen_dataset(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deja solo los campos que sirven para decidir, no el registro completo."""
    return {
        "id": d.get("id"),
        "name": d.get("name"),
        "created": d.get("created"),
        "status": d.get("status"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de entrada
# ─────────────────────────────────────────────────────────────────────────────

class SinArgumentos(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListarDatasets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, description="Máximo de datasets a devolver", ge=1, le=200)


class DescribirDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(..., description="Id del dataset, tal como lo lista dashai_list_datasets", ge=1)
    include_sample: bool = Field(
        default=True,
        description="Incluir ~10 filas de muestra. Ponlo en false si el dataset tiene columnas muy anchas.",
    )


class ListarComponentes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Filtra por tipo de componente. Valores válidos: 'Model', 'Metric', 'Task', "
            "'Optimizer'. Sin filtro devuelve todo el registro, que es largo."
        ),
    )

    @field_validator("types")
    @classmethod
    def validar_tipos(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        desconocidos = [t for t in v if t not in TIPOS_COMPONENTE]
        if desconocidos:
            raise ValueError(
                f"Tipos no reconocidos: {desconocidos}. Usa alguno de {list(TIPOS_COMPONENTE)}."
            )
        return v


class EntrenarModelo(BaseModel):
    """Todo lo necesario para el recorrido completo model-session → run → job."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: int = Field(..., description="Id del dataset a usar", ge=1)
    task_name: str = Field(
        ...,
        description=(
            "Nombre exacto de la tarea, de dashai_list_components(types=['Task']). "
            "Ej: 'TabularClassificationTask', 'TextClassificationTask'."
        ),
        min_length=1,
    )
    model_name: str = Field(
        ...,
        description=(
            "Nombre exacto del modelo, de dashai_list_components(types=['Model']). "
            "Ej: 'RandomForestClassifier', 'DistilBertTransformer'. Sensible a mayúsculas."
        ),
        min_length=1,
    )
    input_columns: List[str] = Field(..., description="Columnas de entrada (features)", min_length=1)
    output_columns: List[str] = Field(..., description="Columnas de salida (target)", min_length=1)
    metrics: List[str] = Field(
        ...,
        description=(
            "Métricas a calcular, de dashai_list_components(types=['Metric']). "
            "Ej: ['Accuracy', 'F1']. Se aplican a train, validación y test."
        ),
        min_length=1,
    )
    goal_metric: str = Field(
        ...,
        description="Métrica que se optimiza y por la que se compara. Debe estar dentro de `metrics`.",
        min_length=1,
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hiperparámetros del modelo. Vacío usa los de dashAI. Ej: {'n_estimators': 100}",
    )
    splits: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15},
        description="Proporciones de división. Las tres claves deben sumar 1.0.",
    )
    optimizer_name: str = Field(
        default="",
        description="Optimizador de hiperparámetros (opcional). Vacío entrena una sola vez.",
    )
    optimizer_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros del optimizador")
    run_name: Optional[str] = Field(default=None, description="Nombre para identificar la corrida", max_length=200)

    @field_validator("splits")
    @classmethod
    def validar_splits(cls, v: Dict[str, float]) -> Dict[str, float]:
        faltan = {"train", "validation", "test"} - set(v)
        if faltan:
            raise ValueError(f"A splits le faltan las claves {sorted(faltan)}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Las proporciones de splits suman {total}, deben sumar 1.0")
        return v


class EstadoJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Id de job devuelto por dashai_train_model o dashai_predict", min_length=1)


class ListarRuns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_session_id: Optional[int] = Field(default=None, description="Filtra por sesión de modelo (experimento)", ge=1)
    limit: int = Field(default=50, description="Máximo de corridas a devolver", ge=1, le=200)


class ObtenerRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(..., description="Id de la corrida", ge=1)


class Predecir(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(..., description="Id de una corrida ya terminada (status FINISHED)", ge=1)


# ─────────────────────────────────────────────────────────────────────────────
# Herramientas
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="dashai_server_info",
    annotations=ToolAnnotations(title="Estado del servidor dashAI", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_server_info(params: SinArgumentos) -> str:
    """Comprueba que dashAI esté corriendo y resume qué hay cargado.

    Llama a esto PRIMERO cuando algo falla o cuando no sabes si el backend está
    levantado: distingue "dashAI está apagado" de "el id no existe", que son dos
    problemas con soluciones distintas.

    Args:
        params (SinArgumentos): sin parámetros.

    Returns:
        str: JSON con el siguiente esquema:
        {
            "base_url": str,      # a qué instancia se está apuntando
            "reachable": bool,    # si respondió
            "datasets": int,      # cantidad de datasets cargados
            "runs": int,          # cantidad de corridas registradas
            "queue_empty": bool   # si la cola de jobs está vacía
        }
        Ante fallo: "Error: <qué pasó y qué hacer>".
    """
    try:
        datasets = await client.get("dataset/")
        runs = await client.get("run/")
        cola = await client.get("job/is_empty")
    except DashAIError as e:
        return _error(e)

    return _json(
        {
            "base_url": base_url(),
            "reachable": True,
            "datasets": len(datasets or []),
            "runs": len(runs or []),
            "queue_empty": cola.get("is_empty") if isinstance(cola, dict) else cola,
        }
    )


@mcp.tool(
    name="dashai_list_datasets",
    annotations=ToolAnnotations(title="Listar datasets", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_datasets(params: ListarDatasets) -> str:
    """Lista los datasets cargados en dashAI.

    Devuelve solo id, nombre, fecha y estado — lo justo para elegir uno. Para ver
    columnas y tipos usa dashai_describe_dataset con el id.

    Args:
        params (ListarDatasets): contiene:
            - limit (int): máximo a devolver, 1-200 (default 50)

    Returns:
        str: JSON {"count": int, "datasets": [{"id", "name", "created", "status"}]}
        Si no hay ninguno: mensaje indicando cómo cargar datos desde la interfaz.
    """
    try:
        datos = await client.get("dataset/")
    except DashAIError as e:
        return _error(e)

    datasets = [_resumen_dataset(d) for d in (datos or [])][: params.limit]
    if not datasets:
        return (
            "No hay datasets cargados en dashAI. Súbelos desde la interfaz gráfica "
            f"({base_url()}) — este servidor no carga archivos a propósito."
        )
    return _json({"count": len(datasets), "datasets": datasets})


@mcp.tool(
    name="dashai_describe_dataset",
    annotations=ToolAnnotations(title="Describir un dataset", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_describe_dataset(params: DescribirDataset) -> str:
    """Devuelve todo lo necesario para configurar un entrenamiento sobre un dataset.

    Junta en una sola llamada lo que en la API cruda son cuatro (`/{id}`, `/info`,
    `/types` y `/sample`), porque para decidir qué columnas son entrada y cuál es
    salida hace falta verlas juntas.

    Args:
        params (DescribirDataset): contiene:
            - dataset_id (int): id del dataset
            - include_sample (bool): incluir filas de muestra (default True)

    Returns:
        str: JSON {"dataset": {...}, "info": {...}, "column_types": {...}, "sample": [...]}
        Si una parte no está disponible, viene como null en vez de fallar entera.
    """
    try:
        dataset = await client.get(f"dataset/{params.dataset_id}")
    except DashAIError as e:
        return _error(e)

    # Las secundarias no deben tumbar la respuesta: un dataset recién creado
    # todavía no tiene tipos ni muestra.
    async def _opcional(path: str) -> Any:
        try:
            return await client.get(path)
        except DashAIError:
            return None

    info = await _opcional(f"dataset/{params.dataset_id}/info")
    tipos = await _opcional(f"dataset/{params.dataset_id}/types")
    muestra = await _opcional(f"dataset/{params.dataset_id}/sample") if params.include_sample else None

    return _json({"dataset": dataset, "info": info, "column_types": tipos, "sample": muestra})


@mcp.tool(
    name="dashai_list_components",
    annotations=ToolAnnotations(title="Listar modelos, métricas y tareas disponibles", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_components(params: ListarComponentes) -> str:
    """Lista los componentes registrados: modelos, métricas, tareas y optimizadores.

    Úsalo SIEMPRE antes de dashai_train_model. Los nombres que espera dashAI son
    exactos y sensibles a mayúsculas, y el catálogo cambia según los plugins que
    tenga instalados esa instancia — no se pueden adivinar.

    Args:
        params (ListarComponentes): contiene:
            - types (Optional[List[str]]): filtro por 'Model', 'Metric', 'Task', 'Optimizer'

    Returns:
        str: JSON {"count": int, "components": [{"name": str, "type": str, "schema": {...}}]}
        El campo `schema` describe los hiperparámetros aceptados por ese componente.
    """
    # httpx serializa una lista como parámetros repetidos
    # (?select_types=Task&select_types=Model), que es lo que el backend espera.
    # OJO: la doc de dashAI muestra ?select_types=["Model","Metric"] y NO funciona.
    query = {"select_types": params.types} if params.types else {}

    try:
        datos = await client.get("component/", params=query)
    except DashAIError as e:
        return _error(e)

    componentes = datos or []
    return _json({"count": len(componentes), "components": componentes})


@mcp.tool(
    name="dashai_train_model",
    annotations=ToolAnnotations(title="Entrenar un modelo", read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True),)
async def dashai_train_model(params: EntrenarModelo) -> str:
    """Entrena un modelo sobre un dataset y devuelve el id del job encolado.

    NO espera a que termine. Entrenar puede tomar minutos u horas, así que dashAI
    lo encola y esta herramienta devuelve de inmediato; el avance se consulta con
    dashai_job_status.

    Colapsa las tres llamadas que exige la API cruda:
      1. POST /model-session/  crea el experimento (dataset, tarea, columnas, métricas)
      2. POST /run/            crea la corrida (modelo, hiperparámetros)
      3. POST /job/            encola el ModelJob

    Args:
        params (EntrenarModelo): contiene:
            - dataset_id (int), task_name (str), model_name (str)
            - input_columns / output_columns (List[str])
            - metrics (List[str]), goal_metric (str)
            - parameters (Dict): hiperparámetros del modelo
            - splits (Dict[str, float]): proporciones que suman 1.0
            - optimizer_name (str), optimizer_parameters (Dict)
            - run_name (Optional[str])

    Returns:
        str: JSON {"job_id": str, "run_id": int, "model_session_id": int, "status": "enqueued", "next_step": str}
        Ante fallo: "Error: ..." indicando qué parámetro rechazó dashAI.

    Examples:
        - "Entrena un random forest sobre el dataset 3 prediciendo 'species'"
        - No lo uses para ver resultados: eso es dashai_get_run, ya con el run_id.
    """
    if params.goal_metric not in params.metrics:
        return (
            f"Error: goal_metric '{params.goal_metric}' no está en metrics {params.metrics}. "
            "La métrica objetivo tiene que ser una de las que se calculan."
        )

    nombre = params.run_name or f"{params.model_name} sobre dataset {params.dataset_id}"

    try:
        # 1. Sesión de modelo (experimento).
        # `splits` viaja como string JSON: dashAI lo declara como str, no como dict,
        # aunque su documentación lo muestre como objeto.
        sesion = await client.post(
            "model-session/",
            json={
                "dataset_id": params.dataset_id,
                "task_name": params.task_name,
                "name": nombre,
                "input_columns": params.input_columns,
                "output_columns": params.output_columns,
                "train_metrics": params.metrics,
                "validation_metrics": params.metrics,
                "test_metrics": params.metrics,
                "splits": json.dumps(params.splits),
            },
        )
        session_id = sesion["id"]

        # 2. Corrida. Los plot_*_path son obligatorios en el esquema pero los
        # rellena el job al optimizar; van vacíos.
        run = await client.post(
            "run/",
            json={
                "model_session_id": session_id,
                "model_name": params.model_name,
                "name": nombre,
                "parameters": params.parameters,
                "optimizer_name": params.optimizer_name,
                "optimizer_parameters": params.optimizer_parameters,
                "plot_history_path": "",
                "plot_slice_path": "",
                "plot_contour_path": "",
                "plot_importance_path": "",
                "goal_metric": params.goal_metric,
            },
        )
        run_id = run["id"]

        # 3. Encolar.
        job = await _encolar_job("ModelJob", {"run_id": run_id})

    except DashAIError as e:
        return _error(e)
    except (KeyError, TypeError) as e:
        return f"Error: dashAI respondió con una forma inesperada ({e}). Revisa su versión."

    job_id = job.get("id") if isinstance(job, dict) else job

    return _json(
        {
            "job_id": str(job_id),
            "run_id": run_id,
            "model_session_id": session_id,
            "status": "enqueued",
            "next_step": (
                f"Consulta el avance con dashai_job_status(job_id='{job_id}'). "
                f"Cuando termine, los resultados están en dashai_get_run(run_id={run_id})."
            ),
        }
    )


@mcp.tool(
    name="dashai_job_status",
    annotations=ToolAnnotations(title="Estado de un job", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_job_status(params: EstadoJob) -> str:
    """Consulta el estado de un job encolado (entrenamiento, predicción, explicación).

    Estados de dashAI: `not_started` (en cola), `started` (corriendo), `finished`
    (listo) y `error` (falló). Distinguir `started` de `error` importa: el primero
    se espera, el segundo no mejora por reintentar la consulta.

    Args:
        params (EstadoJob): contiene:
            - job_id (str): id devuelto al encolar

    Returns:
        str: JSON {"job_id": str, "status": str, "finished": bool, "failed": bool, "raw": {...}}
    """
    try:
        datos = await client.get(f"job/status/{params.job_id}")
    except DashAIError as e:
        return _error(e)

    estado = datos.get("status") if isinstance(datos, dict) else str(datos)
    return _json(
        {
            "job_id": params.job_id,
            "status": estado,
            "finished": estado == "finished",
            "failed": estado == "error",
            "raw": datos,
        }
    )


@mcp.tool(
    name="dashai_list_runs",
    annotations=ToolAnnotations(title="Listar corridas de entrenamiento", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_runs(params: ListarRuns) -> str:
    """Lista las corridas de entrenamiento registradas, con su estado.

    Sirve para comparar modelos entrenados sobre el mismo experimento.

    Args:
        params (ListarRuns): contiene:
            - model_session_id (Optional[int]): filtra por experimento
            - limit (int): máximo a devolver, 1-200 (default 50)

    Returns:
        str: JSON {"count": int, "runs": [{"id", "name", "model_name", "status", "goal_metric"}]}
    """
    query = {}
    if params.model_session_id is not None:
        query["model_session_id"] = params.model_session_id

    try:
        datos = await client.get("run/", params=query)
    except DashAIError as e:
        return _error(e)

    runs = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "model_name": r.get("model_name"),
            "status": r.get("status"),
            "goal_metric": r.get("goal_metric"),
        }
        for r in (datos or [])
    ][: params.limit]

    if not runs:
        return "No hay corridas registradas todavía. Entrena una con dashai_train_model."
    return _json({"count": len(runs), "runs": runs})


@mcp.tool(
    name="dashai_get_run",
    annotations=ToolAnnotations(title="Resultados de una corrida", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_get_run(params: ObtenerRun) -> str:
    """Devuelve la configuración y las métricas de una corrida de entrenamiento.

    Es donde se leen los resultados una vez que dashai_job_status dice `finished`.
    Si la corrida no terminó, las métricas vendrán vacías — eso no es un error.

    Args:
        params (ObtenerRun): contiene:
            - run_id (int): id de la corrida

    Returns:
        str: JSON con la corrida completa: parámetros del modelo, estado y métricas
        por split (train / validation / test).
    """
    try:
        run = await client.get(f"run/{params.run_id}")
    except DashAIError as e:
        return _error(e)

    # `split_indexes` trae la lista completa de índices por partición: en un
    # dataset de 10.000 filas son ~59 KB, el 99% de la respuesta, y no le sirve
    # de nada a quien lee. Se reemplaza por el conteo.
    indices = run.pop("split_indexes", None)
    if indices:
        try:
            cargado = json.loads(indices) if isinstance(indices, str) else indices
            run["split_sizes"] = {k: len(v) for k, v in cargado.items()}
        except (ValueError, TypeError):
            run["split_sizes"] = None

    # El estado viaja como entero; se agrega su nombre sin quitar el original.
    if isinstance(run.get("status"), int):
        run["status_name"] = ESTADO_RUN.get(run["status"], "DESCONOCIDO")

    return _json(run)


@mcp.tool(
    name="dashai_predict",
    annotations=ToolAnnotations(title="Predecir con un modelo entrenado", read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True),)
async def dashai_predict(params: Predecir) -> str:
    """Encola una predicción usando el modelo de una corrida ya terminada.

    Igual que entrenar, es asíncrono: devuelve un job_id y el resultado se sigue
    con dashai_job_status. La corrida debe estar en estado FINISHED; si no, dashAI
    rechaza la petición.

    Args:
        params (Predecir): contiene:
            - run_id (int): id de una corrida terminada

    Returns:
        str: JSON {"job_id": str, "run_id": int, "status": "enqueued", "next_step": str}
    """
    try:
        job = await _encolar_job("PredictJob", {"run_id": params.run_id})
    except DashAIError as e:
        return _error(e)

    job_id = job.get("id") if isinstance(job, dict) else job
    return _json(
        {
            "job_id": str(job_id),
            "run_id": params.run_id,
            "status": "enqueued",
            "next_step": f"Sigue el avance con dashai_job_status(job_id='{job_id}').",
        }
    )


def main() -> None:
    """Punto de entrada. Transporte stdio: dashAI es local y sin autenticación."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
