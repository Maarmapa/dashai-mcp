#!/usr/bin/env python3
"""MCP server for dashAI — open source ML workbench (MIT, DashAISoftware).

dashAI exposes 142 REST endpoints. This server does NOT wrap them all: wrapping
an API endpoint by endpoint produces a catalogue the model cannot navigate and
in which it chooses worse. There are nine tools here, covering the actual
working path — look at data, see which models exist, train, follow the job,
read results, predict.

The centrepiece is `dashai_train_model`. In the raw API, training is THREE
chained calls (model-session → run → job) with fields the GUI fills in on its
own and that nobody documents. Here it is a single call.

What is deliberately absent
---------------------------
No tool deletes anything. dashAI has neither authentication nor undo, and a
`DELETE /dataset/{id}` fired by a model that misread a sentence is
irreversible. Deleting is done from the GUI, looking at what is being deleted.
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

# RunStatus arrives as an integer. Names taken from
# DashAI/back/core/enums/status.py (dashAI 0.9.7.post1).
RUN_STATUS = {
    0: "NOT_STARTED",
    1: "DELIVERED",
    2: "STARTED",
    3: "FINISHED",
    4: "ERROR",
}

# The 13 types in dashAI's component registry, verified against a real
# 0.9.7 instance (the backend's own error message enumerates them).
COMPONENT_TYPES = (
    "Task", "GenerativeTask", "Model", "GenerativeModel", "DataLoader",
    "DatasetSource", "Metric", "Optimizer", "Job", "LocalExplainer",
    "GlobalExplainer", "Explorer", "Converter",
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _error(e: DashAIError) -> str:
    return f"Error: {e}"


async def _enqueue_job(job_type: str, kwargs: Dict[str, Any]) -> Any:
    """Enqueues a job in dashAI.

    NOTE: `POST /job/` does NOT accept JSON. It expects **form data** with
    `job_type` and with `kwargs` serialized as a JSON string. Verified against
    dashAI 0.9.7.post1: the endpoint parses `request` by hand, which is why its
    own `openapi.json` declares no requestBody for this route. Sending `json=`
    returns 422 "Missing job_type or kwargs".
    """
    return await client.post(
        "job/",
        data={"job_type": job_type, "kwargs": json.dumps(kwargs)},
    )


def _summarize_dataset(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps only the fields useful for deciding, not the whole record."""
    return {
        "id": d.get("id"),
        "name": d.get("name"),
        "created": d.get("created"),
        "status": d.get("status"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input models
# ─────────────────────────────────────────────────────────────────────────────

class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListDatasets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, description="Maximum number of datasets to return", ge=1, le=200)


class DescribeDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(..., description="Dataset id, as listed by dashai_list_datasets", ge=1)
    include_sample: bool = Field(
        default=True,
        description="Include ~10 sample rows. Set to false if the dataset has very wide columns.",
    )


class ListComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Filter by component type. Valid values: 'Model', 'Metric', 'Task', "
            "'Optimizer'. With no filter it returns the whole registry, which is long."
        ),
    )

    @field_validator("types")
    @classmethod
    def validate_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        unknown = [t for t in v if t not in COMPONENT_TYPES]
        if unknown:
            raise ValueError(
                f"Unrecognized types: {unknown}. Use one of {list(COMPONENT_TYPES)}."
            )
        return v


class TrainModel(BaseModel):
    """Everything needed for the full model-session → run → job path."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: int = Field(..., description="Id of the dataset to use", ge=1)
    task_name: str = Field(
        ...,
        description=(
            "Exact task name, from dashai_list_components(types=['Task']). "
            "E.g. 'TabularClassificationTask', 'TextClassificationTask'."
        ),
        min_length=1,
    )
    model_name: str = Field(
        ...,
        description=(
            "Exact model name, from dashai_list_components(types=['Model']). "
            "E.g. 'RandomForestClassifier', 'DistilBertTransformer'. Case-sensitive."
        ),
        min_length=1,
    )
    input_columns: List[str] = Field(..., description="Input columns (features)", min_length=1)
    output_columns: List[str] = Field(..., description="Output columns (target)", min_length=1)
    metrics: List[str] = Field(
        ...,
        description=(
            "Metrics to compute, from dashai_list_components(types=['Metric']). "
            "E.g. ['Accuracy', 'F1']. Applied to train, validation and test."
        ),
        min_length=1,
    )
    goal_metric: str = Field(
        ...,
        description="Metric that is optimized and compared against. Must be one of `metrics`.",
        min_length=1,
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters. Empty uses dashAI's defaults. E.g. {'n_estimators': 100}",
    )
    splits: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15},
        description="Split proportions. The three keys must add up to 1.0.",
    )
    optimizer_name: str = Field(
        default="",
        description="Hyperparameter optimizer (optional). Empty trains once.",
    )
    optimizer_parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimizer parameters")
    run_name: Optional[str] = Field(default=None, description="Name to identify the run", max_length=200)

    @field_validator("splits")
    @classmethod
    def validate_splits(cls, v: Dict[str, float]) -> Dict[str, float]:
        missing = {"train", "validation", "test"} - set(v)
        if missing:
            raise ValueError(f"splits is missing the keys {sorted(missing)}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"The split proportions add up to {total}, they must add up to 1.0")
        return v


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Job id returned by dashai_train_model or dashai_predict", min_length=1)


class ListRuns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_session_id: Optional[int] = Field(default=None, description="Filter by model session (experiment)", ge=1)
    limit: int = Field(default=50, description="Maximum number of runs to return", ge=1, le=200)


class GetRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(..., description="Run id", ge=1)


class Predict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(..., description="Id of an already finished run (status FINISHED)", ge=1)
    dataset_id: Optional[int] = Field(
        default=None,
        description=(
            "Dataset to predict on. If omitted, the run's training dataset is used. "
            "Must have the same input columns as the model; pick it from dashai_list_datasets."
        ),
        ge=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="dashai_server_info",
    annotations=ToolAnnotations(title="dashAI server status", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_server_info(params: NoArgs) -> str:
    """Checks that dashAI is running and summarizes what is loaded.

    Call this FIRST when something fails or when you do not know whether the
    backend is up: it tells "dashAI is down" apart from "that id does not
    exist", which are two problems with different fixes.

    Args:
        params (NoArgs): no parameters.

    Returns:
        str: JSON with the following schema:
        {
            "base_url": str,      # which instance is being targeted
            "reachable": bool,    # whether it responded
            "datasets": int,      # number of loaded datasets
            "runs": int,          # number of recorded runs
            "queue_empty": bool   # whether the job queue is empty
        }
        On failure: "Error: <what happened and what to do>".
    """
    try:
        datasets = await client.get("dataset/")
        runs = await client.get("run/")
        queue = await client.get("job/is_empty")
    except DashAIError as e:
        return _error(e)

    return _json(
        {
            "base_url": base_url(),
            "reachable": True,
            "datasets": len(datasets or []),
            "runs": len(runs or []),
            "queue_empty": queue.get("is_empty") if isinstance(queue, dict) else queue,
        }
    )


@mcp.tool(
    name="dashai_list_datasets",
    annotations=ToolAnnotations(title="List datasets", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_datasets(params: ListDatasets) -> str:
    """Lists the datasets loaded in dashAI.

    Returns only id, name, date and status — just enough to pick one. To see
    columns and types use dashai_describe_dataset with the id.

    Args:
        params (ListDatasets): contains:
            - limit (int): maximum to return, 1-200 (default 50)

    Returns:
        str: JSON {"count": int, "datasets": [{"id", "name", "created", "status"}]}
        If there are none: a message explaining how to load data from the GUI.
    """
    try:
        data = await client.get("dataset/")
    except DashAIError as e:
        return _error(e)

    datasets = [_summarize_dataset(d) for d in (data or [])][: params.limit]
    if not datasets:
        return (
            "There are no datasets loaded in dashAI. Upload them from the GUI "
            f"({base_url()}) — this server deliberately does not upload files."
        )
    return _json({"count": len(datasets), "datasets": datasets})


@mcp.tool(
    name="dashai_describe_dataset",
    annotations=ToolAnnotations(title="Describe a dataset", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_describe_dataset(params: DescribeDataset) -> str:
    """Returns everything needed to configure a training run over a dataset.

    Gathers into a single call what the raw API splits into four (`/{id}`,
    `/info`, `/types` and `/sample`), because deciding which columns are input
    and which is output requires seeing them together.

    Args:
        params (DescribeDataset): contains:
            - dataset_id (int): dataset id
            - include_sample (bool): include sample rows (default True)

    Returns:
        str: JSON {"dataset": {...}, "info": {...}, "column_types": {...}, "sample": [...]}
        If one part is unavailable it comes back as null instead of failing whole.
    """
    try:
        dataset = await client.get(f"dataset/{params.dataset_id}")
    except DashAIError as e:
        return _error(e)

    # The secondary calls must not take down the response: a freshly created
    # dataset has neither types nor a sample yet.
    async def _optional(path: str) -> Any:
        try:
            return await client.get(path)
        except DashAIError:
            return None

    info = await _optional(f"dataset/{params.dataset_id}/info")
    types = await _optional(f"dataset/{params.dataset_id}/types")
    sample = await _optional(f"dataset/{params.dataset_id}/sample") if params.include_sample else None

    return _json({"dataset": dataset, "info": info, "column_types": types, "sample": sample})


@mcp.tool(
    name="dashai_list_components",
    annotations=ToolAnnotations(title="List available models, metrics and tasks", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_components(params: ListComponents) -> str:
    """Lists the registered components: models, metrics, tasks and optimizers.

    ALWAYS use this before dashai_train_model. The names dashAI expects are
    exact and case-sensitive, and the catalogue changes with the plugins that
    instance has installed — they cannot be guessed.

    Args:
        params (ListComponents): contains:
            - types (Optional[List[str]]): filter by 'Model', 'Metric', 'Task', 'Optimizer'

    Returns:
        str: JSON {"count": int, "components": [{"name": str, "type": str, "schema": {...}}]}
        The `schema` field describes the hyperparameters that component accepts.
    """
    # httpx serializes a list as repeated parameters
    # (?select_types=Task&select_types=Model), which is what the backend expects.
    # NOTE: dashAI's docs show ?select_types=["Model","Metric"] and it does NOT work.
    query = {"select_types": params.types} if params.types else {}

    try:
        data = await client.get("component/", params=query)
    except DashAIError as e:
        return _error(e)

    components = data or []
    return _json({"count": len(components), "components": components})


@mcp.tool(
    name="dashai_train_model",
    annotations=ToolAnnotations(title="Train a model", read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True),)
async def dashai_train_model(params: TrainModel) -> str:
    """Trains a model on a dataset and returns the id of the enqueued job.

    It does NOT wait for it to finish. Training can take minutes or hours, so
    dashAI enqueues it and this tool returns immediately; progress is polled
    with dashai_job_status.

    Collapses the three calls the raw API demands:
      1. POST /model-session/  creates the experiment (dataset, task, columns, metrics)
      2. POST /run/            creates the run (model, hyperparameters)
      3. POST /job/            enqueues the ModelJob

    Args:
        params (TrainModel): contains:
            - dataset_id (int), task_name (str), model_name (str)
            - input_columns / output_columns (List[str])
            - metrics (List[str]), goal_metric (str)
            - parameters (Dict): model hyperparameters
            - splits (Dict[str, float]): proportions adding up to 1.0
            - optimizer_name (str), optimizer_parameters (Dict)
            - run_name (Optional[str])

    Returns:
        str: JSON {"job_id": str, "run_id": int, "model_session_id": int, "status": "enqueued", "next_step": str}
        On failure: "Error: ..." stating which parameter dashAI rejected.

    Examples:
        - "Train a random forest on dataset 3 predicting 'species'"
        - Do not use it to read results: that is dashai_get_run, with the run_id.
    """
    if params.goal_metric not in params.metrics:
        return (
            f"Error: goal_metric '{params.goal_metric}' is not in metrics {params.metrics}. "
            "The goal metric has to be one of the metrics being computed."
        )

    name = params.run_name or f"{params.model_name} on dataset {params.dataset_id}"

    try:
        # 1. Model session (experiment).
        # `splits` travels as a JSON string: dashAI declares it as str, not dict,
        # even though its documentation shows it as an object.
        session = await client.post(
            "model-session/",
            json={
                "dataset_id": params.dataset_id,
                "task_name": params.task_name,
                "name": name,
                "input_columns": params.input_columns,
                "output_columns": params.output_columns,
                "train_metrics": params.metrics,
                "validation_metrics": params.metrics,
                "test_metrics": params.metrics,
                "splits": json.dumps(params.splits),
            },
        )
        session_id = session["id"]

        # 2. Run. The plot_*_path fields are required by the schema but the job
        # fills them in while optimizing; they go empty.
        run = await client.post(
            "run/",
            json={
                "model_session_id": session_id,
                "model_name": params.model_name,
                "name": name,
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

        # 3. Enqueue.
        job = await _enqueue_job("ModelJob", {"run_id": run_id})

    except DashAIError as e:
        return _error(e)
    except (KeyError, TypeError) as e:
        return f"Error: dashAI responded with an unexpected shape ({e}). Check its version."

    job_id = job.get("id") if isinstance(job, dict) else job

    return _json(
        {
            "job_id": str(job_id),
            "run_id": run_id,
            "model_session_id": session_id,
            "status": "enqueued",
            "next_step": (
                f"Poll progress with dashai_job_status(job_id='{job_id}'). "
                f"When it finishes, the results are in dashai_get_run(run_id={run_id})."
            ),
        }
    )


@mcp.tool(
    name="dashai_job_status",
    annotations=ToolAnnotations(title="Job status", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_job_status(params: JobStatus) -> str:
    """Polls the status of an enqueued job (training, prediction, explanation).

    dashAI's statuses: `not_started` (queued), `started` (running), `finished`
    (done) and `error` (failed). Telling `started` from `error` matters: the
    first is worth waiting on, the second does not improve by polling again.

    Args:
        params (JobStatus): contains:
            - job_id (str): id returned when enqueuing

    Returns:
        str: JSON {"job_id": str, "status": str, "finished": bool, "failed": bool, "raw": {...}}
    """
    try:
        data = await client.get(f"job/status/{params.job_id}")
    except DashAIError as e:
        return _error(e)

    status = data.get("status") if isinstance(data, dict) else str(data)
    return _json(
        {
            "job_id": params.job_id,
            "status": status,
            "finished": status == "finished",
            "failed": status == "error",
            "raw": data,
        }
    )


@mcp.tool(
    name="dashai_list_runs",
    annotations=ToolAnnotations(title="List training runs", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_list_runs(params: ListRuns) -> str:
    """Lists the recorded training runs, with their status.

    Useful for comparing models trained within the same experiment.

    Args:
        params (ListRuns): contains:
            - model_session_id (Optional[int]): filter by experiment
            - limit (int): maximum to return, 1-200 (default 50)

    Returns:
        str: JSON {"count": int, "runs": [{"id", "name", "model_name", "status", "goal_metric"}]}
    """
    query = {}
    if params.model_session_id is not None:
        query["model_session_id"] = params.model_session_id

    try:
        data = await client.get("run/", params=query)
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
        for r in (data or [])
    ][: params.limit]

    if not runs:
        return "There are no runs recorded yet. Train one with dashai_train_model."
    return _json({"count": len(runs), "runs": runs})


@mcp.tool(
    name="dashai_get_run",
    annotations=ToolAnnotations(title="Results of a run", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),)
async def dashai_get_run(params: GetRun) -> str:
    """Returns the configuration and metrics of a training run.

    This is where results are read once dashai_job_status says `finished`. If
    the run did not finish, the metrics will come back empty — that is not an
    error.

    Args:
        params (GetRun): contains:
            - run_id (int): run id

    Returns:
        str: JSON with the full run: model parameters, status and metrics per
        split (train / validation / test).
    """
    try:
        run = await client.get(f"run/{params.run_id}")
    except DashAIError as e:
        return _error(e)

    # `split_indexes` carries the full list of indices per split: on a
    # 10,000-row dataset that is ~59 KB, 99% of the response, and it is of no
    # use to whoever reads it. It is replaced by the counts.
    indexes = run.pop("split_indexes", None)
    if indexes:
        try:
            loaded = json.loads(indexes) if isinstance(indexes, str) else indexes
            run["split_sizes"] = {k: len(v) for k, v in loaded.items()}
        except (ValueError, TypeError):
            run["split_sizes"] = None

    # The status travels as an integer; its name is added without removing the original.
    if isinstance(run.get("status"), int):
        run["status_name"] = RUN_STATUS.get(run["status"], "UNKNOWN")

    return _json(run)


@mcp.tool(
    name="dashai_predict",
    annotations=ToolAnnotations(title="Predict with a trained model", read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True),)
async def dashai_predict(params: Predict) -> str:
    """Enqueues a prediction using the model of an already finished run.

    Like training, it is asynchronous: it returns a job_id and the result is
    followed with dashai_job_status.

    Collapses the two calls the GUI makes (read from
    DatasetPredictionPanel + createPrediction + enqueuePredictionJob, not
    from the docs):
      1. POST /predict/           creates the Prediction row ({run_id, dataset_id})
      2. POST /job/  PredictJob   enqueues with {prediction_id} — NOT run_id

    Passing only run_id to the job raises KeyError 'prediction_id' inside
    PredictJob.run. The Prediction row must exist first.

    Args:
        params (Predict): contains:
            - run_id (int): id of a finished run
            - dataset_id (Optional[int]): dataset to score; defaults to the
              run's training dataset (from its model session)

    Returns:
        str: JSON {"job_id", "run_id", "prediction_id", "dataset_id",
        "status": "enqueued", "next_step"}
    """
    try:
        run = await client.get(f"run/{params.run_id}")
    except DashAIError as e:
        return _error(e)

    status = run.get("status")
    if status != 3:  # FINISHED — DashAI/back/core/enums/status.py
        name = RUN_STATUS.get(status, str(status))
        return (
            f"Error: run {params.run_id} is {name}, not FINISHED. "
            "Wait until dashai_job_status reports finished, then retry."
        )

    dataset_id = params.dataset_id
    if dataset_id is None:
        session_id = run.get("model_session_id")
        if not session_id:
            return (
                f"Error: run {params.run_id} has no model_session_id, so the "
                "training dataset cannot be inferred. Pass dataset_id explicitly."
            )
        try:
            session = await client.get(f"model-session/{session_id}")
        except DashAIError as e:
            return _error(e)
        dataset_id = session.get("dataset_id")
        if not dataset_id:
            return (
                f"Error: model session {session_id} has no dataset_id. "
                "Pass dataset_id from dashai_list_datasets."
            )

    try:
        # 1. Prediction row. JSON body, both fields required by the job even
        # though PredictionCreationParams marks dataset_id optional: without
        # it PredictJob raises "Either dataset_id or manual_input_data must
        # be provided."
        prediction = await client.post(
            "predict/",
            json={"run_id": params.run_id, "dataset_id": dataset_id},
        )
        prediction_id = prediction["id"]

        # 2. Enqueue. Same form-data contract as ModelJob; kwargs carry
        # prediction_id, matching enqueuePredictionJob in the GUI.
        job = await _enqueue_job("PredictJob", {"prediction_id": prediction_id})
    except DashAIError as e:
        return _error(e)
    except (KeyError, TypeError) as e:
        return f"Error: dashAI responded with an unexpected shape ({e}). Check its version."

    job_id = job.get("id") if isinstance(job, dict) else job
    return _json(
        {
            "job_id": str(job_id),
            "run_id": params.run_id,
            "prediction_id": prediction_id,
            "dataset_id": dataset_id,
            "status": "enqueued",
            "next_step": (
                f"Poll progress with dashai_job_status(job_id='{job_id}'). "
                f"The prediction record is id={prediction_id}."
            ),
        }
    )


def main() -> None:
    """Entry point. stdio transport: dashAI is local and unauthenticated."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
