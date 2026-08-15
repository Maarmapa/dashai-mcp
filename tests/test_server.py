"""Deterministic tests for the MCP server.

They need no dashAI instance: the HTTP responses are stubbed with respx. What
is tested is the contract — which calls it makes, in what order, with what
body, and what it tells the agent when something goes wrong.
"""

import json

import httpx
import pytest
import respx

from dashai_mcp import client, config
from dashai_mcp.server import (
    DescribeDataset,
    GetRun,
    JobStatus,
    ListComponents,
    NoArgs,
    TrainModel,
    dashai_describe_dataset,
    dashai_get_run,
    dashai_job_status,
    dashai_list_components,
    dashai_server_info,
    dashai_train_model,
)

API = "http://localhost:8000/api/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from a known environment."""
    monkeypatch.delenv("DASHAI_BASE_URL", raising=False)
    monkeypatch.delenv("DASHAI_ALLOW_REMOTE", raising=False)


# --- The network lock --------------------------------------------------------

def test_local_url_passes():
    assert config.api_url("dataset/") == f"{API}/dataset/"


def test_remote_url_is_blocked(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "http://ml.example.com:8000")
    with pytest.raises(config.ConfigError) as e:
        config.api_url("dataset/")
    # The message has to explain WHY, not just refuse.
    assert "no authentication" in str(e.value)
    assert "DASHAI_ALLOW_REMOTE" in str(e.value)


def test_remote_url_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "http://ml.example.com:8000")
    monkeypatch.setenv("DASHAI_ALLOW_REMOTE", "1")
    assert config.api_url("dataset/").startswith("http://ml.example.com:8000/api/v1")


async def test_the_block_reaches_the_agent_as_an_error(monkeypatch):
    monkeypatch.setenv("DASHAI_BASE_URL", "https://ml.example.com")
    out = await dashai_server_info(NoArgs())
    assert out.startswith("Error:")
    assert "DASHAI_ALLOW_REMOTE" in out


# --- Actionable errors -------------------------------------------------------

@respx.mock
async def test_dashai_down_says_how_to_start_it():
    respx.get(f"{API}/dataset/").mock(side_effect=httpx.ConnectError("refused"))
    out = await dashai_server_info(NoArgs())
    assert "`dashai`" in out
    assert "DASHAI_BASE_URL" in out


@respx.mock
async def test_422_points_at_list_components():
    respx.post(f"{API}/model-session/").mock(
        return_value=httpx.Response(422, json={"detail": "Unknown task 'Clasificacion'"})
    )
    out = await dashai_train_model(
        TrainModel(
            dataset_id=1, task_name="Clasificacion", model_name="RandomForest",
            input_columns=["a"], output_columns=["b"], metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )
    assert "Unknown task" in out
    assert "dashai_list_components" in out


# --- Input validation --------------------------------------------------------

def test_splits_must_add_up_to_one():
    with pytest.raises(ValueError, match="add up to"):
        TrainModel(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy",
            splits={"train": 0.8, "validation": 0.3, "test": 0.1},
        )


def test_incomplete_splits_are_rejected():
    with pytest.raises(ValueError, match="missing the keys"):
        TrainModel(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy", splits={"train": 1.0},
        )


def test_invalid_component_type_is_rejected():
    with pytest.raises(ValueError, match="Unrecognized types"):
        ListComponents(types=["Modelo"])  # Spanish, does not exist


async def test_goal_metric_must_be_in_metrics():
    out = await dashai_train_model(
        TrainModel(
            dataset_id=1, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="F1",
        )
    )
    assert out.startswith("Error:")
    assert "goal_metric" in out


# --- The training path -------------------------------------------------------

@respx.mock
async def test_training_chains_the_three_calls():
    session = respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(201, json={"id": 7}))
    run = respx.post(f"{API}/run/").mock(return_value=httpx.Response(201, json={"id": 42}))
    job = respx.post(f"{API}/job/").mock(return_value=httpx.Response(201, json={"id": "huey-abc"}))

    out = json.loads(
        await dashai_train_model(
            TrainModel(
                dataset_id=3, task_name="TabularClassificationTask", model_name="RandomForestClassifier",
                input_columns=["length", "width"], output_columns=["species"],
                metrics=["Accuracy", "F1"], goal_metric="F1", parameters={"n_estimators": 100},
            )
        )
    )

    assert session.called and run.called and job.called
    assert out == {
        "job_id": "huey-abc", "run_id": 42, "model_session_id": 7,
        "status": "enqueued", "next_step": out["next_step"],
    }
    assert "dashai_job_status" in out["next_step"]

    # splits travels as a JSON STRING: dashAI declares it str, even though its docs draw an object.
    session_body = json.loads(session.calls[0].request.content)
    assert isinstance(session_body["splits"], str)
    assert json.loads(session_body["splits"])["train"] == 0.7
    # The metrics apply to all three splits.
    assert session_body["train_metrics"] == session_body["test_metrics"] == ["Accuracy", "F1"]

    # The run links to the session just created and sends the required plot_path fields.
    run_body = json.loads(run.calls[0].request.content)
    assert run_body["model_session_id"] == 7
    assert run_body["plot_history_path"] == ""
    assert run_body["parameters"] == {"n_estimators": 100}

    # The job points at the run, not the session — and travels as form data (see
    # test_the_job_is_enqueued_as_form_data for why).
    job_body = dict(pair.split("=", 1) for pair in job.calls[0].request.content.decode().split("&"))
    assert job_body["job_type"] == "ModelJob"
    assert "42" in job_body["kwargs"]


@respx.mock
async def test_if_the_session_fails_nothing_is_enqueued():
    respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(404, json={"detail": "Dataset does not exist"}))
    job_route = respx.post(f"{API}/job/")

    out = await dashai_train_model(
        TrainModel(
            dataset_id=999, task_name="T", model_name="M", input_columns=["a"], output_columns=["b"],
            metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )
    assert out.startswith("Error:")
    assert not job_route.called, "a job must not be enqueued if the experiment was not created"


# --- Job status --------------------------------------------------------------

@respx.mock
@pytest.mark.parametrize(
    "status,finished,failed",
    [("not_started", False, False), ("started", False, False), ("finished", True, False), ("error", False, True)],
)
async def test_job_statuses(status, finished, failed):
    respx.get(f"{API}/job/status/j1").mock(return_value=httpx.Response(200, json={"status": status}))
    out = json.loads(await dashai_job_status(JobStatus(job_id="j1")))
    assert out["status"] == status
    assert out["finished"] is finished
    assert out["failed"] is failed


# --- Degradation -------------------------------------------------------------

@respx.mock
async def test_describe_dataset_degrades_if_a_part_is_missing():
    """A freshly created dataset has no types and no sample: that is no reason to error."""
    respx.get(f"{API}/dataset/5").mock(return_value=httpx.Response(200, json={"id": 5, "name": "iris"}))
    respx.get(f"{API}/dataset/5/info").mock(return_value=httpx.Response(200, json={"rows": 150}))
    respx.get(f"{API}/dataset/5/types").mock(return_value=httpx.Response(404, json={"detail": "no types"}))
    respx.get(f"{API}/dataset/5/sample").mock(return_value=httpx.Response(500, text="boom"))

    out = json.loads(await dashai_describe_dataset(DescribeDataset(dataset_id=5)))
    assert out["dataset"]["name"] == "iris"
    assert out["info"] == {"rows": 150}
    assert out["column_types"] is None
    assert out["sample"] is None


@respx.mock
async def test_a_missing_dataset_is_an_error():
    respx.get(f"{API}/dataset/999").mock(return_value=httpx.Response(404, json={"detail": "does not exist"}))
    out = await dashai_describe_dataset(DescribeDataset(dataset_id=999))
    assert out.startswith("Error:")


# --- Server surface ----------------------------------------------------------

async def test_no_tool_deletes():
    """Design invariant: dashAI has neither authentication nor undo."""
    from dashai_mcp.server import mcp

    names = [t.name for t in await mcp.list_tools()]
    assert names, "the server must expose tools"
    for forbidden in ("delete", "remove", "drop", "borrar"):
        assert not any(forbidden in n for n in names), f"a tool with '{forbidden}' appeared"


async def test_tools_declare_annotations():
    from dashai_mcp.server import mcp

    for tool in await mcp.list_tools():
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.description, f"{tool.name} has no description"


# --- Regressions found running against a real dashAI 0.9.7.post1 -------------
# The four bugs below passed the tests with doubles and failed against the real
# instance. Each one has its test so they do not come back.

@respx.mock
async def test_select_types_goes_as_repeated_parameters():
    """dashAI's docs show ?select_types=["Model","Metric"] and it does NOT work.

    The backend receives it as ONE type literally named '["Model","Metric"]'
    and answers 422. They must be repeated parameters.
    """
    route = respx.get(f"{API}/component/").mock(return_value=httpx.Response(200, json=[]))
    await dashai_list_components(ListComponents(types=["Task", "Model"]))

    query = str(route.calls[0].request.url.params)
    assert "select_types=Task" in query and "select_types=Model" in query
    assert "%5B" not in query, "a JSON array is being sent, which dashAI rejects"


def test_accepts_the_13_registry_types():
    """The real registry has 13 types, not 4: under-validating rejected good input."""
    from dashai_mcp.server import COMPONENT_TYPES

    for t in ("Converter", "GlobalExplainer", "Explorer", "DatasetSource", "GenerativeModel"):
        assert t in COMPONENT_TYPES
        ListComponents(types=[t])  # must not raise


@respx.mock
async def test_the_job_is_enqueued_as_form_data():
    """POST /job/ does not accept JSON: it expects form data with kwargs as a JSON string.

    Its own openapi.json declares no requestBody for this route because the
    endpoint parses `request` by hand. Sending json= returns
    422 "Missing job_type or kwargs".
    """
    respx.post(f"{API}/model-session/").mock(return_value=httpx.Response(201, json={"id": 1}))
    respx.post(f"{API}/run/").mock(return_value=httpx.Response(201, json={"id": 1}))
    job = respx.post(f"{API}/job/").mock(return_value=httpx.Response(201, json={"id": "h1"}))

    await dashai_train_model(
        TrainModel(
            dataset_id=1, task_name="TabularClassificationTask", model_name="RandomForestClassifier",
            input_columns=["a"], output_columns=["b"], metrics=["Accuracy"], goal_metric="Accuracy",
        )
    )

    request = job.calls[0].request
    assert "application/x-www-form-urlencoded" in request.headers["content-type"]
    body = dict(p.split("=", 1) for p in request.content.decode().split("&"))
    assert body["job_type"] == "ModelJob"
    # kwargs travels serialized, not as loose fields
    assert "run_id" in body["kwargs"]


@respx.mock
async def test_get_run_prunes_the_indexes_and_translates_the_status():
    """split_indexes are ~59 KB on a 10k-row dataset: 99% of the response."""
    indexes = {"train_indexes": list(range(7000)), "test_indexes": list(range(1500)),
               "val_indexes": list(range(1500))}
    respx.get(f"{API}/run/9").mock(
        return_value=httpx.Response(200, json={
            "id": 9, "status": 3, "split_indexes": json.dumps(indexes),
            "test_metrics": {"Accuracy": 0.91},
        })
    )

    out = await dashai_get_run(GetRun(run_id=9))
    data = json.loads(out)

    assert "split_indexes" not in data, "the raw indexes must not reach the agent"
    assert data["split_sizes"] == {"train_indexes": 7000, "test_indexes": 1500, "val_indexes": 1500}
    assert data["status_name"] == "FINISHED"
    assert data["status"] == 3, "the original value is preserved"
    assert data["test_metrics"] == {"Accuracy": 0.91}
    assert len(out) < 1500, f"the response is still huge: {len(out)} chars"


@respx.mock
async def test_queue_empty_is_unnested():
    """/job/is_empty returns {"is_empty": bool}, not a bare bool."""
    respx.get(f"{API}/dataset/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/run/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/job/is_empty").mock(return_value=httpx.Response(200, json={"is_empty": True}))

    data = json.loads(await dashai_server_info(NoArgs()))
    assert data["queue_empty"] is True
