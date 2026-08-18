# dashai-mcp

An [MCP](https://modelcontextprotocol.io) server for **[dashAI](https://github.com/DashAISoftware/DashAI)**, the open source Machine Learning workbench led by the University of Chile (FCFM), built by students of DCC UChile and UTFSM, with CENIA and IMFD.

> **Unofficial and independent.** This is a third-party project. It is not
> affiliated with, endorsed by, or maintained by the dashAI project or the
> institutions that develop it.

It gives an agent the same surface dashAI gives a person through its GUI: look at datasets, see which models are available, train, follow queued work, and read the metrics.

```
"Train a random forest on dataset 3 predicting 'species' and tell me the F1"
```

## Status

**v0.2.0 — verified against a running dashAI 0.9.7.post1.** 25 deterministic tests
green, plus one complete end-to-end training run: `dashai_train_model` →
`dashai_job_status` → `dashai_get_run` with metrics.

Verifying against a live instance surfaced **four bugs that tests with doubles
could not see**, all of them gaps between dashAI's documentation and its actual
behaviour. They are described below and each one has its own regression test.

## Install

```bash
uv pip install git+https://github.com/Maarmapa/dashai-mcp
# or: pip install git+https://github.com/Maarmapa/dashai-mcp
```

Not published on PyPI yet.

In your MCP client configuration:

```json
{
  "mcpServers": {
    "dashai": {
      "command": "dashai-mcp"
    }
  }
}
```

dashAI has to be running separately (`dashai`, or the desktop app). It is looked up at `http://localhost:8000` by default.

## Tools

| Tool | What it does |
|---|---|
| `dashai_server_info` | Is dashAI up? How many datasets and runs are there |
| `dashai_list_datasets` | Lists the loaded datasets |
| `dashai_describe_dataset` | Columns, types and a sample — all in one call |
| `dashai_list_components` | Available models, metrics, tasks and optimizers |
| `dashai_train_model` | **Trains.** Enqueues and returns `job_id` + `run_id` |
| `dashai_job_status` | Job progress: `not_started` / `started` / `finished` / `error` |
| `dashai_list_runs` | Recorded runs, for comparing models |
| `dashai_get_run` | Configuration and metrics of a run |
| `dashai_predict` | Predicts using the model of a finished run |

## Four things dashAI's documentation gets wrong

Found by running against a real instance. If you are writing a client for this
API, these will bite you:

| What the docs say | What the code does |
|---|---|
| `?select_types=["Model","Metric"]` | Must be **repeated parameters**: `?select_types=Model&select_types=Metric`. The JSON array returns 422. |
| `POST /job/` with a JSON body | It is **form data**, with `kwargs` serialized as a JSON string. Its own `openapi.json` declares no `requestBody` for that route, because the endpoint parses `request` by hand. |
| `splits` as an object | It travels as a **JSON string**: the Pydantic schema declares it `str`. |
| `optimize(model_class, search_space, X, y, n_trials)` | The real signature is `optimize(model, input_dataset, output_dataset, parameters, metric)`, and `model` is an **instance**, not a class. |

The component registry also has **13 types**, not the four the documentation
suggests: `Task`, `GenerativeTask`, `Model`, `GenerativeModel`, `DataLoader`,
`DatasetSource`, `Metric`, `Optimizer`, `Job`, `LocalExplainer`,
`GlobalExplainer`, `Explorer`, `Converter`.

And `GET /run/{id}` returns `split_indexes` with the full list of indices: on a
10,000-row dataset that is **59 KB, 99% of the response**. This server replaces
it with the per-split counts, bringing the response down to ~1 KB.

## Three design decisions

### 1. Nine tools, not 142

dashAI exposes 142 REST endpoints. Generating one tool per endpoint is mechanical and it is a mistake: a model with 140 tools burns context reading the catalogue and chooses worse. These nine cover the actual working path.

### 2. `dashai_train_model` collapses three calls

In the raw API, training is a chained sequence:

```
POST /model-session/   → creates the experiment
POST /run/             → creates the run
POST /job/             → enqueues the ModelJob
```

With required fields the GUI fills in on its own and that are undocumented — `plot_history_path`, `plot_slice_path`, `plot_contour_path`, `plot_importance_path`. On top of that, **`splits` travels as a JSON string, not an object**, even though dashAI's documentation shows it as an object: the backend's Pydantic schema declares it `str`. That kind of detail is exactly what makes an agent fail against the raw API.

Here it is a single call, and it **does not block**: training can take hours, so it returns the `job_id` immediately and progress is polled with `dashai_job_status`.

### 3. No tool deletes anything

dashAI's API has **no authentication** — checked endpoint by endpoint. That is coherent for something local-first, but it means there is no barrier between a misread sentence and an irreversible `DELETE /dataset/{id}`. Deleting is done from the GUI, looking at what is being deleted.

For the same reason, the server **refuses to point at a non-local host**:

```
DASHAI_BASE_URL points to 'ml.example.com', which is not local, and dashAI's API
has no authentication: exposing it to the network leaves the backend open to
anyone who can reach it.
```

This can be disabled on purpose with `DASHAI_ALLOW_REMOTE=1`, if the target is protected some other way.

## Configuration

| Variable | Default | What for |
|---|---|---|
| `DASHAI_BASE_URL` | `http://localhost:8000` | Where the backend is |
| `DASHAI_ALLOW_REMOTE` | *(no)* | Allow a non-local host (see above) |
| `DASHAI_TIMEOUT` | `30` | Seconds to wait per request |

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

The tests stub the HTTP responses with `respx`: **they need neither a dashAI instance nor credentials**. They test the contract — which calls are made, in what order, with what body, and what the agent is told when something fails.

## A note on the SDK

Requires the MCP Python SDK **2.x**. Version 2.0 removed `mcp.server.fastmcp`; it is now `mcp.server.mcpserver.MCPServer`, and annotations are `ToolAnnotations` objects instead of dictionaries. Most tutorials still show the 1.x API.

## License

MIT, same as dashAI. See the note at the top on affiliation.
