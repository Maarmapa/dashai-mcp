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

**v0.2.2 — verified against a running dashAI 0.9.7.post1.** Deterministic tests
(including the predict two-step path) plus live runs: `dashai_train_model` →
`dashai_job_status` → `dashai_get_run` with metrics, and `dashai_predict` →
`dashai_job_status` finished.

### Live tabular run (seed `students`)

Public dashAI seed only. Target `placement_status` (~83% majority).  
`exam_score` was **left out** of the inputs (it leaks the label). Split 70/15/15. Goal metric: **BalancedAccuracy** — Accuracy and F1 on the majority class are traps.

| model | test BalancedAccuracy | test MCC | test F1 | test Accuracy |
|---|---:|---:|---:|---:|
| `DummyClassifier(most_frequent)` | 0.500 | 0.000 | **0.906** | 0.827 |
| `RandomForestClassifier` (`class_weight=balanced`, depth 8, 100 trees) | 0.851 | 0.591 | 0.900 | 0.845 |
| `LogisticRegression` (`class_weight=balanced`, L2) | **0.873** | **0.624** | 0.905 | 0.853 |

The dummy *wins F1* by always answering the majority class. The linear model beats the forest on the metrics that actually measure separation. Each row is its own 70/15/15 draw (not the same test rows) — still enough to stop treating the forest as the default. If a client reports only F1 here, it is lying.

`dashai_predict` on the finished forest run returned `prediction_id` and the job finished (`Saving predictions`). That scores the **same seed dataset the model was trained on**, not a held-out file — do not read it as a generalization check. The MCP does not yet return the prediction table; the proof is the finished job, not a dumped column of labels.

### Live image run (seed `cifar10-subset`)

Public seed: 200 images, frog vs truck (100/100). `LeNet5ImageClassifier`, CPU, 32×32. Split 70/15/15 → **test n=30**. Chance is 0.5.

| run | epochs | train Acc | val Acc | test Acc | test MCC |
|---|---:|---:|---:|---:|---:|
| 10 epochs | 10 | 0.879 | **0.467** | 0.633 | 0.000 |
| 40 epochs | 40 | 0.950 | 0.733 | 0.867 | 0.000 |

The image *path* works (job finished, metrics came back). The *0.867 is not a result*: dashAI defaulted `shuffle=False`, so val/test were 30 trucks and zero frogs. MCC 0 is sklearn on a one-class split, not a broken metric function. Do not quote the 0.867. Retrain with the shuffle/stratify this server now sends.

Verifying against a live instance surfaced **gaps between dashAI's documentation
and its actual behaviour**. Each one has its own regression test. A sixth —
sequential splits with `shuffle=False` — only showed up live because MCC
came back 0 next to a moving Accuracy.
`dashai_predict` sending `run_id` to `PredictJob` — only showed up live
(`KeyError: 'prediction_id'`) because there was no predict test.

## Install

```bash
pip install dashai-mcp
# or: uv pip install dashai-mcp
```

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
| `dashai_get_prediction` | Class counts of a finished prediction — **never the rows** |

## Six things dashAI's documentation (or defaults) get wrong

Found by running against a real instance. If you are writing a client for this
API, these will bite you:

| What the docs say | What the code does |
|---|---|
| `?select_types=["Model","Metric"]` | Must be **repeated parameters**: `?select_types=Model&select_types=Metric`. The JSON array returns 422. |
| `POST /job/` with a JSON body | It is **form data**, with `kwargs` serialized as a JSON string. Its own `openapi.json` declares no `requestBody` for that route, because the endpoint parses `request` by hand. |
| `splits` as an object | It travels as a **JSON string**: the Pydantic schema declares it `str`. |
| `optimize(model_class, search_space, X, y, n_trials)` | The real signature is `optimize(model, input_dataset, output_dataset, parameters, metric)`, and `model` is an **instance**, not a class. |
| Predict by `run_id` on the job | `PredictJob.run` requires `kwargs["prediction_id"]`. The GUI first `POST /predict/` (`{run_id, dataset_id}`) and only then enqueues. Sending `run_id` to the job raises `KeyError: 'prediction_id'`. |
| Split `shuffle` / `stratify` | `prepare_for_model_session` defaults both to **False**. On a class-sorted seed (`cifar10-subset` is 100 frog then 100 truck) a 70/15/15 cut puts val and test in **one class**. Accuracy still moves; sklearn's MCC is defined as 0. This server sends `shuffle=true` and, on classification tasks, `stratify=true`. |

The component registry also has **13 types**, not the four the documentation
suggests: `Task`, `GenerativeTask`, `Model`, `GenerativeModel`, `DataLoader`,
`DatasetSource`, `Metric`, `Optimizer`, `Job`, `LocalExplainer`,
`GlobalExplainer`, `Explorer`, `Converter`.

And `GET /run/{id}` returns `split_indexes` with the full list of indices: on a
10,000-row dataset that is **59 KB, 99% of the response**. This server replaces
it with the per-split counts, bringing the response down to ~1 KB.

## Three design decisions

### 1. Ten tools, not 142

dashAI exposes 142 REST endpoints. Generating one tool per endpoint is mechanical and it is a mistake: a model with 140 tools burns context reading the catalogue and chooses worse. These ten cover the actual working path.

### 2. `dashai_train_model` collapses three calls

In the raw API, training is a chained sequence:

```
POST /model-session/   → creates the experiment
POST /run/             → creates the run
POST /job/             → enqueues the ModelJob
```

With required fields the GUI fills in on its own and that are undocumented — `plot_history_path`, `plot_slice_path`, `plot_contour_path`, `plot_importance_path`. On top of that, **`splits` travels as a JSON string, not an object**, even though dashAI's documentation shows it as an object: the backend's Pydantic schema declares it `str`. That kind of detail is exactly what makes an agent fail against the raw API.

Here it is a single call, and it **does not block**: training can take hours, so it returns the `job_id` immediately and progress is polled with `dashai_job_status`.

`dashai_predict` does the same for the two-step GUI path: `POST /predict/` then `POST /job/` with `prediction_id`.

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

`dashai_get_prediction` needs `pyarrow` in the MCP process to turn the Arrow file into class counts (`pip install 'dashai-mcp[counts]'`, or install the MCP into the same env as dashAI). Without it the tool still returns status and refuses to dump rows.

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
