"""Live smoke: exercises every tool of this server against a real dashAI.

A client written only against documentation is a hypothesis; this script is
how the hypothesis gets tested. It talks to a running dashAI instance through
the same code paths an agent would use, tool by tool, and fails loudly on the
first lie.

Usage:
    python scripts/smoke_live.py               # read-only tools
    python scripts/smoke_live.py --train       # + real train -> predict loop

    DASHAI_BASE_URL   target instance (default http://localhost:8000)
    --dataset-id N    dataset for --train (default: auto-pick a tabular one)
    --timeout S       per-job wait in seconds (default 300)

Exit code 0 only if every exercised tool worked. The --train loop creates a
model session, a run and a prediction on the target instance; run it against
an instance where that garbage is acceptable, not a production one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from dashai_mcp import server
from dashai_mcp.config import base_url

CHECK, CROSS = "ok", "FAIL"
_results: list[tuple[str, str, str]] = []


def _record(tool: str, raw: str) -> dict | None:
    """Parses a tool response, records the verdict, returns the JSON if any.

    Tool responses are JSON on success and either {"error": ...} or a plain
    explanatory string otherwise. A plain string is not automatically a
    failure — list tools answer with prose when there is nothing to list.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        _results.append((tool, CHECK, str(raw)[:120]))
        return None
    if isinstance(data, dict) and "error" in data:
        _results.append((tool, CROSS, str(data["error"])[:200]))
        return None
    if isinstance(raw, str) and raw.startswith("Error:"):
        _results.append((tool, CROSS, raw[:200]))
        return None
    _results.append((tool, CHECK, json.dumps(data, default=str)[:120]))
    return data


async def _wait_for_job(job_id: str, timeout: int) -> dict | None:
    """Polls dashai_job_status until the job finishes, fails or times out."""
    for _ in range(max(1, timeout // 3)):
        raw = await server.dashai_job_status(server.JobStatus(job_id=job_id))
        data = json.loads(raw)
        if data.get("failed"):
            _results.append(("job_status", CROSS, raw[:200]))
            return None
        if data.get("finished"):
            _record("job_status", raw)
            return data
        await asyncio.sleep(3)
    _results.append(("job_status", CROSS, f"job {job_id} still running after {timeout}s"))
    return None


async def _pick_tabular_dataset(datasets: list[dict]) -> tuple[int, list[str], str] | None:
    """First dataset with numeric inputs and a categorical output, described live."""
    for d in datasets:
        raw = await server.dashai_describe_dataset(
            server.DescribeDataset(dataset_id=d["id"])
        )
        try:
            types = json.loads(raw).get("column_types") or {}
        except (json.JSONDecodeError, AttributeError):
            continue
        numeric = [c for c, t in types.items() if t.get("type") in ("Integer", "Float")]
        categorical = [c for c, t in types.items() if t.get("type") == "Categorical"]
        if len(numeric) >= 2 and categorical:
            return d["id"], numeric, categorical[-1]
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="run the full train -> predict loop")
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    print(f"target: {base_url()}\n")

    # ── Read-only tools ──────────────────────────────────────────────────
    info = _record("server_info", await server.dashai_server_info(server.NoArgs()))
    if info is None:
        _print_verdict()
        return 1
    compat = info.get("compatibility", {})
    if compat.get("status") == "mismatch":
        print("compatibility warnings:", *compat.get("warnings", []), sep="\n  - ")

    ds = _record("list_datasets", await server.dashai_list_datasets(server.ListDatasets()))
    datasets = (ds or {}).get("datasets", [])
    if datasets:
        _record(
            "describe_dataset",
            await server.dashai_describe_dataset(
                server.DescribeDataset(dataset_id=datasets[0]["id"])
            ),
        )
    _record(
        "list_components",
        await server.dashai_list_components(server.ListComponents(types=["Model"])),
    )
    _record("list_runs", await server.dashai_list_runs(server.ListRuns()))

    # ── Full loop, only on request: it writes to the instance ────────────
    if args.train:
        ok = await _train_loop(args, datasets)
        if not ok:
            _print_verdict()
            return 1
    else:
        print("(--train not given: train/job/run/predict tools not exercised)\n")

    _print_verdict()
    return 1 if any(v == CROSS for _, v, _ in _results) else 0


async def _train_loop(args, datasets: list[dict]) -> bool:
    if args.dataset_id is not None:
        raw = await server.dashai_describe_dataset(
            server.DescribeDataset(dataset_id=args.dataset_id)
        )
        types = json.loads(raw).get("column_types") or {}
        numeric = [c for c, t in types.items() if t.get("type") in ("Integer", "Float")]
        categorical = [c for c, t in types.items() if t.get("type") == "Categorical"]
        if not (len(numeric) >= 2 and categorical):
            _results.append(("train_model", CROSS, f"dataset {args.dataset_id} has no numeric-inputs/categorical-output shape"))
            return False
        picked = (args.dataset_id, numeric, categorical[-1])
    else:
        picked = await _pick_tabular_dataset(datasets)
        if picked is None:
            _results.append(("train_model", CROSS, "no tabular dataset with numeric inputs and a categorical output; upload one or pass --dataset-id"))
            return False

    dataset_id, inputs, output = picked
    print(f"training on dataset {dataset_id}: {inputs} -> {output}")

    train = _record(
        "train_model",
        await server.dashai_train_model(
            server.TrainModel(
                dataset_id=dataset_id,
                task_name="TabularClassificationTask",
                model_name="SVC",
                input_columns=inputs,
                output_columns=[output],
                metrics=["Accuracy", "F1"],
                goal_metric="Accuracy",
                run_name="smoke-live",
            )
        ),
    )
    if train is None:
        return False

    if await _wait_for_job(train["job_id"], args.timeout) is None:
        return False
    run = _record(
        "get_run", await server.dashai_get_run(server.GetRun(run_id=train["run_id"]))
    )
    if run is None:
        return False

    pred = _record(
        "predict",
        await server.dashai_predict(
            server.Predict(run_id=train["run_id"], dataset_id=dataset_id)
        ),
    )
    if pred is None:
        return False
    if await _wait_for_job(pred["job_id"], args.timeout) is None:
        return False
    got = _record(
        "get_prediction",
        await server.dashai_get_prediction(
            server.GetPrediction(prediction_id=pred["prediction_id"])
        ),
    )
    return got is not None


def _print_verdict() -> None:
    print("\n── verdict ──")
    for tool, verdict, detail in _results:
        print(f"  [{verdict:>4}] {tool}: {detail}")
    failed = sum(1 for _, v, _ in _results if v == CROSS)
    print(f"\n{len(_results) - failed}/{len(_results)} tool calls ok")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
