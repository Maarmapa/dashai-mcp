"""Train a model in dashAI over its REST API, without the MCP server.

Useful when dashai-mcp is out of date for your dashAI version, or when you
want to drive dashAI from a script. It reproduces what dashai_train_model does,
including the fields dashAI 0.11.0 made mandatory: column_roles and
evaluation_strategy.

Usage:
    python scripts/train_rest.py            # defaults to the `students` seed
    python scripts/train_rest.py --dataset 2 --target placement_status
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

BASE = "http://localhost:8000/api/v1"


def _post(path: str, **kw) -> dict:
    r = httpx.post(f"{BASE}/{path}", timeout=30.0, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def _get(path: str, **kw) -> dict:
    r = httpx.get(f"{BASE}/{path}", timeout=30.0, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def train(dataset_id: int, target: str, features: list[str], metrics: list[str],
         goal_metric: str, run_name: str) -> tuple[int, int, str]:
    # 1. Model session (experiment). dashAI 0.11.0 wants a role map, not the
    #    old input_columns/output_columns, and evaluation_strategy is required.
    column_roles = {
        "features": [{"kind": "raw", "name": c} for c in features],
        "target": [{"kind": "raw", "name": target}],
    }
    session = _post(
        "model-session/",
        json={
            "dataset_id": dataset_id,
            "task_name": "TabularClassificationTask",
            "name": run_name,
            "column_roles": column_roles,
            "train_metrics": metrics,
            "validation_metrics": metrics,
            "test_metrics": metrics,
            "evaluation_strategy": "HoldoutEvaluationStrategy",
            "splits": json.dumps(
                {"train": 0.7, "validation": 0.15, "test": 0.15,
                 "shuffle": True, "stratify": True}
            ),
        },
    )
    session_id = session["id"]

    # 2. Run. The plot_*_path fields are required but filled in while optimizing.
    run = _post(
        "run/",
        json={
            "model_session_id": session_id,
            "model_name": "RandomForestClassifier",
            "name": run_name,
            "parameters": {"n_estimators": 100, "class_weight": "balanced"},
            "optimizer_name": "",
            "optimizer_parameters": {},
            "plot_history_path": "",
            "plot_slice_path": "",
            "plot_contour_path": "",
            "plot_importance_path": "",
            "goal_metric": goal_metric,
        },
    )
    run_id = run["id"]

    # 3. Enqueue (form data, not JSON).
    from urllib.parse import urlencode
    job = httpx.post(
        f"{BASE}/job/",
        data={"job_type": "ModelJob", "kwargs": json.dumps({"run_id": run_id})},
        timeout=30.0,
    )
    job.raise_for_status()
    job_id = job.json().get("id") if job.content else None
    return session_id, run_id, job_id


def poll(job_id: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _get(f"job/{job_id}") if False else None
        status = httpx.get(f"{BASE}/job/{job_id}", timeout=30.0).json().get("status")
        if status in ("finished", "error"):
            print(f"job {job_id}: {status}")
            return
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=2)
    ap.add_argument("--target", default="placement_status")
    ap.add_argument("--features", nargs="+",
                    default=["study_hours", "attendance", "sleep_hours",
                             "internet_usage", "assignments_completed", "previous_score"])
    args = ap.parse_args()

    metrics = ["Accuracy", "F1", "MatthewsCorrCoef"]
    run_name = f"rest: RandomForest on dataset {args.dataset}"
    session_id, run_id, job_id = train(
        args.dataset, args.target, args.features, metrics, "MatthewsCorrCoef", run_name
    )
    print(f"session={session_id} run={run_id} job={job_id}")
    poll(job_id)

    run = _get(f"run/{run_id}")
    print("metrics (test):", json.dumps(run.get("test_metrics"), indent=2))
    print("split sizes:", run.get("split_sizes"))


if __name__ == "__main__":
    main()
EOF
echo "creado"; python3 -c "import ast; ast.parse(open('scripts/train_rest.py').read()); print('sintaxis OK')"