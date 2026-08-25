"""Tests for the API compatibility check inside dashai_server_info.

dashAI has no version endpoint, so the check reads openapi.json and compares
the API surface against what this server actually calls. These tests pin the
three verdicts — ok, mismatch, unknown — and the exact situation the check
exists for: a future dashAI requiring a model-session field this server does
not send (its development branch already adds ``evaluation_strategy``).
"""

import json

import httpx
import pytest
import respx

from dashai_mcp.server import (
    _MODEL_SESSION_FIELDS_SENT,
    NoArgs,
    _api_compat,
    dashai_server_info,
)

API = "http://localhost:8000/api/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DASHAI_BASE_URL", raising=False)
    monkeypatch.delenv("DASHAI_ALLOW_REMOTE", raising=False)


def _openapi(required=None, drop_prefix=None):
    """A minimal but honest slice of dashAI 0.9.7's schema."""
    paths = {
        "/dataset/": {},
        "/dataset/{dataset_id}": {},
        "/component/": {},
        "/model-session/": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ModelSessionParams"}
                        }
                    }
                }
            }
        },
        "/run/": {},
        "/job/": {},
        "/job/status/{job_id}": {},
        "/predict/": {},
    }
    if drop_prefix:
        paths = {p: v for p, v in paths.items() if not p.startswith(drop_prefix)}
    return {
        "paths": paths,
        "components": {
            "schemas": {
                "ModelSessionParams": {
                    "required": sorted(required if required is not None else _MODEL_SESSION_FIELDS_SENT)
                }
            }
        },
    }


@respx.mock
async def test_matching_api_reports_ok():
    respx.get(f"{API}/openapi.json").mock(
        return_value=httpx.Response(200, json=_openapi())
    )
    out = await _api_compat()
    assert out["status"] == "ok"
    assert "warnings" not in out


@respx.mock
async def test_new_required_field_is_named():
    """The scenario this check was written for.

    dashAI's development branch adds evaluation_strategy to model sessions;
    the day that ships, server_info has to say so instead of letting
    dashai_train_model fail with a bare 422.
    """
    required = set(_MODEL_SESSION_FIELDS_SENT) | {"evaluation_strategy"}
    respx.get(f"{API}/openapi.json").mock(
        return_value=httpx.Response(200, json=_openapi(required=required))
    )
    out = await _api_compat()
    assert out["status"] == "mismatch"
    assert any("evaluation_strategy" in w for w in out["warnings"])


@respx.mock
async def test_missing_endpoint_is_named():
    respx.get(f"{API}/openapi.json").mock(
        return_value=httpx.Response(200, json=_openapi(drop_prefix="/predict"))
    )
    out = await _api_compat()
    assert out["status"] == "mismatch"
    assert any("/predict" in w for w in out["warnings"])


@respx.mock
async def test_unreadable_schema_degrades_to_unknown():
    """No openapi.json is not an error: the rest of the API may work fine."""
    respx.get(f"{API}/openapi.json").mock(return_value=httpx.Response(404))
    out = await _api_compat()
    assert out["status"] == "unknown"
    assert "note" in out


@respx.mock
async def test_server_info_carries_the_verdict():
    respx.get(f"{API}/dataset/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/run/").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{API}/job/is_empty").mock(
        return_value=httpx.Response(200, json={"is_empty": True})
    )
    respx.get(f"{API}/openapi.json").mock(
        return_value=httpx.Response(200, json=_openapi())
    )
    data = json.loads(await dashai_server_info(NoArgs()))
    assert data["compatibility"]["status"] == "ok"
    assert data["compatibility"]["verified_against"].startswith("dashAI ")
