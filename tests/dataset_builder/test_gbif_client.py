"""Unit tests for GBIFClient's retry behavior on transient failures.

Prompted by a real run hitting a transient 503 from GBIF's backend that
killed the whole 64-species collection after ~4 species had already
succeeded - retries (here) plus per-species fault isolation (in
pipeline.py) are the fix.
"""

from __future__ import annotations

import pytest
import requests

from dataset_builder.config import GBIFClientConfig
from dataset_builder.gbif_client import GBIFClient
from dataset_builder.robots import RobotsChecker


def _no_op_robots() -> RobotsChecker:
    return RobotsChecker("test-bot/1.0", fetch_text=lambda _url: "User-agent: *\nDisallow:\n")


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self):
        return self._json_body


def _client(config: GBIFClientConfig) -> GBIFClient:
    return GBIFClient(config, robots=_no_op_robots())


def test_retries_transient_503_then_succeeds():
    responses = [_FakeResponse(503), _FakeResponse(503), _FakeResponse(200, {"ok": True})]
    calls = {"count": 0}

    class FlakySession:
        def get(self, *args, **kwargs):
            response = responses[calls["count"]]
            calls["count"] += 1
            return response

    client = _client(GBIFClientConfig(request_delay_seconds=0.0, retry_backoff_seconds=0.0))
    client._session = FlakySession()

    result = client._get("/species/match", {"name": "x"})

    assert result == {"ok": True}
    assert calls["count"] == 3


def test_does_not_retry_4xx_errors():
    calls = {"count": 0}

    class AlwaysNotFound:
        def get(self, *args, **kwargs):
            calls["count"] += 1
            return _FakeResponse(404)

    client = _client(GBIFClientConfig(request_delay_seconds=0.0, retry_backoff_seconds=0.0))
    client._session = AlwaysNotFound()

    with pytest.raises(requests.HTTPError):
        client._get("/species/match", {"name": "x"})

    assert calls["count"] == 1  # no retry for a non-transient client error


def test_gives_up_after_max_retries_on_persistent_5xx():
    calls = {"count": 0}

    class AlwaysDown:
        def get(self, *args, **kwargs):
            calls["count"] += 1
            return _FakeResponse(503)

    client = _client(
        GBIFClientConfig(request_delay_seconds=0.0, retry_backoff_seconds=0.0, max_retries=2)
    )
    client._session = AlwaysDown()

    with pytest.raises(requests.HTTPError):
        client._get("/species/match", {"name": "x"})

    assert calls["count"] == 3  # initial attempt + 2 retries
