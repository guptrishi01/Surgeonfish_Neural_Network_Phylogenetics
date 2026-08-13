"""robots.txt is the one hard legal/ethical constraint this pipeline must
never silently bypass, so these tests check it against the actual
robots.txt text pulled from the real hosts (captured verbatim below), not
a synthetic stand-in that might not reflect real-world quirks.
"""

from __future__ import annotations

import pytest

from dataset_builder.config import GBIFClientConfig
from dataset_builder.gbif_client import GBIFClient
from dataset_builder.robots import RobotsChecker, RobotsDisallowedError

# Captured verbatim via `curl https://api.gbif.org/robots.txt` (2026-08-06).
GBIF_ROBOTS_TXT = """\
# robots.txt for api.gbif.org
#

User-agent: *
# Block unsigned images
Disallow: /v1/image/unsafe

# Allow message/social previews
User-agent: Discordbot
User-agent: LinkedInBot
User-agent: Pinterestbot
User-agent: Slackbot
User-agent: Snapchat
User-agent: TelegramBot
User-agent: Twitterbot
User-agent: WhatsApp
User-agent: facebookexternalhit
Disallow:
"""

# Captured verbatim via `curl https://api.inaturalist.org/robots.txt` (2026-08-06).
# Kept here to prove our checker would correctly block the endpoint this
# pipeline deliberately avoids calling directly.
INATURALIST_API_ROBOTS_TXT = """\
User-agent: *
Allow: /favicon.ico
Allow: /v1/docs
Allow: /v2/docs
Disallow: /v2/docs?
Disallow: /*?
Disallow: /
"""


def _checker(robots_txt: str, user_agent: str = "test-bot/1.0") -> RobotsChecker:
    return RobotsChecker(user_agent, fetch_text=lambda _url: robots_txt)


def test_gbif_robots_allows_occurrence_search():
    checker = _checker(GBIF_ROBOTS_TXT)
    checker.ensure_allowed(
        "https://api.gbif.org/v1/occurrence/search?taxonKey=2379869&mediaType=StillImage"
    )  # must not raise


def test_gbif_robots_allows_species_match():
    checker = _checker(GBIF_ROBOTS_TXT)
    checker.ensure_allowed(
        "https://api.gbif.org/v1/species/match?name=Zebrasoma+flavescens"
    )  # must not raise


def test_gbif_robots_blocks_image_unsafe_endpoint():
    checker = _checker(GBIF_ROBOTS_TXT)
    with pytest.raises(RobotsDisallowedError):
        checker.ensure_allowed("https://api.gbif.org/v1/image/unsafe/some-signed-path")


def test_inaturalist_robots_blocks_query_string_api_calls():
    """This is the exact endpoint/shape we chose not to call directly."""
    checker = _checker(INATURALIST_API_ROBOTS_TXT)
    with pytest.raises(RobotsDisallowedError):
        checker.ensure_allowed(
            "https://api.inaturalist.org/v1/observations?taxon_name=Zebrasoma+flavescens"
        )


def test_inaturalist_robots_allows_docs_page():
    checker = _checker(INATURALIST_API_ROBOTS_TXT)
    checker.ensure_allowed("https://api.inaturalist.org/v1/docs")  # must not raise


def test_missing_robots_txt_defaults_to_allow():
    def _raise_not_found(_url: str) -> str:
        raise RuntimeError("404 Not Found")

    checker = RobotsChecker("test-bot/1.0", fetch_text=_raise_not_found)
    checker.ensure_allowed("https://inaturalist-open-data.s3.amazonaws.com/photos/1/original.jpg")


def test_robots_txt_is_fetched_once_per_host_then_cached():
    calls = []

    def _fetch(url: str) -> str:
        calls.append(url)
        return GBIF_ROBOTS_TXT

    checker = RobotsChecker("test-bot/1.0", fetch_text=_fetch)
    checker.ensure_allowed("https://api.gbif.org/v1/species/match?name=a")
    checker.ensure_allowed("https://api.gbif.org/v1/occurrence/search?taxonKey=1")
    checker.ensure_allowed("https://api.gbif.org/v1/species/match?name=b")

    assert calls == ["https://api.gbif.org/robots.txt"]


def test_gbif_client_blocks_before_any_network_call_when_robots_disallows():
    """The check must happen before the real HTTP GET is issued, not after."""

    class ExplodingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("HTTP request was issued despite robots.txt disallowing it")

    disallow_all = "User-agent: *\nDisallow: /\n"
    client = GBIFClient(
        GBIFClientConfig(request_delay_seconds=0.0),
        robots=_checker(disallow_all),
    )
    client._session = ExplodingSession()

    with pytest.raises(RobotsDisallowedError):
        client.match_taxon_key("Zebrasoma flavescens")


def test_gbif_client_download_skips_disallowed_host_instead_of_raising():
    """GBIF aggregates images from many independent hosts, each with its own
    robots.txt - one host disallowing us should skip that photo, not kill
    the whole collection run for every other species queued behind it."""

    class ExplodingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("HTTP request was issued despite robots.txt disallowing it")

    disallow_all = "User-agent: *\nDisallow: /\n"
    client = GBIFClient(
        GBIFClientConfig(request_delay_seconds=0.0),
        robots=_checker(disallow_all),
    )
    client._session = ExplodingSession()

    result = client.download("https://example-museum.org/photos/1.jpg", max_bytes=1_000_000)

    assert result is None


def test_gbif_client_allows_real_gbif_robots_txt():
    """End-to-end sanity check with the real GBIF robots.txt content."""

    calls = {"count": 0}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            calls["count"] += 1
            return {"rank": "SPECIES", "matchType": "EXACT", "usageKey": 123}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = GBIFClient(
        GBIFClientConfig(request_delay_seconds=0.0),
        robots=_checker(GBIF_ROBOTS_TXT),
    )
    client._session = FakeSession()

    key = client.match_taxon_key("Zebrasoma flavescens")
    assert key == 123
    assert calls["count"] == 1
