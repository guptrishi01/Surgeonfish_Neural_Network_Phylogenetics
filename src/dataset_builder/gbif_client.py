"""Rate-limited client for the parts of the GBIF API this pipeline needs.

Only ``species/match`` (name -> taxon key) and ``occurrence/search`` (media
records for a taxon key) are used. Every request - including each image
download, which may land on a different host entirely - is checked against
that host's robots.txt at runtime via RobotsChecker before it is sent; see
robots.py.

GBIF's own guidance is that scripts running the search API for more than
~15 minutes should use a bulk occurrence download instead - this pipeline's
per-species incremental review workflow doesn't fit the download API, so
search is used deliberately despite exceeding that threshold across a full
run. See ``dataset_builder``'s package docstring for the resulting citation
follow-up (a GBIF "derived dataset" DOI, registered from the occurrence
keys logged in ``reports/image_sourcing_log.csv``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import requests

from dataset_builder.config import GBIFClientConfig
from dataset_builder.robots import RobotsChecker, RobotsDisallowedError

logger = logging.getLogger(__name__)


@dataclass
class MediaRecord:
    """One candidate image for a species, sourced from a GBIF occurrence.

    Attributes:
        occurrence_key: GBIF occurrence ID the image came from. Used as the
            dedup/seen-tracking key, since a single occurrence can list
            several photos.
        image_url: Direct URL to the original image file.
        license_url: License URL as published by GBIF.
        rights_holder: Attribution string (photographer / rights holder).
    """

    occurrence_key: int
    image_url: str
    license_url: str
    rights_holder: str


class GBIFClient:
    """Thin, rate-limited wrapper around the GBIF REST API."""

    def __init__(self, config: GBIFClientConfig, robots: RobotsChecker | None = None) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers["User-Agent"] = config.user_agent
        self._last_request_time = 0.0
        self._robots = robots or RobotsChecker(config.user_agent)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self._config.request_delay_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def _get_with_retry(self, url: str):
        """Issues a rate-limited GET, retrying transient failures.

        Retries on connection/timeout errors and 5xx responses (the GBIF
        backend occasionally returns a transient 503) with exponential
        backoff. 4xx responses are not retried - they won't succeed on a
        second attempt.

        Raises:
            requests.HTTPError: If retries are exhausted, or on a non-5xx
                error response.
            requests.RequestException: If retries are exhausted on a
                connection-level failure.
        """
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            if attempt > 0:
                time.sleep(self._config.retry_backoff_seconds * (2 ** (attempt - 1)))
            self._throttle()
            try:
                response = self._session.get(url, timeout=self._config.timeout_seconds)
                self._last_request_time = time.monotonic()
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                self._last_request_time = time.monotonic()
                if exc.response is not None and exc.response.status_code < 500:
                    raise
                last_exc = exc
                logger.warning(
                    "Transient error on attempt %d/%d for %s: %s",
                    attempt + 1, self._config.max_retries + 1, url, exc,
                )
            except requests.RequestException as exc:
                self._last_request_time = time.monotonic()
                last_exc = exc
                logger.warning(
                    "Connection error on attempt %d/%d for %s: %s",
                    attempt + 1, self._config.max_retries + 1, url, exc,
                )
        raise last_exc

    def _get(self, path: str, params: dict) -> dict:
        """Issues a rate-limited, robots.txt-checked GET and returns JSON.

        The full URL (including the query string built from ``params``) is
        checked against the host's robots.txt before the request is sent -
        some hosts key their rules off the presence of a query string, so
        checking the bare path would not be sufficient.

        Args:
            path: API path relative to the configured base URL.
            params: Query parameters.

        Returns:
            Parsed JSON response body.

        Raises:
            RobotsDisallowedError: If robots.txt forbids this URL.
            requests.HTTPError: If the response status is not successful
                (after retries, for transient 5xx errors).
        """
        base = f"{self._config.base_url}{path}"
        prepared = requests.Request(url=base, params=params).prepare()
        full_url = prepared.url
        self._robots.ensure_allowed(full_url)

        response = self._get_with_retry(full_url)
        return response.json()

    def match_taxon_key(self, scientific_name: str) -> int | None:
        """Resolves a binomial name to a GBIF taxon key.

        Args:
            scientific_name: e.g. "Zebrasoma flavescens".

        Returns:
            The matched taxon key, or None if GBIF could not confidently
            match the name to a species-rank taxon.
        """
        data = self._get("/species/match", {"name": scientific_name})
        if data.get("rank") != "SPECIES" or data.get("matchType") == "NONE":
            logger.warning(
                "No confident species-rank match for %r (matchType=%s)",
                scientific_name,
                data.get("matchType"),
            )
            return None
        return data.get("usageKey")

    def _is_licensed(self, license_url: str | None) -> bool:
        if not license_url:
            return False
        return any(
            fragment in license_url
            for fragment in self._config.allowed_license_fragments
        )

    def iter_media(
        self, taxon_key: int, seen_occurrence_keys: set[int]
    ) -> Iterator[MediaRecord]:
        """Yields still-image media records for a taxon, newest-crawled first.

        Paginates through ``occurrence/search`` until GBIF's results are
        exhausted. Occurrences already in ``seen_occurrence_keys`` are
        skipped without re-fetching their images, so callers can resume a
        partially-completed species without reprocessing prior candidates.

        Args:
            taxon_key: GBIF usageKey for the target species.
            seen_occurrence_keys: Occurrence keys to skip (already
                accepted or rejected in a prior run).

        Yields:
            MediaRecord for each licensed still image not yet seen.
        """
        offset = 0
        while True:
            data = self._get(
                "/occurrence/search",
                {
                    "taxonKey": taxon_key,
                    "mediaType": "StillImage",
                    "limit": self._config.page_size,
                    "offset": offset,
                },
            )
            results = data.get("results", [])
            if not results:
                return

            for occurrence in results:
                key = occurrence.get("key")
                if key in seen_occurrence_keys:
                    continue
                for media in occurrence.get("media", []):
                    if media.get("type") != "StillImage":
                        continue
                    image_url = media.get("identifier")
                    license_url = media.get("license")
                    if not image_url or not self._is_licensed(license_url):
                        continue
                    yield MediaRecord(
                        occurrence_key=key,
                        image_url=image_url,
                        license_url=license_url,
                        rights_holder=media.get("rightsHolder")
                        or media.get("creator")
                        or "unknown",
                    )
                    break  # one photo per occurrence keeps sourcing diverse

            if data.get("endOfRecords", True):
                return
            offset += self._config.page_size

    def download(self, image_url: str, max_bytes: int) -> bytes | None:
        """Downloads image bytes, rate-limited like any other GBIF-driven call.

        Args:
            image_url: Direct URL to the image (not necessarily on
                gbif.org - GBIF aggregates media hosted by publishers such
                as iNaturalist's open-data bucket).
            max_bytes: Reject (return None) if Content-Length exceeds this.

        Returns:
            Raw image bytes, or None if the download was skipped/failed.
            GBIF aggregates media from many independent hosting providers,
            each with its own robots.txt - unlike the GBIF API calls
            themselves, a disallow here means "skip this one photo's host,"
            not "our whole sourcing approach is wrong," so it is handled
            the same way as any other per-image failure rather than raised.
        """
        try:
            self._robots.ensure_allowed(image_url)
        except RobotsDisallowedError as exc:
            logger.warning("Skipping (robots.txt): %s", exc)
            return None
        self._throttle()

        try:
            response = self._session.get(
                image_url, timeout=self._config.timeout_seconds, stream=True
            )
            self._last_request_time = time.monotonic()
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                logger.warning("Skipping oversized image: %s", image_url)
                return None
            content = response.content
            if len(content) > max_bytes:
                logger.warning("Skipping oversized image: %s", image_url)
                return None
            return content
        except requests.RequestException as exc:
            logger.warning("Download failed for %s: %s", image_url, exc)
            return None
