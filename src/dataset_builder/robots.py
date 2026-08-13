"""robots.txt compliance, enforced before every outbound HTTP request.

This is a runtime check, not a one-time manual read: each host's
robots.txt is fetched and parsed the first time it's needed, cached, and
consulted via the standard library's own rule-matching (``RobotFileParser``)
before any request to that host is issued.
"""

from __future__ import annotations

import logging
from typing import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

logger = logging.getLogger(__name__)


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt forbids fetching a URL for our user agent."""


def _default_fetch_text(robots_url: str) -> str:
    response = requests.get(robots_url, timeout=10)
    response.raise_for_status()
    return response.text


class RobotsChecker:
    """Fetches, caches, and enforces robots.txt rules per host.

    A missing robots.txt (any fetch failure - 404, timeout, connection
    error) is treated as allow-all, matching the standard convention that
    the absence of a robots.txt imposes no restriction.
    """

    def __init__(
        self,
        user_agent: str,
        fetch_text: Callable[[str], str] = _default_fetch_text,
    ) -> None:
        self._user_agent = user_agent
        self._fetch_text = fetch_text
        self._parsers: dict[str, RobotFileParser] = {}

    def _parser_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._parsers:
            return self._parsers[origin]

        robots_url = f"{origin}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            text = self._fetch_text(robots_url)
            parser.parse(text.splitlines())
            logger.info("Loaded robots.txt from %s", robots_url)
        except Exception as exc:  # noqa: BLE001 - any fetch failure -> allow-all
            logger.info("No robots.txt at %s (%s) - treating as allow-all", robots_url, exc)
            parser.parse([])
        self._parsers[origin] = parser
        return parser

    def ensure_allowed(self, url: str) -> None:
        """Raises RobotsDisallowedError if robots.txt forbids fetching url.

        Args:
            url: Full URL, including query string - some sites (e.g.
                iNaturalist's API host) key their rules off the presence of
                a query string, so the path alone is not enough.
        """
        parser = self._parser_for(url)
        if not parser.can_fetch(self._user_agent, url):
            raise RobotsDisallowedError(
                f"robots.txt at {parser.url} disallows {url!r} for UA={self._user_agent!r}"
            )
