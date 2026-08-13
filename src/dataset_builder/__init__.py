"""Programmatic image sourcing for the surgeonfish visual phenomics dataset.

This package's automated requests are bound by the following external
guidelines, each with a concrete implementation and/or an open follow-up
action noted below:

- **Robots Exclusion Protocol (robots.txt, RFC 9309).** Enforced at
  runtime, not read once by hand - see ``robots.RobotsChecker``, used by
  every request in ``gbif_client.GBIFClient`` (API calls *and* image
  downloads, which can land on hosts other than GBIF's own). Tested against
  the real robots.txt text of both api.gbif.org and api.inaturalist.org in
  ``tests/dataset_builder/test_robots.py``.
- **GBIF's own API usage guidance.** GBIF asks that scripts expected to run
  more than ~15 minutes against the occurrence-search endpoint use a bulk
  occurrence *download* instead (lighter server load, and downloads carry
  an automatic citable DOI). This pipeline's per-species, incrementally
  reviewed workflow is a poor fit for the download API (it needs to inspect
  and accept/reject individual candidate photos, not receive a static bulk
  export), so occurrence search is used deliberately here - but a full
  64-species run comfortably exceeds 15 minutes. See the "Citation"
  follow-up below for how this is resolved for anything eventually
  published.
- **GBIF Data User Agreement (citation).** Data pulled via the search API
  (as opposed to a bulk download) does not come with an automatic DOI.
  GBIF's own recommendation is to register a *derived dataset*
  (gbif.org/derived-dataset) listing the occurrence keys actually used, to
  obtain a citable DOI. ``reports/image_sourcing_log.csv`` logs the
  ``occurrence_key`` for every accepted image specifically so that list can
  be assembled later. **Open follow-up**: register the derived dataset
  before this dataset is cited in any publication - the pipeline code
  itself cannot do this (it requires the GBIF web UI).
- **License compliance for derivative works.** Only Creative Commons /
  public-domain images are accepted (``config.ALLOWED_LICENSE_FRAGMENTS``);
  no-derivatives (ND) variants are excluded because this pipeline produces
  derivative works (segmentation masks, crops, figure overlays). License
  URL and rights-holder attribution are logged per image.
- **Rate limiting and identification.** One request per second by default
  (``GBIFClientConfig.request_delay_seconds``), a descriptive `User-Agent`
  naming the project and a contact address, and exponential-backoff retry
  (not a hammering retry loop) on transient 5xx/connection failures - see
  ``gbif_client.GBIFClient._get_with_retry``.
- **Prefer the official API over HTML scraping.** All requests go through
  GBIF's documented public REST API; nothing here parses or scrapes
  rendered web pages.

See CLAUDE.md's "Data capture" section for the developer-facing version of
this policy, and README.md for the user-facing summary.
"""
