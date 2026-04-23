from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in limited runtime environments.
    httpx = None

ScalarQueryValue = str | int | float | bool
QueryValue = ScalarQueryValue | Sequence[ScalarQueryValue]


class HttpDownloadError(RuntimeError):
    """Raised when a remote dataset request cannot be completed successfully."""


@dataclass(frozen=True, slots=True)
class HttpRequestSpec:
    url: str
    params: Mapping[str, QueryValue] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    response_format: str = "text"
    provider: str = "generic"


@dataclass(frozen=True, slots=True)
class DownloadedPayload:
    spec: HttpRequestSpec
    cache_key: str
    requested_at: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    from_cache: bool = False

    @property
    def text(self) -> str:
        charset = _extract_charset(self.headers.get("content-type", ""))
        return self.body.decode(charset or "utf-8", errors="replace")


class FileResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, spec: HttpRequestSpec) -> DownloadedPayload | None:
        cache_key = self.cache_key(spec)
        body_path = self._body_path(cache_key)
        metadata_path = self._metadata_path(cache_key)
        if not body_path.exists() or not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return DownloadedPayload(
            spec=spec,
            cache_key=cache_key,
            requested_at=str(metadata.get("requested_at", "")),
            final_url=str(metadata.get("final_url", spec.url)),
            status_code=int(metadata.get("status_code", 200)),
            headers={str(key): str(value) for key, value in dict(metadata.get("headers", {})).items()},
            body=body_path.read_bytes(),
            from_cache=True,
        )

    def save(self, payload: DownloadedPayload) -> None:
        body_path = self._body_path(payload.cache_key)
        metadata_path = self._metadata_path(payload.cache_key)
        body_path.parent.mkdir(parents=True, exist_ok=True)

        body_path.write_bytes(payload.body)
        metadata = {
            "requested_at": payload.requested_at,
            "final_url": payload.final_url,
            "status_code": payload.status_code,
            "headers": payload.headers,
            "spec": asdict(payload.spec),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def cache_key(self, spec: HttpRequestSpec) -> str:
        payload = {
            "url": spec.url,
            "params": _canonicalize_params(spec.params),
            "headers": {key.lower(): value for key, value in sorted(spec.headers.items())},
            "response_format": spec.response_format,
            "provider": spec.provider,
        }
        digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _body_path(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / f"{cache_key}.body"

    def _metadata_path(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / f"{cache_key}.json"


class RetryingHttpClient:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        user_agent: str = "dataset-ingestion/0.1.0",
        http_client: object | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.user_agent = user_agent
        self._owns_http_client = http_client is None and httpx is not None
        self.http_client = http_client or self._build_default_http_client()

    def close(self) -> None:
        if self._owns_http_client and self.http_client is not None:
            self.http_client.close()

    def fetch(
        self,
        spec: HttpRequestSpec,
        cache: FileResponseCache | None = None,
        force_refresh: bool = False,
        logger: logging.Logger | None = None,
    ) -> DownloadedPayload:
        active_logger = logger or logging.getLogger(__name__)

        if cache is not None and not force_refresh:
            cached = cache.load(spec)
            if cached is not None:
                active_logger.info("Cache hit for %s", cached.final_url)
                return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._perform_request(spec, cache_key=cache.cache_key(spec) if cache else None)
                if cache is not None:
                    cache.save(payload)
                return payload
            except Exception as exc:  # pragma: no cover - retried failure path is hard to unit test deterministically.
                last_error = exc
                if attempt >= self.max_retries or not _is_retriable_error(exc):
                    break
                delay = self.retry_backoff_seconds * (2**attempt)
                active_logger.warning(
                    "Request failed for %s on attempt %s/%s; retrying in %.1fs",
                    spec.url,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)

        raise HttpDownloadError(f"Failed to download {spec.url}") from last_error

    def _perform_request(self, spec: HttpRequestSpec, cache_key: str | None) -> DownloadedPayload:
        headers = {"User-Agent": self.user_agent, **dict(spec.headers)}
        requested_at = datetime.now(UTC).isoformat()

        if self.http_client is not None:
            response = self.http_client.get(spec.url, params=dict(spec.params), headers=headers)
            response.raise_for_status()
            return DownloadedPayload(
                spec=spec,
                cache_key=cache_key or sha256(response.content).hexdigest(),
                requested_at=requested_at,
                final_url=str(response.url),
                status_code=int(response.status_code),
                headers={str(key).lower(): str(value) for key, value in response.headers.items()},
                body=bytes(response.content),
            )

        final_url = build_request_url(spec.url, spec.params)
        request = Request(final_url, headers=headers)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return DownloadedPayload(
                spec=spec,
                cache_key=cache_key or sha256(final_url.encode("utf-8")).hexdigest(),
                requested_at=requested_at,
                final_url=final_url,
                status_code=int(getattr(response, "status", 200)),
                headers={str(key).lower(): str(value) for key, value in response.headers.items()},
                body=response.read(),
            )

    def _build_default_http_client(self) -> object | None:
        if httpx is None:
            return None
        return httpx.Client(timeout=self.timeout_seconds, follow_redirects=True)


def build_request_url(url: str, params: Mapping[str, QueryValue]) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(_expand_params(params), doseq=True)}"


def write_raw_response(payload: DownloadedPayload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.body)

    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    metadata = {
        "cache_key": payload.cache_key,
        "requested_at": payload.requested_at,
        "final_url": payload.final_url,
        "status_code": payload.status_code,
        "from_cache": payload.from_cache,
        "headers": payload.headers,
        "spec": asdict(payload.spec),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_dataframe_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:  # pragma: no cover - depends on runtime extras.
        raise RuntimeError(
            "Writing parquet files requires an installed parquet engine such as pyarrow."
        ) from exc
    return path


def _canonicalize_params(params: Mapping[str, QueryValue]) -> dict[str, list[str]]:
    return {key: [str(value) for value in values] for key, values in _expand_params(params).items()}


def _expand_params(params: Mapping[str, QueryValue]) -> dict[str, list[str]]:
    expanded: dict[str, list[str]] = {}
    for key, value in params.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            expanded[key] = [str(item) for item in value]
        else:
            expanded[key] = [str(value)]
    return expanded


def _extract_charset(content_type: str) -> str | None:
    for part in content_type.split(";"):
        token = part.strip()
        if token.casefold().startswith("charset="):
            return token.split("=", maxsplit=1)[1].strip()
    return None


def _is_retriable_error(exc: Exception) -> bool:
    if httpx is not None:
        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 429, 500, 502, 503, 504}
        if isinstance(exc, httpx.TransportError):
            return True

    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, URLError):
        return True
    return False
