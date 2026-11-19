from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from ..rolling_hnsw import RollingSearchConfig, RollingTemporalHNSW


@dataclass
class _LoadedSnapshot:
    generation: str
    directory: Path
    manifest: dict[str, Any]
    vectors: np.ndarray
    index: RollingTemporalHNSW
    lease_handle: Any
    loaded_at_ms: int
    references: int = 0
    retired: bool = False

    @property
    def last_sequence(self) -> int:
        return int(self.manifest["last_index_sequence"])

    def close(self) -> None:
        fcntl.flock(self.lease_handle.fileno(), fcntl.LOCK_UN)
        self.lease_handle.close()


class SnapshotHandle(AbstractContextManager):
    def __init__(self, manager: "LiveSnapshotManager", snapshot: _LoadedSnapshot):
        self._manager = manager
        self._snapshot = snapshot
        self._released = False

    @property
    def generation(self) -> str:
        return self._snapshot.generation

    @property
    def last_sequence(self) -> int:
        return self._snapshot.last_sequence

    @property
    def index(self) -> RollingTemporalHNSW:
        return self._snapshot.index

    @property
    def manifest(self) -> dict[str, Any]:
        return self._snapshot.manifest

    def search(
        self,
        query: np.ndarray,
        intervals: list[list[int]],
        config: RollingSearchConfig,
    ) -> dict[str, Any]:
        if self._released:
            raise RuntimeError("snapshot handle has been released")
        return self._snapshot.index.search(query, intervals, config)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self._snapshot)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class LiveSnapshotManager:
    """Read-copy-update manager for atomically published RT-HNSW generations."""

    def __init__(self, root: Path, poll_interval: float = 0.25):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.root = Path(root)
        self.poll_interval = poll_interval
        self._condition = threading.Condition(threading.RLock())
        self._current: _LoadedSnapshot | None = None
        self._retired: dict[str, _LoadedSnapshot] = {}
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_refresh_error: str | None = None
        self._refresh_count = 0
        self.refresh()

    def _current_name(self) -> str:
        value = (self.root / "CURRENT").read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError("CURRENT does not name a generation")
        return value

    def _load(self, generation: str) -> _LoadedSnapshot:
        directory = self.root / "generations" / generation
        lease_path = directory / "READERS.lock"
        lease_handle = lease_path.open("a+")
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_SH)
        try:
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            files = manifest["files"]
            for filename, expected in manifest.get("sha256", {}).items():
                digest = hashlib.sha256()
                with (directory / filename).open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"snapshot checksum mismatch: {filename}")
            vectors = np.load(directory / files["vectors"], mmap_mode="r")
            index = RollingTemporalHNSW(directory, vectors)
            return _LoadedSnapshot(
                generation=generation,
                directory=directory,
                manifest=manifest,
                vectors=vectors,
                index=index,
                lease_handle=lease_handle,
                loaded_at_ms=int(time.time() * 1000),
            )
        except BaseException:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
            lease_handle.close()
            raise

    def refresh(self) -> bool:
        generation = self._current_name()
        with self._condition:
            if self._current is not None and self._current.generation == generation:
                return False
        loaded = self._load(generation)
        # CURRENT may advance while a large generation is loading. Never publish
        # that stale load as current; retry on the next poll.
        if self._current_name() != generation:
            loaded.close()
            return self.refresh()
        with self._condition:
            if self._current is not None and self._current.generation == generation:
                loaded.close()
                return False
            previous = self._current
            self._current = loaded
            self._last_refresh_error = None
            self._refresh_count += 1
            if previous is not None:
                previous.retired = True
                if previous.references:
                    self._retired[previous.generation] = previous
                else:
                    previous.close()
            self._condition.notify_all()
            return True

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._thread = threading.Thread(
                target=self._watch, name="rt-hnsw-snapshot-watcher", daemon=True
            )
            self._thread.start()

    def _watch(self) -> None:
        while not self._stopping.wait(self.poll_interval):
            try:
                self.refresh()
            except Exception as error:
                with self._condition:
                    self._last_refresh_error = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()

    def acquire(
        self,
        min_sequence: int | None = None,
        timeout: float | None = None,
    ) -> SnapshotHandle:
        if (min_sequence is not None
                and (isinstance(min_sequence, bool) or not isinstance(min_sequence, int))):
            raise ValueError("min_sequence must be an integer")
        if min_sequence is not None and min_sequence < -1:
            raise ValueError("min_sequence must be at least -1")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                self.refresh()
            except Exception as error:
                with self._condition:
                    self._last_refresh_error = f"{type(error).__name__}: {error}"
            with self._condition:
                snapshot = self._current
                if snapshot is not None and (
                    min_sequence is None or snapshot.last_sequence >= min_sequence
                ):
                    snapshot.references += 1
                    return SnapshotHandle(self, snapshot)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    current = snapshot.last_sequence if snapshot else None
                    raise TimeoutError(
                        f"minimum sequence {min_sequence} unavailable; current={current}"
                    )
                self._condition.wait(
                    self.poll_interval if remaining is None else min(self.poll_interval, remaining)
                )

    def _release(self, snapshot: _LoadedSnapshot) -> None:
        with self._condition:
            if snapshot.references <= 0:
                raise RuntimeError("snapshot reference count underflow")
            snapshot.references -= 1
            if snapshot.retired and snapshot.references == 0:
                self._retired.pop(snapshot.generation, None)
                snapshot.close()
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            current = self._current
            return {
                "root": str(self.root),
                "generation": current.generation if current else None,
                "last_index_sequence": current.last_sequence if current else None,
                "vector_count": int(current.manifest["vector_count"]) if current else 0,
                "loaded_at_ms": current.loaded_at_ms if current else None,
                "active_references": current.references if current else 0,
                "retired_generations": {
                    name: snapshot.references
                    for name, snapshot in self._retired.items()
                },
                "refresh_count": self._refresh_count,
                "last_refresh_error": self._last_refresh_error,
            }

    def collect_generations(self, retain: int = 2) -> list[str]:
        """Delete old generations only when no process holds a reader lease."""
        if retain < 1:
            raise ValueError("retain must be positive")
        current_name = self._current_name()
        directories = sorted(
            path for path in (self.root / "generations").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        protected = {path.name for path in directories[-retain:]}
        protected.add(current_name)
        deleted: list[str] = []
        for directory in directories:
            if directory.name in protected:
                continue
            lease_path = directory / "READERS.lock"
            handle = lease_path.open("a+")
            try:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Recheck CURRENT after locking to close the publication race.
                if directory.name == self._current_name():
                    continue
                shutil.rmtree(directory)
                deleted.append(directory.name)
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                handle.close()
        return deleted

    def close(self) -> None:
        self._stopping.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.poll_interval * 4))
        with self._condition:
            snapshots = list(self._retired.values())
            if self._current is not None:
                snapshots.append(self._current)
            if any(snapshot.references for snapshot in snapshots):
                raise RuntimeError("cannot close snapshot manager with active queries")
            for snapshot in snapshots:
                snapshot.close()
            self._retired.clear()
            self._current = None


class _QueryHandler(BaseHTTPRequestHandler):
    server: "LiveQueryHTTPServer"

    def _json_response(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path in ("/health", "/status"):
            status = self.server.manager.status()
            healthy = status["generation"] is not None and status["last_refresh_error"] is None
            self._json_response(
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                {"healthy": healthy, **status},
            )
        else:
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/refresh":
            try:
                changed = self.server.manager.refresh()
                self._json_response(
                    HTTPStatus.OK, {"changed": changed, **self.server.manager.status()}
                )
            except Exception as error:
                self._json_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)}
                )
            return
        if self.path != "/search":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.max_request_bytes:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            query = np.asarray(request["query"], dtype=np.float32)
            intervals = request["intervals"]
            min_sequence = request.get("min_sequence")
            timeout = float(request.get("wait_timeout_ms", 0)) / 1000.0
            config = RollingSearchConfig(
                k=int(request.get("k", 100)),
                ef_search=int(request.get("ef_search", 128)),
                exact_threshold=int(request.get("exact_threshold", 5_000)),
                gateway_candidates=int(request.get("gateway_candidates", 16)),
                gateway_seeds=int(request.get("gateway_seeds", 4)),
                use_temporal_edges=bool(request.get("use_temporal_edges", True)),
                use_gateway_edges=bool(request.get("use_gateway_edges", False)),
                use_range_edges=bool(request.get("use_range_edges", True)),
                use_hierarchy=False,
            )
            with self.server.manager.acquire(min_sequence, timeout) as snapshot:
                result = snapshot.search(query, intervals, config)
                response = {
                    "generation": snapshot.generation,
                    "last_index_sequence": snapshot.last_sequence,
                    "ids": result["ids"].tolist(),
                    "distances": result["distances"].tolist(),
                    "method": result["method"],
                    "eligible_count": result["eligible_count"],
                    "candidate_count": result["candidate_count"],
                    "distance_computations": result["distance_computations"],
                    "routing_ms": result["routing_ms"],
                    "search_ms": result["search_ms"],
                    "rerank_ms": result["rerank_ms"],
                    "total_ms": result["total_ms"],
                }
            self._json_response(HTTPStatus.OK, response)
        except TimeoutError as error:
            self._json_response(HTTPStatus.GATEWAY_TIMEOUT, {"error": str(error)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)}
            )

    def log_message(self, format: str, *args: Any) -> None:
        if not self.server.quiet:
            super().log_message(format, *args)


class LiveQueryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        manager: LiveSnapshotManager,
        max_request_bytes: int = 2_000_000,
        quiet: bool = False,
    ):
        self.manager = manager
        self.max_request_bytes = max_request_bytes
        self.quiet = quiet
        super().__init__(address, _QueryHandler)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Serve live immutable RT-HNSW snapshots")
    value.add_argument("--index", type=Path, default=Path("Twitter_data/ann/streaming_rt_hnsw"))
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8080)
    value.add_argument("--poll-interval-ms", type=int, default=250)
    value.add_argument("--max-request-bytes", type=int, default=2_000_000)
    value.add_argument("--quiet", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    manager = LiveSnapshotManager(
        args.index, poll_interval=args.poll_interval_ms / 1000.0
    )
    manager.start()
    server = LiveQueryHTTPServer(
        (args.host, args.port), manager,
        max_request_bytes=args.max_request_bytes,
        quiet=args.quiet,
    )
    print(f"serving {args.index} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()
