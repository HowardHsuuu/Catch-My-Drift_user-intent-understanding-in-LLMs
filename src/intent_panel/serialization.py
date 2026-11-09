"""Canonical identities and exclusive, durable experiment artifacts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .errors import IntegrityError


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def identity(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest(path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def loads_json(text):
    # Duplicate keys must not silently change a protocol or submitted judgment.
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise IntegrityError("Repeated JSON field: " + key)
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda v: (_ for _ in ()).throw(
            IntegrityError("Nonfinite JSON value: " + v)
        ),
    )


def read_json(path):
    return loads_json(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise IntegrityError(message)
