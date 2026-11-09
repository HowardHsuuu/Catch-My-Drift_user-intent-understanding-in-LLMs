"""Append-only experiment records with checked identities and ordering."""

from pathlib import Path
import os
from .errors import IntegrityError
from .serialization import identity, loads_json, now, require


class EventWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.count = 0
        self.previous = None
        self.identities = set()

    def emit(self, value):
        if (
            value.get("id") in self.identities
            or not isinstance(value.get("id"), str)
            or not value["id"]
        ):
            raise IntegrityError("Events require distinct explicit IDs")
        event = dict(value, sequence=self.count, recorded_at=now(), previous_sha256=self.previous)
        event["record_sha256"] = identity(event)
        self.stream.write(
            __import__("json").dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.identities.add(value["id"])
        self.previous = event["record_sha256"]
        self.count += 1
        return event

    def close(self):
        if not self.stream.closed:
            self.stream.close()


def read_events(path):
    events = []
    previous = None
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for sequence, line in enumerate(stream):
            if not line.endswith("\n"):
                raise IntegrityError("Partial trailing event record")
            event = loads_json(line)
            recorded = event.pop("record_sha256", None)
            require(
                type(event.get("sequence")) is int
                and event.get("sequence") == sequence
                and event.get("previous_sha256") == previous
                and recorded == identity(event),
                "Broken event chain or order",
            )
            require(event.get("id") not in ids, "Repeated event identity")
            event["record_sha256"] = recorded
            events.append(event)
            ids.add(event["id"])
            previous = recorded
    return events
