"""Run directories preserve inputs, partial outputs and completed stages."""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .config import load_config, resolve_path
from .data import load_works
from .errors import ResourceError, StageError
from .serialization import digest, identity, now, read_json, require, write_new
from .store import read_events


def write_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return digest(path)


def stage_complete(folder):
    folder = Path(folder)
    seal = read_json(folder / "complete.json")
    require(seal["events_sha256"] == digest(folder / "events.jsonl"), "Stage records changed")
    events = read_events(folder / "events.jsonl")
    require(seal["events"] == len(events), "Stage event count differs")
    for event in events:
        if "array_file" in event:
            file = (folder / event["array_file"]).resolve()
            require(file.is_relative_to(folder.resolve()), "Array path escapes the run")
            require(digest(file) == event["array_sha256"], "Saved activation array changed")
    return seal, events


def finish_stage(folder, writer, **metadata):
    writer.close()
    value = dict(
        metadata,
        completed_at=now(),
        events=writer.count,
        events_sha256=digest(Path(folder) / "events.jsonl"),
    )
    write_new(Path(folder) / "complete.json", value)
    return value


def initialize(config_path, phase, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load_config(config_path)
    works = load_works(config, config_path)
    if phase not in config.phases:
        raise StageError("Unknown data phase")
    protected = [
        Path(__file__).resolve().parents[2],
        resolve_path(config.dataset, config_path),
        resolve_path(config.agent.path, config_path),
    ]
    if config.nla:
        protected += [
            resolve_path(s.path, config_path)
            for s in (config.nla.verbalizer, config.nla.reconstructor)
        ]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            # Runs in the repository's ignored runs directory are allowed.
            if path == protected[0] and path / "runs" in output.parents:
                continue
            raise StageError(
                "Output must be separate from source, input data and model directories"
            )
    document = asdict(config)
    document["dataset"] = str(output / "works.json")
    document["agent"]["path"] = str(resolve_path(config.agent.path, config_path))
    if config.nla:
        for role in ("verbalizer", "reconstructor"):
            document["nla"][role]["path"] = str(
                resolve_path(document["nla"][role]["path"], config_path)
            )
        document["nla"]["reconstruction"]["head_path"] = str(
            resolve_path(config.nla.reconstruction.head_path, config_path)
        )
    if config.runtime.worker_lock_path:
        document["runtime"]["worker_lock_path"] = str(
            resolve_path(config.runtime.worker_lock_path, config_path)
        )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(resolve_path(config.dataset, config_path), output / "works.json")
    write_new(output / "config.json", document)
    write_new(
        output / "run.json",
        dict(
            phase=phase,
            model=config.model_key,
            created_at=now(),
            config_sha256=digest(output / "config.json"),
            dataset_sha256=digest(output / "works.json"),
            source_sha256=identity(
                {
                    str(p.relative_to(Path(__file__).parent)): digest(p)
                    for p in sorted(Path(__file__).parent.rglob("*.py"))
                }
            ),
        ),
    )
    return load_config(output / "config.json"), works


def open_run(path):
    path = Path(path).resolve()
    record = read_json(path / "run.json")
    require(digest(path / "config.json") == record["config_sha256"], "Run configuration changed")
    require(digest(path / "works.json") == record["dataset_sha256"], "Run dataset changed")
    return load_config(path / "config.json"), record


@contextmanager
def model_lease(config, path):
    """Serialize local model work and reject an occupied CUDA device."""
    lock = (
        resolve_path(config.runtime.worker_lock_path, path)
        if config.runtime.worker_lock_path
        else (Path(tempfile.gettempdir()) / ("catch-my-drift-" + str(os.getuid()) + ".lock"))
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourceError("Another local model stage holds the worker lock") from error
        if (
            shutil.disk_usage(Path(path).resolve().parent).free
            < config.runtime.minimum_free_disk_gib * 1024**3
        ):
            raise ResourceError("Insufficient free disk space for this run")
        if config.agent.device.startswith("cuda"):
            command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
            check = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            if check.stdout.strip():
                raise ResourceError("CUDA is occupied; finish other GPU work before collecting")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
