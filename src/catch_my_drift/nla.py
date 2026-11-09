"""Describe activation vectors and retain every reconstruction."""

from pathlib import Path

import numpy as np

from .backends import load_backend
from .contracts import model_inventory, resolve_contract
from .runs import finish_stage, model_lease, open_run, stage_complete, write_arrays
from .serialization import digest, identity, require, write_new
from .store import EventWriter


def run_nla(run, role):
    run = Path(run).resolve()
    config, record = open_run(run)
    require(config.nla is not None, "No matched NLA is configured for this model")
    require(role in ("av", "ar"), "NLA stage must be av or ar")
    resolve_contract(config, run / "config.json")
    _, events = stage_complete(run / "collect")
    captures = {e["id"]: e for e in events if e["kind"] == "capture"}
    selected = sorted(
        {
            e["capture_id"]
            for e in events
            if e["kind"] == "context" and e["position"] in config.nla_positions
        }
    )
    descriptions = {}
    if role == "ar":
        _, av = stage_complete(run / "av")
        descriptions = {e["capture_token_id"]: e for e in av if e["kind"] == "NLA_description"}
    output = run / role
    spec = config.nla.verbalizer if role == "av" else config.nla.reconstructor
    with model_lease(config, run / "config.json"):
        inventory = model_inventory(spec, run / "config.json")
        output.mkdir(exist_ok=False)
        write_new(output / "model.json", inventory)
        writer = EventWriter(output / "events.jsonl")
        backend = None
        count, tokens = 0, 0
        try:
            backend = load_backend(spec, config.runtime, run / "config.json")
            for capture_id in selected:
                capture = captures[capture_id]
                with np.load(run / "collect" / capture["array_file"], allow_pickle=False) as arrays:
                    vectors = arrays["positions"].copy()
                if not capture["middle"]:
                    writer.emit(
                        dict(
                            id="unavailable-" + capture_id,
                            kind="NLA_unavailable",
                            capture_id=capture_id,
                            reason="Fewer than three contribution tokens",
                        )
                    )
                for index in capture["middle"]:
                    token = capture_id + ":" + str(index)
                    if role == "av":
                        require(
                            tokens + config.generation.nla.max_tokens
                            <= config.runtime.maximum_generated_tokens,
                            "NLA generated-token budget exhausted",
                        )
                        response = backend.verbalize(
                            vectors[index], config.nla.injection, config.generation.nla, 42
                        )
                        require(
                            response["source_text_supplied"] is False, "NLA received source text"
                        )
                        tokens += response["generation_tokens"]
                        writer.emit(
                            dict(
                                id="av-" + token,
                                kind="NLA_description",
                                capture_id=capture_id,
                                capture_token_id=token,
                                position_index=index,
                                response=response,
                                source_text_supplied=False,
                            )
                        )
                    else:
                        description = descriptions[token]
                        vector, metadata = backend.reconstruct(
                            description["response"]["text"], config.nla.reconstruction
                        )
                        original = vectors[index].astype(float)
                        denominator = float(np.linalg.norm(original) * np.linalg.norm(vector))
                        require(
                            denominator > 0 and np.isfinite(vector).all(),
                            "Invalid NLA reconstruction",
                        )
                        filename = "arrays/" + identity(token) + ".npz"
                        checksum = write_arrays(output / filename, reconstructed=vector)
                        writer.emit(
                            dict(
                                id="ar-" + token,
                                kind="NLA_reconstruction",
                                capture_id=capture_id,
                                capture_token_id=token,
                                position_index=index,
                                array_file=filename,
                                array_sha256=checksum,
                                cosine=float(original @ vector / denominator),
                                description_event_sha256=description["record_sha256"],
                                metadata=metadata,
                            )
                        )
                    count += 1
            return finish_stage(
                output,
                writer,
                token_vectors=count,
                generated_tokens=tokens,
                config_sha256=record["config_sha256"],
                no_fidelity_filter=True,
                collection_sha256=digest(run / "collect/complete.json"),
            )
        finally:
            writer.close()
            if backend is not None:
                backend.close()
