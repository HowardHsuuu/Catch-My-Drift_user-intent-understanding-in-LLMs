"""Command-line interface for saved results and new panel experiments."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from . import __version__
from .config import load_config, resolve_path
from .data import load_works
from .errors import ConfigurationError, IntegrityError, ResourceError, StageError
from .serialization import read_json, require, write_new


def parser():
    root = argparse.ArgumentParser(
        description="Investigate user-intent reports, readouts and replies in agent panels."
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "reproduce", help="Recompute paper results from included measurements; CPU only"
    )
    p.add_argument("--out", required=True, type=Path)
    for name, help_text in [
        ("validate", "Check a configuration and dataset without loading models"),
        ("download", "Explicitly download pinned model snapshots"),
        ("doctor", "Inspect local checkpoints and optionally run neutral runtime checks"),
    ]:
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, type=Path)
        if name in ("download", "doctor"):
            p.add_argument("--include-nla", action="store_true")
        if name == "doctor":
            p.add_argument(
                "--runtime",
                action="store_true",
                help="Load the agent and test native prefixes, hooks and caches",
            )
    p = commands.add_parser("collect", help="Collect one independent training or held-out run")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--phase", choices=("training", "held_out"), required=True)
    p.add_argument("--out", required=True, type=Path)
    p = commands.add_parser("nla", help="Describe or reconstruct captured activations")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--stage", choices=("av", "ar"), required=True)
    p = commands.add_parser(
        "review", help="Prepare, read and submit condition-hidden semantic labels"
    )
    p.add_argument("action", choices=("prepare", "show", "submit"))
    p.add_argument("--run", required=True, type=Path)
    p.add_argument(
        "--weights", type=Path, help="Fix a model-matched probe bundle before test review"
    )
    p.add_argument("--labels", type=Path)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=16)
    p = commands.add_parser("fit", help="Fit probes from a reviewed training run only")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = commands.add_parser(
        "analyze", help="Analyze a reviewed run, optionally using its fixed probe bundle"
    )
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--family-size",
        type=int,
        default=1,
        help="Number of prespecified model comparisons; Qwen paper cohort uses 3",
    )
    return root


def validate(path):
    config = load_config(path)
    works = load_works(config, path)
    phases = {}
    for name, phase in config.phases.items():
        count = sum(w.split == phase.split for w in works)
        cases = count * len(phase.seeds) * 2
        phases[name] = dict(
            works=count, seeds=phase.seeds, cases=cases, logical_contexts=cases * 27
        )
    return dict(model=config.model_key, valid=True, model_loaded=False, phases=phases)


def download(path, include_nla):
    config = load_config(path)
    from huggingface_hub import snapshot_download

    models = [config.agent]
    if include_nla:
        require(config.nla is not None, "No matched NLA is configured")
        models += [config.nla.verbalizer, config.nla.reconstructor]
    for model in models:
        folder = resolve_path(model.path, path)
        marker = folder / "source.json"
        expected = dict(model_id=model.model_id, revision=model.revision)
        if marker.exists():
            require(
                read_json(marker) == expected,
                "Existing model folder identifies a different revision",
            )
        snapshot_download(repo_id=model.model_id, revision=model.revision, local_dir=str(folder))
        if not marker.exists():
            write_new(marker, expected)
    return dict(downloaded=[m.model_id for m in models])


def doctor(path, include_nla, runtime):
    from .contracts import resolve_contract

    config = load_config(path)
    contract = resolve_contract(config if include_nla else replace(config, nla=None), path)
    if runtime:
        from .backends import load_backend
        from .runs import model_lease

        with model_lease(config, path):
            backend = load_backend(config.agent, config.runtime, path)
            try:
                contract["runtime"] = backend.technical_check(config.state)
            finally:
                backend.close()
    return contract


def analyze(run, weights, output, family_size):
    from .analysis import analyze_rows
    from .probes import add_predictions
    from .review import joined_rows

    config, record, rows, events = joined_rows(run)
    require(family_size >= 1, "family-size must be positive")
    packet = read_json(run / "review/packet.json")
    if weights is not None:
        require(
            packet["weights_sha256"] is not None,
            "Probe bundle was not fixed before semantic review",
        )
        rows = add_predictions(run, weights, rows, events, packet["weights_sha256"])
    else:
        require(
            packet["weights_sha256"] is None, "Supply the probe bundle fixed in the review packet"
        )
    if record["phase"] == "training":
        raise StageError("Use fit for training data; panel-effect analysis expects held_out")
    result = analyze_rows(rows, family_size=family_size)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "rows.json", rows)
    write_new(output / "summary.json", result)
    return dict(model=config.model_key, summary=result)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command not in ("download", "validate", "reproduce"):
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
                os.environ[name] = "1"
        if args.command == "reproduce":
            from .analysis import reproduce

            result = reproduce(args.out)
            result = {
                k: {
                    "estimate_bounds": v["primary"]["estimate_bounds"],
                    "outer_interval": v["primary"]["outer_interval"],
                }
                for k, v in result.items()
            }
        elif args.command == "validate":
            result = validate(args.config)
        elif args.command == "download":
            result = download(args.config, args.include_nla)
        elif args.command == "doctor":
            result = doctor(args.config, args.include_nla, args.runtime)
        elif args.command == "collect":
            from .collect import collect

            result = collect(args.config, args.phase, args.out)
        elif args.command == "nla":
            from .nla import run_nla

            result = run_nla(args.run, args.stage)
        elif args.command == "review":
            from .review import prepare_review, show_review, submit_review

            if args.action == "prepare":
                result = prepare_review(args.run, args.weights)
            elif args.action == "show":
                result = show_review(args.run, args.offset, args.limit)
            else:
                if args.labels is None:
                    raise StageError("review submit requires --labels")
                result = submit_review(args.run, args.labels)
        elif args.command == "fit":
            from .probes import fit

            result = fit(args.run, args.out)
        elif args.command == "analyze":
            result = analyze(args.run, args.weights, args.out, args.family_size)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (
        ConfigurationError,
        IntegrityError,
        ResourceError,
        StageError,
        ValueError,
        OSError,
        ImportError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
