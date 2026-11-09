"""Work-balanced ridge readouts fitted only on training states."""

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .analysis import unique_states
from .config import load_config
from .review import joined_rows, read_weights
from .runs import write_arrays
from .serialization import digest, require, write_new

KNOWN = ("support", "assess")


def normalize(value, center):
    vector = np.asarray(value, dtype=np.float64) - center
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def score(vector, weight, bias):
    return math.fsum(float(a) * float(b) for a, b in zip(vector, weight)) + float(bias)


def sample_weights(rows, target):
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        key = (row["current"], row["inferred_report"]) if target == "I" else row["target"]
        groups[str(key)][row["work_id"]].append(index)
    weights = np.zeros(len(rows), dtype=float)
    for group in groups.values():
        for indices in group.values():
            weights[indices] = len(rows) / (len(groups) * len(group) * len(indices))
    require(abs(weights.sum() - len(rows)) < 1e-8, "Training weights must sum to the sample count")
    return weights


def features(run, rows, events, field):
    require(field in ("last", "suffix_mean"), "Unknown contribution feature")
    vectors = {}
    for row in rows:
        capture = events[row["capture_id"]]
        if not capture["natural_content_available"]:
            continue
        file = Path(run) / "collect" / capture["array_file"]
        require(digest(file) == capture["array_sha256"], "Probe source array changed")
        with np.load(file, allow_pickle=False) as arrays:
            vectors[capture["id"]] = arrays[field].astype(np.float64)
    return vectors


def fit(run, output):
    config, record, rows, events = joined_rows(run)
    require(record["phase"] == "training", "Probe fitting only accepts a training run")
    rows = unique_states(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    for feature in config.probes.features:
        vectors = features(run, rows, events, feature)
        available = [r for r in rows if r["capture_id"] in vectors]
        require(available, "No training activations are available")
        center = np.mean(
            np.stack([vectors[r["capture_id"]] for r in available]), axis=0, dtype=np.float64
        )
        normalized = {key: normalize(x, center) for key, x in vectors.items()}
        for target in config.probes.targets:
            selected = []
            for row in available:
                if target == "G":
                    value = row["current"] == "assess"
                elif target == "I":
                    if row["inferred_report"] not in KNOWN:
                        continue
                    value = row["inferred_report"] == "assess"
                else:
                    if row["inferred_report"] not in KNOWN or row["reply_purpose"] not in KNOWN:
                        continue
                    value = row["reply_purpose"] != row["inferred_report"]
                selected.append(dict(row, target=bool(value)))
            key = feature + "-" + target
            class_works = {
                str(y): sorted({r["work_id"] for r in selected if r["target"] == y})
                for y in (False, True)
            }
            head = dict(
                target=target,
                feature=feature,
                threshold=0.0,
                training_samples=len(selected),
                class_works=class_works,
                available=all(
                    len(v) >= config.probes.minimum_works_per_class for v in class_works.values()
                ),
            )
            models[key] = head
            if not head["available"]:
                head["reason"] = "Fewer than two independent training works in a required class"
                head["file"] = None
                continue
            weights = sample_weights(selected, target)
            matrix = np.stack([normalized[r["capture_id"]] for r in selected])
            y = np.asarray([1.0 if r["target"] else -1.0 for r in selected])
            xbar = np.average(matrix, axis=0, weights=weights)
            ybar = float(np.average(y, weights=weights))
            design = (matrix - xbar) * np.sqrt(weights)[:, None]
            outcome = (y - ybar) * np.sqrt(weights)
            dual = np.linalg.solve(
                np.einsum("ik,jk->ij", design, design, optimize=False)
                + np.eye(len(design)) * config.probes.ridge,
                outcome,
            )
            weight = np.einsum("ij,i->j", design, dual, optimize=False)
            bias = ybar - score(xbar, weight, 0.0)
            residual = np.einsum("ij,j->i", matrix, weight, optimize=False) + bias - y
            gradient = (
                np.einsum("ij,i->j", matrix, weights * residual, optimize=False)
                + config.probes.ridge * weight
            )
            require(
                np.isfinite(weight).all()
                and math.isfinite(bias)
                and max(
                    float(np.max(np.abs(gradient))),
                    abs(math.fsum(float(a) * float(b) for a, b in zip(weights, residual))),
                )
                < 1e-6,
                "Ridge fit does not satisfy its optimality equations",
            )
            filename = key + ".npz"
            head.update(
                file=filename,
                sha256=write_arrays(
                    output / filename, center=center, weight=weight, bias=np.asarray(bias)
                ),
                ridge=config.probes.ridge,
            )
    result = dict(
        model=config.model_key,
        model_id=config.agent.model_id,
        revision=config.agent.revision,
        layer=config.state.layer,
        models=models,
        fit_split="training",
        training_works=sorted({r["work_id"] for r in rows}),
        config_sha256=record["config_sha256"],
        training_review_sha256=digest(Path(run) / "review/complete.json"),
        training_collection_sha256=digest(Path(run) / "collect/complete.json"),
    )
    write_new(output / "models.json", result)
    return result


def add_predictions(run, weights, rows, events, expected_hash):
    weights = Path(weights)
    require(
        digest(weights / "models.json") == expected_hash,
        "Use the exact probe bundle fixed before semantic test review",
    )
    models = read_weights(weights)
    config = load_config(Path(run) / "config.json")
    require(
        (models["model"], models["model_id"], models["revision"], models["layer"])
        == (config.model_key, config.agent.model_id, config.agent.revision, config.state.layer),
        "Probe weights belong to another model, revision or extraction layer",
    )
    values = {
        f: features(run, unique_states(rows), events, f)
        for f in {h["feature"] for h in models["models"].values()}
    }
    predictions = defaultdict(dict)
    for name, head in models["models"].items():
        if not head["available"]:
            for row in rows:
                predictions[row["capture_id"]][name] = dict(prediction=None, score=None)
            continue
        with np.load(weights / head["file"], allow_pickle=False) as arrays:
            center, weight, bias = arrays["center"], arrays["weight"], float(arrays["bias"])
        require(center.shape == weight.shape and center.ndim == 1, "Invalid probe dimensions")
        for row in unique_states(rows):
            key = row["capture_id"]
            vector = values[head["feature"]].get(key)
            if vector is None:
                predictions[key][name] = dict(prediction=None, score=None)
                continue
            require(vector.shape == weight.shape, "Probe feature width differs from training")
            value = score(normalize(vector, center), weight, bias)
            predictions[key][name] = dict(prediction=value >= head["threshold"], score=value)
    return [dict(row, probes=predictions[row["capture_id"]]) for row in rows]
