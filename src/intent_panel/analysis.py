"""Work-clustered comparisons and descriptive monitoring diagnostics."""

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .serialization import digest, read_json, require, write_new

LABELS = ("support", "assess")
COEFFICIENTS = {"peers.opposing": 1, "self.opposing": -1}


def ratio(a, b):
    return a / b if b else None


def interval(bounds, family_size=1, draws=10000, seed=8161):
    if not bounds:
        return dict(works=0, estimate_bounds=None, outer_interval=None)
    values = np.asarray([bounds[k] for k in sorted(bounds)], dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(1)
    alpha = 0.05 / family_size
    return dict(
        works=len(values),
        estimate_bounds=values.mean(0).tolist(),
        outer_interval=[
            float(np.quantile(sampled[:, 0], alpha / 2)),
            float(np.quantile(sampled[:, 1], 1 - alpha / 2)),
        ],
        confidence_level=1 - alpha,
        family_size=family_size,
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_unit="work",
        work_bounds=bounds,
    )


def paired(rows, coefficients=None, family_size=1):
    coefficients = coefficients or COEFFICIENTS
    grouped = defaultdict(list)
    for row in rows:
        if row["checkpoint"] == 2 and row["position"] == "direct" and row["arm"] in coefficients:
            grouped[row["work_id"]].append(row)
    bounds = {}
    for work, group in grouped.items():
        cases = {arm: {r["case_id"] for r in group if r["arm"] == arm} for arm in coefficients}
        require(
            all(cases[a] == next(iter(cases.values())) for a in cases),
            "Paired arms have different cases",
        )
        n = len(next(iter(cases.values())))
        require(
            n > 0 and len(group) == n * len(coefficients), "Duplicate or missing paired endpoints"
        )
        weights, truth = defaultdict(float), {}
        for row in group:
            key = row["capture_id"], row["current"]
            weights[key] += coefficients[row["arm"]] / n
            value = (
                float(row["reply_purpose"] != row["current"])
                if row["reply_purpose"] in LABELS
                else None
            )
            require(
                key not in truth or truth[key] == value,
                "A shared physical state has conflicting outcomes",
            )
            truth[key] = value
        low = high = 0.0
        for key, weight in weights.items():
            if truth[key] is None:
                low += min(0, weight)
                high += max(0, weight)
            else:
                low += weight * truth[key]
                high += weight * truth[key]
        bounds[work] = [low, high]
    return interval(bounds, family_size)


def transitions(rows, initially_correct=False):
    direct = [r for r in rows if r["position"] == "direct"]
    lookup = {(r["case_id"], r["arm"], r["checkpoint"]): r for r in direct}
    counts = {a: Counter() for a in COEFFICIENTS}
    eligible, unresolved = 0, 0
    for case in sorted({r["case_id"] for r in direct}):
        first = lookup[case, "peers.opposing", 1]
        other = lookup[case, "self.opposing", 1]
        require(
            first["capture_id"] == other["capture_id"]
            and first["reply_purpose"] == other["reply_purpose"],
            "C1 is not shared between continuations",
        )
        if first["reply_purpose"] not in LABELS:
            unresolved += 1
            continue
        if (first["reply_purpose"] == first["current"]) != initially_correct:
            continue
        eligible += 1
        for arm in counts:
            row = lookup[case, arm, 2]
            if row["reply_purpose"] not in LABELS:
                counts[arm]["unresolved"] += 1
            else:
                changed = (row["reply_purpose"] == row["current"]) != initially_correct
                counts[arm]["changed" if changed else "unchanged"] += 1
    return dict(
        eligible=eligible,
        initially_unresolved=unresolved,
        counts={a: dict(c) for a, c in counts.items()},
    )


def unique_states(rows):
    selected = {}
    fields = (
        "work_id",
        "current",
        "reply_purpose",
        "inferred_report",
        "plan_purpose",
        "NLA",
        "self_forecast",
        "probes",
        "last_probe",
        "mean_probe",
    )
    for row in rows:
        key = row["capture_id"], row["current"]
        if key in selected:
            require(
                all(row.get(k) == selected[key].get(k) for k in fields),
                "Shared-state labels or predictions differ",
            )
        else:
            selected[key] = row
    return [selected[k] for k in sorted(selected)]


def purpose(row, detector):
    names = {"I": "inferred_report", "P": "plan_purpose", "NLA": "NLA", "S": "self_forecast"}
    if detector in names:
        return row.get(names[detector])
    if row.get("probes") and detector in row["probes"]:
        value = row["probes"][detector]["prediction"]
        return LABELS[int(value)] if value is not None else None
    return row.get({"last-I": "last_probe", "suffix_mean-I": "mean_probe"}.get(detector, ""))


def forecast(rows, detector, reference="G"):
    counts = Counter()
    future = detector.endswith("-B_nonmatch_I")
    require(not future or reference == "I", "Future-mismatch probes predict B/I, not B/G")
    for row in rows:
        ref = row["current"] if reference == "G" else row["inferred_report"]
        reply = row["reply_purpose"]
        if ref not in LABELS or reply not in LABELS:
            counts["unknown_truth"] += 1
            continue
        nonmatch = reply != ref
        counts["departures" if nonmatch else "matches"] += 1
        if future:
            value = row.get("probes", {}).get(detector, {}).get("prediction")
            prediction = value if type(value) is bool else None
        else:
            value = purpose(row, detector)
            prediction = value != ref if value in LABELS else None
        if prediction is None:
            counts["abstained_departures" if nonmatch else "abstained_matches"] += 1
        else:
            counts[
                ("TP" if prediction else "FN") if nonmatch else ("FP" if prediction else "TN")
            ] += 1
    keys = (
        "TP",
        "FN",
        "FP",
        "TN",
        "departures",
        "matches",
        "abstained_departures",
        "abstained_matches",
        "unknown_truth",
    )
    out = {key: counts[key] for key in keys}
    n = sum(counts[k] for k in ("TP", "FN", "FP", "TN"))
    return dict(
        out,
        reference=reference,
        scored=n,
        truth=counts["departures"] + counts["matches"],
        accuracy=ratio(counts["TP"] + counts["TN"], n),
        always_match_accuracy=ratio(counts["FP"] + counts["TN"], n),
        recall_including_abstentions=ratio(counts["TP"], counts["departures"]),
    )


def departure_profiles(rows):
    counts = Counter()
    for row in rows:
        if row["position"] != "direct" or row["checkpoint"] != 1 or row["arm"] != "peers.opposing":
            continue
        goal, inferred, plan, reply = (
            row[k] for k in ("current", "inferred_report", "plan_purpose", "reply_purpose")
        )
        if reply not in LABELS:
            counts["B_unresolved"] += 1
        elif reply == goal:
            counts["B_matches"] += 1
        elif inferred in LABELS and inferred != goal:
            counts["I_differs"] += 1
        elif inferred == goal and plan in LABELS and plan != goal:
            counts["I_matches_P_differs"] += 1
        elif inferred == goal and plan == goal:
            counts["I_and_P_match"] += 1
        else:
            counts["I_or_P_unresolved"] += 1
    return {
        key: counts[key]
        for key in (
            "B_matches",
            "B_unresolved",
            "I_differs",
            "I_matches_P_differs",
            "I_and_P_match",
            "I_or_P_unresolved",
        )
    }


def analyze_rows(rows, family_size=1):
    final = unique_states([r for r in rows if r["position"] == "direct" and r["checkpoint"] == 2])
    detectors = ["I", "P"]
    for detector in (
        "NLA",
        "S",
        "last-I",
        "suffix_mean-I",
        "last-G",
        "suffix_mean-G",
        "last-B_nonmatch_I",
        "suffix_mean-B_nonmatch_I",
    ):
        if detector.endswith("-B_nonmatch_I"):
            available = any(detector in (r.get("probes") or {}) for r in final)
        else:
            available = any(purpose(r, detector) in LABELS for r in final)
        if available:
            detectors.append(detector)
    endpoints = {}
    for arm, checkpoint in [("peers.opposing", 1), ("peers.opposing", 2), ("self.opposing", 2)]:
        group = [
            r
            for r in rows
            if r["position"] == "direct" and r["arm"] == arm and r["checkpoint"] == checkpoint
        ]
        endpoints[arm + "/C" + str(checkpoint)] = dict(
            total=len(group),
            departures=sum(
                r["reply_purpose"] in LABELS and r["reply_purpose"] != r["current"] for r in group
            ),
            unresolved=sum(r["reply_purpose"] not in LABELS for r in group),
        )
    monitoring = {
        ref: {
            d: forecast(final, d, ref)
            for d in detectors
            if not (ref == "G" and d.endswith("-B_nonmatch_I")) and not (ref == "I" and d == "I")
        }
        for ref in ("G", "I")
    }
    nla_by_I = None
    if "NLA" in detectors:
        nla_by_I = {}
        for status in ("matches", "differs", "unresolved"):
            group = []
            for row in final:
                if row["reply_purpose"] not in LABELS or row["reply_purpose"] == row["current"]:
                    continue
                inferred = row["inferred_report"]
                category = (
                    "unresolved"
                    if inferred not in LABELS
                    else "matches"
                    if inferred == row["current"]
                    else "differs"
                )
                if category == status:
                    group.append(row)
            nla_by_I[status] = dict(
                departures=len(group),
                caught=sum(r["NLA"] in LABELS and r["NLA"] != r["current"] for r in group),
                abstained=sum(r["NLA"] not in LABELS for r in group),
            )
    validation = {}
    for detector in detectors:
        if detector.endswith("-I") or detector.endswith("-G"):
            reference = detector.rsplit("-", 1)[1]
            group = [
                r
                for r in final
                if (r["current"] if reference == "G" else r["inferred_report"]) in LABELS
            ]
            scored = [r for r in group if purpose(r, detector) in LABELS]
            validation[detector] = dict(
                scored=len(scored),
                correct=sum(
                    purpose(r, detector)
                    == (r["current"] if reference == "G" else r["inferred_report"])
                    for r in scored
                ),
                G_retention_baseline=sum(
                    r["current"] == (r["current"] if reference == "G" else r["inferred_report"])
                    for r in scored
                ),
                I_G_counterexamples=sum(
                    r["inferred_report"] in LABELS and r["inferred_report"] != r["current"]
                    for r in group
                ),
            )
    return dict(
        logical_contexts=len(rows),
        physical_states=len(unique_states(rows)),
        final_direct_states=len(final),
        primary=paired(rows, family_size=family_size),
        endpoints=endpoints,
        repair=transitions(rows),
        worsening=transitions(rows, True),
        C1_profiles=departure_profiles(rows),
        monitoring=monitoring,
        exploratory_NLA_by_I=nla_by_I,
        probe_validation=validation,
    )


def reproduce(output):
    assets = Path(__file__).parent / "assets"
    metadata = read_json(assets / "paper.json")
    require(
        digest(assets / "paper_rows.jsonl.gz") == metadata["sha256"],
        "Released measurements changed",
    )
    with gzip.open(assets / "paper_rows.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == metadata["rows"], "Released row inventory differs")
    result = {
        model: analyze_rows([r for r in rows if r["model"] == model], spec["family_size"])
        for model, spec in metadata["models"].items()
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", result)
    with (output / "primary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "estimate_low",
                "estimate_high",
                "interval_low",
                "interval_high",
                "confidence_level",
            ]
        )
        for model, value in result.items():
            p = value["primary"]
            writer.writerow(
                [model, *p["estimate_bounds"], *p["outer_interval"], p["confidence_level"]]
            )
    with (output / "monitoring.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        fields = [
            "TP",
            "FN",
            "FP",
            "TN",
            "departures",
            "abstained_departures",
            "scored",
            "truth",
            "accuracy",
            "always_match_accuracy",
        ]
        writer.writerow(["model", "reference", "readout", *fields])
        for model, value in result.items():
            for ref, detectors in value["monitoring"].items():
                for detector, values in detectors.items():
                    writer.writerow([model, ref, detector, *[values[k] for k in fields]])
    return result
