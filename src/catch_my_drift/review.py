"""Condition-hidden semantic review with exact textual evidence."""

from pathlib import Path

from .data import load_works
from .runs import open_run, stage_complete
from .serialization import digest, identity, read_json, require, write_new

KNOWN = ("support", "assess")
MODES = (*KNOWN, "mixed", "unclear")


def item(kind, text, work=None):
    value = dict(kind=kind, text=text)
    if work is not None:
        value["work"] = work
    return dict(id=identity(value), **value)


def read_weights(path):
    root = Path(path).resolve()
    models = read_json(root / "models.json")
    for head in models["models"].values():
        if head["available"]:
            file = (root / head["file"]).resolve()
            require(file.is_relative_to(root), "Probe path escapes the weights folder")
            require(digest(file) == head["sha256"], "Probe weights changed")
    return models


def prepare_review(run, weights=None):
    run = Path(run).resolve()
    config, record = open_run(run)
    _, events = stage_complete(run / "collect")
    lookup = {e["id"]: e for e in events}
    works = {w.id: w.content for w in load_works(config, run / "config.json")}
    items = {}
    for event in events:
        if event["kind"] != "context":
            continue
        capture = lookup[event["capture_id"]]
        current = [item("plan", capture["opening"])]
        current += [
            item("report", lookup[key]["response"]["text"]) for key in event["inference_reports"]
        ]
        current.append(
            item("reply", lookup[event["reply_id"]]["response"]["text"], works[event["work_id"]])
        )
        if "self_forecast_id" in event:
            current.append(item("forecast", lookup[event["self_forecast_id"]]["response"]["text"]))
        for value in current:
            items[value["id"]] = value
    sources = {"collect/complete.json": digest(run / "collect/complete.json")}
    if config.nla and record["phase"] == "held_out":
        for stage in ("av", "ar"):
            _, values = stage_complete(run / stage)
            sources[stage + "/complete.json"] = digest(run / stage / "complete.json")
            if stage == "av":
                for event in values:
                    if event["kind"] == "NLA_description":
                        value = item("NLA", event["response"]["text"])
                        items[value["id"]] = value
    weight_hash = None
    if weights is not None:
        models = read_weights(weights)
        require(models["model"] == config.model_key, "Probe model does not match the collection")
        weight_hash = digest(Path(weights) / "models.json")
    folder = run / "review"
    folder.mkdir(exist_ok=False)
    ordered = [items[k] for k in sorted(items)]
    write_new(folder / "items.json", ordered)
    write_new(
        folder / "labels-template.json",
        [
            dict(
                id=x["id"], mode=None, evidence=None, audience=None, described_response_purpose=None
            )
            for x in ordered
        ],
    )
    packet = dict(
        items=len(ordered),
        items_sha256=digest(folder / "items.json"),
        sources=sources,
        config_sha256=record["config_sha256"],
        weights_sha256=weight_hash,
    )
    write_new(folder / "packet.json", packet)
    return packet


def show_review(run, offset=0, limit=16):
    folder = Path(run) / "review"
    packet = read_json(folder / "packet.json")
    require(digest(folder / "items.json") == packet["items_sha256"], "Review items changed")
    require(offset >= 0 and 1 <= limit <= 128, "Invalid review slice")
    return read_json(folder / "items.json")[offset : offset + limit]


def submit_review(run, submission):
    folder = Path(run) / "review"
    packet = read_json(folder / "packet.json")
    require(digest(folder / "items.json") == packet["items_sha256"], "Review items changed")
    items = {value["id"]: value for value in read_json(folder / "items.json")}
    decisions = read_json(submission)
    require(
        isinstance(decisions, list) and len(decisions) == len(items),
        "Every unique item requires one explicit decision",
    )
    labels = {}
    for label in decisions:
        require(
            set(label) == {"id", "mode", "evidence", "audience", "described_response_purpose"},
            "Unexpected label fields",
        )
        key = label["id"]
        require(key in items and key not in labels, "Unknown or repeated review item")
        source = items[key]
        require(label["mode"] in MODES, "Use support, assess, mixed or unclear")
        evidence = label["evidence"]
        require(
            isinstance(evidence, str)
            and evidence in source["text"]
            and (evidence or not source["text"]),
            "Evidence must be an exact nonempty excerpt, except for empty output",
        )
        if source["kind"] == "reply":
            require(
                type(label["audience"]) is bool or label["audience"] is None,
                "Reply audience must be true, false or null",
            )
        else:
            require(label["audience"] is None, "Audience is only judged for creator-facing replies")
        if source["kind"] == "NLA":
            require(
                label["described_response_purpose"] in MODES,
                "NLA requires a separate described-purpose judgment",
            )
        else:
            require(
                label["described_response_purpose"] is None,
                "Described response purpose is NLA-only",
            )
        labels[key] = label
    write_new(folder / "labels.json", labels)
    seal = dict(
        labels_sha256=digest(folder / "labels.json"),
        packet_sha256=digest(folder / "packet.json"),
        items=len(labels),
    )
    write_new(folder / "complete.json", seal)
    return seal


def joined_rows(run):
    run = Path(run).resolve()
    config, record = open_run(run)
    folder = run / "review"
    seal, packet = read_json(folder / "complete.json"), read_json(folder / "packet.json")
    require(
        seal["labels_sha256"] == digest(folder / "labels.json")
        and seal["packet_sha256"] == digest(folder / "packet.json"),
        "Semantic decisions changed",
    )
    require(
        packet["config_sha256"] == record["config_sha256"],
        "Review belongs to another configuration",
    )
    for name, checksum in packet["sources"].items():
        require(digest(run / name) == checksum, "Review source changed")
    labels = read_json(folder / "labels.json")
    _, events = stage_complete(run / "collect")
    lookup = {e["id"]: e for e in events}
    works = {w.id: w.content for w in load_works(config, run / "config.json")}
    descriptions = {}
    if "av/complete.json" in packet["sources"]:
        _, av = stage_complete(run / "av")
        descriptions = {
            (e["capture_id"], e["position_index"]): e for e in av if e["kind"] == "NLA_description"
        }

    def mode(kind, response, work=None, field="mode"):
        decision = labels[item(kind, response["text"], work)["id"]]
        value = decision[field]
        if response["finish_reason"] != "stop" or (
            kind == "reply" and decision["audience"] is not True
        ):
            return None
        return value if value in KNOWN else None

    rows = []
    for event in events:
        if event["kind"] != "context":
            continue
        capture = lookup[event["capture_id"]]
        generation = dict(
            lookup[event["discussion_generation_id"]]["response"], text=capture["opening"]
        )
        answers = [mode("report", lookup[key]["response"]) for key in event["inference_reports"]]
        inferred = answers[0] if answers[0] in KNOWN and len(set(answers)) == 1 else None
        values, expectations = [], []
        for index in capture["middle"]:
            description = descriptions.get((capture["id"], index))
            if description:
                expectations.append(mode("NLA", description["response"]))
                values.append(
                    mode("NLA", description["response"], field="described_response_purpose")
                )
        consensus = next(
            (
                v
                for v in KNOWN
                if values.count(v) >= 2 and not any(x in KNOWN and x != v for x in values)
            ),
            None,
        )
        rows.append(
            dict(
                event,
                inferred_report=inferred,
                inference_answers=answers,
                plan_purpose=mode("plan", generation)
                if capture["natural_content_available"]
                else None,
                reply_purpose=mode(
                    "reply", lookup[event["reply_id"]]["response"], works[event["work_id"]]
                ),
                self_forecast=mode("forecast", lookup[event["self_forecast_id"]]["response"])
                if "self_forecast_id" in event
                else None,
                NLA=consensus,
                NLA_expectation=next(
                    (
                        v
                        for v in KNOWN
                        if expectations.count(v) >= 2
                        and not any(x in KNOWN and x != v for x in expectations)
                    ),
                    None,
                ),
                natural_content_available=capture["natural_content_available"],
            )
        )
    return config, record, rows, lookup
