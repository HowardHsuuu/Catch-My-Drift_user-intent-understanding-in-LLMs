"""Paired panel collection with fresh, non-feedback measurement branches."""

from dataclasses import asdict, replace
from pathlib import Path
import hashlib

import numpy as np

from .backends import load_backend
from .contracts import model_inventory, resolve_contract
from .data import case_for
from .frames import token_frame
from .panel import Turn, action_seed, arm_order, messages_for, position, source_member
from .runs import finish_stage, initialize, model_lease, write_arrays
from .serialization import identity, require, write_new
from .store import EventWriter


class Engine:
    def __init__(self, config, path, folder, backend, writer):
        self.config, self.path, self.folder = config, path, Path(folder)
        self.backend, self.writer = backend, writer
        self.generated, self.observed = {}, {}
        self.tokens = 0

    def reserve(self, sampling):
        require(
            self.tokens + sampling.max_tokens <= self.config.runtime.maximum_generated_tokens,
            "Generated-token budget exhausted before the next model call",
        )

    def generate(self, messages, seed):
        sampling = self.config.generation.discussion
        key = identity([messages, asdict(sampling), seed])
        if key not in self.generated:
            self.reserve(sampling)
            response = self.backend.generate(messages, sampling, seed)
            self.tokens += response["generation_tokens"]
            self.generated[key] = self.writer.emit(
                dict(
                    id="g-" + key,
                    kind="generation",
                    action="discussion",
                    messages=messages,
                    seed=seed,
                    sampling=asdict(sampling),
                    response=response,
                )
            )
        return self.generated[key]

    def frame(self, messages, opening, question):
        spec = replace(
            self.config.state,
            bridge=opening,
            position="bridge_last_content" if opening else "prefix_last_token",
        )
        reserve = max(
            self.config.generation.report.max_tokens, self.config.generation.reply.max_tokens
        )
        frame = token_frame(
            self.backend.tokenizer,
            messages,
            spec,
            question,
            self.config.runtime.maximum_context_tokens - reserve,
            self.config.agent.chat_template_options,
        )
        return frame, spec

    def observe(self, messages, opening):
        key = identity([messages, opening])
        if key in self.observed:
            return self.observed[key]
        frame, spec = self.frame(messages, opening, self.config.measurement.inference_questions[0])
        positions = tuple(range(frame.bridge_start, frame.bridge_end))
        require(
            positions or self.config.allow_empty_contributions,
            "Empty contribution: preserve this partial run",
        )
        active = replace(frame, positions=positions) if positions else frame
        state = self.backend.prefill(active, spec)
        vectors, last = state.positions_after.copy(), state.vector_after.copy()
        state.cache = None
        if positions:
            require(np.array_equal(last, vectors[-1]), "Last contribution state differs")
        arrays = dict(positions=vectors, base_token_ids=np.asarray(frame.base_ids, dtype=np.int32))
        if positions:
            arrays.update(
                last=last, suffix_mean=vectors[-min(3, len(vectors)) :].mean(0, dtype=np.float32)
            )
        else:
            arrays["context_last"] = last
        filename = "captures/" + key + ".npz"
        checksum = write_arrays(self.folder / filename, **arrays)
        n = len(vectors)
        middle = list(range((n - 3) // 2, (n - 3) // 2 + 3)) if n >= 3 and positions else []
        capture = self.writer.emit(
            dict(
                id="h-" + key,
                kind="capture",
                messages=messages,
                opening=opening,
                array_file=filename,
                array_sha256=checksum,
                natural_content_available=bool(positions),
                content_positions=list(positions),
                middle=middle,
                metadata=state.metadata,
                base_token_ids_sha256=identity(list(frame.base_ids)),
                measurement_question_processed=False,
            )
        )

        def branch(question, action):
            framed, state_spec = self.frame(messages, opening, question)
            require(framed.base_ids == frame.base_ids, "Question changed the common prefix")
            sampling = (
                self.config.generation.reply if action == "reply" else self.config.generation.report
            )
            self.reserve(sampling)
            response, before = self.backend.branch(messages, question, state_spec, sampling, 42)
            require(
                np.array_equal(last, before.vector_after),
                "Question changed the pre-question activation",
            )
            self.tokens += response["generation_tokens"]
            return self.writer.emit(
                dict(
                    id="b-" + identity([key, question, action]),
                    kind="generation",
                    action=action,
                    capture_id=capture["id"],
                    question=question,
                    response=response,
                    actual_base_token_ids_sha256=identity(list(framed.base_ids)),
                    actual_last_vector_sha256=hashlib.sha256(
                        before.vector_after.tobytes()
                    ).hexdigest(),
                )
            )["id"]

        value = dict(
            capture_id=capture["id"],
            inference_reports=[
                branch(q, "report") for q in self.config.measurement.inference_questions
            ],
        )
        if self.config.forecast_question:
            value["self_forecast_id"] = branch(self.config.forecast_question, "self_forecast")
        value["reply_id"] = branch(self.config.measurement.reply_question, "reply")
        self.observed[key] = value
        return value


def collect_panel(config, path, folder, backend, writer, works, phase):
    engine = Engine(config, path, folder, backend, writer)
    selected = [w for w in works if w.split == config.phases[phase].split]
    contexts, cases = 0, 0
    for seed in config.phases[phase].seeds:
        cfg = replace(config, generation=replace(config.generation, seed=seed))
        engine.config = cfg
        for work in selected:
            for purpose in cfg.measurement.labels:
                case = case_for(cfg, work, purpose)
                case_id = case.id + ":seed" + str(seed)
                opening = []

                def emit(member, checkpoint, arm, generation, observed):
                    nonlocal contexts
                    contexts += 1
                    writer.emit(
                        dict(
                            id="c-" + identity([case_id, arm, member, checkpoint]),
                            kind="context",
                            model=cfg.model_key,
                            case_id=case_id,
                            work_id=work.id,
                            form=work.form,
                            current=purpose,
                            seed=seed,
                            arm=arm,
                            member=member,
                            position=position(cfg, case, member),
                            checkpoint=checkpoint,
                            discussion_generation_id=generation["id"],
                            source_member=source_member(cfg, case),
                            **observed,
                        )
                    )

                def contribution(messages, member, checkpoint):
                    generation = engine.generate(
                        messages, action_seed(cfg, case, member, "discussion", checkpoint)
                    )
                    text = generation["response"]["text"]
                    return generation, text.strip() if cfg.trim_contributions else text

                for member in range(3):
                    messages = messages_for(cfg, case, member, [], cfg.panel.arms[0])
                    generation, text = contribution(messages, member, 0)
                    opening.append(Turn(case_id + ".m" + str(member) + ".r0", member, 0, text))
                    emit(member, 0, "baseline", generation, engine.observe(messages, text))
                writer.emit(
                    dict(
                        id="opening-" + identity(case_id),
                        kind="opening_group",
                        case_id=case_id,
                        turns=[asdict(t) for t in opening],
                    )
                )
                for arm in arm_order(cfg, case):
                    history = [opening]
                    for checkpoint in (1, 2):
                        group = []
                        for member in range(3):
                            messages = messages_for(cfg, case, member, history, arm)
                            generation, text = contribution(messages, member, checkpoint)
                            group.append(
                                Turn(
                                    case_id
                                    + "."
                                    + arm.name
                                    + ".m"
                                    + str(member)
                                    + ".r"
                                    + str(checkpoint),
                                    member,
                                    checkpoint,
                                    text,
                                )
                            )
                            emit(
                                member,
                                checkpoint,
                                arm.name,
                                generation,
                                engine.observe(messages, text),
                            )
                        writer.emit(
                            dict(
                                id="round-" + identity([case_id, arm.name, checkpoint]),
                                kind="round_group",
                                case_id=case_id,
                                arm=arm.name,
                                checkpoint=checkpoint,
                                turns=[asdict(t) for t in group],
                            )
                        )
                        history.append(group)
                cases += 1
    require(contexts == cases * 27, "Panel endpoint inventory differs from the protocol")
    return dict(
        cases=cases,
        logical_contexts=contexts,
        physical_states=len(engine.observed),
        generated_tokens=engine.tokens,
    )


def collect(config_path, phase, output):
    from .config import load_config

    original = load_config(config_path)
    resolve_contract(replace(original, nla=None), config_path)
    with model_lease(original, config_path):
        config, works = initialize(config_path, phase, output)
        output = Path(output)
        path = output / "config.json"
        write_new(output / "model.json", model_inventory(config.agent, path))
        folder = output / "collect"
        folder.mkdir()
        writer = EventWriter(folder / "events.jsonl")
        backend = None
        try:
            backend = load_backend(config.agent, config.runtime, path)
            result = collect_panel(config, path, folder, backend, writer, works, phase)
            return finish_stage(folder, writer, **result)
        finally:
            writer.close()
            if backend is not None:
                backend.close()
