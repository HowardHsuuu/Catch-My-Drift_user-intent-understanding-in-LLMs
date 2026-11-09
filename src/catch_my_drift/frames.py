"""Exact content-span and question-prefix contracts across tokenizer templates."""

from dataclasses import dataclass
import re
from typing import Tuple

from .config import StateConfig
from .errors import ConfigurationError
from .serialization import identity


def reconstruction_prompt(text, reconstruction):
    """Apply a declared AV wrapper convention without discarding raw evidence.

    Match the reference NLA client's raw-output fallback when tags are missing.
    This never fabricates or filters a description based on its content/fidelity.
    """
    if reconstruction.strip_explanation_tags:
        match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
        if match is not None:
            text = match.group(1)
    return reconstruction.prompt.format(explanation=text)


@dataclass(frozen=True)
class TokenFrame:
    base_ids: Tuple[int, ...]
    full_ids: Tuple[int, ...]
    suffix_ids: Tuple[int, ...]
    bridge_start: int
    bridge_end: int
    positions: Tuple[int, ...]

    def metadata(self):
        return dict(
            base_token_ids_sha256=identity(list(self.base_ids)),
            full_token_ids_sha256=identity(list(self.full_ids)),
            capture_positions=list(self.positions),
            bridge_span=[self.bridge_start, self.bridge_end],
            base_tokens=len(self.base_ids),
            question_tokens=len(self.suffix_ids),
        )


def token_frame(
    tokenizer,
    messages,
    state: StateConfig,
    question: str,
    maximum_tokens: int,
    template_options=None,
):
    options = dict(template_options or {})
    if set(options) & {"tokenize", "add_generation_prompt", "conversation"}:
        raise ConfigurationError("Chat template options cannot replace framing arguments")
    prefix = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **options
        )
    )
    bridge = list(tokenizer.encode(state.bridge, add_special_tokens=False))
    base = list(messages) + [dict(role="assistant", content=state.bridge)]
    ids = list(
        tokenizer.apply_chat_template(base, tokenize=True, add_generation_prompt=False, **options)
    )
    start, end = len(prefix), len(prefix) + len(bridge)
    if (
        (not bridge and state.position != "prefix_last_token")
        or ids[:start] != prefix
        or ids[start:end] != bridge
        or not end <= len(ids)
    ):
        raise ConfigurationError(
            "Tokenizer template does not preserve the exact assistant content span"
        )
    full = list(
        tokenizer.apply_chat_template(
            base + [dict(role="user", content=question)],
            tokenize=True,
            add_generation_prompt=True,
            **options,
        )
    )
    if full[: len(ids)] != ids or len(full) <= len(ids):
        raise ConfigurationError("Question must extend the identical pre-question prefix")
    if len(full) > maximum_tokens:
        raise ConfigurationError("Measurement context exceeds the declared token budget")
    if state.position == "bridge_last_content":
        positions = (end - 1,)
    elif state.position == "bridge_content":
        positions = tuple(range(start, end))
    elif state.position == "prefix_last_token":
        positions = (len(ids) - 1,)
    else:
        raise ConfigurationError("Unknown capture position")
    return TokenFrame(tuple(ids), tuple(full), tuple(full[len(ids) :]), start, end, positions)


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ConfigurationError("A nonempty message list is required")
    for item in messages:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item["role"] not in ("system", "user", "assistant")
            or not isinstance(item["content"], str)
        ):
            raise ConfigurationError("Messages must contain only role and text content")
