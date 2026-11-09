"""Explicit configuration for the panel experiment and matched readouts."""

from dataclasses import dataclass, field, fields, is_dataclass
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from .errors import ConfigurationError
from .serialization import read_json


@dataclass(frozen=True)
class ModelConfig:
    path: str
    model_id: str
    revision: str
    backend: str = "transformers"
    dtype: str = "bfloat16"
    device: str = "cpu"
    attention: str = "eager"
    tokenizer_options: Dict[str, Any] = field(default_factory=dict)
    chat_template_options: Dict[str, Any] = field(default_factory=dict)
    final_norm: str = "checkpoint"
    language_model_head: str = "checkpoint"
    stop_token_ids: Optional[List[int]] = None


@dataclass(frozen=True)
class StateConfig:
    layer: int
    bridge: str = "I have considered the creator's message."
    position: str = "bridge_last_content"
    pooling: str = "last"
    hook_point: str = "block_output"


@dataclass(frozen=True)
class NLATarget:
    model_id: str
    revision: str
    hidden_size: int
    layer: int
    hook_point: str = "block_output"
    pooling: str = "last"
    dtype: str = "bfloat16"
    quantization: Optional[Dict[str, Any]] = None
    final_norm: str = "checkpoint"
    metadata_layer_index_base: int = 1


@dataclass(frozen=True)
class InjectionConfig:
    prompt: str
    token: str
    token_id: int
    scale: float
    normalize: bool = True
    left_neighbor_id: Optional[int] = None
    right_neighbor_id: Optional[int] = None


@dataclass(frozen=True)
class ReconstructionConfig:
    prompt: str
    head_path: str
    readout: str = "final_hidden"
    head_weight_key: str = "weight"
    head_bias_key: Optional[str] = "bias"
    head_orientation: str = "output_input"
    add_special_tokens: bool = True
    normalize_output: bool = True
    layer: Optional[int] = None
    strip_explanation_tags: bool = False


@dataclass(frozen=True)
class NLAConfig:
    target: NLATarget
    verbalizer: ModelConfig
    reconstructor: ModelConfig
    injection: InjectionConfig
    reconstruction: ReconstructionConfig
    adapter: str = "activation_autoencoder"


@dataclass(frozen=True)
class ArmConfig:
    name: str
    communication: str
    interpretation: str = "none"


@dataclass(frozen=True)
class PanelConfig:
    arms: List[ArmConfig]
    system_prompt: str
    interpretation_templates: Dict[str, str]
    members: int = 3
    rounds: int = 2
    checkpoints: List[int] = field(default_factory=lambda: [1, 2])
    continue_instruction: str = (
        "Continue the private discussion without writing the creator-facing reply."
    )


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class GenerationConfig:
    discussion: SamplingConfig = field(default_factory=SamplingConfig)
    reply: SamplingConfig = field(default_factory=SamplingConfig)
    report: SamplingConfig = field(default_factory=SamplingConfig)
    nla: SamplingConfig = field(default_factory=SamplingConfig)
    seed: int = 42


@dataclass(frozen=True)
class RuntimeConfig:
    memory_limit_gib: float = 8.0
    minimum_free_disk_gib: float = 2.0
    maximum_context_tokens: int = 8192
    maximum_generated_tokens: int = 100000
    offline: bool = True
    native_absolute_tolerance: float = 1e-5
    worker_lock_path: Optional[str] = None


def _parse(expected, value, name):
    if expected is Any:
        return value
    origin, args = get_origin(expected), get_args(expected)
    if origin is Union:
        if value is None and type(None) in args:
            return None
        for choice in args:
            if choice is type(None):
                continue
            try:
                return _parse(choice, value, name)
            except ConfigurationError:
                pass
        raise ConfigurationError(name + " has an invalid type")
    if is_dataclass(expected):
        if not isinstance(value, dict):
            raise ConfigurationError(name + " must be an object")
        declared = {f.name for f in fields(expected)}
        extra = set(value) - declared
        if extra:
            raise ConfigurationError(name + " has unknown fields: " + ", ".join(sorted(extra)))
        hints = get_type_hints(expected)
        try:
            return expected(**{k: _parse(hints[k], v, name + "." + k) for k, v in value.items()})
        except TypeError as error:
            raise ConfigurationError(name + ": " + str(error)) from error
    if origin is list:
        if not isinstance(value, list):
            raise ConfigurationError(name + " must be a list")
        return [_parse(args[0], item, name + "[]") for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigurationError(name + " must be an object")
        return {
            _parse(args[0], k, name + ".key"): _parse(args[1], v, name + "." + str(k))
            for k, v in value.items()
        }
    if expected is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if type(value) is not expected:
        raise ConfigurationError(name + " must be " + expected.__name__)
    return value


@dataclass(frozen=True)
class MeasurementConfig:
    labels: List[str]
    creator_templates: Dict[str, str]
    inference_questions: List[str]
    reply_question: str


@dataclass(frozen=True)
class PhaseConfig:
    split: str
    seeds: List[int]


@dataclass(frozen=True)
class ProbeSettings:
    targets: List[str] = field(default_factory=lambda: ["G", "I", "B_nonmatch_I"])
    features: List[str] = field(default_factory=lambda: ["last", "suffix_mean"])
    ridge: float = 1.0
    minimum_works_per_class: int = 2
    threshold: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    model_key: str
    dataset: str
    agent: ModelConfig
    state: StateConfig
    panel: PanelConfig
    measurement: MeasurementConfig
    phases: Dict[str, PhaseConfig]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    nla: Optional[NLAConfig] = None
    nla_positions: List[str] = field(default_factory=lambda: ["direct"])
    forecast_question: Optional[str] = None
    trim_contributions: bool = False
    allow_empty_contributions: bool = False
    probes: ProbeSettings = field(default_factory=ProbeSettings)


def resolve_path(value, config_path):
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (Path(config_path).resolve().parent / path).resolve()
    )


def load_config(path):
    config = _parse(ExperimentConfig, read_json(path), "configuration")
    check(config)
    return config


def check(config):
    def ensure(ok, message):
        if not ok:
            raise ConfigurationError(message)

    ensure(
        bool(re.fullmatch(r"[a-zA-Z0-9_-]+", config.model_key)),
        "model_key must be a simple identifier",
    )
    ensure(
        config.measurement.labels == ["support", "assess"],
        "The protocol uses support and assess, in that order",
    )
    ensure(
        set(config.measurement.creator_templates) == set(config.measurement.labels),
        "One creator template per purpose is required",
    )
    ensure(
        len(config.measurement.inference_questions) == 2
        and all(config.measurement.inference_questions),
        "Two nonempty intent questions are required",
    )
    ensure(len(set(config.measurement.inference_questions)) == 2, "Intent questions must differ")
    ensure(bool(config.measurement.reply_question), "A reply question is required")
    for template in config.measurement.creator_templates.values():
        try:
            template.format(work="work", title="title", form="form")
        except (KeyError, ValueError, IndexError) as error:
            raise ConfigurationError("Invalid creator template") from error
    ensure(
        config.panel.members == 3
        and config.panel.rounds == 2
        and config.panel.checkpoints == [1, 2],
        "This protocol measures three members at C0, C1 and C2",
    )
    expected = {c + "." + i for c in ("peers", "self") for i in ("aligned", "opposing")}
    ensure(
        len(config.panel.arms) == 4 and {a.name for a in config.panel.arms} == expected,
        "All four paired arms are required",
    )
    for arm in config.panel.arms:
        ensure(
            arm.name == arm.communication + "." + arm.interpretation,
            "Arm names must match their conditions; creator requests stay fixed",
        )
    ensure(
        set(config.panel.interpretation_templates) == set(config.measurement.labels),
        "Peer framings must cover both purposes",
    )
    ensure(
        config.state.layer > 0
        and config.state.hook_point == "block_output"
        and config.state.pooling == "last",
        "State layers are one-based decoder block outputs with last-token pooling",
    )
    ensure(
        config.state.position == "bridge_last_content",
        "Capture the actual contribution before the measurement questions",
    )
    ensure(set(config.phases) == {"training", "held_out"}, "Declare training and held_out phases")
    ensure(
        len({p.split for p in config.phases.values()}) == 2, "Training and test splits must differ"
    )
    for phase in config.phases.values():
        ensure(
            phase.seeds
            and len(set(phase.seeds)) == len(phase.seeds)
            and all(0 <= n < 2**31 for n in phase.seeds),
            "Each phase needs distinct nonnegative seeds",
        )
    ensure(
        not set(config.phases["training"].seeds) & set(config.phases["held_out"].seeds),
        "Training and test seeds must differ",
    )
    for sampling in (
        config.generation.discussion,
        config.generation.reply,
        config.generation.report,
        config.generation.nla,
    ):
        ensure(
            sampling.max_tokens > 0
            and math.isfinite(sampling.temperature)
            and sampling.temperature >= 0
            and 0 < sampling.top_p <= 1,
            "Invalid sampling settings",
        )
    ensure(
        config.runtime.offline
        and config.runtime.maximum_context_tokens > 0
        and config.runtime.maximum_generated_tokens > 0,
        "Offline inference and positive token budgets are required",
    )
    ensure(
        config.runtime.memory_limit_gib > 0 and config.runtime.minimum_free_disk_gib >= 0,
        "Invalid resource limits",
    )
    ensure(
        set(config.probes.targets) <= {"G", "I", "B_nonmatch_I"} and config.probes.targets,
        "Unknown probe targets",
    )
    ensure(
        set(config.probes.features) <= {"last", "suffix_mean"} and config.probes.features,
        "Unknown probe features",
    )
    ensure(
        config.probes.ridge > 0
        and config.probes.minimum_works_per_class >= 2
        and config.probes.threshold == 0,
        "Use positive ridge, two training works per class and threshold zero",
    )
    specs = [config.agent] + (
        [config.nla.verbalizer, config.nla.reconstructor] if config.nla else []
    )
    for model in specs:
        ensure(
            model.path and model.model_id and re.fullmatch(r"[a-f0-9]{40}", model.revision),
            "Models require a local path, model ID and pinned commit SHA",
        )
        ensure(
            model.backend == "transformers" and re.fullmatch(r"cpu|cuda(?::[0-9]+)?", model.device),
            "Use the local Transformers backend on cpu or CUDA",
        )
        ensure(
            model.dtype in ("float32", "float16", "bfloat16")
            and model.final_norm in ("checkpoint", "identity"),
            "Unsupported numerical layout",
        )
        ensure(
            model.language_model_head in ("checkpoint", "absent"),
            "Unsupported language-model head layout",
        )
        ensure(
            not set(model.tokenizer_options) & {"token", "trust_remote_code", "local_files_only"},
            "Tokenizer options cannot replace offline settings",
        )
    ensure(
        config.agent.language_model_head == "checkpoint",
        "The agent must have a language-model head",
    )
    if config.nla:
        nla = config.nla
        ensure(
            (nla.target.model_id, nla.target.revision, nla.target.layer, nla.target.hook_point)
            == (
                config.agent.model_id,
                config.agent.revision,
                config.state.layer,
                config.state.hook_point,
            ),
            "NLA training target must match the agent and extraction layer",
        )
        ensure(
            nla.target.metadata_layer_index_base in (0, 1) and nla.target.pooling == "last",
            "Invalid NLA layer convention",
        )
        ensure(
            nla.injection.scale > 0 and nla.injection.token_id >= 0,
            "Invalid AV injection parameters",
        )
        ensure(nla.reconstruction.readout in ("final_hidden", "block_output"), "Invalid AR readout")
        ensure(
            nla.reconstruction.head_orientation in ("output_input", "input_output"),
            "Invalid AR head orientation",
        )
        ensure(
            nla.reconstruction.readout != "block_output"
            or (nla.reconstruction.layer is not None and nla.reconstruction.layer > 0),
            "AR block readout requires a layer",
        )
        ensure(
            set(config.nla_positions) <= {"source", "direct", "indirect"} and config.nla_positions,
            "Declare NLA exposure positions",
        )
