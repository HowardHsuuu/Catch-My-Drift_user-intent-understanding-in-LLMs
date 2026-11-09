"""Resolve local checkpoint contracts before allocating model memory."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .config import ExperimentConfig, ModelConfig, resolve_path
from .errors import ConfigurationError
from .serialization import digest, identity, read_json


@dataclass(frozen=True)
class ModelShape:
    model_type: str
    hidden_size: int
    layers: int
    vocabulary: int
    maximum_positions: Optional[int]
    quantization: Optional[Dict[str, Any]]


def text_configuration(raw):
    """Resolve Gemma's omitted defaults from the pinned Transformers schema."""
    if raw.get("model_type") in ("gemma3", "gemma3_text"):
        from transformers import AutoConfig

        values = dict(raw)
        family = values.pop("model_type")
        config = AutoConfig.for_model(family, **values)
        return config.get_text_config().to_dict()
    return raw


def inspect_model(spec: ModelConfig, config_path):
    path = resolve_path(spec.path, config_path)
    if not path.is_dir():
        raise ConfigurationError("Local model directory is missing: " + str(path))
    marker = path / "source.json"
    if marker.exists() and read_json(marker) != dict(
        model_id=spec.model_id, revision=spec.revision
    ):
        raise ConfigurationError(
            "Local checkpoint provenance differs from the configured model revision"
        )
    raw = read_json(path / "config.json")
    # These aliases describe metadata, not an assertion that all model families
    # implement the same runtime hooks. Runtime capability checks are separate.
    text = text_configuration(raw)
    hidden = text.get("hidden_size", text.get("n_embd", text.get("d_model")))
    layers = text.get("num_hidden_layers", text.get("n_layer", text.get("n_layers")))
    vocabulary = text.get("vocab_size")
    if not all(type(v) is int and v > 0 for v in (hidden, layers, vocabulary)):
        raise ConfigurationError(
            "Model must declare positive hidden size, layer count and vocabulary"
        )
    shape = ModelShape(
        str(raw.get("model_type", "")),
        hidden,
        layers,
        vocabulary,
        text.get("max_position_embeddings", text.get("n_positions")),
        raw.get("quantization", raw.get("quantization_config")),
    )
    from .backends import validate_capability

    validate_capability(spec, shape)
    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise ConfigurationError("Only local safetensors checkpoints are supported: " + str(path))
    return path, raw, shape


def model_inventory(spec, config_path, hash_weights=True):
    path, _, shape = inspect_model(spec, config_path)
    names = set()
    for pattern in ("*.json", "*.yaml", "*.model", "*.txt", "*.jinja", "*.safetensors"):
        names.update(p for p in path.glob(pattern) if p.is_file())
    files = {}
    for item in sorted(names):
        weight = item.suffix == ".safetensors"
        files[item.name] = dict(
            bytes=item.stat().st_size, sha256=digest(item) if hash_weights or not weight else None
        )
    return dict(
        spec=asdict(spec),
        path=str(path),
        shape=asdict(shape),
        files=files,
        fingerprint=identity(dict(shape=asdict(shape), files=files, model=asdict(spec))),
    )


def resolve_contract(config: ExperimentConfig, config_path):
    path, _, shape = inspect_model(config.agent, config_path)
    if config.state.layer > shape.layers:
        raise ConfigurationError("Captured layer exceeds the agent's layer count")
    if shape.maximum_positions and config.runtime.maximum_context_tokens > shape.maximum_positions:
        raise ConfigurationError("Configured context budget exceeds checkpoint positions")
    result = dict(agent_path=str(path), agent_shape=asdict(shape), nla=None)
    if config.nla is None:
        return result
    nla = config.nla
    if nla.target.hidden_size != shape.hidden_size or nla.target.quantization != shape.quantization:
        raise ConfigurationError(
            "NLA training target dimension or quantization differs from the agent checkpoint"
        )
    avpath, _, avshape = inspect_model(nla.verbalizer, config_path)
    arpath, _, arshape = inspect_model(nla.reconstructor, config_path)
    for role_path in (avpath, arpath):
        sidecar = role_path / "nla_meta.yaml"
        if sidecar.exists():
            import yaml

            meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            known = {
                "base_model": nla.target.model_id,
                "d_model": nla.target.hidden_size,
                "extraction_layer_index": nla.target.layer
                - (1 - nla.target.metadata_layer_index_base),
            }
            for key, expected in known.items():
                if key in meta and meta[key] != expected:
                    raise ConfigurationError(
                        "NLA target contradicts checkpoint sidecar field: " + key
                    )
    if avshape.hidden_size != nla.target.hidden_size:
        raise ConfigurationError(
            "Direct embedding injection requires matching verbalizer and target dimensions"
        )
    if nla.reconstruction.layer is not None and not 1 <= nla.reconstruction.layer <= arshape.layers:
        raise ConfigurationError("AR readout layer exceeds the reconstructor checkpoint")
    head = resolve_path(nla.reconstruction.head_path, config_path)
    if not head.is_file():
        raise ConfigurationError("AR value head is missing: " + str(head))
    from safetensors import safe_open

    with safe_open(str(head), framework="np") as archive:
        expected = {nla.reconstruction.head_weight_key}
        if nla.reconstruction.head_bias_key is not None:
            expected.add(nla.reconstruction.head_bias_key)
        if set(archive.keys()) != expected:
            raise ConfigurationError("AR head keys differ from the declared weight/bias contract")
        dimensions = archive.get_slice(nla.reconstruction.head_weight_key).get_shape()
        target = [shape.hidden_size, arshape.hidden_size]
        if nla.reconstruction.head_orientation == "input_output":
            target.reverse()
        if dimensions != target:
            raise ConfigurationError(
                "AR head shape differs from target and reconstructor dimensions"
            )
        if nla.reconstruction.head_bias_key is not None:
            if archive.get_slice(nla.reconstruction.head_bias_key).get_shape() != [
                shape.hidden_size
            ]:
                raise ConfigurationError("AR bias dimension differs from target dimension")
    result["nla"] = dict(
        verbalizer_path=str(avpath),
        reconstructor_path=str(arpath),
        verbalizer_shape=asdict(avshape),
        reconstructor_shape=asdict(arshape),
        head_path=str(head),
        head_sha256=digest(head),
        target=asdict(nla.target),
    )
    return result
