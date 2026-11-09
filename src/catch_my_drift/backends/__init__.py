"""Load the local Transformers runtime only when inference is requested."""

import re
from ..errors import ConfigurationError


def validate_capability(spec, shape):
    if spec.backend != "transformers":
        raise ConfigurationError("Only the Transformers backend is supported")
    if shape.model_type not in ("qwen2", "llama", "gemma3", "gemma3_text"):
        raise ConfigurationError("Unsupported decoder family: " + shape.model_type)
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", spec.device):
        raise ConfigurationError("Use cpu or one CUDA device")
    if shape.quantization is not None:
        raise ConfigurationError("This runtime requires unquantized safetensors weights")


def load_backend(spec, runtime, config_path):
    from .torch_backend import TransformersBackend

    return TransformersBackend(spec, runtime, config_path)
