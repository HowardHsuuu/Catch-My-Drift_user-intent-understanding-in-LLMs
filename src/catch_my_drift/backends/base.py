"""Backend boundary for local generation, raw states and private perturbations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class PrefillState:
    cache: Any
    vector_before: np.ndarray
    vector_after: np.ndarray
    metadata: Dict[str, Any]
    positions_before: np.ndarray
    positions_after: np.ndarray


class ModelBackend(ABC):
    """Implementations must isolate caches and never persist modified weights."""

    @abstractmethod
    def encode_messages(self, messages, add_generation_prompt=True):
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages, sampling, seed):
        raise NotImplementedError

    @abstractmethod
    def prefill(self, frame, state_spec, delta=None):
        raise NotImplementedError

    @abstractmethod
    def branch(self, messages, question, state_spec, sampling, seed, delta=None):
        raise NotImplementedError

    @abstractmethod
    def verbalize(self, activation, injection, sampling, seed):
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, text, reconstruction):
        raise NotImplementedError

    @abstractmethod
    def technical_check(self, state_spec):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
