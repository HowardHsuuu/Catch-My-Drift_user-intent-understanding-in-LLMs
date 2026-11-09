"""Explicit boundaries between invalid configuration and failed measurement."""


class ConfigurationError(ValueError):
    """The requested experiment is ambiguous, incompatible or unsupported."""


class IntegrityError(RuntimeError):
    """An input, event or stage no longer matches its recorded provenance."""


class StageError(RuntimeError):
    """The requested stage has not satisfied its prerequisites."""


class ResourceError(RuntimeError):
    """The local resource budget cannot safely admit the requested job."""
