"""Synthetic works and paired creator requests."""

from dataclasses import dataclass
from .config import resolve_path
from .errors import ConfigurationError
from .serialization import identity, read_json


@dataclass(frozen=True)
class Work:
    id: str
    title: str
    text: str
    form: str
    split: str

    @property
    def content(self):
        return self.title + "\n" + self.text


@dataclass(frozen=True)
class Case:
    id: str
    work: Work
    requested: str
    creator_message: str


def load_works(config, config_path):
    document = read_json(resolve_path(config.dataset, config_path))
    if not document.get("synthetic") or not isinstance(document.get("works"), list):
        raise ConfigurationError("The dataset must declare synthetic works")
    works, ids, texts = [], set(), set()
    for value in document["works"]:
        if set(value) != {"id", "title", "text", "form", "split"} or not all(
            isinstance(x, str) and x.strip() for x in value.values()
        ):
            raise ConfigurationError("Each work requires nonempty id, title, text, form and split")
        fingerprint = identity(value["text"].strip())
        if value["id"] in ids or fingerprint in texts:
            raise ConfigurationError("Duplicate work ID or content, including across splits")
        ids.add(value["id"])
        texts.add(fingerprint)
        works.append(Work(**value))
    splits = {p.split for p in config.phases.values()}
    if {w.split for w in works} != splits:
        raise ConfigurationError("Dataset splits must match the declared phases")
    return works


def case_for(config, work, purpose):
    if purpose not in config.measurement.labels:
        raise ConfigurationError("Unknown creator purpose")
    message = config.measurement.creator_templates[purpose].format(
        work=work.content, title=work.title, form=work.form
    )
    return Case(work.id + ":" + purpose, work, purpose, message)
