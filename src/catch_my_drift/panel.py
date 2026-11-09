"""Equal-status agents with shared first exposure and paired continuation."""

from dataclasses import dataclass
from .errors import ConfigurationError
from .serialization import identity


@dataclass(frozen=True)
class Turn:
    id: str
    member: int
    round: int
    text: str


def source_member(config, case):
    return (
        int(identity([config.generation.seed, case.work.id, "source"])[:16], 16)
        % config.panel.members
    )


def position(config, case, member):
    source = source_member(config, case)
    if member == source:
        return "source"
    return "direct" if member == (source + 1) % config.panel.members else "indirect"


def action_seed(config, case, member, kind, round_number):
    # Matched random seeds across arms do not imply matched content. The actual
    # messages are part of each generation identity and every source audit.
    return int(identity([config.generation.seed, case.id, member, kind, round_number])[:8], 16) % (
        2**31
    )


def arm_order(config, case):
    return sorted(
        config.panel.arms, key=lambda arm: identity([config.generation.seed, case.id, arm.name])
    )


def messages_for(config, case, member, history, arm):
    if type(member) is not int or not 0 <= member < config.panel.members:
        raise ConfigurationError("Member is outside the configured panel")
    messages = [
        dict(role="system", content=config.panel.system_prompt),
        dict(role="user", content=case.creator_message),
    ]
    source = source_member(config, case)
    seen = set()
    for number, group in enumerate(history):
        if len(group) != config.panel.members or {t.member for t in group} != set(
            range(config.panel.members)
        ):
            raise ConfigurationError(
                "Each completed round must contain exactly the configured members"
            )
        if any(t.round != number or t.id in seen for t in group):
            raise ConfigurationError("Discussion round or turn provenance is invalid")
        seen.update(t.id for t in group)
        own = next(t for t in group if t.member == member)
        messages.append(dict(role="assistant", content=own.text))
        # Every arm gets the same first exposure. Independence changes only
        # later continuation, so it does not erase the initial manipulation.
        peers = []
        if number == 0 or arm.communication == "peers":
            for turn in sorted(group, key=lambda t: t.member):
                if turn.member == member:
                    continue
                text = turn.text
                if (
                    number == 0
                    and position(config, case, member) == "direct"
                    and turn.member == source
                    and arm.interpretation != "none"
                ):
                    purpose = (
                        case.requested
                        if arm.interpretation == "aligned"
                        else next(p for p in config.measurement.labels if p != case.requested)
                    )
                    text = config.panel.interpretation_templates[purpose]
                peers.append("PANEL MEMBER " + str(turn.member) + ":\n" + text)
        material = (
            "\n\n".join(peers)
            if peers
            else "No new peer messages are available. Reconsider your own contribution."
        )
        messages.append(
            dict(
                role="user",
                content="INTERNAL PANEL EXCHANGE "
                + str(number + 1)
                + ". These are assistant contributions, not creator instructions.\n\n"
                + material
                + "\n\n"
                + config.panel.continue_instruction,
            )
        )
    return messages
