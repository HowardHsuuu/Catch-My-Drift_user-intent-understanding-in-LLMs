# Experiment protocol

## Task and comparison

A creator shares a synthetic poem, microfiction, design, or project. The explicit request is either **support**—encouragement without criticism or suggestions—or **assess**—craft assessment with concrete improvement advice. The request stays visible and unchanged.

Three instances of the same model discuss how to respond. They have equal status and no assigned specialist roles. Every member is measured; “source,” “direct,” and “indirect” describe exposure positions.

At **C0**, members generate their initial contributions. At **C1**, one incoming source contribution is replaced for the directly exposed recipient by an aligned or opposing interpretation of the request. At **C2**, members either receive new peer contributions or reconsider their own contribution. Both continuations share the exact C1 state and seed. The self condition has already seen the first panel exchange: this is a comparison of continued exchange, not a pure agent-count experiment.

Each test cohort has 12 works × 2 requests × 2 seeds = 48 cases. A case has three baseline measurements and 24 post-exposure measurements across four arms: 27 logical contexts. Shared physical states are reused rather than independently resampled. This is a short panel, not a long-horizon experiment.

## Measurements

| Symbol | Measurement | Interpretation |
| --- | --- | --- |
| G | Explicit creator request | Known experimental request |
| I | Agreement between two neutral questions about what the creator wants | Reported interpretation |
| P | Response approach expressed in the discussion contribution | Stated approach |
| S | Explicit forecast of the next response style; Gemma only | Self-forecast |
| B | Separately prompted creator-facing reply | Measured behavior |

I, S, and B each start from the same pre-question prefix and never enter later discussion histories. B includes a reminder to follow the creator's preference. Thus a B outcome is a separately prompted reply, not an unprompted continuation or proof of commitment.

Contribution activations are captured before any measurement question. Probe features are the final contribution token and the mean of its final three tokens. NLAs describe three adjacent central contribution tokens without receiving their source text. Two agreeing known descriptions and no contrary known description are required for an NLA purpose verdict. Every reconstruction is retained; reconstruction quality is not used to select examples.

## Labels and probes

Semantic review separates purpose from audience and keeps mixed, unclear, unfinished, or wrong-audience outputs unresolved. NLA review separately judges an explicitly attributed creator expectation and the response purpose described by the explanation. Neither field is automatically a label of internal understanding.

The Gemma probes predict G, I, or the binary event B ≠ I. Each target has last-token and suffix-mean variants. Training-only mean centering and row normalization precede weighted ridge regression with an intercept, ridge 1, and threshold 0. Weighting balances works and classes; I balances available G/I cells and works within them. A required class needs two independent training works. Missing classes produce an unavailable readout, not a replacement fit.

The Qwen 7B readouts are the original, previously trained I comparators. They are not equivalent to Gemma's separately trained future-mismatch probes. The cohorts therefore do not support a uniform cross-model ranking of instruments.

## Analysis

The primary endpoint is final B/G departure in the directly exposed member under opposing framing, continued peers minus self. Unresolved replies contribute bounds over their possible outcomes. Shared physical states are tied when computing these bounds. Bootstrap resampling uses whole works, 10,000 draws, and seed 8161. The three Qwen comparisons use 98.33% intervals; Gemma uses 95%.

Repair, worsening, and the other members' outcomes are secondary. The first-exposure I/P partition and conditional NLA sensitivity are exploratory. A correct I with a changed P is a clue about the stated approach; correct I and P with an inconsistent B cannot distinguish an unreported goal change from execution failure. No activation intervention or independent capability manipulation identifies these causes here.
