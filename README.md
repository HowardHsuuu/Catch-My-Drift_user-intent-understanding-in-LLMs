# Catch My Drift

**Investigating Intent Understanding and Drift in Multi-Agent Panels**

Chih-Hao Howard Hsu · Independent Researcher · Swarm Alignment Hackathon 2026

When an agent panel gives a reply that conflicts with a user's request, has its interpretation changed, has its stated approach shifted, or has execution failed? This project compares peer discussion with self-reconsideration and examines what activation probes, natural language autoencoders (NLAs), and intent reports can detect.

[Experiment protocol](docs/protocol.md) · [Run new experiments](docs/experiments.md)

## Findings

- **Peer discussion can help repair replies.** Qwen 7B ends with 11/48 departures under continued peer exchange versus 23/48 under self-reconsideration. Gemma 12B ends with 15/48 in both; its uncertainty interval does not establish equivalence.
- **Departures have different observable patterns.** At first opposing exposure, 11 Qwen 7B departures accompany an incorrect intent report. In Gemma, 20 departures preserve both the reported expectation and stated response approach.
- **Readouts have different blind spots.** Qwen NLA flags 15/18 final departures with an incorrect intent report, but 1/7 with a correct report. Gemma's future-mismatch probe flags 18/29 reply–report mismatches, with 21 false alarms.

These are observations about short, controlled panels. Reports and activation readouts do not establish internal understanding, psychological commitment, or the cause of a failure.

## Reproduce the results without a GPU

Python 3.9 or newer is required. The default installation does not install PyTorch or download models.

```bash
git clone https://github.com/HowardHsuuu/Catch-My-Drift_user-intent-understanding-in-LLMs.git
cd Catch-My-Drift_user-intent-understanding-in-LLMs
python -m venv .venv
source .venv/bin/activate
python -m pip install .
catch-my-drift reproduce --out outputs/paper-results
```

The command recomputes the four models' primary effects, work-bootstrap intervals, departure profiles, and detector confusion counts from 5,184 included measurement rows. It writes `summary.json`, `primary.csv`, and `monitoring.csv`. No model calls or new fitting occur. Use a new output folder for each run.

## Run a panel experiment

Install the inference extra and validate a preset:

```bash
python -m pip install '.[inference]'
catch-my-drift validate --config configs/gemma12b.json
```

Presets cover Qwen2.5 3B, 7B, 14B, and Gemma 3 12B IT. Inference uses local, unquantized safetensors checkpoints on one CUDA device or CPU. Model downloads are a separate, explicit command; all inference stages run offline. Follow the [experiment guide](docs/experiments.md) for downloads, runtime checks, collection, semantic review, probe fitting, and analysis.

The Qwen 7B and Gemma 12B presets include matched released NLAs. Qwen 3B/14B have no substituted NLA or probe bundle. Switching the agent alone does not make an NLA compatible: the target model, revision, layer, and dimensions must match.

## What is included

| Path | Contents |
| --- | --- |
| `src/catch_my_drift/` | Collection, NLA, review, probe fitting, and analysis code |
| `configs/` | Pinned model and experiment settings |
| `data/works.json` | Eight training and twelve test works, all synthetic |
| `weights/` | Small fitted readout arrays for Qwen 7B and Gemma 12B |

The [data guide](docs/data.md) explains the included measurement rows, labels, shared states, and uncertainty calculations. Semantic labels were supplied by an AI reviewer; there is no independent human semantic ground truth.

The panel framework and original Qwen 7B probes predate the sprint. The reported model-size comparison, Gemma extension, and diagnostic analyses were conducted for the sprint. This public runner consolidates the experiment scripts; the included measurements preserve the reported results. Fresh inference can vary with hardware and numerical libraries.

## License

Code, synthetic tasks, and readout coefficients are released under the [MIT license](LICENSE). Downloaded language models and NLAs retain their original licenses and access requirements.
