# Running experiments

Run commands from the repository root after installing `.[inference]`. The presets reproduce the task, prompts, seeds, layers, and sampling settings. Read the [protocol](protocol.md) before changing them.

## 1. Prepare local checkpoints

Validation needs no model files or GPU:

```bash
catch-my-drift validate --config configs/gemma12b.json
```

Downloading is explicit and may require substantial disk space. Gemma access requires accepting its model terms and authenticating with Hugging Face, for example through `hf auth login`.

```bash
catch-my-drift download --config configs/gemma12b.json --include-nla
catch-my-drift doctor --config configs/gemma12b.json --include-nla --runtime
```

The doctor command checks local checkpoint layouts, NLA target/head compatibility, and neutral native-prefix/cache correspondence. These runtime checks do not validate an intent decoder's semantics. Models are loaded one stage at a time. A worker lock prevents overlapping jobs, and an occupied CUDA device is rejected.

The presets use BF16 and one CUDA device. Adjust `agent.device` and the NLA devices consistently if selecting another device. The agent, KV cache, and temporary tensors must fit the declared memory limit. CPU inference is supported but is expensive for these checkpoints. Models and generated runs belong in ignored `models/` and `runs/` folders.

## 2. Collect and label training data

For Gemma's six probe heads:

```bash
catch-my-drift collect --config configs/gemma12b.json --phase training --out runs/gemma-training
catch-my-drift review prepare --run runs/gemma-training
catch-my-drift review show --run runs/gemma-training --offset 0 --limit 16
```

The review folder contains condition-hidden `items.json` and `labels-template.json`. Copy the template to a working file, fill every decision, and submit that file. Use `review show` with successive offsets to inspect manageable slices. Label the text itself, not its presumed experimental condition. Do not infer labels from injected framing, keywords, or probe predictions.

Each decision has these fields:

| Field | Required value |
| --- | --- |
| `id` | Unchanged item identifier |
| `mode` | `support`, `assess`, `mixed`, or `unclear` |
| `evidence` | Exact excerpt from the item; empty only for empty output |
| `audience` | For replies: `true` if directly addressing the creator, `false` otherwise, or `null` if unclear; `null` for other item types |
| `described_response_purpose` | For NLA descriptions: the same four purpose categories; `null` for other types |

For a report, judge what response the creator is said to want. For a plan or forecast, judge the operative response approach. For a reply, distinguish encouragement from actual criticism or improvement advice and judge its audience. For NLA, `mode` requires explicit attribution of a creator expectation; `described_response_purpose` concerns the response described, even without that attribution. Do not fill missing information from surrounding conditions. “Mixed” means both incompatible purposes are expressed; “unclear” means no definite purpose can be assigned.

```bash
catch-my-drift review submit --run runs/gemma-training --labels my-training-labels.json
catch-my-drift fit --run runs/gemma-training --out weights/new-gemma
```

Fitting rejects held-out runs. Complete the fit before inspecting test labels. You may instead use the included `weights/gemma12b` bundle to evaluate the released heads on a fresh matching-model run.

## 3. Collect held-out data and run NLA

```bash
catch-my-drift collect --config configs/gemma12b.json --phase held_out --out runs/gemma-test
catch-my-drift nla --run runs/gemma-test --stage av
catch-my-drift nla --run runs/gemma-test --stage ar
catch-my-drift review prepare --run runs/gemma-test --weights weights/new-gemma
catch-my-drift review show --run runs/gemma-test --offset 0 --limit 16
```

Fill the new review template, then submit and analyze:

```bash
catch-my-drift review submit --run runs/gemma-test --labels my-test-labels.json
catch-my-drift analyze --run runs/gemma-test --weights weights/new-gemma --out outputs/gemma-test
```

The review packet fixes the probe-bundle hash before test annotation. Analysis rejects a different or newly substituted bundle. Analysis writes scored rows and a summary; it does not fit or tune anything.

## Other presets

For Qwen 7B, use `configs/qwen7b.json`, skip new training, and pass `--weights weights/qwen7b` when preparing test review and analyzing. Use `--family-size 3` in analysis to retain the paper's correction across Qwen sizes.

For Qwen 3B and 14B, omit NLA and weights commands. Their presets collect reports and behavior but do not pretend that another model's NLA or probes are compatible. Their role in the paper is the behavioral size comparison.

Output directories are created exclusively. A failed stage is kept for diagnosis rather than overwritten or silently restarted. Use a new run directory after resolving a failure. Configuration and event hashes detect altered inputs; raw activations use non-pickled NumPy arrays.
