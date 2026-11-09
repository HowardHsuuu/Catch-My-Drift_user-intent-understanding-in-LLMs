# Included data and readouts

`src/catch_my_drift/assets/paper_rows.jsonl.gz` contains 5,184 logical measurement rows: 1,296 each for Qwen 3B, 7B, 14B, and Gemma 12B. It contains scientific labels and saved readout predictions, rather than execution logs. It has no machine paths, credentials, model weights, or operational notes. `paper.json` records its checksum and the original multiple-comparison settings.

| Fields | Meaning |
| --- | --- |
| `model`, `work_id`, `case_id`, `seed` | Cohort, synthetic work, paired request/seed case |
| `arm`, `checkpoint`, `member`, `position` | Discussion condition, C0/C1/C2, panel member, exposure position |
| `capture_id` | Opaque identity of a shared physical state |
| `current` | G, the explicit request: `support` or `assess` |
| `inference_answers`, `inferred_report` | Two labeled reports and their unanimous I verdict |
| `plan_purpose`, `reply_purpose`, `self_forecast` | P, B, and optional S labels |
| `NLA`, `NLA_expectation` | Described response purpose and separate creator-expectation field |
| `last_probe`, `mean_probe` | Original Qwen 7B I comparator verdicts |
| `probes` | Gemma head names, saved scores, and Boolean predictions |

Unavailable measurements are `null`. A Gemma G/I probe Boolean means `assess` when true and `support` when false. A `B_nonmatch_I` Boolean directly predicts whether B will differ from I. Do not treat it as a purpose label or compare it with G by changing its interpretation.

Several logical endpoints can share the same `capture_id` and request. They must remain tied when computing uncertainty and must be deduplicated within a monitoring population. Predictions are propagated from the saved canonical physical state to its logical aliases. Do not count the aliases as independent measurements.

`catch-my-drift reproduce` recalculates the primary contrasts, intervals, initial departure profiles, pooled repair/worsening counts, and G/I-reference confusion matrices. Detector accuracy and the always-match baseline use the same scored subset. Detection recall over all known departures includes detector abstentions in its denominator. Unknown reference or behavior labels are counted separately.

The files in `weights/` are small linear readouts, not language-model checkpoints. Each bundle identifies its target model revision and extraction layer and hashes its arrays. Qwen heads are preexisting I comparators without an intercept. Gemma heads use a fitted intercept and training-only centering. Exact training traces and activation files are not bundled with this lightweight release; the public training pipeline can create a new bundle from reviewed development runs.

All creative works are synthetic. Labels were assigned by one AI reviewer with textual evidence; they are not independent human validation or psychological ground truth. The high request/I decoding accuracy in Gemma does not demonstrate understanding beyond request retention, because its known I labels agree with G.
