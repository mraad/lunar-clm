# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Tests (stdlib only; no model weights or optional deps needed)
python3 -m unittest discover -s tests -v
python3 -m unittest tests.test_lunar.LunarTests.test_seed_and_gravity   # single test
node tests/test_replay.cjs                                              # replay + interactive page logic

# Interactive app: pick start/tilt/pad in the browser, flights stream live
.venv/bin/python -m lunar_clm.serve [--pilot baseline|small|...] [--port 8000]

# Run a flight (default pilot is `small`; needs .venv with [small] extras + models/)
.venv/bin/python -m lunar_clm --pilot baseline            # no model, instant physics demo
.venv/bin/python -m lunar_clm --pilot small --seed 4000 --out dist/small.json --replay dist/small.html
.venv/bin/python -m lunar_clm --pilot clm --seed 3000     # needs clm-serve on :8700 (see README)

# Train / evaluate / benchmark the small model
.venv/bin/python -m lunar_clm.train_small [--reuse-embeddings] [--epochs N]
.venv/bin/python -m lunar_clm.evaluate_small              # held-out seeds 4000+, writes results/small-flights.json
.venv/bin/python -m lunar_clm.benchmark --backend baseline|clm|small|laya --out results/X.json
```

Environment: `uv venv --python 3.12 && uv pip install -e '.[small,train]'`. Encoder download and
pinned revision are in README "Trained small CLM". Type-check with `pyright lunar_clm tests`; the
existing errors are all missing optional imports (mlx, torch, lunar_laya) and Optional narrowing.

## Architecture

Import layering, bottom to top. Keep it acyclic in this order:

```
game.py  →  guidance.py  →  small.py  →  pilot.py  →  cli.py  →  serve.py / benchmark.py / evaluate_small.py / train_small.py
```

- **`pyproject` declares `dependencies = []` on purpose.** `small.py` imports numpy/mlx *inside*
  functions so `pilot.py` can import it at module level and baseline/clm modes plus the whole
  test suite run with stdlib only. Do not hoist those imports.
- **Physics, `QUESTIONS` text, prompt formats and the recording format keep parity with
  `../lunar-laya`** (commit in README). The terrain and guidance no longer do: pads sit on hills
  and guidance clears the highest terrain between lander and pad (`game.ceiling`). Everything in
  `results/` predates that change. `benchmark --backend laya` refuses to run while Laya's
  `PADS`/`TERRAIN` differ, because Laya's pilot computes guidance from its own world.
- **Custom starts go through `game.custom_start`** (wall margin, clearance above terrain, height
  cap, tilt wrap). `serve.py` validates with it; `replay.html` mirrors the same limits from the
  `start` block the server injects, and its `ground`/`lowest` mirror `game.ground`/`ceiling`.
  Change the Python rule and those JS mirrors together.

### Two model "dialects"

| | full (`clm`, `assisted`) | small (`small`, `small-assisted`) |
|---|---|---|
| prompt | `guidance.observation` | `small.short_observation` |
| questions | `QUESTIONS`: two choices (rotation, engine) | `QUESTION`: one choice over nine `turn/power` `ACTIONS` |
| agent | `CLMClient` → HTTP `/v1/systemone` on clm-serve | `SmallAgent`: MLX Qwen3-Embedding-0.6B + NumPy heads, in-process |

`Pilot.__init__` binds `observe`/`questions`/`agent` from `mode` (`MODES` in `pilot.py`; cli
`--pilot choices` reuse it). `*-assisted` modes execute guidance whenever the model disagrees and
record `intervened`; plain modes execute the model's choice, errors included. Invalid model output
raises, never silently falls back (`validate_answers`).

### Decision and recording schema

`Pilot.decide` returns `(Command, decision)` where decision carries `reference` (guidance),
`proposed` (model), `executed`, `intervened`, `answers`, `prompt`, `latency_ms`. `cli.run_episode`
derives `guidance_matches`, `interventions`, p50/p95 and `wall_seconds` from these frames; do not
recompute guidance downstream. `new_recording` is the shared `schema_version: 1` envelope used by
both `cli` and `evaluate_small`. `export_replay` injects the JSON into `replay.html` at the
`/* RECORDING */null` marker with `<` escaped; `replay.html` is a dependency-free canvas page.

### Interactive app

`serve.make_server` injects `new_recording(pilot)` plus `interactive: true` and the start limits
into the same `replay.html`; the page switches to setup mode on that flag. `POST /fly` with
`{x, y, angle, target}` streams NDJSON, one `{"frame": ...}` per decision via
`run_episode(..., start=, on_frame=)`, then `{"summary": ...}` or `{"error": ...}`. One
single-threaded `HTTPServer` bound to 127.0.0.1: one flight at a time, pilot loaded once. The page
text is built at startup, so restart the server after editing `replay.html`.

### Small-model checkpoint contract

`models/lunar-small/head.npz` + `head.json`. `SmallAgent` refuses to load unless `head.json`
`actions`/`action_texts` equal the current `ACTIONS`/`ACTION_TEXTS` order and `head_sha256` matches.
`encoder_path` is relative to the head directory. `train_small.train` asserts the NumPy inference
path (`small.project`) reproduces the torch logits before writing the checkpoint; keep that check.
`train_seeds`/`validation_seeds` in metadata are derived from the dataset and consumed by
`evaluate_small` to refuse overlapping evaluation seeds.

### Seed discipline

| range | use |
|---|---|
| 1000–1031 | training flights (`train_small.dataset`) |
| 2000–2007 | validation (model selection uses validation loss only) |
| 3000–3002 (+ `seed-100` warmup) | benchmark corpus, 96 distinct states across all pads |
| 4000+ | held-out evaluation; never used for selection |

### Benchmark caveats

`benchmark.py` times each `pilot.decide` with its own clock over unique states and raises on
duplicate prompts: clm-serve caches states, so restart it between runs and never time a replay.
Run backends sequentially. The `laya` backend imports `../lunar-laya` via `--laya-root` and must be
run with that repo's venv; it asserts Laya's `QUESTIONS` equal ours.
