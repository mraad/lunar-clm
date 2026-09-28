# Lunar CLM

A small lunar lander driven by [Contrastive Language Models](https://github.com/Contrastive-LM/CLM).
Python simulates a flight; a standalone canvas page replays it with controls,
fuel, model probabilities, latency, and any guidance overrides. No frontend
framework, build step, CDN, or Python dependencies in the game client.

The physics, guidance, two typed questions, and recording format come from
`../lunar-laya` at `74e66c202a568a4ee6d32fe440d6f34653c05c38`.
CLM replaces Laya inference. The game runs independently of that checkout;
only the optional Laya benchmark imports it.

## How it works

Every control step runs the same pipeline, whichever pilot is flying:

1. **Simulate.** `Game` holds the lander state. Each decision advances ten
   0.02 s physics substeps, so one decision covers 0.2 simulation seconds.
2. **Guide.** A deterministic PD controller computes the requested rotation
   (`left`, `hold`, `right`) and engine power (`off`, `half`, `full`), plus the
   metrics it used: altitude, pad offset, desired tilt and vertical speed.
3. **Describe.** The telemetry and guidance request are written as text. The
   8B CLM receives a long observation and two questions. The small CLM receives
   a short observation and one question over nine combined actions.
4. **Decide.** The model returns a choice and a probability distribution for
   each question. Malformed answers stop the run instead of falling back.
5. **Execute.** Plain modes execute the model's choice, errors included.
   Assisted modes execute guidance whenever the model disagrees and record that
   as an intervention.
6. **Record.** Each frame stores the state before, the full decision, and the
   state after. The run is saved as JSON and embedded into a replay page.

The model never sees future states, and inference time does not affect the
simulation. Playback therefore runs at simulation speed regardless of model
latency.

## Pilot modes

| Mode | Model | Executes | Needs |
|---|---|---|---|
| `baseline` | none | guidance | nothing; stdlib only |
| `small` (default) | Qwen3-Embedding-0.6B + trained heads, in process | model choice | MLX environment and `models/lunar-small/` |
| `small-assisted` | same as `small` | guidance on disagreement | same as `small` |
| `clm` | official CLM heads + Qwen3-8B over HTTP | model choice | running `clm-serve` and `llama-server` |
| `assisted` | same as `clm` | guidance on disagreement | same as `clm` |

When `--model` is omitted, `small` modes load `models/lunar-small/head.npz`
and `clm` modes request the `clm-latest` model from the server.

## Trained small CLM on this Mac

The additional `small` pilot uses a frozen **Qwen3-Embedding-0.6B, MLX 8-bit**
encoder with new lunar-specific contrastive projection heads. It embeds a short
observation once, then scores nine cached rotation/thrust combinations. Inference
runs locally in one process; no CLM or embedding HTTP server is needed.
The original `clm` and `assisted` pilots remain available.
The trained small model is now the CLI default. See
[training and measured performance](results/SMALL-CLM.md): 19.8 ms median,
24.0 ms p95, and 9/9 unseen unassisted landings in this local evaluation.

```sh
.venv/bin/python -m lunar_clm --pilot small --seed 4000 \
  --out dist/small.json --replay dist/small.html
open dist/small.html
```

`small` executes the trained model's choice directly. `small-assisted` is the
separate, explicit guidance override mode. Both still receive requested control
labels: this remains the same guidance-following problem as the original Laya
experiment, not an independently learned navigation policy.

To reproduce from scratch on Apple Silicon:

```sh
uv venv --python 3.12
uv pip install -e '.[small,train]'
HF_HUB_DISABLE_XET=1 .venv/bin/hf download \
  mlx-community/Qwen3-Embedding-0.6B-8bit \
  --revision 407ad2329cd30702720aafe83f74a1ba30fdfbca \
  --local-dir models/qwen3-embedding-0.6b-8bit
echo 407ad2329cd30702720aafe83f74a1ba30fdfbca > models/qwen3-embedding-0.6b-8bit/revision.txt
.venv/bin/python -m lunar_clm.train_small
.venv/bin/python -m lunar_clm.benchmark --backend small --out results/small.json
.venv/bin/python -m lunar_clm.evaluate_small
```

The training script generates guidance flights for seeds 1000–1031 and validates
on 2000–2007. Labeled request substitutions cover rare action combinations on
both splits; their validation accuracy is reported separately from natural
guidance examples. It precomputes embeddings once and trains only two small
projection heads using supervised InfoNCE over the nine candidates. Use
`--reuse-embeddings` to retrain the heads without repeating encoding. The
standalone evaluator reserves seeds 4000 onward, cycles through all three pads,
and refuses overlap with training/validation seeds. Model selection uses only
validation loss, never these evaluation flights.

The resulting `models/lunar-small/head.npz` is paired with `head.json`, which
records the encoder revision, label order, training splits and head checksum.
The encoder path in that metadata is relative to the head directory. Loading
fails if the checksum or the action label order does not match the code. Both
heads are learned from scratch for this smaller encoder; the released 8B CLM
heads are not compatible with it. Before saving, training checks that the NumPy
inference path reproduces the PyTorch logits. The optimized benchmark uses the
same physical states as before but shorter text and a combined action, so
compare it as a complete deployment, not as an isolated effect of fine-tuning.

The encoder weights retain their Apache-2.0 license; `mlx-embeddings` is a
separate GPL-3.0 dependency. Downloaded models and generated datasets are local
ignored artifacts. The Python client and baseline still need no dependencies.

## Run

Python 3.11+ is sufficient for the baseline and HTTP client. The default small
model uses the MLX environment described above. From this directory:

```sh
# Immediate physics demo; no model needed.
python3 -m lunar_clm --pilot baseline
open dist/replay.html

# With the local CLM server below running: real, unshielded model controls.
python3 -m lunar_clm --pilot clm --seed 3000
open dist/replay.html

# Explicit assistance: every model disagreement is replaced by guidance.
python3 -m lunar_clm --pilot assisted --seed 3000 \
  --out dist/assisted.json --replay dist/assisted.html
```

| Option | Default | Meaning |
|---|---|---|
| `--pilot` | `small` | one of the pilot modes above |
| `--model` | per mode | CLM model name, or a small-head `.npz` path |
| `--base-url` | `CLM_BASE_URL` or `http://127.0.0.1:8700` | CLM server for `clm`/`assisted` |
| `--timeout` | `30` | HTTP timeout in seconds |
| `--episodes` | `1` | consecutive seeds to fly, starting at `--seed` |
| `--seed` | `0` | first seed; sets start position, drift and tilt |
| `--target` | `1` | landing pad `0`, `1` or `2` |
| `--steps` | `900` | maximum decisions per episode |
| `--out` | `dist/run.json` | recording JSON |
| `--replay` | `dist/replay.html` | standalone replay page |

Optional `CLM_API_KEY` is read from the environment and never saved. Existing
output files are replaced. The default 900-step limit is 180 simulation
seconds; inference waits between steps, so playback speed is independent of
inference speed. This follows the original application's recorded-flight
design, not keyboard or live play. Each finished episode prints its summary
as one JSON line.

## Replay page

The replay is a single HTML file with the recording embedded. Open it directly
from disk; it needs no server or network access.

- **Viewport.** Terrain, the three pads with the target highlighted, the
  flight trail, the lander and its engine flame, plus altitude, vertical and
  horizontal speed, and fuel.
- **Controls.** Play or pause, restart, a scrub bar over every decision, speed
  of 0.5×, 1×, 4× or 16×, and a flight selector for multi-episode runs.
- **Decision panel.** Per-label model probabilities, the proposed and executed
  commands, decision latency, and whether guidance intervened.
- **Experiment panel.** An explanation of the pilot mode and the exact
  observation text sent to the model at the current frame.

Model paths and prompts are JSON-escaped before embedding, so a recording
cannot inject script into the page.

## Recording format

`--out` writes one JSON object with `schema_version: 1`:

| Key | Contents |
|---|---|
| `pilot` | mode, model, and agent provenance such as endpoint, head checksum or training metadata |
| `runtime` | Python version, platform, installed package versions, model load time |
| `world` | terrain vertices, pads, physics timestep and substeps per decision |
| `episodes` | one entry per flight, each with a `summary` and a list of `frames` |

Each frame holds `before` and `after` states and a `decision`:

| Decision key | Meaning |
|---|---|
| `reference` | guidance command for this state |
| `proposed` | model command, or guidance for `baseline` |
| `executed` | command actually applied |
| `intervened` | true when an assisted mode replaced the model's command |
| `answers` | validated model choices and probabilities, or `null` |
| `guidance` | guidance metrics used to build the prompt |
| `prompt` | observation text sent to the model |
| `latency_ms` | wall time for guidance, prompt, inference and validation |
| `server_latency_ms`, `usage` | server timing and token counts when reported |

An episode `summary` holds the final state and status, plus decision count,
interventions, guidance matches, p50/p95 decision latency and wall seconds.
A flight still flying at the step limit is recorded as `truncated`.

## Physics and scoring

The world is 1000 m wide with a fixed piecewise terrain and three flat pads.
Lunar gravity is 1.62 m/s², full thrust gives 5 m/s², and attitude jets turn
at 30°/s. Main engine and attitude jets draw on one 100-unit fuel tank. When
the tank runs dry mid-step, thrust and turning scale down to the fuel left.

| Pad | Span (m) | Height (m) | Landing score |
|---|---|---|---|
| 0 | 150–240 | 30 | 100 |
| 1 (default) | 440–560 | 20 | 50 |
| 2 | 775–825 | 40 | 200 |

Contact is a landing only when the whole hull is over a pad, horizontal speed
is at most 2 m/s, downward speed is at most 3 m/s, and tilt is within 8°.
Any other contact is a crash. Leaving the world or rising above 750 m is
`out_of_bounds`, and 180 simulation seconds is a `timeout`. Terminal states
are absorbing. Each seed places the lander within 100 m of the target pad
center at a height of 450 m, descending at 8 m/s, with random
horizontal drift and tilt.

## Local CLM on Apple Silicon

The official CLM service supplies projection heads and caching. The installed
llama.cpp supplies the Qwen3-8B **last-token** embeddings on Metal. The Q8
quantized encoder is a local deployment variant, not the official vLLM BF16
reference; accuracy parity with that reference has not been established.

```sh
uv venv --python 3.12
# The upstream package pulls in Linux/CUDA vLLM by default. Install only its
# head/API dependencies on this Mac; llama.cpp provides the encoder instead.
uv pip install --no-deps contrastive-lm==0.1.0
uv pip install -r requirements-macos.txt
.venv/bin/clm-download --dest models
```

Install llama.cpp if needed (`brew install llama.cpp`). In one terminal:

```sh
llama-server -m /path/to/Qwen3-8B-Q8_0.gguf \
  --embedding --pooling last --alias qwen3-8b \
  --host 127.0.0.1 --port 8091 -c 4096 -b 2048 -ub 2048 -np 2
```

In a second terminal:

```sh
OMP_NUM_THREADS=1 .venv/bin/clm-serve --host 127.0.0.1 --port 8700 \
  --emb-url http://127.0.0.1:8091/v1/embeddings \
  --ckpt models/CLM_v0.1-8B.pt --device cpu --action-cache 16MiB --no-ui
```

The validated machine already had `Qwen/Qwen3-8B-GGUF` Q8 weights cached.
For another machine, obtain that file from
[the official Qwen repository](https://huggingface.co/Qwen/Qwen3-8B-GGUF).
Use the matching Qwen3-8B backbone and last-token pooling: arbitrary embedding
models do not work with these heads. On CUDA, the
[upstream setup](https://github.com/Contrastive-LM/CLM#quickstart) uses vLLM
instead; the game client does not change.

## What the models do

The shared feedback controller computes requested rotation (`left`, `hold`,
`right`) and power (`off`, `half`, `full`). Both models receive these requests
alongside identical telemetry and answer identical questions. This measures
**guidance following**, not discovery of a landing policy.

Laya scores choices with its small task-oriented encoder. CLM separately
embeds question-conditioned states and candidate action descriptions, applies
trained projection heads, then scores similarity. Fixed action descriptions
are cached by the official server. The client sends both questions in one
request over one persistent connection. It does not cache changing states or
replace failed requests with the baseline.

The small CLM follows the same idea in one process. Its nine action
descriptions are embedded and projected once at load time. Each decision then
costs one short state embedding, a projection, and a dot product against the
nine cached actions, followed by a softmax.

`clm` executes the model's choices, including errors. `assisted` records the
model's proposal but executes guidance on every disagreement. Assisted landing
rates measure integration and the guidance controller, not model piloting skill.
Invalid responses or unavailable inference stop the run with a clear error.

## Reproduce the speed comparison

See [the measured comparison](results/COMPARISON.md) and the raw JSON in
`results/`. Run backends sequentially to avoid competing GPU workloads:

```sh
python3 -m lunar_clm.benchmark --backend baseline --out results/baseline.json
python3 -m lunar_clm.benchmark --backend clm --out results/clm.json

# Uses Laya's existing environment and original public model.
HF_HOME=../lunar-laya/.cache/huggingface HF_HUB_OFFLINE=1 \
  ../lunar-laya/.venv/bin/python -m lunar_clm.benchmark \
  --backend laya --out results/laya-public.json

# Additional task-trained, compact Q6 checkpoint from the original project.
HF_HUB_OFFLINE=1 ../lunar-laya/.venv/bin/python -m lunar_clm.benchmark \
  --backend laya --model ../lunar-laya/models/lunar-laya-q6 \
  --out results/laya-q6.json
```

Each backend gets the same 96 distinct states sampled across complete guidance
flights on all three pads (seeds 3000–3002), with three separate warmup states.
Reports include the corpus/question hash, individual timings, p50/p95, mean
throughput, fraction meeting the 200 ms control interval, and agreement with
guidance. Timing includes prompt construction, model execution, response parsing,
and CLM's loopback HTTP; excludes physics, rendering and model/server startup.
`load_seconds` for CLM is only client initialization. `first_call_ms` includes
the first excluded warmup call; it is cold only if the server was newly started.
Restart CLM before repeating the benchmark; its state cache otherwise makes
identical reruns much faster. CLM's recorded `usage.input_tokens` allows checking
that timed requests really executed the encoder.

## Tests

```sh
python3 -m unittest discover -s tests -v
node tests/test_replay.cjs
```

The dependency-free checks cover physics and landing rules, seeded landings,
adapter wire format, invalid answers, explicit assistance, recording safety,
distinct benchmark states, statistics, and replay controls. They use a fake model
only for adapter tests; the measurements and flight artifacts use real weights.

## Project layout

| Path | Role |
|---|---|
| `lunar_clm/game.py` | physics, terrain, pads, landing rules |
| `lunar_clm/guidance.py` | PD guidance, question text, full observation |
| `lunar_clm/small.py` | small CLM encoder, heads, nine actions, short observation |
| `lunar_clm/pilot.py` | pilot modes, CLM HTTP client, answer validation |
| `lunar_clm/cli.py` | episode loop, recording, replay export |
| `lunar_clm/replay.html` | replay page template |
| `lunar_clm/benchmark.py` | fixed-corpus latency benchmark for all backends |
| `lunar_clm/train_small.py` | dataset generation, embedding cache, head training |
| `lunar_clm/evaluate_small.py` | held-out unassisted flights for the small CLM |
| `results/` | committed measurements and write-ups |
| `models/`, `dist/` | downloaded weights and generated runs; not committed |

## Sources

Sources: the requested [Notion blog](https://contrastive-lm.notion.site), read
in Safari, the official
[CLM source and API documentation](https://github.com/Contrastive-LM/CLM),
[released heads](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), and local
Laya implementation. Upstream CLM speed claims compare against
Jev, not this Laya deployment. CLM code/heads retain their Apache-2.0 terms;
encoder weights retain their upstream terms. No Atari ROM or assets are used.
