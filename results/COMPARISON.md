# CLM versus Laya: measured local execution

This records the original 8B deployment. The subsequent
[trained small CLM comparison](SMALL-CLM.md) achieved 19.79 ms median latency
and nine unassisted held-out landings; the measurements below are preserved.

On this Apple M4 Max (128 GiB), **CLM is slower than both Laya deployments**
for this small, two-question task. CLM's median decision took 869.0 ms:
47.0× public Laya's latency and 32.7× trained Q6 Laya's latency.
This is a deployment comparison, not an architecture-only result.

| Backend | Median ms | p95 ms | Decisions/s¹ | Within 200 ms | Both controls agree with guidance |
|---|---:|---:|---:|---:|---:|
| Deterministic guidance | 0.0042 | 0.0047 | 231,373 | 100% | 96/96 |
| Public Laya, MLX FP16 | 18.50 | 37.39 | 47.05 | 100% | 53/96 |
| Lunar-trained Laya, MLX Q6 | 26.56 | 39.47 | 35.59 | 100% | 96/96 |
| CLM reference heads + Qwen3-8B Q8, Metal | 869.00 | 1,378.07 | 1.07 | 0% | 15/96 |

¹ Reciprocal of mean latency, not median. The baseline row measures decision
construction alone, not complete simulation throughput. All rows use 96
identical states, identical question descriptions, and identical guidance.
The matching corpus SHA-256 is saved in each JSON report.

## Why this result makes sense

The local Laya implementation uses a small encoder specialized for typed
decisions. The trained Q6 version also knows this exact guidance-following
task. Its compact runtime processes questions sequentially to reduce memory;
Q6 is not automatically faster than the original FP16 path, as these timings show.

CLM uses a much larger Qwen3-8B encoder. The six fixed candidate descriptions
are cached, but every new flight state still produces **two question-conditioned
state embeddings**. This removes autoregressive token generation and repeated
action encoding, but does not remove the large encoder pass. Laya also avoids
autoregressive generation. Thus “CLM avoids generation” alone predicts no
speed advantage over Laya. The upstream CLM speed claims compare against Jev,
not this small Laya implementation.

CLM's server latency median was about 868 ms; loopback HTTP is not the main
cost. Every timed CLM request reported 250–266 encoder input tokens, confirming
that the run did not measure cached-state answers. CLM's recorded first warmup
call took 1,815 ms; three warmup calls were excluded from steady-state results.
The server and model were already started, so this is not application cold start.

The CLM encoder file is 8.71 GB plus 75.6 MB of heads; trained Q6 Laya's model
file is 261 MB. These are disk sizes, not measured peak RAM. See
[environment.json](environment.json) for exact sizes, hashes, hardware and
runtime settings. The client has zero third-party dependencies; the local
inference services naturally require their model runtimes.

## Method and limitations

- One sequential run per backend on September 28, 2026; 96 unique samples
  spread across baseline flights on pads 0–2, seeds 3000–3002. Warmup uses
  three distinct states from separate seeds. No simultaneous model workloads
  were launched during these benchmarks. This is a small local measurement,
  not a confidence interval or a dedicated-hardware laboratory result.
- End-to-end decision wall time includes guidance, prompt formatting, completed
  inference, response conversion, and CLM HTTP. It excludes physics, playback,
  downloads and server startup. The separate `load_seconds` field measures
  Laya model loading but only CLM client construction; **do not compare it**
  as total startup time. MLX materializes answers before the timer ends.
- CLM is the official `contrastive-lm==0.1.0` engine and released projection
  heads, with a Q8_0 llama.cpp/Metal encoder and CPU heads (one OpenMP thread).
  Its two-slot encoder uses last-token pooling. This differs from the official
  vLLM GPU reference; quantization/backend parity has not been established.
- Laya public weights are the original cached `aac6fef/laya-multilingual-mlx`.
  The extra Q6 weights were already trained specifically for lunar guidance
  in the sibling project. Their accuracy is not a fair zero-shot training
  comparison with CLM; both rows are provided to make that difference visible.
- The state explicitly contains requested control labels. Agreement measures
  following guidance, not learning physics. Probabilities are model outputs,
  not calibrated landing-safety estimates.
- The simulator advances 0.2 seconds per decision and waits for inference.
  Laya met that interval for all sampled states; CLM met it for none. CLM
  still works for offline simulation and replay, which displays simulation
  time independently of wall time.
- Shorter prompts, another encoder precision, fewer questions, native MLX
  CLM, or task fine-tuning may change results; none is measured here. The
  tested model's low agreement should not be disguised with an implicit
  fallback controller.

## Complete-flight check

Seed 3000, central pad, same initial state:

| Pilot | Outcome | Decisions | Overrides | Simulation seconds | Wall seconds |
|---|---|---:|---:|---:|---:|
| Guidance | Landed | 382 | 0 | 76.22 | 0.015 |
| Raw CLM | Crashed | 93 | 0 | 18.42 | 74.34 |
| Assisted CLM | Landed | 382 | 318 | 76.22 | 293.90 |

Raw CLM left thrust off and struck the pad too fast. The assisted trajectory
matches the baseline **at every recorded transition**; its landing is evidence
of guidance, not CLM skill. These single flights are integration checks, not
landing-rate estimates. They ran after the benchmarks with mixed cache history;
use the matched-state table above for speed comparisons. The full recordings
and standalone replays are in `dist/`; [flights.json](flights.json) preserves
summaries and recording hashes. Replay layout, probabilities, terminal outcome
and timeline scrubbing were checked in Safari; automated replay controls and
10 Python tests also pass.

## Reproduce and inspect

Use the commands in [README.md](../README.md). Restart the CLM service before
rerunning the same corpus to clear its state cache. Raw timings and decisions:
[CLM](clm.json), [public Laya](laya-public.json),
[trained Q6 Laya](laya-q6.json), [guidance](baseline.json).
The preserved source and questions match the original Laya simulation; exact
state/prompt parity was checked over three full guidance flights.

The comparison follows the mechanism documented in the
[official CLM source](https://github.com/Contrastive-LM/CLM),
particularly `engine.py`, `embedder.py` and `heads.py`, and the local
`lunar-laya/lunar_laya/pilot.py` and `compact.py` implementations.
