# Trained small CLM: local results

The optimized controller takes **19.79 ms median / 23.95 ms p95** on this
Apple M4 Max, compared with 869.00 / 1378.07 ms for the original CLM deployment.
That is **43.9× lower median latency**. It landed all nine unseen evaluation
flights without overrides and matched guidance on all 3,351 flight decisions.
The CLI now defaults to this `small` pilot; `--pilot clm` retains the 8B version.

| Deployment | Median ms | p95 ms | Decisions/s¹ | Guidance agreement² |
|---|---:|---:|---:|---:|
| Original CLM, Qwen3-8B Q8 / Metal + CPU heads | 869.00 | 1378.07 | 1.07 | 15/96 |
| Public Laya / MLX FP16 | 18.50 | 37.39 | 47.05 | 53/96 |
| Lunar-trained Laya / MLX Q6 | 26.56 | 39.47 | 35.59 | 96/96 |
| **Lunar-trained small CLM / MLX 8-bit** | **19.79** | **23.95** | **51.82** | **96/96** |

¹ Reciprocal of mean latency. ² Both requested controls correct.
All 96 small-CLM decisions were below the 200 ms control interval. The small
model is comparable to public Laya in median latency and faster than the tested
trained Q6 Laya deployment. These are individual local runs, not confidence
intervals or universal hardware rankings.

## What changed

- Frozen `Qwen3-Embedding-0.6B`, using its pinned MLX 8-bit conversion.
- Two newly trained projection heads, each `1024 → 128 → 64` with tanh and
  output normalization: **278,912 trainable parameters total**.
- One combined choice over nine rotation/thrust pairs. Nine action embeddings
  and their projections are cached once when the model loads.
- One short state embedding per decision, consuming 63–69 tokens in the timed
  corpus, versus 250–266 total encoder tokens in the original two-question call.
- In-process MLX encoding and NumPy head scoring; no HTTP services or state
  embedding cache. Every timed call actually executes the encoder.

The encoder weights occupy 633.15 MB, and the learned heads 1.12 MB. These are
weight-file sizes, not peak memory or the size of the entire Python environment.
This is a new task-specific CLM using the state/action contrastive design.
It does not reuse or fine-tune the incompatible released 8B projection heads.

The speed improvement comes from the smaller encoder, shorter input, combined
question and local runtime. **Training improves decision accuracy; it does not
make an unchanged dense encoder perform fewer operations.** This comparison
does not isolate the contribution of each optimization.

## Training and checks

The frozen encoder precomputed embeddings for 2,527 training examples from
flight seeds 1000–1031 and 632 validation examples from seeds 2000–2007.
Natural guidance observations are sampled every eight control steps. Additional
explicit request substitutions cover all nine combinations, including rare
full-thrust commands. These examples teach execution of the requested labels;
they are not claimed to be optimal actions for their displayed telemetry.

Only the heads were trained: 100 epochs, AdamW, learning rate 0.001, weight
decay 0.001, batch size 128, seed 42, fixed logit scale 25. The loss is supervised
InfoNCE with the requested action as the positive and the other eight distinct
actions as negatives. Checkpoint selection uses validation loss.

The best checkpoint scored **100%** on both the 376 natural validation examples
and the 256 request-substitution examples. Raw encoder similarity without trained
heads scored 80.38% on their combined set. Precomputing embeddings and training
took **42 seconds**, excluding dependency installation and model downloading.

The initial left-padded batching path produced non-finite values in the upstream
runtime's causal attention. The final implementation uses right padding and
last-non-padding-token pooling. All training embeddings were checked for finite
values. Mixed-length batched versus single embeddings had minimum cosine
similarity 0.999958. Exported NumPy heads matched PyTorch scores within
0.0000144 maximum absolute error, with identical validation choices.

## Unseen flights

The model was frozen before evaluating seeds **4000–4008**, cycling pads 0, 1,
and 2. None overlaps training or validation. All **9/9 landed**, with **zero
overrides**, and all **3,351/3,351** decisions matched guidance. Together, the
flights took 67.54 seconds of wall time, excluding model loading.

The observation still contains the guidance controller's requested labels, as
in the original experiment. Therefore “unassisted” means no replacement of the
model's output after inference. It does **not** mean the model navigates without
guidance. Nine successful flights are a finite integration check, not a general
landing guarantee or evidence of robustness to other physics or unseen formats.

## Reproduction and evidence

See the [README commands](../README.md#trained-small-clm-on-this-mac).
The trained head and downloaded encoder are local artifacts under `models/`.
The nine-flight recording and standalone replay are at
`dist/small-flights.json` and `dist/small-flights.html`.

- [Benchmark samples](small.json): three excluded warmups, 96 distinct physical
  states from the original corpus, sequential decision wall timing. Each seed,
  pad, simulation time and expected command was checked against the original
  CLM/Laya rows. The text/questions differ, so their workload hashes differ.
- [Flight results](small-flights.json): per-flight outcomes and timings.
- [Training metadata](small-training.json) and [epoch history](small-training-history.json).
- [Environment and hashes](small-environment.json): hardware, runtime versions,
  encoder/head checksums and source fingerprints.
- Twelve dependency-free Python tests cover the simulation, interfaces, joint
  action mapping, explicit assistance and seed separation; replay control tests
  also pass. Packaging and the installed CLI were checked. Safari displayed
  the nine-action probabilities and a landed terminal frame with zero overrides.

Sources: [Qwen's encoder](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[the pinned MLX conversion](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-8bit/tree/407ad2329cd30702720aafe83f74a1ba30fdfbca),
[MLX-Embeddings](https://github.com/Blaizzy/mlx-embeddings), and
[CLM's contrastive state/action design](https://github.com/Contrastive-LM/CLM).
