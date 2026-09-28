"""Generate seed-disjoint guidance data, cache MLX embeddings, train contrastive heads."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .cli import positive_int
from .game import PADS, Game
from .guidance import guidance
from .small import ACTIONS, ACTION_TEXTS, Encoder, ENCODER_ID, LABELS, short_observation, project


def dataset(first_seed, episodes, stride, augment=False):
    rows = []
    for seed in range(first_seed, first_seed + episodes):
        game = Game(seed, (seed - first_seed) % len(PADS))
        step = 0
        while game.state.status == "flying":
            command, metrics = guidance(game)
            if step % stride == 0:
                rows.append({"seed": seed, "target": game.target,
                             "state": short_observation(game, command, metrics),
                             "label": LABELS[command]})
                # Balance rare combinations: independent command-following examples.
                # These are labeled request substitutions, not additional physics rollouts.
                if augment and step % (stride * 12) == 0:
                    for label, requested in enumerate(ACTIONS.values()):
                        if requested == command:
                            continue
                        rows.append({"seed": seed, "target": game.target, "augmented": True,
                                     "state": short_observation(game, requested, metrics), "label": label})
            game.step(command)
            step += 1
    return rows


def prepare(encoder_path, out):
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": dataset(1000, 32, 8, True), "validation": dataset(2000, 8, 8, True)}
    assert not {r["seed"] for r in rows["train"]} & {r["seed"] for r in rows["validation"]}
    payload = json.dumps(rows, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (out / "dataset.json").write_text(payload + "\n")
    encoder = Encoder(encoder_path)
    arrays = {"actions": encoder.encode(ACTION_TEXTS)[0]}
    probes = [rows["train"][0]["state"], ACTION_TEXTS[0]]
    batch = encoder.encode(probes)[0]
    singles = np.concatenate([encoder.encode([text])[0] for text in probes])
    cosine = (batch * singles).sum(axis=-1)
    if float(cosine.min()) < 0.999:
        raise ValueError("batched and single-state encoder pooling disagree")
    arrays["batch_single_min_cosine"] = np.array(cosine.min())
    for split, items in rows.items():
        chunks = []
        for i in range(0, len(items), 16):
            vectors, _ = encoder.encode([r["state"] for r in items[i:i + 16]])
            chunks.append(vectors)
            if i % 160 == 0:
                print(f"{split}: embedded {min(i + 16, len(items))}/{len(items)}", flush=True)
        arrays[split] = np.concatenate(chunks)
        arrays[split + "_labels"] = np.array([r["label"] for r in items], dtype=np.int64)
    arrays["dataset_sha256"] = np.array(digest)
    arrays["encoder_revision"] = np.array((encoder_path / "revision.txt").read_text().strip())
    np.savez(out / "embeddings.npz", **arrays)


def train(encoder_path, out, epochs):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rows = json.loads((out / "dataset.json").read_text())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    revision = (encoder_path / "revision.txt").read_text().strip()
    with np.load(out / "embeddings.npz", allow_pickle=False) as archive:
        if str(archive["dataset_sha256"]) != digest:
            raise ValueError("cached embeddings do not match dataset")
        if str(archive["encoder_revision"]) != revision:
            raise ValueError("cached embeddings do not match encoder revision")
        batch_single_cosine = float(archive["batch_single_min_cosine"])
        data = {k: torch.from_numpy(archive[k]) for k in
                ("train", "validation", "train_labels", "validation_labels", "actions")}
    heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(1024, 128), nn.Tanh(), nn.Linear(128, 64))
                           for name in ("state", "action")})
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.001, weight_decay=0.001)
    scale = 25.0

    def logits(x):
        return scale * F.normalize(heads["state"](x), dim=-1) @ F.normalize(heads["action"](data["actions"]), dim=-1).T

    best, best_loss, history = None, float("inf"), []
    for epoch in range(epochs):
        heads.train()
        order = torch.randperm(len(data["train"]))
        for indices in order.split(128):
            # Supervised InfoNCE over nine distinct candidates: eight negatives per state.
            loss = F.cross_entropy(logits(data["train"][indices]), data["train_labels"][indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        heads.eval()
        with torch.no_grad():
            scores = logits(data["validation"])
            val_loss = F.cross_entropy(scores, data["validation_labels"]).item()
            accuracy = (scores.argmax(-1) == data["validation_labels"]).float().mean().item()
        row = {"epoch": epoch + 1, "validation_loss": val_loss, "validation_accuracy": accuracy}
        history.append(row)
        if val_loss < best_loss:
            best_loss, best = val_loss, {k: v.detach().numpy().copy() for k, v in heads.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(json.dumps(row), flush=True)
    path = out / "head.npz"
    np.savez(path, **best)
    heads.load_state_dict({k: torch.from_numpy(v) for k, v in best.items()})
    with torch.no_grad():
        expected = logits(data["validation"]).numpy()
    actual = scale * project(data["validation"].numpy(), best, "state.") @ project(data["actions"].numpy(), best, "action.").T
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    assert np.array_equal(actual.argmax(-1), expected.argmax(-1))
    correct = actual.argmax(-1) == data["validation_labels"].numpy()
    augmented = np.array([r.get("augmented", False) for r in rows["validation"]])
    raw_choices = (data["validation"] @ data["actions"].T).argmax(-1)
    config = json.loads((encoder_path / "config.json").read_text())
    seeds = {split: sorted({r["seed"] for r in items}) for split, items in rows.items()}
    metadata = {"encoder": {"id": ENCODER_ID, "revision": revision,
                             "quantization": config.get("quantization"), "pooling": "last-token"},
                "encoder_path": os.path.relpath(encoder_path.resolve(), out.resolve()),
                "actions": list(ACTIONS), "action_texts": ACTION_TEXTS, "scale": scale,
                "head_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "training": {"method": "supervised contrastive InfoNCE over nine candidate actions",
                             "teacher": "guidance; requested labels included in observation",
                             "encoder_frozen": True, "head_parameters": sum(p.numel() for p in heads.parameters()),
                             "batch_single_min_cosine": batch_single_cosine,
                             "dataset_sha256": digest,
                             "train_seeds": [seeds["train"][0], seeds["train"][-1]],
                             "validation_seeds": [seeds["validation"][0], seeds["validation"][-1]],
                             "split_rows": {s: len(r) for s, r in rows.items()},
                             "label_counts": {s: dict(Counter(r["label"] for r in rs)) for s, rs in rows.items()},
                             "epochs": epochs, "seed": 42, "best_validation_loss": best_loss,
                             "validation_accuracy": float(correct.mean()),
                             "validation_natural_accuracy": float(correct[~augmented].mean()),
                             "validation_augmented_accuracy": float(correct[augmented].mean()),
                             "raw_encoder_validation_accuracy": float((raw_choices == data["validation_labels"]).float().mean()),
                             "numpy_torch_max_error": float(np.abs(actual - expected).max())}}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(metadata["training"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=Path("models/qwen3-embedding-0.6b-8bit"))
    parser.add_argument("--out", type=Path, default=Path("models/lunar-small"))
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--reuse-embeddings", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    if not args.reuse_embeddings:
        prepare(args.encoder, args.out)
    train(args.encoder, args.out, args.epochs)
    print(f"Training pipeline: {perf_counter() - start:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
