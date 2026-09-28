"""Small CLM: frozen MLX encoder, trained contrastive heads, nine cached actions."""

import hashlib
import json
from pathlib import Path

from .game import Command
from .guidance import THROTTLE_NAMES, THROTTLES, TURN_NAMES, TURNS

ENCODER_ID = "mlx-community/Qwen3-Embedding-0.6B-8bit"
ACTIONS = {f"{turn}/{power}": Command(t, p)
           for turn, t in TURNS.items() for power, p in THROTTLES.items()}
ACTION_TEXTS = [f"Rotate {turn}. Set thrust to {power}."
                for turn in TURNS for power in THROTTLES]
QUESTION = {"action": {"type": "choice", "instructions": "Execute the requested controls.",
                       "criteria": dict(zip(ACTIONS, ACTION_TEXTS))}}
LABELS = {command: i for i, command in enumerate(ACTIONS.values())}
DEFAULT_HEAD = "models/lunar-small/head.npz"


def short_observation(game, requested, metrics):
    s = game.state
    turn, power = TURN_NAMES[requested.turn], THROTTLE_NAMES[requested.throttle]
    return (f"Instruct: Execute requested lunar lander controls.\nQuery: "
            f"Altitude {metrics['altitude']:.1f}; offset {metrics['target_dx']:.1f}; "
            f"vx {s.vx:.1f}; vy {s.vy:.1f}; tilt {s.angle:.1f}; fuel {s.fuel:.1f}. "
            f"Requested rotation: {turn}. Requested thrust: {power}.")


class Encoder:
    def __init__(self, path):
        import mlx.core as mx
        from mlx_embeddings import load
        self.mx = mx
        self.model, self.tokenizer = load(str(path))
        self.tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.model.eval()
        mx.eval(self.model.parameters())
        # Avoid fully masked left-pad queries in this runtime's causal attention.
        # Qwen's pooling function selects the last non-padding token on the right.
        self.tokenizer.padding_side = "right"

    def encode(self, texts):
        import numpy as np
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors="np")
        if inputs["input_ids"].shape[1] > 256:
            raise ValueError("small CLM supports at most 256 tokens per observation")
        outputs = self.model(self.mx.array(inputs["input_ids"]),
                             attention_mask=self.mx.array(inputs["attention_mask"]))
        embeddings = outputs.text_embeds.astype(self.mx.float32)
        self.mx.eval(embeddings)
        vectors = np.asarray(embeddings)
        if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
            raise ValueError("encoder must produce finite Qwen3-Embedding-0.6B vectors")
        return vectors, int(inputs["attention_mask"].sum())


def project(x, weights, prefix):
    import numpy as np
    h = np.tanh(x @ weights[prefix + "0.weight"].T + weights[prefix + "0.bias"])
    z = h @ weights[prefix + "2.weight"].T + weights[prefix + "2.bias"]
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


class SmallAgent:
    def __init__(self, checkpoint=DEFAULT_HEAD):
        import numpy as np
        path = Path(checkpoint)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["actions"] != list(ACTIONS) or metadata["action_texts"] != ACTION_TEXTS:
            raise ValueError("checkpoint actions do not match the controller")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["head_sha256"]:
            raise ValueError("head checksum does not match its metadata")
        with np.load(path, allow_pickle=False) as weights:
            self.weights = {key: weights[key] for key in weights.files}
        self.encoder = Encoder(path.parent / metadata["encoder_path"])
        actions, _ = self.encoder.encode(ACTION_TEXTS)
        self.candidates = project(actions, self.weights, "action.")
        self.scale = float(metadata["scale"])
        self.provenance = {"model": "lunar-small-clm", "head_sha256": digest,
                           "encoder": metadata["encoder"], "inference": "MLX 8-bit encoder; NumPy heads",
                           "training": metadata["training"], "state_cache": False}

    def predict(self, state, questions):
        import numpy as np
        if questions != QUESTION:
            raise ValueError("small CLM expects the nine combined actions")
        x, tokens = self.encoder.encode([state])
        logits = self.scale * (project(x, self.weights, "state.") @ self.candidates.T)[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        if not np.isfinite(probabilities).all():
            raise ValueError("non-finite small CLM probabilities")
        return {"answers": {"action": {"choice": list(ACTIONS)[int(probabilities.argmax())],
                                        "probabilities": dict(zip(ACTIONS, map(float, probabilities)))}},
                "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def close(self):
        pass
