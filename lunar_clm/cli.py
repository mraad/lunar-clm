"""Finite, reproducible runs and portable browser replays."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

from .game import DT, CONTROL_STEPS, PADS, TERRAIN, Game
from .pilot import MODES, Pilot


def percentiles(times):
    times = sorted(times)
    return statistics.median(times), times[math.ceil(len(times) * 0.95) - 1]


def write_json(path, obj, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False, **kw) + "\n")


def run_episode(pilot, seed, target, steps):
    start = perf_counter()
    game = Game(seed, target)
    frames = []
    while game.state.status == "flying" and len(frames) < steps:
        before = game.snapshot()
        command, decision = pilot.decide(game)
        after = game.step(command)
        frames.append({"before": before, "decision": decision, "after": after})
    terminal = game.snapshot()
    if terminal["status"] == "flying":
        terminal["status"] = "truncated"
    p50, p95 = percentiles(f["decision"]["latency_ms"] for f in frames)
    summary = {"seed": seed, "target": target, **terminal,
               "decisions": len(frames),
               "interventions": sum(f["decision"]["intervened"] for f in frames),
               "guidance_matches": sum(f["decision"]["executed"] == f["decision"]["reference"] for f in frames),
               "latency_p50_ms": p50, "latency_p95_ms": p95,
               "wall_seconds": perf_counter() - start}
    return {"summary": summary, "frames": frames}


def new_recording(pilot):
    """Common schema and runtime provenance for CLI and training evaluation."""
    packages = {}
    for name in ("contrastive-lm", "mlx", "mlx-embeddings", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"schema_version": 1, "pilot": dict(pilot.provenance),
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": packages},
            "world": {"terrain": TERRAIN, "pads": PADS, "dt": DT,
                      "control_steps": CONTROL_STEPS}, "episodes": []}


def export_replay(record, path):
    # Escape '<' so even a model path containing </script> remains inert JSON.
    payload = json.dumps(record, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("replay.html").read_text()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("/* RECORDING */null", payload))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=MODES, default="small")
    parser.add_argument("--model", help="CLM API model name or a small-head .npz path")
    parser.add_argument("--base-url", help="CLM server URL; defaults to CLM_BASE_URL or localhost:8700")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=int, choices=range(len(PADS)), default=1)
    parser.add_argument("--steps", type=positive_int, default=900, help="maximum decisions per episode")
    parser.add_argument("--out", type=Path, default=Path("dist/run.json"))
    parser.add_argument("--replay", type=Path, default=Path("dist/replay.html"))
    args = parser.parse_args(argv)
    if args.out.resolve() == args.replay.resolve():
        parser.error("--out and --replay must be different files")
    pilot = None
    try:
        start = perf_counter()
        pilot = Pilot(args.pilot, args.model, args.base_url, args.timeout)
        load_seconds = perf_counter() - start
        record = new_recording(pilot)
        record["runtime"]["load_seconds"] = load_seconds
        for seed in range(args.seed, args.seed + args.episodes):
            episode = run_episode(pilot, seed, args.target, args.steps)
            record["episodes"].append(episode)
            print(json.dumps(episode["summary"], allow_nan=False), flush=True)
        write_json(args.out, record, indent=2)
        export_replay(record, args.replay)
        print(f"Recording: {args.out}\nReplay: {args.replay}", file=sys.stderr)
    except (RuntimeError, OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"lunar-clm: {exc}\n")
    finally:
        if pilot is not None:
            pilot.close()


if __name__ == "__main__":
    main()
