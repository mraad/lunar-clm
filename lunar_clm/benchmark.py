"""Time identical, distinct flight states; never time a cached replay as fresh inference."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

from .cli import percentiles, positive_int, write_json
from .game import PADS, Game
from .guidance import QUESTIONS, guidance, observation
from .pilot import Pilot
from .small import QUESTION, short_observation


def corpus(samples, seed):
    states = []
    for target in range(len(PADS)):
        game = Game(seed + target, target)
        while game.state.status == "flying":
            states.append(copy.deepcopy(game))
            game.step(guidance(game)[0])
    if samples > len(states):
        raise ValueError(f"at most {len(states)} distinct samples available")
    return [states[i * len(states) // samples] for i in range(samples)]


def measure(pilot, states):
    rows = []
    for game in states:
        expected = asdict(guidance(game)[0])
        start = perf_counter()
        command, decision = pilot.decide(game)
        latency = (perf_counter() - start) * 1000
        rows.append({"seed": game.seed, "target": game.target, "time": game.state.time,
                     "latency_ms": latency, "expected": expected, "actual": asdict(command),
                     "guidance_match": asdict(command) == expected,
                     "server_latency_ms": decision.get("server_latency_ms"), "usage": decision.get("usage")})
    return rows


def summarize(rows):
    times = [row["latency_ms"] for row in rows]
    p50, p95 = percentiles(times)
    return {"samples": len(times), "p50_ms": p50, "p95_ms": p95,
            "mean_ms": statistics.mean(times), "decisions_per_second": 1000 / statistics.mean(times),
            "within_200ms_fraction": sum(t <= 200 for t in times) / len(times),
            "guidance_agreement": sum(row["guidance_match"] for row in rows) / len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "clm", "small", "laya"), required=True)
    parser.add_argument("--samples", type=positive_int, default=96)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--base-url")
    parser.add_argument("--model", help="CLM model name or Laya checkpoint")
    parser.add_argument("--laya-root", type=Path, default=Path("../lunar-laya"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        states = corpus(args.samples, args.seed)
        observe, questions = (short_observation, QUESTION) if args.backend == "small" else (observation, QUESTIONS)
        prompts = [observe(g, *guidance(g)) for g in states]
        if len(set(prompts)) != len(prompts):
            raise ValueError("duplicate prompts would bias the warm-cache benchmark")
        start = perf_counter()
        if args.backend == "laya":
            sys.path.insert(0, str(args.laya_root.resolve()))
            from lunar_laya.pilot import Pilot as LayaPilot, QUESTIONS as LAYA_QUESTIONS
            if LAYA_QUESTIONS != QUESTIONS:
                raise ValueError("Laya questions differ; comparison would not be matched")
            pilot = LayaPilot("laya", args.model or "aac6fef/laya-multilingual-mlx")
        else:
            pilot = Pilot(args.backend, args.model, args.base_url)
        load_seconds = perf_counter() - start
        # Different states from the timed corpus warm kernels and all fixed candidates.
        warmup = measure(pilot, corpus(3, args.seed - 100))
        rows = measure(pilot, states)
        report = {"backend": args.backend, "pilot": pilot.provenance,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "seed": args.seed, "corpus_sha256": hashlib.sha256(json.dumps(
                      {"prompts": prompts, "questions": questions}, sort_keys=True).encode()).hexdigest(),
                  "state_corpus_sha256": hashlib.sha256(json.dumps([
                      {"target": g.target, "state": g.snapshot()} for g in states], sort_keys=True).encode()).hexdigest(),
                  "load_seconds": load_seconds, "first_call_ms": warmup[0]["latency_ms"],
                  "warmup": warmup, "summary": summarize(rows), "rows": rows,
                  "method": "3 excluded warmup calls; unique states; fixed cached actions; sequential wall latency including guidance, formatting, inference and transport; excludes model/server startup and physics"}
        write_json(args.out, report, indent=2)
        print(json.dumps(report["summary"], indent=2))
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"benchmark: {exc}\n")


if __name__ == "__main__":
    main()
