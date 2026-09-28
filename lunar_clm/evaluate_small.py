"""Unassisted held-out flights on all three pads; no model selection on these seeds."""

import argparse
import json
from pathlib import Path

from .cli import export_replay, new_recording, positive_int, run_episode, write_json
from .game import PADS
from .pilot import Pilot
from .small import DEFAULT_HEAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_HEAD)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--episodes", type=positive_int, default=9)
    parser.add_argument("--out", type=Path, default=Path("dist/small-flights.json"))
    parser.add_argument("--report", type=Path, default=Path("results/small-flights.json"))
    args = parser.parse_args()
    if args.out.resolve() == args.report.resolve():
        parser.error("recording and report must use different paths")
    pilot = Pilot("small", model=args.model)
    training = pilot.provenance["training"]
    evaluation = set(range(args.seed, args.seed + args.episodes))
    for split in ("train_seeds", "validation_seeds"):
        lo, hi = training[split]
        if evaluation.intersection(range(lo, hi + 1)):
            parser.error("evaluation seeds overlap training or validation")
    record = new_recording(pilot)
    for i, seed in enumerate(range(args.seed, args.seed + args.episodes)):
        episode = run_episode(pilot, seed, i % len(PADS), 900)
        record["episodes"].append(episode)
        print(json.dumps(episode["summary"]), flush=True)
    write_json(args.out, record, separators=(",", ":"))
    export_replay(record, args.out.with_suffix(".html"))
    summaries = [e["summary"] for e in record["episodes"]]
    report = {"pilot": pilot.provenance, "runtime": record["runtime"],
              "episodes": summaries, "landed": sum(s["status"] == "landed" for s in summaries),
              "total": len(summaries), "interventions": sum(s["interventions"] for s in summaries),
              "guidance_matches": sum(s["guidance_matches"] for s in summaries),
              "decisions": sum(s["decisions"] for s in summaries)}
    write_json(args.report, report, indent=2)
    print(f"Landed {report['landed']}/{report['total']}; overrides {report['interventions']}")


if __name__ == "__main__":
    main()
