import json
import math
from pathlib import Path
import tempfile
import unittest

from lunar_clm.cli import export_replay, main, run_episode
from lunar_clm.game import (Command, DT, Game, GRAVITY, PADS, RADIUS, START_MARGIN, State,
                            ceiling, custom_start, ground)
from lunar_clm.pilot import Pilot, QUESTIONS, validate_answers


class FakeAgent:
    def predict(self, state, questions):
        assert questions == QUESTIONS
        assert "Altitude" in state
        return {"answers": {
            "rotation": {"choice": "left", "probabilities": {"left": 1, "hold": 0, "right": 0}},
            "engine": {"choice": "off", "probabilities": {"off": 1, "half": 0, "full": 0}},
        }}


class LunarTests(unittest.TestCase):
    def test_seed_and_gravity(self):
        a, b = Game(42), Game(42)
        self.assertEqual(a.snapshot(), b.snapshot())
        old = a.state.vy
        a.step(Command())
        self.assertAlmostEqual(a.state.vy, old - GRAVITY * 0.2)
        self.assertAlmostEqual(a.state.time, 0.2)
        self.assertEqual(a.state.fuel, 100)

    def test_thrust_direction_and_dry_tank(self):
        g = Game()
        g.state = State(500, 400, 0, 0, 90)
        g.step(Command(0, 1))
        self.assertAlmostEqual(g.state.vx, 1)
        self.assertLess(g.state.fuel, 100)
        g.state.fuel = 0
        vx, angle = g.state.vx, g.state.angle
        g.step(Command(1, 1))
        self.assertEqual(g.state.vx, vx)
        self.assertEqual(g.state.angle, angle)
        self.assertEqual(g.state.fuel, 0)

    def test_partial_fuel_and_validation(self):
        g = Game()
        g.state = State(500, 400, 0, 0, 90, fuel=0.65 * DT / 2)
        g.step(Command(0, 1))
        self.assertAlmostEqual(g.state.vx, 5 * DT / 2)
        self.assertEqual(g.state.fuel, 0)
        for throttle in (-1, 2, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Command(0, throttle)
        with self.assertRaises(ValueError):
            Command(2)

    def test_contacts_and_terminal_absorption(self):
        for vx, vy, tilt, status in ((0, -1, 0, "landed"), (3, -1, 0, "crashed"),
                                     (0, -4, 0, "crashed"), (0, -1, 10, "crashed")):
            g = Game()
            g.state = State(500, PADS[1][2] + RADIUS + 0.01, vx, vy, tilt)
            g.step(Command())
            self.assertEqual(g.state.status, status)
            final = g.snapshot()
            g.step(Command(1, 1))
            self.assertEqual(g.snapshot(), final)
        for target, pad in enumerate(PADS):
            g = Game(target=target)
            g.state = State((pad[0]+pad[1])/2, pad[2]+RADIUS+0.01, 0, -1, 0)
            g.step(Command())
            self.assertEqual(g.state.score, 50 * pad[3])
        g = Game()
        g.state = State(PADS[1][0], PADS[1][2] + RADIUS + 0.01, 0, -1, 0)  # hull straddles hill edge
        g.step(Command())
        self.assertEqual(g.state.status, "crashed")

    def test_bounds_timeout_terrain(self):
        self.assertEqual(ground(500), PADS[1][2])
        # Every pad tops its own hill: terrain falls away on both sides, heights differ.
        self.assertEqual(len({pad[2] for pad in PADS}), len(PADS))
        for x0, x1, height, _ in PADS:
            self.assertLess(ground(x0 - 30), height)
            self.assertLess(ground(x1 + 30), height)
        g = Game()
        g.state.x = 1
        g.step(Command())
        self.assertEqual(g.state.status, "out_of_bounds")
        g = Game()
        g.state.time = 180
        g.step(Command())
        self.assertEqual(g.state.status, "timeout")

    def test_guidance_lands_seed_matrix(self):
        for target in range(3):
            for seed in range(10):
                with self.subTest(target=target, seed=seed):
                    e = run_episode(Pilot("baseline"), seed, target, 900)
                    self.assertEqual(e["summary"]["status"], "landed")
                    self.assertEqual(e["summary"]["interventions"], 0)

    def test_guidance_lands_from_custom_starts(self):
        for target in range(len(PADS)):
            for x in (50, 330, 640, 950):
                lowest = ceiling(x - RADIUS, x + RADIUS) + RADIUS + START_MARGIN
                for y in (lowest, 400):
                    for angle in (-180, -90, 0, 120):
                        with self.subTest(target=target, x=x, y=y, angle=angle):
                            e = run_episode(Pilot("baseline"), 0, target, 900, start=custom_start(x, y, angle))
                            self.assertEqual(e["summary"]["status"], "landed")
        for bad in ((10, 400, 0), (500, 60, 0), (500, 400, math.nan), (500, 700, 0), (True, 400, 0)):
            with self.assertRaises(ValueError):
                custom_start(*bad)

    def test_model_proposal_and_intervention_are_separate(self):
        game = Game()
        raw, raw_info = Pilot("clm", agent=FakeAgent()).decide(game)
        guided, info = Pilot("assisted", agent=FakeAgent()).decide(game)
        expected, _ = Pilot("baseline").decide(game)
        self.assertEqual(raw, Command(-1, 0))
        self.assertEqual(guided, expected)
        self.assertEqual(info["proposed"], raw_info["executed"])
        self.assertEqual(info["intervened"], raw != expected)
        self.assertFalse(raw_info["intervened"])

    def test_recording_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, replay = Path(tmp)/"run.json", Path(tmp)/"replay.html"
            main(["--pilot", "baseline", "--steps", "2", "--out", str(out), "--replay", str(replay)])
            record = json.loads(out.read_text())
            e = record["episodes"][0]
            self.assertEqual(e["summary"]["status"], "truncated")
            self.assertEqual(e["frames"][0]["after"], e["frames"][1]["before"])
            self.assertIsNone(e["frames"][0]["decision"]["answers"])
            self.assertIn("const recording = {", replay.read_text())
            record["pilot"]["model"] = "</script><script>alert(1)</script>"
            export_replay(record, replay)
            self.assertNotIn(record["pilot"]["model"], replay.read_text())


if __name__ == "__main__":
    unittest.main()
