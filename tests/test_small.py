import unittest

from lunar_clm.game import Game
from lunar_clm.guidance import guidance
from lunar_clm.pilot import Pilot
from lunar_clm.small import ACTIONS, QUESTION, short_observation
from lunar_clm.train_small import dataset


class Stub:
    def predict(self, state, questions):
        assert questions == QUESTION
        assert "Requested rotation:" in state
        return {"answers": {"action": {"choice": "left/full",
                "probabilities": {label: float(label == "left/full") for label in ACTIONS}}}}


class SmallTests(unittest.TestCase):
    def test_joint_choice_is_executed_without_hidden_guidance(self):
        game = Game(3000)
        raw, info = Pilot("small", agent=Stub()).decide(game)
        assisted, shield = Pilot("small-assisted", agent=Stub()).decide(game)
        self.assertEqual(raw, ACTIONS["left/full"])
        self.assertNotEqual(raw, guidance(game)[0])
        self.assertFalse(info["intervened"])
        self.assertEqual(assisted, guidance(game)[0])
        self.assertTrue(shield["intervened"])

    def test_training_split_and_all_nine_actions(self):
        train = dataset(1000, 2, 20, True)
        validation = dataset(2000, 2, 20)
        self.assertFalse({r["seed"] for r in train} & {r["seed"] for r in validation})
        self.assertEqual({r["label"] for r in train}, set(range(9)))
        self.assertTrue(any(r.get("augmented") for r in train))
        self.assertFalse(any(r.get("augmented") for r in validation))
        game = Game(3000)
        text = short_observation(game, *guidance(game))
        game.state.vx += 1
        self.assertNotEqual(text, short_observation(game, *guidance(game)))


if __name__ == "__main__":
    unittest.main()
