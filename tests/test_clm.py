import copy
import json
import math
import unittest
from unittest.mock import patch

from lunar_clm.benchmark import corpus, summarize
from lunar_clm.guidance import QUESTIONS, guidance, observation
from lunar_clm.pilot import CLMClient, validate_answers


class CLMTests(unittest.TestCase):
    def test_wire_contract_and_persistent_connection(self):
        answers = {name: {"choice": next(iter(q["criteria"])),
                          "probabilities": {label: float(i == 0) for i, label in enumerate(q["criteria"])}}
                   for name, q in QUESTIONS.items()}
        with patch("lunar_clm.pilot.http.client.HTTPConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = json.dumps({"answers": answers}).encode()
            response.getheader.return_value = "12.5"
            client = CLMClient("http://localhost:8700")
            for _ in range(2):
                result = client.predict("test state", QUESTIONS)
                self.assertEqual(validate_answers(result), answers)
                self.assertEqual(result["server_latency_ms"], 12.5)
            factory.assert_called_once()
            args, kwargs = factory.return_value.request.call_args
            self.assertEqual(args, ("POST", "/v1/systemone"))
            self.assertEqual(json.loads(kwargs["body"]), {"state": "test state", "questions": QUESTIONS, "model": "clm-latest"})
            response.status = 502
            with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
                client.predict("state", QUESTIONS)
        for bad in ({}, {"answers": None}, {"answers": {}}):
            with self.assertRaises(ValueError):
                validate_answers(bad)
        for bad in (math.nan, -1, 1.1, True, "0.5"):
            malformed = copy.deepcopy(answers)
            malformed["rotation"]["probabilities"]["left"] = bad
            with self.assertRaises(ValueError):
                validate_answers({"answers": malformed})
        malformed = copy.deepcopy(answers)
        malformed["rotation"]["probabilities"] = ["left", "hold", "right"]
        with self.assertRaises(ValueError):
            validate_answers({"answers": malformed})

    def test_benchmark_corpus_and_statistics(self):
        states = corpus(96, 3000)
        prompts = [observation(g, *guidance(g)) for g in states]
        self.assertEqual(len(set(prompts)), 96)
        self.assertEqual({g.target for g in states}, {0, 1, 2})
        self.assertTrue(all(g.state.status == "flying" for g in states))
        self.assertEqual(prompts, [observation(g, *guidance(g)) for g in corpus(96, 3000)])
        summary = summarize([{"latency_ms": i, "guidance_match": i < 10} for i in range(1, 21)])
        self.assertEqual(summary["p50_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19)
        self.assertEqual(summary["guidance_agreement"], 9 / 20)


if __name__ == "__main__":
    unittest.main()
