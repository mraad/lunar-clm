"""One persistent HTTP connection; two typed choices per control stage."""

from dataclasses import asdict
import http.client
import json
import math
import os
from time import perf_counter
from urllib.parse import urlsplit

from .game import Command
from .guidance import QUESTIONS, THROTTLES, TURNS, guidance, observation
from .small import ACTIONS, DEFAULT_HEAD, QUESTION, SmallAgent, short_observation

MODES = ("baseline", "clm", "assisted", "small", "small-assisted")


class CLMClient:
    def __init__(self, base_url=None, model="clm-latest", timeout=30):
        url = urlsplit(base_url or os.getenv("CLM_BASE_URL", "http://127.0.0.1:8700"))
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("CLM URL must be http(s), without credentials, query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        connection = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        self.connection = connection(url.hostname, url.port, timeout=timeout)
        self.path = url.path.rstrip("/") + "/v1/systemone"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if key := os.getenv("CLM_API_KEY"):
            self.headers["Authorization"] = f"Bearer {key}"
        self.provenance = {"model": model, "transport": "HTTP", "endpoint": url.geturl()}

    def predict(self, state, questions):
        body = json.dumps({"model": self.model, "state": state, "questions": questions}, allow_nan=False)
        try:
            self.connection.request("POST", self.path, body=body, headers=self.headers)
            response = self.connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f"CLM returned HTTP {response.status}; check the model, key and encoder server")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("CLM response must be a JSON object")
            server_ms = response.getheader("X-CLM-Latency-Ms")
            if server_ms is not None:
                value = float(server_ms)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid server latency")
                result["server_latency_ms"] = value
            return result
        except (OSError, http.client.HTTPException) as exc:
            self.close()
            raise RuntimeError("CLM unavailable; start clm-serve or set CLM_BASE_URL") from exc

    def close(self):
        self.connection.close()


def validate_answers(result, questions=QUESTIONS):
    """Reject malformed model output instead of silently substituting guidance."""
    try:
        answers = result["answers"]
        for name, question in questions.items():
            labels = question["criteria"]
            answer = answers[name]
            if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
                raise ValueError("choice answer and probabilities must be objects")
            probabilities = answer["probabilities"]
            if answer["choice"] not in labels or set(probabilities) != set(labels):
                raise ValueError("unknown choice or incomplete distribution")
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.002):
                raise ValueError("invalid probabilities")
        return answers
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CLM answer: {exc}") from exc


class Pilot:
    def __init__(self, mode="clm", model=None, base_url=None, timeout=30, agent=None):
        if mode not in MODES:
            raise ValueError("unknown pilot mode")
        self.mode = mode
        self.small = mode.startswith("small")
        self.observe, self.questions = (short_observation, QUESTION) if self.small else (observation, QUESTIONS)
        if mode == "baseline":
            model = None
        else:
            model = model or (DEFAULT_HEAD if self.small else "clm-latest")
            if agent is None:
                agent = SmallAgent(model) if self.small else CLMClient(base_url, model, timeout)
        self.agent = agent
        self.provenance = {"mode": mode, "model": model, **getattr(agent, "provenance", {})}

    def decide(self, game):
        start = perf_counter()
        reference, metrics = guidance(game)
        prompt = self.observe(game, reference, metrics)
        answers, result, proposed = None, {}, reference
        if self.mode != "baseline":
            result = self.agent.predict(prompt, self.questions)
            answers = validate_answers(result, self.questions)
            proposed = (ACTIONS[answers["action"]["choice"]] if self.small else
                        Command(TURNS[answers["rotation"]["choice"]], THROTTLES[answers["engine"]["choice"]]))
        intervened = self.mode.endswith("assisted") and proposed != reference
        executed = reference if intervened else proposed
        return executed, {"reference": asdict(reference), "proposed": asdict(proposed), "executed": asdict(executed),
                          "intervened": intervened, "answers": answers, "guidance": metrics,
                          "prompt": prompt, "latency_ms": (perf_counter() - start) * 1000,
                          "server_latency_ms": result.get("server_latency_ms"), "usage": result.get("usage")}

    def close(self):
        if self.agent is not None:
            self.agent.close()
