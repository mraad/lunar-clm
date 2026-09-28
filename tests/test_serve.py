import http.client
import json
import threading
import unittest

from lunar_clm.game import PADS
from lunar_clm.pilot import Pilot
from lunar_clm.serve import make_server


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.server = make_server(Pilot("baseline"), port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)
        self.addCleanup(connection.close)
        connection.request(method, path, body=body and json.dumps(body), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()

    def test_page_and_live_flight(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn('"interactive":true', page)
        status, body = self.request("POST", "/fly", {"x": 900, "y": 600, "angle": 170, "target": 0})
        self.assertEqual(status, 200)
        messages = [json.loads(line) for line in body.splitlines()]
        frames = [m["frame"] for m in messages[:-1]]
        self.assertEqual((frames[0]["before"]["x"], frames[0]["before"]["y"], frames[0]["before"]["angle"]), (900, 600, 170))
        summary = messages[-1]["summary"]
        self.assertEqual((summary["status"], summary["target"], summary["decisions"]), ("landed", 0, len(frames)))
        self.assertTrue(PADS[0][0] <= summary["x"] <= PADS[0][1])

    def test_rejects_invalid_starts(self):
        for body in ({"x": 900, "y": 600, "angle": 0}, {"x": 900, "y": 600, "angle": 0, "target": 1.0},
                     {"x": 10, "y": 600, "angle": 0, "target": 1}, {"x": 500, "y": 70, "angle": 0, "target": 1},
                     [1, 2]):
            with self.subTest(body=body):
                status, text = self.request("POST", "/fly", body)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(text))
        self.assertEqual(self.request("GET", "/missing")[0], 404)


if __name__ == "__main__":
    unittest.main()
