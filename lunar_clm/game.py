"""Original arcade-inspired physics; no ROM, Gymnasium or Box2D required."""

from dataclasses import asdict, dataclass
import math
import random

GRAVITY = 1.62
THRUST = 5.0
TURN_RATE = 30.0  # degrees/second; positive tilts toward +x
DT = 0.02
CONTROL_STEPS = 10
RADIUS = 8.0
# Each pad tops its own hill; heights differ so approaches cross ridges and valleys.
TERRAIN = ((0, 45), (60, 70), (120, 140), (210, 140), (270, 45), (330, 25),
           (380, 40), (430, 70), (550, 70), (600, 30), (680, 40),
           (780, 210), (830, 210), (900, 80), (1000, 60))
PADS = ((120, 210, 140, 2), (430, 550, 70, 1), (780, 830, 210, 4))


def ground(x):
    x = max(0, min(1000, x))
    for (x0, y0), (x1, y1) in zip(TERRAIN, TERRAIN[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return TERRAIN[-1][1]


def ceiling(x0, x1):
    """Highest terrain over [x0, x1], including vertices between the ends."""
    return max(ground(x0), ground(x1), *(y for px, y in TERRAIN if x0 <= px <= x1))


@dataclass(frozen=True)
class Command:
    turn: int = 0
    throttle: float = 0

    def __post_init__(self):
        if self.turn not in (-1, 0, 1) or not math.isfinite(self.throttle) or not 0 <= self.throttle <= 1:
            raise ValueError("turn must be -1/0/1 and throttle must be finite in [0, 1]")


@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    fuel: float = 100
    time: float = 0
    status: str = "flying"
    score: int = 0


# Custom starts: away from the walls, clear of the terrain and inside the visible sky.
START_MARGIN = 50
START_CEILING = 690


def custom_start(x, y, angle):
    """Validated resting start; guidance recovers from any tilt inside these limits."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y, angle)):
        raise ValueError("start x, y and angle must be finite numbers")
    if not START_MARGIN <= x <= 1000 - START_MARGIN:
        raise ValueError(f"start x must be within {START_MARGIN}..{1000 - START_MARGIN} m")
    lowest = ceiling(x - RADIUS, x + RADIUS) + RADIUS + START_MARGIN
    if not lowest <= y <= START_CEILING:
        raise ValueError(f"start y at x={x:.0f} must be within {lowest:.0f}..{START_CEILING} m")
    return State(x, y, 0, 0, (angle + 180) % 360 - 180)


class Game:
    def __init__(self, seed=0, target=1):
        if target not in range(len(PADS)):
            raise ValueError("target must be 0, 1 or 2")
        self.seed, self.target = seed, target
        rng = random.Random(seed)
        center = sum(PADS[target][:2]) / 2
        self.state = State(center + rng.uniform(-100, 100), 450,
                           rng.uniform(-8, 8), -8, rng.uniform(-12, 12))

    def snapshot(self):
        return asdict(self.state)

    def step(self, command):
        """Advance at most 0.2 simulation seconds; terminal states are absorbing."""
        s = self.state
        for _ in range(CONTROL_STEPS):
            if s.status != "flying":
                break
            # Both attitude jets and the main engine consume the same finite tank.
            cost = (command.throttle * 0.65 + abs(command.turn) * 0.04) * DT
            fraction = min(1.0, s.fuel / cost) if cost else 1.0
            s.fuel = max(0.0, s.fuel - cost)
            s.angle = (s.angle + command.turn * TURN_RATE * DT * fraction + 180) % 360 - 180
            thrust = command.throttle * THRUST * fraction
            s.vx += math.sin(math.radians(s.angle)) * thrust * DT
            s.vy += (math.cos(math.radians(s.angle)) * thrust - GRAVITY) * DT
            s.x += s.vx * DT
            s.y += s.vy * DT
            s.time += DT
            if s.x < RADIUS or s.x > 1000 - RADIUS or s.y > 750:
                s.status = "out_of_bounds"
            elif s.y - RADIUS <= self.surface_height():
                pad = next((p for p in PADS if p[0] + RADIUS <= s.x <= p[1] - RADIUS), None)
                safe = (pad is not None and abs(s.vx) <= 2 and -3 <= s.vy <= 0
                        and abs(s.angle) <= 8)
                s.status = "landed" if safe else "crashed"
                s.score = 50 * pad[3] if safe else 0
            elif s.time >= 180:
                s.status = "timeout"
        return self.snapshot()

    def surface_height(self):
        """Conservative horizontal footprint, including terrain vertices under the hull."""
        return ceiling(self.state.x - RADIUS, self.state.x + RADIUS)
