"""Unchanged flight guidance and question text from lunar-laya."""

import math

from .game import Command, GRAVITY, PADS, RADIUS, THRUST, ceiling

QUESTIONS = {
    "rotation": {
        "type": "choice",
        "instructions": "Control lunar lander tilt. Follow the requested tilt correction.",
        "criteria": {"left": "Decrease tilt angle", "hold": "Keep current tilt", "right": "Increase tilt angle"},
    },
    "engine": {
        "type": "choice",
        "instructions": "Control lunar lander engine. Follow the requested engine power.",
        "criteria": {"off": "Zero thrust", "half": "Half thrust", "full": "Full thrust"},
    },
}
TURNS = {"left": -1, "hold": 0, "right": 1}
THROTTLES = {"off": 0.0, "half": 0.5, "full": 1.0}
TURN_NAMES = {v: k for k, v in TURNS.items()}
THROTTLE_NAMES = {v: k for k, v in THROTTLES.items()}


def guidance(game):
    """PD navigation produces a desired tilt and vertical acceleration, not a search."""
    s = game.state
    pad = PADS[game.target]
    center = (pad[0] + pad[1]) / 2
    dx = center - s.x
    altitude = max(0, s.y - pad[2] - RADIUS)
    # Height of the hull above the highest terrain between the lander and the pad.
    clearance = s.y - RADIUS - ceiling(min(s.x, center) - RADIUS, max(s.x, center) + RADIUS)
    desired_vx = max(-10, min(10, dx * 0.15))
    desired_vy = -min(12, max(0.7, altitude * 0.12))
    # Hold above every ridge on the way until horizontal alignment is recovered;
    # below a ridge, climb first and only then close the distance.
    if abs(dx) > 35 and clearance < 120:
        desired_vy = max(desired_vy, min(12, (120 - clearance) * 0.15))
        desired_vx *= max(0, min(1, clearance / 40))
    ax = max(-1.8, min(1.8, (desired_vx - s.vx) * 0.65))
    ay = GRAVITY + (desired_vy - s.vy) * 0.8
    desired_angle = max(-30, min(30, math.degrees(math.atan2(ax, max(0.8, ay)))))
    error = (desired_angle - s.angle + 180) % 360 - 180
    turn = 1 if error > 3 else -1 if error < -3 else 0
    # Beyond 60 degrees of tilt the engine pushes sideways or down: rotate first.
    cos = math.cos(math.radians(s.angle))
    power = 0 if cos < 0.5 else max(0, min(1, ay / (THRUST * cos)))
    throttle = 0 if power < 0.25 else 0.5 if power < 0.75 else 1.0
    return Command(turn, throttle), {"target_dx": dx, "altitude": altitude,
                                      "desired_angle": desired_angle, "desired_vy": desired_vy}


def observation(game, requested, metrics):
    s = game.state
    rotation, engine = TURN_NAMES[requested.turn], THROTTLE_NAMES[requested.throttle]
    return (f"Lunar landing. Requested tilt correction: {rotation}. Requested engine power: {engine}. "
            f"Altitude {metrics['altitude']:.1f} m. Pad offset {metrics['target_dx']:.1f} m. "
            f"Horizontal velocity {s.vx:.1f} m/s. Vertical velocity {s.vy:.1f} m/s. "
            f"Tilt {s.angle:.1f} degrees; desired {metrics['desired_angle']:.1f}. "
            f"Desired vertical velocity {metrics['desired_vy']:.1f} m/s. Fuel {s.fuel:.1f}. "
            "Positive x is right, positive y is up, positive tilt is right. "
            "Land upright with horizontal speed <=2 and downward speed <=3 m/s.")


