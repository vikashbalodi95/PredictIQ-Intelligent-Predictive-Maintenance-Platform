"""
simulator/simulator.py
----------------------
A live CNC sensor simulator.

Real factories have PLCs/sensors that emit readings continuously. We don't have
a real machine, so this script *pretends* to be one (actually several). For each
machine it:

  * keeps an internal state (current tool wear, drift, whether it's degrading),
  * generates a fresh, realistic reading every few seconds,
  * POSTs that reading to the backend's /machines/{pk}/readings endpoint,
  * occasionally injects a slow degradation so you get to SEE the health score
    fall, alerts fire, and RUL shrink — then a "maintenance reset" recovers it.

Run it (with the backend already running):
    python -m simulator.simulator

The numbers mirror the AI4I 2020 dataset's distributions, so the trained model
treats them as plausible real machine data.
"""

import random
import time
from dataclasses import dataclass, field

import requests

from core.settings import settings
from utils.logger import get_logger

log = get_logger("simulator")

API = settings.API_BASE_URL


@dataclass
class MachineState:
    """In-memory state for one simulated machine between readings."""
    pk: int                 # database primary key (used in the URL)
    machine_id: str         # human code, e.g. "CNC-001"
    type: str               # L / M / H
    tool_wear: float = 0.0  # minutes — climbs over time, resets on maintenance
    degrading: bool = False # are we currently in a fault-injection episode?
    wear_rate: float = field(default_factory=lambda: random.uniform(1.5, 3.0))


def _wait_for_backend(retries: int = 30, delay: float = 1.0) -> None:
    """Poll the API until it answers, so we don't crash if it starts slightly late."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{API}/health", timeout=3)
            if r.ok:
                log.info("Backend is up.")
                return
        except requests.RequestException:
            pass
        log.info("Waiting for backend (%d/%d) …", attempt, retries)
        time.sleep(delay)
    raise RuntimeError(f"Backend not reachable at {API} after {retries} tries.")


def _load_machines() -> list[MachineState]:
    """Fetch the seeded machines from the API and wrap each in a MachineState."""
    r = requests.get(f"{API}/machines", timeout=10)
    r.raise_for_status()
    machines = r.json()
    if not machines:
        raise RuntimeError("No machines found. Run `python -m scripts.init_db` first.")
    return [MachineState(pk=m["id"], machine_id=m["machine_id"], type=m["type"])
            for m in machines]


def _next_reading(state: MachineState) -> dict:
    """
    Produce the next realistic reading for a machine and advance its state.

    Baseline distributions come from the AI4I 2020 dataset. We add small random
    noise each tick, plus a degradation episode now and then to create
    interesting, alert-worthy behaviour.
    """
    # 1) Tool wear always climbs a little each reading.
    state.tool_wear += state.wear_rate * random.uniform(0.8, 1.2)

    # 2) Randomly enter or leave a "degrading" episode (~3% chance to start).
    if not state.degrading and random.random() < 0.03:
        state.degrading = True
        log.info("%s entering a degradation episode.", state.machine_id)
    elif state.degrading and random.random() < 0.10:
        state.degrading = False  # episode ends naturally

    # 3) Baseline sensor values with noise.
    air_temp = random.gauss(300.0, 1.5)
    process_temp = air_temp + random.gauss(10.0, 1.0)
    rot_speed = random.gauss(1538, 120)
    torque = abs(random.gauss(40.0, 8.0))

    # 4) During a degradation episode, push the stress signals upward so the
    #    model and alert rules react (higher torque, hotter process, more wear).
    if state.degrading:
        torque += random.uniform(15, 30)
        process_temp += random.uniform(4, 9)
        state.tool_wear += random.uniform(2, 5)

    # 5) Maintenance reset: once the tool is worn out, "replace" it.
    if state.tool_wear > 245:
        log.info("%s tool replaced — wear reset.", state.machine_id)
        state.tool_wear = random.uniform(0, 10)
        state.degrading = False
        state.wear_rate = random.uniform(1.5, 3.0)

    return {
        "type": state.type,
        "air_temp": round(air_temp, 2),
        "process_temp": round(process_temp, 2),
        "rotational_speed": round(rot_speed, 1),
        "torque": round(torque, 2),
        "tool_wear": round(state.tool_wear, 1),
    }


def _post_reading(state: MachineState, reading: dict) -> None:
    """Send one reading to the backend and log the resulting status."""
    try:
        r = requests.post(f"{API}/machines/{state.pk}/readings",
                          json=reading, timeout=10)
        r.raise_for_status()
        data = r.json()
        pred = data["prediction"]
        log.info(
            "%s | wear=%5.1f torque=%5.1f | status=%-8s prob=%.0f%% health=%3d RUL=%3dm alerts=%d",
            state.machine_id, reading["tool_wear"], reading["torque"],
            pred["status"], pred["failure_probability"] * 100,
            pred["health_score"], pred["rul_minutes"], data["alerts_created"],
        )
    except requests.RequestException as exc:
        log.error("Failed to post reading for %s: %s", state.machine_id, exc)


def run(interval: float | None = None) -> None:
    """Main loop: forever, generate and post a reading for every machine."""
    interval = interval or settings.SIM_INTERVAL_SECONDS
    _wait_for_backend()
    machines = _load_machines()
    log.info("Simulating %d machine(s) every %.1fs. Press Ctrl+C to stop.",
             len(machines), interval)

    try:
        while True:
            for state in machines:
                reading = _next_reading(state)
                _post_reading(state, reading)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("Simulator stopped by user.")


if __name__ == "__main__":
    run()
