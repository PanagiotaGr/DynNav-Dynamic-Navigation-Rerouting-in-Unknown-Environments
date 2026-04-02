"""
Contribution 15: Neuromorphic Sensing for Navigation
=====================================================
Simulates event-camera (DVS — Dynamic Vision Sensor) output and a lightweight
Spiking Neural Network (SNN) for ultra-low-latency obstacle detection.

Event cameras fire asynchronously per-pixel when brightness changes exceed a
threshold — yielding μs-resolution timestamps instead of global frame captures.
SNNs process spikes temporally, matching the event-driven paradigm perfectly.

Research Question (RQ-Neuro): Does event-camera-based obstacle detection
reduce reaction latency and false-negative rate in high-speed scenarios
compared to frame-based LiDAR or depth camera pipelines?

References:
    Gallego et al. (2020) "Event-based Vision: A Survey" IEEE TPAMI
    Mahowald (1994) "VLSI Analogs of Neuronal Visual Processing"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------

class DVSEvent(NamedTuple):
    """A single DVS event: pixel (x,y), timestamp (µs), polarity."""
    x: int
    y: int
    timestamp_us: float    # microseconds
    polarity: int          # +1 (ON) or -1 (OFF)


# ---------------------------------------------------------------------------
# Event camera simulator
# ---------------------------------------------------------------------------

@dataclass
class DVSSimulatorConfig:
    height: int = 240
    width: int = 320
    threshold_pos: float = 0.15    # log-intensity change to fire ON event
    threshold_neg: float = 0.15    # log-intensity change to fire OFF event
    noise_rate: float = 0.001      # fraction of pixels firing spurious events/frame
    refractory_us: float = 500.0   # refractory period in µs


class DVSSimulator:
    """
    Converts a sequence of greyscale frames (H×W, float [0,1]) into
    asynchronous DVS events by tracking log-intensity changes per pixel.
    """

    def __init__(self, cfg: DVSSimulatorConfig | None = None):
        self.cfg = cfg or DVSSimulatorConfig()
        self._log_ref = np.full(
            (self.cfg.height, self.cfg.width), -np.inf
        )
        self._last_fire_us = np.zeros(
            (self.cfg.height, self.cfg.width)
        )
        self._t_us = 0.0

    def process_frame(self, frame: np.ndarray,
                      dt_us: float = 10_000.0) -> list[DVSEvent]:
        """
        Process one greyscale frame. dt_us = inter-frame gap in microseconds.
        Returns list of DVSEvent fired since the last frame.
        """
        assert frame.shape == (self.cfg.height, self.cfg.width), \
            f"Frame shape mismatch: got {frame.shape}"

        self._t_us += dt_us
        log_frame = np.log(np.clip(frame, 1e-6, 1.0))

        events: list[DVSEvent] = []
        rng = np.random.default_rng()

        # Determine which pixels have a valid reference
        has_ref = self._log_ref > -np.inf
        delta = np.where(has_ref, log_frame - self._log_ref, 0.0)

        # Refractory mask
        ready = (self._t_us - self._last_fire_us) >= self.cfg.refractory_us

        on_mask = (delta >= self.cfg.threshold_pos) & ready
        off_mask = (delta <= -self.cfg.threshold_neg) & ready

        for mask, pol in [(on_mask, +1), (off_mask, -1)]:
            ys, xs = np.where(mask)
            for y, x in zip(ys.tolist(), xs.tolist()):
                # Jitter timestamp within the frame interval
                t_event = self._t_us - dt_us + rng.uniform(0, dt_us)
                events.append(DVSEvent(int(x), int(y), t_event, pol))
                self._last_fire_us[y, x] = t_event
                self._log_ref[y, x] = log_frame[y, x]

        # Add noise events
        n_noise = int(self.cfg.noise_rate * self.cfg.height * self.cfg.width)
        if n_noise > 0:
            nx = rng.integers(0, self.cfg.width, n_noise)
            ny = rng.integers(0, self.cfg.height, n_noise)
            nt = rng.uniform(self._t_us - dt_us, self._t_us, n_noise)
            np_pol = rng.choice([-1, 1], n_noise)
            for x, y, t, p in zip(nx, ny, nt, np_pol):
                events.append(DVSEvent(int(x), int(y), float(t), int(p)))

        events.sort(key=lambda e: e.timestamp_us)
        return events


# ---------------------------------------------------------------------------
# Event surface (time-surface representation)
# ---------------------------------------------------------------------------

def event_to_time_surface(events: list[DVSEvent],
                            h: int, w: int,
                            tau_us: float = 30_000.0) -> np.ndarray:
    """
    Convert a list of events to a 2-channel time surface (H×W×2).
    Channel 0: ON polarity exponential decay surface.
    Channel 1: OFF polarity exponential decay surface.
    """
    surface = np.zeros((h, w, 2), dtype=np.float32)
    if not events:
        return surface

    t_ref = events[-1].timestamp_us
    for ev in events:
        ch = 0 if ev.polarity == 1 else 1
        v = np.exp(-(t_ref - ev.timestamp_us) / tau_us)
        surface[ev.y, ev.x, ch] = max(surface[ev.y, ev.x, ch], v)

    return surface


# ---------------------------------------------------------------------------
# Spiking Neural Network (Leaky Integrate-and-Fire)
# ---------------------------------------------------------------------------

@dataclass
class LIFConfig:
    tau_mem: float = 20.0    # membrane time constant (ms)
    v_thresh: float = 1.0    # spike threshold
    v_reset: float = 0.0     # reset potential
    dt: float = 1.0          # simulation time step (ms)


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron."""

    def __init__(self, cfg: LIFConfig | None = None):
        self.cfg = cfg or LIFConfig()
        self.v = self.cfg.v_reset
        self.spike = False

    def step(self, i_ext: float) -> bool:
        """One dt step. Returns True if neuron fired."""
        decay = np.exp(-self.cfg.dt / self.cfg.tau_mem)
        self.v = decay * self.v + (1 - decay) * i_ext
        if self.v >= self.cfg.v_thresh:
            self.spike = True
            self.v = self.cfg.v_reset
        else:
            self.spike = False
        return self.spike


class SNNObstacleDetector:
    """
    Single-layer SNN that maps an event time-surface to obstacle
    likelihood per spatial region (grid of N×M cells).
    """

    def __init__(self, grid_n: int = 4, grid_m: int = 4,
                 lif_cfg: LIFConfig | None = None):
        self.gn, self.gm = grid_n, grid_m
        self.lif_cfg = lif_cfg or LIFConfig()
        rng = np.random.default_rng(0)
        # Random weights (train offline; here initialised randomly)
        self.W = rng.standard_normal((grid_n * grid_m, 2)) * 0.5
        self.neurons = [LIFNeuron(self.lif_cfg) for _ in range(grid_n * grid_m)]

    def detect(self, surface: np.ndarray,
               n_steps: int = 10) -> np.ndarray:
        """
        Parameters
        ----------
        surface : (H, W, 2) time surface
        Returns  : (N, M) obstacle probability map
        """
        h, w, _ = surface.shape
        ph, pw = h // self.gn, w // self.gm
        spike_counts = np.zeros(self.gn * self.gm)

        for step in range(n_steps):
            for idx in range(self.gn * self.gm):
                r, c = divmod(idx, self.gm)
                cell = surface[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw, :]
                # Aggregate event energy
                i_on = cell[:, :, 0].mean()
                i_off = cell[:, :, 1].mean()
                i_ext = float(np.array([i_on, i_off]) @ self.W[idx])
                if self.neurons[idx].step(i_ext):
                    spike_counts[idx] += 1

        # Normalise to [0, 1]
        prob = spike_counts / n_steps
        return prob.reshape(self.gn, self.gm)

    def reset(self) -> None:
        for n in self.neurons:
            n.v = n.cfg.v_reset
            n.spike = False
