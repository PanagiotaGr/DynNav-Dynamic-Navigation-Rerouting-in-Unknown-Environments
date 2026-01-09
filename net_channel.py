from collections import deque
import numpy as np

class PacketChannel:
    """
    Simulates packet loss + fixed delay on a stream of measurements.
    - p_loss: probability a packet is dropped (Bernoulli)
    - delay_steps: fixed latency in discrete steps
    """
    def __init__(self, p_loss: float = 0.0, delay_steps: int = 0, seed: int = 0):
        self.p_loss = float(p_loss)
        self.delay_steps = int(delay_steps)
        self.rng = np.random.default_rng(seed)

        # Buffer holds past packets so we can output a delayed one
        self.buf = deque(maxlen=self.delay_steps + 1 if self.delay_steps >= 0 else 1)

        # Pre-fill buffer with "no packet"
        for _ in range(self.delay_steps + 1):
            self.buf.append(None)

    def push(self, packet):
        """Push new packet into channel. Returns (out_packet, dropped_flag)."""
        dropped = False

        # packet loss
        if self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            packet = None
            dropped = True

        self.buf.append(packet)

        # output delayed packet
        out = self.buf[0]  # oldest
        return out, dropped
