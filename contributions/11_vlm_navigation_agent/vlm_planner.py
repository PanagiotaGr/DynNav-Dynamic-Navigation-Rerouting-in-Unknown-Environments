"""
Contribution 11: VLM Navigation Agent
======================================
Uses a Vision-Language Model (VLM) as a high-level semantic planner.
The VLM interprets camera frames, identifies semantic regions (room, corridor,
obstacle type) and generates natural-language goals that are translated into
metric waypoints for the downstream A* planner.

Research Question (RQ-VLM): Can a foundation VLM replace hand-crafted
semantic labelling and improve goal-specification in unknown environments?

Author: DynNav Project
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SemanticGoal:
    """A goal produced by the VLM planner."""
    description: str               # e.g. "navigate to the exit door on the left"
    region_label: str              # e.g. "corridor", "doorway", "open_space"
    confidence: float              # VLM confidence in [0, 1]
    pixel_hint: Optional[tuple[int, int]] = None   # (u, v) pixel of region centroid
    metric_waypoint: Optional[tuple[float, float]] = None  # (x, y) in map frame


@dataclass
class VLMPlannerConfig:
    model_name: str = "llava-1.6"          # or "gpt-4-vision-preview"
    api_endpoint: str = "http://localhost:11434/api/chat"  # Ollama default
    confidence_threshold: float = 0.55
    max_retries: int = 3
    image_resize: tuple[int, int] = (640, 480)
    prompt_template: str = (
        "You are a robot navigation assistant. "
        "Analyse the image and identify: "
        "(1) the main navigable regions, "
        "(2) any obstacles or hazards, "
        "(3) the best next waypoint direction. "
        "Reply ONLY with JSON: "
        '{"region": "<label>", "goal": "<short instruction>", '
        '"confidence": <0-1>, "pixel_u": <int>, "pixel_v": <int>}'
    )
    system_prompt: str = "You are an expert mobile-robot navigation assistant."


# ---------------------------------------------------------------------------
# VLM interface (provider-agnostic)
# ---------------------------------------------------------------------------

class VLMNavigationPlanner:
    """
    Sends camera frames to a VLM and converts semantic goals into metric
    waypoints compatible with DynNav's existing A* backend.

    Supports:
        - Local Ollama (LLaVA, BakLLaVA)
        - OpenAI GPT-4V (set api_endpoint accordingly)
        - Any OpenAI-compatible server
    """

    def __init__(self, config: VLMPlannerConfig | None = None,
                 depth_scale: float = 1.0):
        self.cfg = config or VLMPlannerConfig()
        self.depth_scale = depth_scale   # metres per depth unit
        self._session_goals: list[SemanticGoal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, rgb_frame: np.ndarray,
             depth_frame: Optional[np.ndarray] = None,
             current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
             ) -> Optional[SemanticGoal]:
        """
        Main entry point. Feed an RGB (H×W×3 uint8) frame, get a SemanticGoal.
        """
        encoded = self._encode_frame(rgb_frame)
        raw = self._query_vlm(encoded)
        if raw is None:
            return None

        goal = self._parse_response(raw)
        if goal is None or goal.confidence < self.cfg.confidence_threshold:
            logger.warning("VLM goal below confidence threshold: %s", goal)
            return None

        if depth_frame is not None and goal.pixel_hint is not None:
            goal.metric_waypoint = self._pixel_to_metric(
                goal.pixel_hint, depth_frame, current_pose
            )

        self._session_goals.append(goal)
        logger.info("VLM goal accepted: %s (conf=%.2f)", goal.description, goal.confidence)
        return goal

    def session_summary(self) -> dict:
        """Return stats on goals produced this session."""
        if not self._session_goals:
            return {"total": 0}
        confs = [g.confidence for g in self._session_goals]
        labels = [g.region_label for g in self._session_goals]
        return {
            "total": len(self._session_goals),
            "mean_confidence": float(np.mean(confs)),
            "region_distribution": {
                lbl: labels.count(lbl) for lbl in set(labels)
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Base64-encode an RGB frame for transmission to the VLM API."""
        try:
            from PIL import Image
            import io
            img = Image.fromarray(frame).resize(self.cfg.image_resize)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            # Fallback: raw numpy bytes
            return base64.b64encode(frame.tobytes()).decode("utf-8")

    def _query_vlm(self, encoded_image: str) -> Optional[str]:
        """Send request to VLM endpoint; returns raw text response or None."""
        import urllib.request, urllib.error

        payload = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {
                    "role": "user",
                    "content": self.cfg.prompt_template,
                    "images": [encoded_image],
                },
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")

        for attempt in range(self.cfg.max_retries):
            try:
                req = urllib.request.Request(
                    self.cfg.api_endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    # Support both Ollama and OpenAI response shapes
                    if "message" in result:
                        return result["message"]["content"]
                    if "choices" in result:
                        return result["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError) as e:
                logger.warning("VLM attempt %d failed: %s", attempt + 1, e)

        return None

    def _parse_response(self, raw: str) -> Optional[SemanticGoal]:
        """Parse JSON from the VLM response."""
        try:
            # Strip markdown fences if present
            text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(text)
            return SemanticGoal(
                description=str(data.get("goal", "")),
                region_label=str(data.get("region", "unknown")),
                confidence=float(data.get("confidence", 0.0)),
                pixel_hint=(
                    int(data["pixel_u"]), int(data["pixel_v"])
                ) if "pixel_u" in data else None,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse VLM response: %s | raw=%s", e, raw[:200])
            return None

    def _pixel_to_metric(self, pixel: tuple[int, int],
                         depth: np.ndarray,
                         pose: tuple[float, float, float]) -> tuple[float, float]:
        """
        Back-project a pixel (u, v) using the depth map to a (x, y) world point.
        Uses a simple pinhole model; replace with full camera intrinsics for accuracy.
        """
        u, v = pixel
        h, w = depth.shape[:2]
        fx = fy = max(w, h)   # rough focal length
        cx, cy = w / 2, h / 2
        d = float(depth[min(v, h - 1), min(u, w - 1)]) * self.depth_scale

        if d < 0.05:
            logger.warning("Depth at pixel %s too small (%.3f m), skipping", pixel, d)
            return (pose[0], pose[1])

        # Camera-frame 3-D point
        x_cam = (u - cx) / fx * d
        z_cam = d

        # Rotate into world frame using yaw from pose
        yaw = pose[2]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        x_world = pose[0] + cos_y * z_cam - sin_y * x_cam
        y_world = pose[1] + sin_y * z_cam + cos_y * x_cam

        return (float(x_world), float(y_world))
