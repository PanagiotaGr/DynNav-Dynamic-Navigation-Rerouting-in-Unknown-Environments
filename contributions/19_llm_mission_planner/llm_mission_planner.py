"""
Contribution 19: LLM Mission Planner
=====================================
Translates natural-language mission instructions into structured waypoint
sequences compatible with DynNav's A* / topological planner backend.

"Go to the kitchen, then check the corridor and return to start"
→ [("kitchen", priority=1), ("corridor_A", priority=2), ("start", priority=3)]

Supports:
- Local Ollama (llama3, mistral)
- OpenAI-compatible APIs (GPT-4, Claude via API)
- Offline fallback with keyword extraction

Research Question (RQ-LLM): Can natural-language mission specifications
reduce task-completion time and error rate vs. manual waypoint programming?

Author: DynNav Project
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    label: str                          # semantic zone name
    priority: int                       # execution order
    action: str = "navigate"           # "navigate" | "inspect" | "wait"
    duration_s: float = 0.0            # wait duration if action=="wait"
    metric_xy: Optional[tuple[float, float]] = None


@dataclass
class Mission:
    raw_instruction: str
    waypoints: list[Waypoint] = field(default_factory=list)
    estimated_duration_s: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "instruction": self.raw_instruction,
            "confidence": round(self.confidence, 3),
            "waypoints": [
                {
                    "label": w.label,
                    "priority": w.priority,
                    "action": w.action,
                    "duration_s": w.duration_s,
                }
                for w in self.waypoints
            ],
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMPlannerConfig:
    model: str = "llama3"
    endpoint: str = "http://localhost:11434/api/chat"
    temperature: float = 0.1
    max_tokens: int = 512
    confidence_threshold: float = 0.5
    known_zones: list[str] = field(default_factory=lambda: [
        "kitchen", "corridor", "living_room", "office",
        "bedroom", "bathroom", "entrance", "start", "exit",
        "charging_station", "storage_room",
    ])
    system_prompt: str = (
        "You are a robot mission planner. "
        "Parse the user's instruction into an ordered list of waypoints. "
        "Reply ONLY with valid JSON — no other text. Format:\n"
        '{"confidence": 0.9, "waypoints": ['
        '{"label": "kitchen", "priority": 1, "action": "navigate", "duration_s": 0},'
        '{"label": "corridor", "priority": 2, "action": "inspect", "duration_s": 5}'
        "]}"
    )


# ---------------------------------------------------------------------------
# LLM Mission Planner
# ---------------------------------------------------------------------------

class LLMMissionPlanner:
    """
    Parses natural-language instructions into Mission objects.
    Falls back to keyword extraction if LLM is unavailable.
    """

    def __init__(self, config: LLMPlannerConfig | None = None):
        self.cfg = config or LLMPlannerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, instruction: str) -> Mission:
        """
        Main entry: parse a natural-language instruction into a Mission.
        Tries LLM first, falls back to keyword extractor.
        """
        raw = self._query_llm(instruction)
        if raw:
            mission = self._parse_llm_response(instruction, raw)
            if mission and mission.confidence >= self.cfg.confidence_threshold:
                logger.info("LLM mission parsed: %d waypoints (conf=%.2f)",
                            len(mission.waypoints), mission.confidence)
                return mission

        logger.warning("LLM unavailable or low confidence — using keyword fallback")
        return self._keyword_fallback(instruction)

    def resolve_to_metric(self, mission: Mission,
                           zone_map: dict[str, tuple[float, float]]) -> Mission:
        """
        Attach metric (x, y) coordinates to each waypoint using a zone→coord map.
        """
        for wp in mission.waypoints:
            if wp.label in zone_map:
                wp.metric_xy = zone_map[wp.label]
            else:
                # Fuzzy match: find closest zone name
                best = min(zone_map.keys(),
                           key=lambda z: _edit_distance(z, wp.label),
                           default=None)
                if best:
                    wp.metric_xy = zone_map[best]
                    logger.debug("Fuzzy matched '%s' → '%s'", wp.label, best)
        return mission

    # ------------------------------------------------------------------
    # LLM query
    # ------------------------------------------------------------------

    def _query_llm(self, instruction: str) -> Optional[str]:
        import urllib.request, urllib.error

        payload = json.dumps({
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user",   "content": instruction},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }).encode()

        try:
            req = urllib.request.Request(
                self.cfg.endpoint, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                return (data.get("message") or
                        data.get("choices", [{}])[0].get("message", {})).get("content")
        except Exception as e:
            logger.debug("LLM query failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, instruction: str,
                             raw: str) -> Optional[Mission]:
        try:
            text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(text)
            wps = [
                Waypoint(
                    label=str(w["label"]).lower().replace(" ", "_"),
                    priority=int(w.get("priority", i + 1)),
                    action=str(w.get("action", "navigate")),
                    duration_s=float(w.get("duration_s", 0.0)),
                )
                for i, w in enumerate(data.get("waypoints", []))
            ]
            wps.sort(key=lambda w: w.priority)
            return Mission(
                raw_instruction=instruction,
                waypoints=wps,
                confidence=float(data.get("confidence", 0.7)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse LLM response: %s", e)
            return None

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    def _keyword_fallback(self, instruction: str) -> Mission:
        """
        Simple keyword extractor: finds known zone names in the instruction
        and builds a mission in order of appearance.
        """
        text = instruction.lower()
        found = []
        for zone in self.cfg.known_zones:
            if zone.replace("_", " ") in text or zone in text:
                pos = text.find(zone.replace("_", " "))
                if pos == -1:
                    pos = text.find(zone)
                found.append((pos, zone))

        found.sort()
        wps = [
            Waypoint(label=zone, priority=i + 1)
            for i, (_, zone) in enumerate(found)
        ]
        conf = min(0.6, 0.4 + 0.1 * len(wps)) if wps else 0.1
        logger.info("Keyword fallback: found %d zones", len(wps))
        return Mission(raw_instruction=instruction, waypoints=wps,
                       confidence=conf)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance for fuzzy zone matching."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
