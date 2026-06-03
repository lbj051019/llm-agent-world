"""
Three-layer memory system for the Agent LLM.

  WorkingMemory   — current step context (cleared each cycle)
  EpisodicMemory  — ordered log of significant past events (capped, summarisable)
  SemanticMemory  — extracted facts and beliefs that persist indefinitely
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemory:
    """Holds context for the current agent step only."""

    current_room: str = ""
    current_room_description: str = ""
    visible_items: list[str] = field(default_factory=list)
    visible_npcs: list[str] = field(default_factory=list)
    available_exits: list[str] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)
    last_observation: str = ""
    last_tool: str = ""
    last_tool_args: dict = field(default_factory=dict)
    last_tool_success: bool = True
    current_plan: str = ""
    pending_subgoals: list[str] = field(default_factory=list)
    step_number: int = 0

    def to_prompt_block(self) -> str:
        lines = [
            f"Step: {self.step_number}",
            f"Location: {self.current_room}",
            f"Room description: {self.current_room_description}",
            f"Inventory: {', '.join(self.inventory) or 'empty'}",
            f"Visible items: {', '.join(self.visible_items) or 'none'}",
            f"Visible NPCs: {', '.join(self.visible_npcs) or 'none'}",
            f"Exits: {', '.join(self.available_exits) or 'none'}",
        ]
        if self.last_observation:
            lines.append(f"Last observation: {self.last_observation}")
        if self.current_plan:
            lines.append(f"Current plan: {self.current_plan}")
        if self.pending_subgoals:
            lines.append(f"Pending subgoals: {'; '.join(self.pending_subgoals)}")
        return "\n".join(lines)

    def clear_step(self) -> None:
        self.last_observation = ""
        self.last_tool = ""
        self.last_tool_args = {}
        self.last_tool_success = True


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    step: int
    timestamp: float
    tool: str
    tool_args: dict
    observation: str
    significance: float  # 0.0–1.0; higher = more likely to be retained after compression
    tags: list[str] = field(default_factory=list)

    def to_summary_line(self) -> str:
        args_str = ", ".join(f"{k}={v}" for k, v in self.tool_args.items())
        return f"[Step {self.step}] {self.tool}({args_str}) → {self.observation[:120]}"


class EpisodicMemory:
    """
    Ordered log of agent actions and observations.
    When it exceeds `max_episodes`, low-significance entries are compressed
    into a running summary string.
    """

    def __init__(self, max_episodes: int = 50, summary_keep: int = 20):
        self.episodes: list[Episode] = []
        self.compressed_summary: str = ""
        self.max_episodes = max_episodes
        self.summary_keep = summary_keep  # keep this many recent episodes after compression

    def record(
        self,
        step: int,
        tool: str,
        tool_args: dict,
        observation: str,
        significance: float = 0.5,
        tags: list[str] | None = None,
    ) -> None:
        episode = Episode(
            step=step,
            timestamp=time.time(),
            tool=tool,
            tool_args=tool_args,
            observation=observation,
            significance=significance,
            tags=tags or [],
        )
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self._compress()

    def _compress(self) -> None:
        """Keep the most significant + most recent episodes; summarise the rest."""
        by_significance = sorted(self.episodes, key=lambda e: e.significance, reverse=True)
        to_keep_ids = {id(e) for e in by_significance[: self.summary_keep]}
        # always keep the most recent summary_keep regardless of significance
        for e in self.episodes[-self.summary_keep :]:
            to_keep_ids.add(id(e))

        to_compress = [e for e in self.episodes if id(e) not in to_keep_ids]
        if to_compress:
            summary_lines = [e.to_summary_line() for e in to_compress]
            self.compressed_summary += "\n".join(summary_lines) + "\n"

        self.episodes = [e for e in self.episodes if id(e) in to_keep_ids]

    def recent(self, n: int = 10) -> list[Episode]:
        return self.episodes[-n:]

    def search(self, keyword: str) -> list[Episode]:
        kw = keyword.lower()
        return [
            e
            for e in self.episodes
            if kw in e.observation.lower() or kw in str(e.tool_args).lower()
        ]

    def to_prompt_block(self, max_lines: int = 15) -> str:
        lines = []
        if self.compressed_summary:
            lines.append("=== Earlier history (compressed) ===")
            # show last few lines of summary to keep prompt size bounded
            summary_tail = self.compressed_summary.strip().split("\n")[-5:]
            lines.extend(summary_tail)
            lines.append("=== Recent events ===")
        for ep in self.recent(max_lines):
            lines.append(ep.to_summary_line())
        return "\n".join(lines)

    def get_significant_events(self, threshold: float = 0.7) -> list[Episode]:
        return [e for e in self.episodes if e.significance >= threshold]


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    key: str          # canonical identifier, e.g. "cursed_tome_location"
    value: Any        # the belief
    confidence: float  # 0.0–1.0
    source: str       # "observation" | "npc_dialogue" | "item_read" | "inference"
    step_recorded: int
    last_updated: int

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "step_recorded": self.step_recorded,
            "last_updated": self.last_updated,
        }


class SemanticMemory:
    """
    Long-term factual beliefs about the world.
    Keyed by a string identifier; newer observations overwrite older ones
    only if their confidence is >= the existing entry.
    """

    def __init__(self):
        self.facts: dict[str, Fact] = {}

    def store(
        self,
        key: str,
        value: Any,
        confidence: float = 0.8,
        source: str = "observation",
        step: int = 0,
    ) -> None:
        existing = self.facts.get(key)
        if existing and existing.confidence > confidence:
            return  # don't overwrite with lower-confidence information
        self.facts[key] = Fact(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            step_recorded=step if not existing else existing.step_recorded,
            last_updated=step,
        )

    def retrieve(self, key: str) -> Any | None:
        fact = self.facts.get(key)
        return fact.value if fact else None

    def retrieve_fact(self, key: str) -> Fact | None:
        return self.facts.get(key)

    def search(self, keyword: str) -> list[Fact]:
        kw = keyword.lower()
        return [
            f
            for f in self.facts.values()
            if kw in f.key.lower() or kw in str(f.value).lower()
        ]

    def update_confidence(self, key: str, delta: float) -> None:
        if key in self.facts:
            self.facts[key].confidence = max(0.0, min(1.0, self.facts[key].confidence + delta))

    def to_prompt_block(self, max_facts: int = 30) -> str:
        if not self.facts:
            return "No established facts yet."
        # sort by confidence desc, then recency
        sorted_facts = sorted(
            self.facts.values(),
            key=lambda f: (f.confidence, f.last_updated),
            reverse=True,
        )[:max_facts]
        lines = []
        for f in sorted_facts:
            conf_str = f"{f.confidence:.0%}"
            lines.append(f"  [{conf_str}] {f.key}: {f.value}  (via {f.source})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.facts.items()}

    def from_dict(self, data: dict) -> None:
        for k, v in data.items():
            self.facts[k] = Fact(**v)


# ---------------------------------------------------------------------------
# MemorySystem — unified facade
# ---------------------------------------------------------------------------

class MemorySystem:
    """Combines all three memory layers with convenience methods."""

    def __init__(self, max_episodes: int = 50):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(max_episodes=max_episodes)
        self.semantic = SemanticMemory()

    # --- update helpers ---

    def update_working(self, world_state_snapshot: dict) -> None:
        """Sync working memory from current world state snapshot."""
        w = self.working
        w.current_room = world_state_snapshot.get("agent_location", "")
        room = world_state_snapshot.get("current_room_data", {})
        w.current_room_description = room.get("description", "")
        w.visible_items = world_state_snapshot.get("visible_item_names", [])
        w.visible_npcs = world_state_snapshot.get("visible_npc_names", [])
        w.available_exits = world_state_snapshot.get("available_exits", [])
        w.inventory = world_state_snapshot.get("inventory_names", [])

    def record_action(
        self,
        tool: str,
        tool_args: dict,
        observation: str,
        success: bool,
        significance: float | None = None,
    ) -> None:
        step = self.working.step_number
        if significance is None:
            significance = _estimate_significance(tool, success, observation)
        self.episodic.record(
            step=step,
            tool=tool,
            tool_args=tool_args,
            observation=observation,
            significance=significance,
            tags=[tool, "success" if success else "failure"],
        )
        self.working.last_tool = tool
        self.working.last_tool_args = tool_args
        self.working.last_observation = observation
        self.working.last_tool_success = success

    def infer_fact(self, key: str, value: Any, source: str = "observation", confidence: float = 0.8) -> None:
        self.semantic.store(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            step=self.working.step_number,
        )

    # --- prompt assembly ---

    def build_agent_context(self) -> str:
        """Assemble a full memory context block for injection into the agent prompt."""
        sections = [
            "## Working Memory\n" + self.working.to_prompt_block(),
            "## Recent History\n" + self.episodic.to_prompt_block(),
            "## Known Facts\n" + self.semantic.to_prompt_block(),
        ]
        return "\n\n".join(sections)

    # --- persistence ---

    def save(self, path: str) -> None:
        data = {
            "semantic": self.semantic.to_dict(),
            "episodic_compressed_summary": self.episodic.compressed_summary,
            "episodic_recent": [
                {
                    "step": e.step,
                    "timestamp": e.timestamp,
                    "tool": e.tool,
                    "tool_args": e.tool_args,
                    "observation": e.observation,
                    "significance": e.significance,
                    "tags": e.tags,
                }
                for e in self.episodic.episodes
            ],
            "working_step_number": self.working.step_number,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.semantic.from_dict(data.get("semantic", {}))
        self.episodic.compressed_summary = data.get("episodic_compressed_summary", "")
        self.episodic.episodes = [
            Episode(**ep) for ep in data.get("episodic_recent", [])
        ]
        self.working.step_number = data.get("working_step_number", 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_significance(tool: str, success: bool, observation: str) -> float:
    """Heuristic significance score for automatic tagging."""
    if not success:
        return 0.3
    high_sig_tools = {"take", "use", "finish"}
    if tool in high_sig_tools:
        return 0.9
    if tool == "talk":
        return 0.7
    if tool == "move":
        return 0.4
    if tool == "examine":
        # examining something that reveals a clue is significant
        if any(kw in observation.lower() for kw in ("clue", "key", "secret", "hidden", "must", "code")):
            return 0.8
        return 0.4
    if tool == "think":
        return 0.2
    return 0.5
