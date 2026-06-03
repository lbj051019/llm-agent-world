"""
Scenario validator for the interactive-fiction LLM agent project.

Usage:
    python scenarios/validator.py scenarios/haunted_library.json
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Scenario validation result: {status}"]
        if self.errors:
            lines.append(f"  {len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"    [ERROR] {e}")
        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"    [WARN]  {w}")
        if not self.errors and not self.warnings:
            lines.append("  No issues found.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ScenarioValidator:
    def __init__(self, scenario: dict) -> None:
        self.s = scenario
        self.report = ValidationReport()

    # -- helpers -------------------------------------------------------------

    @property
    def rooms(self) -> dict:
        return self.s.get("rooms", {})

    @property
    def items(self) -> dict:
        return self.s.get("items", {})

    @property
    def npcs(self) -> dict:
        return self.s.get("npcs", {})

    @property
    def events(self) -> dict:
        return self.s.get("events", {})

    def _err(self, msg: str) -> None:
        self.report.error(msg)

    def _warn(self, msg: str) -> None:
        self.report.warn(msg)

    # -- structural checks ---------------------------------------------------

    def _check_top_level_keys(self) -> bool:
        required = {"id", "name", "description", "start_room", "goal", "rooms", "items", "npcs"}
        missing = required - set(self.s.keys())
        for key in sorted(missing):
            self._err(f"Missing required top-level key: '{key}'")
        return not missing

    def _check_start_room(self) -> None:
        start = self.s.get("start_room")
        if start and start not in self.rooms:
            self._err(f"start_room '{start}' does not exist in rooms")

    def _check_exit_targets(self) -> None:
        for room_id, room in self.rooms.items():
            for direction, exit_def in room.get("exits", {}).items():
                if not isinstance(exit_def, dict):
                    continue
                to = exit_def.get("to")
                if to and to not in self.rooms:
                    self._err(
                        f"Room '{room_id}' exit '{direction}' references unknown room '{to}'"
                    )

    def _check_room_item_refs(self) -> None:
        for room_id, room in self.rooms.items():
            for item_id in room.get("items", []):
                if item_id not in self.items:
                    self._err(
                        f"Room '{room_id}' references unknown item '{item_id}'"
                    )

    def _check_room_npc_refs(self) -> None:
        for room_id, room in self.rooms.items():
            for npc_id in room.get("npcs", []):
                if npc_id not in self.npcs:
                    self._err(
                        f"Room '{room_id}' references unknown NPC '{npc_id}'"
                    )

    def _check_npc_initial_states(self) -> None:
        for npc_id, npc in self.npcs.items():
            initial = npc.get("initial_state")
            states = npc.get("dialogue", {}).get("states", {})
            if initial and initial not in states:
                self._err(
                    f"NPC '{npc_id}' initial_state '{initial}' not found in dialogue.states"
                )

    # -- logical checks ------------------------------------------------------

    def _check_locked_exits(self) -> None:
        for room_id, room in self.rooms.items():
            for direction, exit_def in room.get("exits", {}).items():
                if not isinstance(exit_def, dict):
                    continue
                if exit_def.get("locked"):
                    key = exit_def.get("key_item")
                    if not key:
                        self._err(
                            f"Room '{room_id}' exit '{direction}' is locked but has no key_item"
                        )
                    elif key not in self.items:
                        self._err(
                            f"Room '{room_id}' exit '{direction}' key_item '{key}' does not exist in items"
                        )

    def _check_use_targets(self) -> None:
        all_ids = set(self.items) | set(self.npcs)
        for item_id, item in self.items.items():
            for target in item.get("use_targets", []):
                if target not in all_ids:
                    self._warn(
                        f"Item '{item_id}' use_target '{target}' not found in items or npcs "
                        f"(may be a fixture string — ignored at runtime)"
                    )

    def _check_goal_conditions(self) -> None:
        goal = self.s.get("goal", {})
        all_ids = set(self.rooms) | set(self.items) | set(self.npcs) | set(self.events)
        for i, cond in enumerate(goal.get("conditions", [])):
            ctype = cond.get("type", f"<condition[{i}]>")
            for ref_key in ("item_id", "room_id", "npc_id", "event_id"):
                ref_val = cond.get(ref_key)
                if ref_val and ref_val not in all_ids:
                    self._err(
                        f"Goal condition[{i}] (type='{ctype}') references unknown {ref_key} '{ref_val}'"
                    )

    def _check_reachability(self) -> None:
        start = self.s.get("start_room")
        if not start or start not in self.rooms:
            return  # already reported above

        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            room_id = queue.popleft()
            if room_id in visited:
                continue
            visited.add(room_id)
            room = self.rooms.get(room_id, {})
            for exit_def in room.get("exits", {}).values():
                if not isinstance(exit_def, dict):
                    continue
                if exit_def.get("hidden"):
                    continue
                to = exit_def.get("to")
                if to and to in self.rooms and to not in visited:
                    queue.append(to)

        for room_id in self.rooms:
            if room_id not in visited:
                self._err(f"Room '{room_id}' is unreachable from start_room '{start}'")

    def _check_event_effect_refs(self) -> None:
        all_rooms = set(self.rooms)
        all_items = set(self.items)
        all_npcs = set(self.npcs)
        for event_id, event in self.events.items():
            for effect in event.get("effects", []):
                for ref_key, pool, label in (
                    ("npc_id", all_npcs, "npcs"),
                    ("item_id", all_items, "items"),
                    ("room_id", all_rooms, "rooms"),
                ):
                    ref_val = effect.get(ref_key)
                    if ref_val and ref_val not in pool:
                        self._err(
                            f"Event '{event_id}' effect references unknown {ref_key} '{ref_val}' (not in {label})"
                        )

    def _check_event_cycles(self) -> None:
        """
        Build a directed graph: event A -> event B if an effect of A sets an NPC state,
        and some trigger for event B is item_used/npc_state that matches.
        Then detect cycles with DFS.
        """
        if not self.events:
            return

        # Map (npc_id, new_state) -> [event_ids triggered by that state]
        state_to_events: dict[tuple[str, str], list[str]] = {}
        for event_id, event in self.events.items():
            trigger = event.get("trigger", {})
            if trigger.get("type") == "npc_state":
                npc_id = trigger.get("npc_id", "")
                state = trigger.get("state", "")
                state_to_events.setdefault((npc_id, state), []).append(event_id)

        # Build adjacency: event -> set of events it can trigger
        graph: dict[str, set[str]] = {eid: set() for eid in self.events}
        for event_id, event in self.events.items():
            for effect in event.get("effects", []):
                if effect.get("type") == "npc_state":
                    npc_id = effect.get("npc_id", "")
                    new_state = effect.get("new_state", "")
                    for triggered in state_to_events.get((npc_id, new_state), []):
                        if triggered != event_id:
                            graph[event_id].add(triggered)

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {eid: WHITE for eid in self.events}

        def dfs(node: str, path: list[str]) -> bool:
            color[node] = GRAY
            path.append(node)
            for neighbor in graph.get(node, set()):
                if color[neighbor] == GRAY:
                    cycle = path[path.index(neighbor):] + [neighbor]
                    self._err(f"Event cycle detected: {' -> '.join(cycle)}")
                    return True
                if color[neighbor] == WHITE:
                    if dfs(neighbor, path):
                        return True
            path.pop()
            color[node] = BLACK
            return False

        for event_id in self.events:
            if color[event_id] == WHITE:
                dfs(event_id, [])

    # -- NPC persona checks --------------------------------------------------

    def _check_npc_personas(self) -> None:
        room_names = {r.get("name", "").lower() for r in self.rooms.values() if r.get("name")}
        for npc_id, npc in self.npcs.items():
            persona = npc.get("persona", "")
            if not persona:
                continue
            words_in_persona = set(persona.lower().split())
            # Check multi-word room names via substring
            for word in list(words_in_persona):
                # Crude: skip short/common words
                if len(word) < 4:
                    continue
                # If a significant word appears in persona but no room name contains it,
                # and it looks like it might be a location word, skip silently.
                # Only warn when persona contains a phrase like "the X room" or "X library"
                # that doesn't match any actual room name.
            # More targeted: look for quoted or capitalised terms in persona
            import re
            candidates = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Z][a-z]+(?: [A-Z][a-z]+)*)', persona)
            for groups in candidates:
                candidate = next((g for g in groups if g), None)
                if not candidate:
                    continue
                candidate_lower = candidate.lower()
                if not any(candidate_lower in rn or rn in candidate_lower for rn in room_names):
                    self._warn(
                        f"NPC '{npc_id}' persona mentions '{candidate}' which doesn't match any room name"
                    )

    # -- main entry point ----------------------------------------------------

    def run(self) -> ValidationReport:
        if not self._check_top_level_keys():
            # Without the basic keys many later checks would crash or mislead
            return self.report

        self._check_start_room()
        self._check_exit_targets()
        self._check_room_item_refs()
        self._check_room_npc_refs()
        self._check_npc_initial_states()
        self._check_locked_exits()
        self._check_use_targets()
        self._check_goal_conditions()
        self._check_reachability()
        self._check_event_effect_refs()
        self._check_event_cycles()
        self._check_npc_personas()

        return self.report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(scenario: dict) -> ValidationReport:
    """Validate a scenario dict and return a ValidationReport."""
    return ScenarioValidator(scenario).run()


def validate_file(path: str) -> ValidationReport:
    """Load a JSON file from *path* and validate it."""
    p = Path(path)
    report = ValidationReport()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        report.error(f"File not found: {path}")
        return report
    except OSError as exc:
        report.error(f"Could not read file '{path}': {exc}")
        return report

    try:
        scenario = json.loads(text)
    except json.JSONDecodeError as exc:
        report.error(f"Invalid JSON in '{path}': {exc}")
        return report

    return validate(scenario)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scenarios/validator.py <scenario.json>", file=sys.stderr)
        sys.exit(1)

    report = validate_file(sys.argv[1])
    print(report.summary())
    sys.exit(0 if report.valid else 1)
