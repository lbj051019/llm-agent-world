"""
World Engine — owns all mutable world state and calls the World LLM.

Responsibilities:
  - Load and hold scenario data
  - Maintain live world state (room items, NPC states, triggered events, etc.)
  - Process event triggers and apply effects
  - Generate NPC dialogue via the World LLM (Claude)
  - Expose a snapshot dict for the Agent and MemorySystem
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic


# ---------------------------------------------------------------------------
# WorldState — the single source of truth for the simulation
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    # static scenario data (treat as read-only after load)
    scenario_id: str = ""
    scenario_name: str = ""
    scenario_description: str = ""
    rooms: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    npcs: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    goal: dict = field(default_factory=dict)

    # mutable runtime state
    agent_location: str = ""
    inventory: set = field(default_factory=set)
    room_items: dict = field(default_factory=dict)      # room_id → [item_id, ...]
    triggered_events: set = field(default_factory=set)
    used_items: set = field(default_factory=set)        # set of (item_id, target_id) tuples
    revealed_exits: dict = field(default_factory=dict)  # "room_id:direction" → bool
    item_flags: dict = field(default_factory=dict)      # misc runtime flags

    # reference back to the engine (set by WorldEngine after construction)
    _engine: Any = field(default=None, repr=False)

    def fire_event_trigger(self, trigger_type: str, _depth: int = 0, **kwargs) -> list[str]:
        """Called by tools to fire events matching a trigger type. Returns effect descriptions."""
        if self._engine:
            return self._engine.fire_event_trigger(trigger_type, self, _depth=_depth, **kwargs)
        return []

    def snapshot(self) -> dict:
        """Return a serialisable snapshot for logging and agent prompts."""
        current_room = self.rooms.get(self.agent_location, {})
        visible_items = [
            self.items[i]["name"]
            for i in self.room_items.get(self.agent_location, [])
            if i in self.items
        ]
        visible_npcs = [
            self.npcs[n]["name"]
            for n in current_room.get("npcs", [])
            if n in self.npcs
        ]
        available_exits = []
        for direction, exit_data in current_room.get("exits", {}).items():
            if not exit_data.get("hidden", False) or self.revealed_exits.get(
                f"{self.agent_location}:{direction}", False
            ):
                locked = " (locked)" if exit_data.get("locked") else ""
                available_exits.append(f"{direction}{locked}")

        return {
            "agent_location": self.agent_location,
            "current_room_data": current_room,
            "visible_item_names": visible_items,
            "visible_npc_names": visible_npcs,
            "available_exits": available_exits,
            "inventory_names": [
                self.items[i]["name"] for i in self.inventory if i in self.items
            ],
            "triggered_events": list(self.triggered_events),
            "npc_states": {
                npc_id: npc.get("current_state", "") for npc_id, npc in self.npcs.items()
            },
        }


# ---------------------------------------------------------------------------
# WorldEngine
# ---------------------------------------------------------------------------

class WorldEngine:
    """Loads a scenario, maintains WorldState, drives World LLM for NPC dialogue."""

    def __init__(self, model: str = "claude-sonnet-4-5", temperature: float = 0.8):
        self.model = model
        self.temperature = temperature
        self._client: anthropic.Anthropic | None = None
        self.state: WorldState | None = None
        self._npc_dialogue_history: dict[str, list[dict]] = {}  # npc_id → message list

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def load_scenario(self, path: str | Path) -> WorldState:
        with open(path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        state = WorldState(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            scenario_description=scenario["description"],
            rooms=scenario["rooms"],
            items=scenario["items"],
            npcs=copy.deepcopy(scenario.get("npcs", {})),
            events=scenario.get("events", {}),
            goal=scenario["goal"],
            agent_location=scenario["start_room"],
        )

        # initialise room_items from scenario data
        for room_id, room_data in state.rooms.items():
            state.room_items[room_id] = list(room_data.get("items", []))

        state._engine = self
        self.state = state
        return state

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def fire_event_trigger(
        self, trigger_type: str, state: WorldState, _depth: int = 0, **kwargs
    ) -> list[str]:
        """Check all events for matching triggers and apply effects."""
        if _depth > 5:
            raise RuntimeError(
                f"Event chain depth exceeded 5 — possible cycle in events "
                f"(trigger_type={trigger_type!r}, kwargs={kwargs})"
            )
        effect_descriptions = []
        for event_id, event in state.events.items():
            if event_id in state.triggered_events:
                continue  # already fired (events fire once)
            trigger = event.get("trigger", {})
            if not self._trigger_matches(trigger, trigger_type, **kwargs):
                continue
            state.triggered_events.add(event_id)
            for effect in event.get("effects", []):
                desc = self._apply_effect(state, effect, _depth=_depth + 1)
                if desc:
                    effect_descriptions.append(desc)
        return effect_descriptions

    def _trigger_matches(self, trigger: dict, trigger_type: str, **kwargs) -> bool:
        if trigger.get("type") != trigger_type:
            return False
        if trigger_type == "item_used":
            return trigger.get("item_id") == kwargs.get("item_id")
        if trigger_type == "room_entered":
            return trigger.get("room_id") == kwargs.get("room_id")
        if trigger_type == "npc_state_reached":
            return (
                trigger.get("npc_id") == kwargs.get("npc_id")
                and trigger.get("state") == kwargs.get("npc_state")
            )
        return False

    def _apply_effect(self, state: WorldState, effect: dict, _depth: int = 0) -> str:
        etype = effect.get("type")

        if etype == "change_npc_state":
            npc_id = effect["npc_id"]
            new_state = effect["new_state"]
            if npc_id in state.npcs:
                state.npcs[npc_id]["current_state"] = new_state
                npc_name = state.npcs[npc_id]["name"]
                # fire npc_state_reached triggers
                self.fire_event_trigger(
                    "npc_state_reached", state, _depth=_depth + 1, npc_id=npc_id, npc_state=new_state
                )
                state_hint = state.npcs[npc_id]["dialogue"]["states"].get(new_state, "")
                return f"{npc_name}'s state changed to '{new_state}'. {state_hint}"

        elif etype == "unlock_exit":
            room_id = effect["room_id"]
            direction = effect["direction"]
            room = state.rooms.get(room_id, {})
            exit_data = room.get("exits", {}).get(direction, {})
            if exit_data.get("locked"):
                exit_data["locked"] = False
                return f"The {direction} exit from '{room_id}' is now unlocked."

        elif etype == "reveal_item":
            room_id = effect["room_id"]
            item_id = effect["item_id"]
            if item_id not in state.room_items.get(room_id, []):
                state.room_items.setdefault(room_id, []).append(item_id)
                item_name = state.items.get(item_id, {}).get("name", item_id)
                return f"A new item appears in '{room_id}': {item_name}."

        elif etype == "add_item_to_room":
            room_id = effect["room_id"]
            item_id = effect["item_id"]
            if item_id not in state.room_items.get(room_id, []):
                state.room_items.setdefault(room_id, []).append(item_id)
                item_name = state.items.get(item_id, {}).get("name", item_id)
                return f"Item '{item_name}' added to '{room_id}'."

        return ""

    # ------------------------------------------------------------------
    # NPC dialogue via World LLM
    # ------------------------------------------------------------------

    def generate_npc_dialogue(
        self, state: WorldState, npc_id: str, player_message: str
    ) -> str:
        """Call the World LLM to generate a contextual NPC response."""
        npc = state.npcs.get(npc_id)
        if not npc:
            return "[NPC not found]"

        current_npc_state = npc.get("current_state", npc.get("initial_state", ""))
        state_hint = npc["dialogue"]["states"].get(current_npc_state, "")
        greeting = npc["dialogue"].get("greeting", "")
        default_response = npc["dialogue"].get("default", "")

        system_prompt = (
            f"You are roleplaying as {npc['name']} in an interactive fiction game.\n\n"
            f"Character: {npc['persona']}\n\n"
            f"Current emotional/narrative state: {current_npc_state}\n"
            f"State guidance: {state_hint}\n\n"
            f"Greeting (first contact only): {greeting}\n"
            f"Default response style: {default_response}\n\n"
            f"Scenario context: {state.scenario_description}\n\n"
            "Rules:\n"
            "- Stay in character at all times.\n"
            "- Respond only with what this character would say or do.\n"
            "- Keep responses concise (1-4 sentences) unless the character naturally rambles.\n"
            "- Do not break the fourth wall or reference game mechanics.\n"
            "- Incorporate the character's current state into your response naturally.\n"
        )

        history = self._npc_dialogue_history.setdefault(npc_id, [])

        # first message: use greeting if history is empty
        if not history and greeting:
            history.append({"role": "assistant", "content": greeting})

        history.append({"role": "user", "content": player_message})

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=history,
            temperature=self.temperature,
            max_tokens=300,
        )
        reply = response.content[0].text.strip()
        history.append({"role": "assistant", "content": reply})

        return f"{npc['name']}: {reply}"

    def reset_npc_dialogue(self, npc_id: str) -> None:
        self._npc_dialogue_history.pop(npc_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def describe_current_room(self, state: WorldState) -> str:
        room = state.rooms.get(state.agent_location, {})
        name = room.get("name", state.agent_location)
        description = room.get("description", "")
        items_here = [
            state.items[i]["name"]
            for i in state.room_items.get(state.agent_location, [])
            if i in state.items
        ]
        npcs_here = [
            state.npcs[n]["name"]
            for n in room.get("npcs", [])
            if n in state.npcs
        ]
        exits = []
        for direction, exit_data in room.get("exits", {}).items():
            if not exit_data.get("hidden", False):
                locked = " (locked)" if exit_data.get("locked") else ""
                exits.append(f"{direction}{locked}")

        lines = [f"**{name}**", description]
        if items_here:
            lines.append(f"Items: {', '.join(items_here)}")
        if npcs_here:
            lines.append(f"Present: {', '.join(npcs_here)}")
        if exits:
            lines.append(f"Exits: {', '.join(exits)}")
        return "\n".join(lines)

    def get_scenario_list(self, scenarios_dir: str | Path) -> list[dict]:
        """Return id/name/description for all JSON scenarios in a directory."""
        results = []
        for p in Path(scenarios_dir).glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "description": data.get("description", "")[:100],
                    "path": str(p),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results
