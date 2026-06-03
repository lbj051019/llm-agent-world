"""
Tests for engine/world_engine.py — WorldEngine state management only.
No LLM calls; only load_scenario (file I/O) and manual event firing are tested.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.world_engine import WorldState, WorldEngine

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
HAUNTED_LIBRARY = SCENARIOS_DIR / "haunted_library.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine():
    """Return a WorldEngine with no API client instantiated."""
    return WorldEngine()


def make_state_with_engine(
    agent_location="room1",
    rooms=None,
    items=None,
    npcs=None,
    events=None,
):
    """Build a WorldState wired to a WorldEngine, without loading a file."""
    if rooms is None:
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A plain room.",
                "exits": {
                    "north": {"to": "room2", "locked": True},
                },
                "items": [],
                "npcs": [],
            },
            "room2": {
                "id": "room2",
                "name": "Room Two",
                "description": "Another room.",
                "exits": {},
                "items": [],
                "npcs": [],
            },
        }
    engine = make_engine()
    state = WorldState(
        scenario_id="test",
        scenario_name="Test",
        scenario_description="",
        rooms=rooms,
        items=items or {},
        npcs=npcs or {},
        events=events or {},
        goal={},
        agent_location=agent_location,
        inventory=set(),
        room_items={rid: [] for rid in (rooms or {})},
        triggered_events=set(),
        used_items=set(),
        revealed_exits={},
        item_flags={},
    )
    state._engine = engine
    engine.state = state
    return engine, state


# ---------------------------------------------------------------------------
# load_scenario
# ---------------------------------------------------------------------------

class TestLoadScenario(unittest.TestCase):

    def test_load_scenario(self):
        engine = make_engine()
        state = engine.load_scenario(HAUNTED_LIBRARY)
        self.assertEqual(state.scenario_id, "haunted_library")
        # start_room must be set and exist in rooms dict
        self.assertIn(state.agent_location, state.rooms)
        self.assertIsNotNone(state.agent_location)
        self.assertNotEqual(state.agent_location, "")


# ---------------------------------------------------------------------------
# Event triggering
# ---------------------------------------------------------------------------

class TestFireEventTriggerItemUsed(unittest.TestCase):

    def _setup(self):
        npcs = {
            "guard": {
                "id": "guard",
                "name": "Guard",
                "description": "A stern guard.",
                "persona": "Stoic.",
                "initial_state": "hostile",
                "current_state": "hostile",
                "dialogue": {"greeting": "", "default": "", "states": {"pacified": "The guard lowers his weapon."}},
            }
        }
        events = {
            "pacify_event": {
                "id": "pacify_event",
                "description": "Pacify the guard with the amulet.",
                "trigger": {"type": "item_used", "item_id": "amulet"},
                "effects": [
                    {"type": "change_npc_state", "npc_id": "guard", "new_state": "pacified"}
                ],
            }
        }
        items = {
            "amulet": {"id": "amulet", "name": "Amulet", "description": "A glowing amulet."},
        }
        engine, state = make_state_with_engine(npcs=npcs, events=events, items=items)
        return engine, state

    def test_fire_event_trigger_item_used(self):
        engine, state = self._setup()
        engine.fire_event_trigger("item_used", state, item_id="amulet")
        self.assertEqual(state.npcs["guard"]["current_state"], "pacified")

    def test_event_fires_once(self):
        engine, state = self._setup()
        engine.fire_event_trigger("item_used", state, item_id="amulet")
        # Reset NPC state manually to check that a second fire does not re-apply
        state.npcs["guard"]["current_state"] = "hostile"
        engine.fire_event_trigger("item_used", state, item_id="amulet")
        # Event is already in triggered_events, so effect should not re-run
        self.assertIn("pacify_event", state.triggered_events)
        self.assertEqual(state.npcs["guard"]["current_state"], "hostile")


class TestFireEventTriggerRoomEntered(unittest.TestCase):

    def test_fire_event_trigger_room_entered(self):
        rooms = {
            "room1": {
                "id": "room1", "name": "Room One", "description": "",
                "exits": {}, "items": [], "npcs": [],
            },
            "room2": {
                "id": "room2", "name": "Room Two", "description": "",
                "exits": {}, "items": [], "npcs": [],
            },
        }
        npcs = {
            "spirit": {
                "id": "spirit",
                "name": "Spirit",
                "description": "A wandering spirit.",
                "persona": "Mysterious.",
                "initial_state": "sleeping",
                "current_state": "sleeping",
                "dialogue": {"greeting": "", "default": "", "states": {"awakened": "The spirit stirs."}},
            }
        }
        events = {
            "awaken_spirit": {
                "id": "awaken_spirit",
                "description": "Spirit awakens when player enters room2.",
                "trigger": {"type": "room_entered", "room_id": "room2"},
                "effects": [
                    {"type": "change_npc_state", "npc_id": "spirit", "new_state": "awakened"}
                ],
            }
        }
        engine, state = make_state_with_engine(
            agent_location="room1", rooms=rooms, npcs=npcs, events=events
        )
        engine.fire_event_trigger("room_entered", state, room_id="room2")
        self.assertEqual(state.npcs["spirit"]["current_state"], "awakened")
        self.assertIn("awaken_spirit", state.triggered_events)


# ---------------------------------------------------------------------------
# Effect types
# ---------------------------------------------------------------------------

class TestUnlockExitEffect(unittest.TestCase):

    def test_unlock_exit_effect(self):
        rooms = {
            "room1": {
                "id": "room1", "name": "Room One", "description": "",
                "exits": {
                    "north": {"to": "room2", "locked": True},
                },
                "items": [], "npcs": [],
            },
            "room2": {
                "id": "room2", "name": "Room Two", "description": "",
                "exits": {}, "items": [], "npcs": [],
            },
        }
        events = {
            "open_gate": {
                "id": "open_gate",
                "description": "Lever opens the north gate.",
                "trigger": {"type": "item_used", "item_id": "lever"},
                "effects": [
                    {"type": "unlock_exit", "room_id": "room1", "direction": "north"}
                ],
            }
        }
        engine, state = make_state_with_engine(rooms=rooms, events=events)
        self.assertTrue(state.rooms["room1"]["exits"]["north"]["locked"])
        engine.fire_event_trigger("item_used", state, item_id="lever")
        self.assertFalse(state.rooms["room1"]["exits"]["north"]["locked"])


class TestNpcStateChangeEffect(unittest.TestCase):

    def test_npc_state_change_effect(self):
        npcs = {
            "sage": {
                "id": "sage",
                "name": "Sage",
                "description": "An old sage.",
                "persona": "Wise.",
                "initial_state": "neutral",
                "current_state": "neutral",
                "dialogue": {"greeting": "", "default": "", "states": {"grateful": "The sage bows."}},
            }
        }
        events = {
            "gift_event": {
                "id": "gift_event",
                "description": "Give a gift to the sage.",
                "trigger": {"type": "item_used", "item_id": "gift"},
                "effects": [
                    {"type": "change_npc_state", "npc_id": "sage", "new_state": "grateful"}
                ],
            }
        }
        engine, state = make_state_with_engine(npcs=npcs, events=events)
        engine.fire_event_trigger("item_used", state, item_id="gift")
        self.assertEqual(state.npcs["sage"]["current_state"], "grateful")


if __name__ == "__main__":
    unittest.main()
