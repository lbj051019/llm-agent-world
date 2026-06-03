"""
Tests for engine/tools.py — all seven tools.
No API calls; builds minimal WorldState objects directly.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.world_engine import WorldState
from engine.tools import (
    tool_move,
    tool_take,
    tool_use,
    tool_talk,
    tool_think,
    tool_finish,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    agent_location="room1",
    rooms=None,
    items=None,
    npcs=None,
    inventory=None,
    room_items=None,
):
    """Build a minimal WorldState without touching the file system."""
    if rooms is None:
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A plain room.",
                "exits": {
                    "north": {"to": "room2", "locked": False},
                },
                "items": [],
                "npcs": [],
            },
            "room2": {
                "id": "room2",
                "name": "Room Two",
                "description": "Another plain room.",
                "exits": {
                    "south": {"to": "room1", "locked": False},
                },
                "items": [],
                "npcs": [],
            },
        }
    state = WorldState(
        scenario_id="test",
        scenario_name="Test",
        scenario_description="",
        rooms=rooms,
        items=items or {},
        npcs=npcs or {},
        events={},
        goal={},
        agent_location=agent_location,
        inventory=set(inventory or []),
        room_items=dict(room_items or {}),
        triggered_events=set(),
        used_items=set(),
        revealed_exits={},
        item_flags={},
    )
    # no engine attached — fire_event_trigger is a no-op
    return state


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------

class TestToolMove(unittest.TestCase):

    def test_move_valid(self):
        state = make_state()
        result = tool_move(state, "north")
        self.assertTrue(result.success)
        self.assertEqual(state.agent_location, "room2")
        self.assertTrue(result.state_changed)

    def test_move_invalid_direction(self):
        state = make_state()
        result = tool_move(state, "east")
        self.assertFalse(result.success)
        self.assertEqual(state.agent_location, "room1")

    def test_move_locked_no_key(self):
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A room.",
                "exits": {
                    "north": {"to": "room2", "locked": True, "key_item": "gold_key"},
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
        items = {
            "gold_key": {"id": "gold_key", "name": "Gold Key", "description": "A gold key."},
        }
        state = make_state(rooms=rooms, items=items)
        result = tool_move(state, "north")
        self.assertFalse(result.success)
        self.assertIn("locked", result.observation.lower())
        self.assertEqual(state.agent_location, "room1")

    def test_move_locked_with_key(self):
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A room.",
                "exits": {
                    "north": {"to": "room2", "locked": True, "key_item": "gold_key"},
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
        items = {
            "gold_key": {"id": "gold_key", "name": "Gold Key", "description": "A gold key."},
        }
        state = make_state(rooms=rooms, items=items, inventory=["gold_key"])
        result = tool_move(state, "north")
        self.assertTrue(result.success)
        self.assertEqual(state.agent_location, "room2")


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------

class TestToolTake(unittest.TestCase):

    def _state_with_item(self, takeable=True):
        items = {
            "key1": {
                "id": "key1",
                "name": "Small Key",
                "description": "A small key.",
                "takeable": takeable,
            }
        }
        state = make_state(
            items=items,
            room_items={"room1": ["key1"]},
        )
        return state

    def test_take_success(self):
        state = self._state_with_item(takeable=True)
        result = tool_take(state, "Small Key")
        self.assertTrue(result.success)
        self.assertIn("key1", state.inventory)
        self.assertNotIn("key1", state.room_items.get("room1", []))

    def test_take_not_in_room(self):
        state = self._state_with_item(takeable=True)
        state.room_items["room1"] = []  # remove it from room
        result = tool_take(state, "Small Key")
        self.assertFalse(result.success)

    def test_take_not_takeable(self):
        state = self._state_with_item(takeable=False)
        result = tool_take(state, "Small Key")
        self.assertFalse(result.success)
        self.assertNotIn("key1", state.inventory)


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------

class TestToolUse(unittest.TestCase):

    def _state_with_usable_item(self):
        items = {
            "torch": {
                "id": "torch",
                "name": "Torch",
                "description": "A burning torch.",
                "takeable": True,
                "usable": True,
                "use_targets": ["brazier"],
                "use_effect": "You light the brazier with the torch.",
            },
            "brazier": {
                "id": "brazier",
                "name": "Brazier",
                "description": "An iron brazier.",
                "takeable": False,
                "usable": False,
                "use_targets": [],
            },
        }
        state = make_state(
            items=items,
            inventory=["torch"],
            room_items={"room1": ["brazier"]},
        )
        return state

    def test_use_item_on_item(self):
        state = self._state_with_usable_item()
        result = tool_use(state, "torch", "brazier")
        self.assertTrue(result.success)
        self.assertIn(("torch", "brazier"), state.used_items)

    def test_use_exit_unlock(self):
        """Using a key item on a direction should unlock that exit."""
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A room.",
                "exits": {
                    "north": {
                        "to": "room2",
                        "locked": True,
                        "key_item": "silver_key",
                        "description": "A locked gate.",
                    }
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
        items = {
            "silver_key": {
                "id": "silver_key",
                "name": "Silver Key",
                "description": "A silver key.",
                "takeable": True,
                "usable": True,
                "use_targets": [],
                "use_effect": "The gate clicks open.",
            }
        }
        state = make_state(rooms=rooms, items=items, inventory=["silver_key"])
        result = tool_use(state, "silver_key", "north")
        self.assertTrue(result.success)
        self.assertFalse(state.rooms["room1"]["exits"]["north"]["locked"])

    def test_use_wrong_target(self):
        state = self._state_with_usable_item()
        result = tool_use(state, "torch", "wall")
        self.assertFalse(result.success)

    def test_use_not_in_inventory(self):
        state = self._state_with_usable_item()
        state.inventory.discard("torch")
        result = tool_use(state, "torch", "brazier")
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# talk
# ---------------------------------------------------------------------------

class TestToolTalk(unittest.TestCase):

    def _state_with_npc(self, npc_in_room=True):
        rooms = {
            "room1": {
                "id": "room1",
                "name": "Room One",
                "description": "A room.",
                "exits": {},
                "items": [],
                "npcs": ["ghost"] if npc_in_room else [],
            },
        }
        npcs = {
            "ghost": {
                "id": "ghost",
                "name": "Ghost",
                "description": "A translucent spectre.",
                "persona": "A sad ghost.",
                "initial_state": "wandering",
                "current_state": "wandering",
                "dialogue": {
                    "greeting": "...",
                    "default": "The ghost moans.",
                    "states": {},
                },
            }
        }
        state = make_state(rooms=rooms, npcs=npcs)
        return state

    def test_talk_npc_present(self):
        state = self._state_with_npc(npc_in_room=True)
        result = tool_talk(state, "Ghost", "Hello")
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("requires_llm"))
        self.assertEqual(result.metadata.get("npc_id"), "ghost")

    def test_talk_npc_not_present(self):
        state = self._state_with_npc(npc_in_room=False)
        result = tool_talk(state, "Ghost", "Hello")
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# think
# ---------------------------------------------------------------------------

class TestToolThink(unittest.TestCase):

    def test_think_no_effect(self):
        state = make_state()
        result = tool_think(state, thought="I should go north.")
        self.assertTrue(result.success)
        self.assertFalse(result.state_changed)


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------

class TestToolFinish(unittest.TestCase):

    def test_finish_returns_finish(self):
        state = make_state()
        result = tool_finish(state, reason="I completed the goal.")
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("finish"))


if __name__ == "__main__":
    unittest.main()
