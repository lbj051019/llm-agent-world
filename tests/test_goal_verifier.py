"""
Tests for engine/goal_verifier.py.
Builds minimal WorldState objects and scenario dicts directly — no file I/O.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.world_engine import WorldState
from engine.goal_verifier import GoalVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    agent_location="room1",
    inventory=None,
    npcs=None,
    triggered_events=None,
    rooms=None,
    items=None,
    events=None,
):
    if rooms is None:
        rooms = {
            "room1": {"id": "room1", "name": "Room One", "description": ""},
            "room2": {"id": "room2", "name": "Room Two", "description": ""},
        }
    return WorldState(
        scenario_id="test",
        scenario_name="Test",
        scenario_description="",
        rooms=rooms,
        items=items or {},
        npcs=npcs or {},
        events=events or {},
        goal={},
        agent_location=agent_location,
        inventory=set(inventory or []),
        room_items={},
        triggered_events=set(triggered_events or []),
        used_items=set(),
        revealed_exits={},
        item_flags={},
    )


def scenario_with(conditions, logic="AND"):
    """Return a minimal scenario dict containing the given conditions list."""
    return {"goal": {"conditions": conditions}}


# ---------------------------------------------------------------------------
# agent_at_location
# ---------------------------------------------------------------------------

class TestAgentAtLocation(unittest.TestCase):

    def test_agent_at_location_pass(self):
        state = make_state(agent_location="room1")
        verifier = GoalVerifier(scenario_with([
            {"type": "agent_at_location", "room_id": "room1"}
        ]))
        result = verifier.verify(state)
        self.assertEqual(result.satisfied_count, 1)
        self.assertTrue(result.condition_results[0].satisfied)

    def test_agent_at_location_fail(self):
        state = make_state(agent_location="room2")
        verifier = GoalVerifier(scenario_with([
            {"type": "agent_at_location", "room_id": "room1"}
        ]))
        result = verifier.verify(state)
        self.assertEqual(result.satisfied_count, 0)
        self.assertFalse(result.condition_results[0].satisfied)


# ---------------------------------------------------------------------------
# item_in_inventory
# ---------------------------------------------------------------------------

class TestItemInInventory(unittest.TestCase):

    def test_item_in_inventory_pass(self):
        state = make_state(inventory=["magic_wand"])
        verifier = GoalVerifier(scenario_with([
            {"type": "item_in_inventory", "item_id": "magic_wand"}
        ]))
        result = verifier.verify(state)
        self.assertTrue(result.condition_results[0].satisfied)

    def test_item_in_inventory_fail(self):
        state = make_state(inventory=[])
        verifier = GoalVerifier(scenario_with([
            {"type": "item_in_inventory", "item_id": "magic_wand"}
        ]))
        result = verifier.verify(state)
        self.assertFalse(result.condition_results[0].satisfied)


# ---------------------------------------------------------------------------
# npc_state
# ---------------------------------------------------------------------------

class TestNpcState(unittest.TestCase):

    def _state_with_npc(self, current_state):
        npcs = {
            "wizard": {
                "name": "Wizard",
                "initial_state": "hostile",
                "current_state": current_state,
            }
        }
        return make_state(npcs=npcs)

    def test_npc_state_pass(self):
        state = self._state_with_npc("friendly")
        verifier = GoalVerifier(scenario_with([
            {"type": "npc_state", "npc_id": "wizard", "state": "friendly"}
        ]))
        result = verifier.verify(state)
        self.assertTrue(result.condition_results[0].satisfied)

    def test_npc_state_fail(self):
        state = self._state_with_npc("hostile")
        verifier = GoalVerifier(scenario_with([
            {"type": "npc_state", "npc_id": "wizard", "state": "friendly"}
        ]))
        result = verifier.verify(state)
        self.assertFalse(result.condition_results[0].satisfied)


# ---------------------------------------------------------------------------
# event_triggered
# ---------------------------------------------------------------------------

class TestEventTriggered(unittest.TestCase):

    def test_event_triggered_pass(self):
        state = make_state(triggered_events=["big_explosion"])
        verifier = GoalVerifier(scenario_with([
            {"type": "event_triggered", "event_id": "big_explosion"}
        ]))
        result = verifier.verify(state)
        self.assertTrue(result.condition_results[0].satisfied)

    def test_event_triggered_fail(self):
        state = make_state(triggered_events=[])
        verifier = GoalVerifier(scenario_with([
            {"type": "event_triggered", "event_id": "big_explosion"}
        ]))
        result = verifier.verify(state)
        self.assertFalse(result.condition_results[0].satisfied)


# ---------------------------------------------------------------------------
# AND / OR logic
# ---------------------------------------------------------------------------

class TestGoalLogic(unittest.TestCase):

    def _two_condition_scenario(self):
        return scenario_with([
            {"type": "agent_at_location", "room_id": "room1"},
            {"type": "item_in_inventory", "item_id": "sword"},
        ])

    def test_and_logic_all_pass(self):
        state = make_state(agent_location="room1", inventory=["sword"])
        verifier = GoalVerifier(self._two_condition_scenario(), logic="AND")
        result = verifier.verify(state)
        self.assertTrue(result.won)
        self.assertEqual(result.satisfied_count, 2)

    def test_and_logic_partial(self):
        # at room1 but no sword
        state = make_state(agent_location="room1", inventory=[])
        verifier = GoalVerifier(self._two_condition_scenario(), logic="AND")
        result = verifier.verify(state)
        self.assertFalse(result.won)
        self.assertEqual(result.satisfied_count, 1)

    def test_or_logic_any_pass(self):
        # has sword but not at room1
        state = make_state(agent_location="room2", inventory=["sword"])
        verifier = GoalVerifier(self._two_condition_scenario(), logic="OR")
        result = verifier.verify(state)
        self.assertTrue(result.won)
        self.assertGreaterEqual(result.satisfied_count, 1)


if __name__ == "__main__":
    unittest.main()
