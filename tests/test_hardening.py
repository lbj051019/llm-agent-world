"""
Tests for the 6 hardening fixes:
  Fix 1: JSON parsing (markdown fences, fallback)
  Fix 2: Observation truncation
  Fix 3: Repeat action detection
  Fix 4: Fuzzy item name matching with suggestions
  Fix 5: Event chain cycle detection
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.agent import _extract_json_object, _truncate_observation, _OBS_MAX_CHARS
from engine.world_engine import WorldState, WorldEngine
from engine.tools import (
    tool_take, tool_use, _match_item, _suggest_items,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fix 1: JSON parsing hardening
# ---------------------------------------------------------------------------

class TestExtractJsonObject(unittest.TestCase):

    def test_plain_json(self):
        text = '{"tool": "move", "args": {"direction": "north"}}'
        result = _extract_json_object(text)
        self.assertEqual(result["tool"], "move")

    def test_json_with_prefix_text(self):
        text = 'Sure, here is the action: {"tool": "take", "args": {"item_name": "key"}}'
        result = _extract_json_object(text)
        self.assertEqual(result["tool"], "take")

    def test_markdown_code_fence(self):
        text = '```json\n{"tool": "finish", "args": {"reason": "done"}}\n```'
        result = _extract_json_object(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "finish")

    def test_markdown_code_fence_no_lang(self):
        text = '```\n{"tool": "examine", "args": {"target": "room"}}\n```'
        result = _extract_json_object(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "examine")

    def test_nested_json(self):
        text = '{"tool": "use", "args": {"item_name": "key", "target_name": "door"}}'
        result = _extract_json_object(text)
        self.assertEqual(result["args"]["item_name"], "key")

    def test_no_json_returns_none(self):
        result = _extract_json_object("just some plain text with no braces")
        self.assertIsNone(result)

    def test_broken_json_returns_none(self):
        result = _extract_json_object('{"tool": "move", "args": {incomplete')
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Fix 2: Observation truncation
# ---------------------------------------------------------------------------

class TestObservationTruncation(unittest.TestCase):

    def test_short_obs_unchanged(self):
        obs = "You move north.\nSmall room."
        self.assertEqual(_truncate_observation(obs), obs)

    def test_long_obs_truncated(self):
        obs = "A" * (_OBS_MAX_CHARS + 200)
        result = _truncate_observation(obs)
        self.assertLessEqual(len(result), _OBS_MAX_CHARS + 20)
        self.assertIn("[truncated]", result)

    def test_exactly_at_limit(self):
        obs = "B" * _OBS_MAX_CHARS
        result = _truncate_observation(obs)
        self.assertEqual(result, obs)

    def test_truncation_preserves_newline_boundary(self):
        lines = ["Line " + str(i) for i in range(200)]
        obs = "\n".join(lines)
        result = _truncate_observation(obs)
        self.assertIn("[truncated]", result)
        # should end with a complete line, not mid-word
        self.assertTrue(result.endswith("[truncated]"))


# ---------------------------------------------------------------------------
# Fix 3: Repeat action detection (tested indirectly via agent internals)
# ---------------------------------------------------------------------------

class TestRepeatActionDetection(unittest.TestCase):
    """Test the repeat-action bookkeeping logic without running the full agent."""

    def _run_repeat_check(self, actions):
        """Simulate the in-loop repeat detection logic."""
        recent = []
        warnings = []
        for tool_name, tool_args in actions:
            action_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            recent.append(action_key)
            if len(recent) > 5:
                recent.pop(0)
            if len(recent) >= 3 and all(a == action_key for a in recent[-3:]):
                warnings.append(f"repeat_{tool_name}")
        return warnings

    def test_no_repeat_no_warning(self):
        actions = [
            ("move", {"direction": "north"}),
            ("move", {"direction": "south"}),
            ("take", {"item_name": "key"}),
        ]
        self.assertEqual(self._run_repeat_check(actions), [])

    def test_three_identical_triggers_warning(self):
        actions = [("move", {"direction": "north"})] * 3
        warnings = self._run_repeat_check(actions)
        self.assertEqual(len(warnings), 1)

    def test_two_identical_no_warning(self):
        actions = [("move", {"direction": "north"})] * 2
        self.assertEqual(self._run_repeat_check(actions), [])

    def test_reset_after_different_action(self):
        actions = [
            ("move", {"direction": "north"}),
            ("move", {"direction": "north"}),
            ("take", {"item_name": "key"}),
            ("move", {"direction": "north"}),
            ("move", {"direction": "north"}),
        ]
        self.assertEqual(self._run_repeat_check(actions), [])


# ---------------------------------------------------------------------------
# Fix 4: Fuzzy item name matching
# ---------------------------------------------------------------------------

def _make_items():
    return {
        "brass_key": {"name": "Brass Key", "takeable": True},
        "old_lantern": {"name": "Old Lantern", "takeable": True},
        "silver_coin": {"name": "Silver Coin", "takeable": True},
    }


class TestFuzzyItemMatching(unittest.TestCase):

    def setUp(self):
        self.items = _make_items()
        self.candidates = list(self.items.keys())

    def test_exact_id_match(self):
        self.assertEqual(_match_item(self.items, self.candidates, "brass_key"), "brass_key")

    def test_exact_name_substring_match(self):
        self.assertEqual(_match_item(self.items, self.candidates, "brass key"), "brass_key")

    def test_case_insensitive(self):
        self.assertEqual(_match_item(self.items, self.candidates, "BRASS KEY"), "brass_key")

    def test_partial_token_match(self):
        self.assertEqual(_match_item(self.items, self.candidates, "lantern"), "old_lantern")

    def test_no_match_returns_none(self):
        self.assertIsNone(_match_item(self.items, self.candidates, "magic sword"))

    def test_suggest_items_lists_names(self):
        suggestion = _suggest_items(self.items, self.candidates)
        self.assertIn("Brass Key", suggestion)
        self.assertIn("Old Lantern", suggestion)

    def test_take_failure_includes_suggestions(self):
        """tool_take failure message should list available items."""
        state = WorldState(
            agent_location="r1",
            rooms={"r1": {"name": "R1", "exits": {}, "items": ["brass_key"]}},
            items=self.items,
            npcs={},
            events={},
            room_items={"r1": ["brass_key"]},
        )
        result = tool_take(state, "nonexistent thing")
        self.assertFalse(result.success)
        self.assertIn("Brass Key", result.observation)


# ---------------------------------------------------------------------------
# Fix 5: Event chain cycle detection
# ---------------------------------------------------------------------------

class TestEventCycleDetection(unittest.TestCase):

    def _make_engine_with_cycle(self):
        """Build a WorldEngine + WorldState with a trivially deep event chain."""
        engine = WorldEngine.__new__(WorldEngine)
        engine.model = "test"
        engine.temperature = 0.0
        engine._client = None
        engine._npc_dialogue_history = {}

        # Events A → triggers B → triggers A (cycle via npc_state_reached)
        # We test only depth guard, not a true infinite loop
        state = WorldState(
            scenario_id="test",
            scenario_name="Test",
            scenario_description="",
            rooms={},
            items={},
            npcs={},
            events={},
            goal={},
            agent_location="",
        )
        state._engine = engine
        engine.state = state
        return engine, state

    def test_depth_guard_raises_at_limit(self):
        engine, state = self._make_engine_with_cycle()
        with self.assertRaises(RuntimeError) as ctx:
            engine.fire_event_trigger("room_entered", state, _depth=6)
        self.assertIn("depth exceeded 5", str(ctx.exception))

    def test_normal_depth_does_not_raise(self):
        engine, state = self._make_engine_with_cycle()
        # depth=0, no matching events → should return empty list without error
        result = engine.fire_event_trigger("room_entered", state, _depth=0, room_id="x")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
