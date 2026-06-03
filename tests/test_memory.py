"""
Tests for engine/memory.py — WorkingMemory, EpisodicMemory, SemanticMemory, MemorySystem.
No API calls; purely in-process.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.memory import MemorySystem, EpisodicMemory, SemanticMemory


class TestWorkingMemory(unittest.TestCase):

    def test_working_memory_update(self):
        mem = MemorySystem()
        snapshot = {
            "agent_location": "dungeon",
            "current_room_data": {"description": "Dark and damp."},
            "visible_item_names": ["torch", "rope"],
            "visible_npc_names": ["goblin"],
            "available_exits": ["north", "east"],
            "inventory_names": ["sword"],
        }
        mem.update_working(snapshot)
        w = mem.working
        self.assertEqual(w.current_room, "dungeon")
        self.assertEqual(w.current_room_description, "Dark and damp.")
        self.assertEqual(w.visible_items, ["torch", "rope"])
        self.assertEqual(w.visible_npcs, ["goblin"])
        self.assertEqual(w.available_exits, ["north", "east"])
        self.assertEqual(w.inventory, ["sword"])


class TestEpisodicMemory(unittest.TestCase):

    def test_episodic_record(self):
        em = EpisodicMemory(max_episodes=50)
        for i in range(3):
            em.record(step=i, tool="move", tool_args={"direction": "north"},
                      observation=f"Moved north step {i}.")
        self.assertEqual(len(em.episodes), 3)

    def test_episodic_compression(self):
        max_ep = 10
        em = EpisodicMemory(max_episodes=max_ep, summary_keep=5)
        for i in range(max_ep + 5):
            em.record(step=i, tool="examine", tool_args={"target": "room"},
                      observation=f"You see a room. Step {i}.", significance=0.5)
        # After compression, episode count should be at or below max
        self.assertLessEqual(len(em.episodes), max_ep)
        # Some content should have been compressed into the summary
        self.assertTrue(len(em.compressed_summary) > 0)


class TestSemanticMemory(unittest.TestCase):

    def test_semantic_store_retrieve(self):
        sm = SemanticMemory()
        sm.store("key_location", "drawer", confidence=0.8, source="observation", step=1)
        value = sm.retrieve("key_location")
        self.assertEqual(value, "drawer")

    def test_semantic_confidence_no_overwrite(self):
        sm = SemanticMemory()
        sm.store("npc_mood", "angry", confidence=0.9, source="observation", step=1)
        sm.store("npc_mood", "calm", confidence=0.5, source="inference", step=2)
        # Lower-confidence update must not overwrite the existing higher-confidence fact
        value = sm.retrieve("npc_mood")
        self.assertEqual(value, "angry")


class TestMemorySystemBuildContext(unittest.TestCase):

    def test_memory_system_build_context(self):
        mem = MemorySystem()
        mem.update_working({
            "agent_location": "hall",
            "current_room_data": {"description": "A long hall."},
            "visible_item_names": [],
            "visible_npc_names": [],
            "available_exits": ["south"],
            "inventory_names": [],
        })
        context = mem.build_agent_context()
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("Working Memory", context)


if __name__ == "__main__":
    unittest.main()
