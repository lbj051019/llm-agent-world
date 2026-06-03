"""
Objective goal verification layer.

Evaluates scenario win conditions against live WorldState without
relying on the Agent LLM's self-assessment.  Supports AND (all conditions)
and OR (any condition) logic, plus individual condition introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.world_engine import WorldState


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    condition_type: str
    satisfied: bool
    detail: str  # human-readable explanation of pass/fail


@dataclass
class VerificationResult:
    won: bool
    logic: str  # "AND" | "OR"
    condition_results: list[ConditionResult]
    satisfied_count: int
    total_count: int

    def summary(self) -> str:
        lines = [
            f"Goal {'ACHIEVED' if self.won else 'NOT YET ACHIEVED'} "
            f"({self.satisfied_count}/{self.total_count} conditions, logic={self.logic})"
        ]
        for cr in self.condition_results:
            mark = "✓" if cr.satisfied else "✗"
            lines.append(f"  {mark} [{cr.condition_type}] {cr.detail}")
        return "\n".join(lines)

    def unsatisfied(self) -> list[ConditionResult]:
        return [cr for cr in self.condition_results if not cr.satisfied]

    def satisfied_conditions(self) -> list[ConditionResult]:
        return [cr for cr in self.condition_results if cr.satisfied]


# ---------------------------------------------------------------------------
# Individual condition checkers
# ---------------------------------------------------------------------------

def check_agent_at_location(state: "WorldState", condition: dict) -> ConditionResult:
    room_id = condition["room_id"]
    satisfied = state.agent_location == room_id
    room_name = state.rooms.get(room_id, {}).get("name", room_id)
    current_name = state.rooms.get(state.agent_location, {}).get("name", state.agent_location)
    return ConditionResult(
        condition_type="agent_at_location",
        satisfied=satisfied,
        detail=(
            f"Agent in '{room_name}'" if satisfied
            else f"Agent is in '{current_name}', needs '{room_name}'"
        ),
    )


def check_item_in_inventory(state: "WorldState", condition: dict) -> ConditionResult:
    item_id = condition["item_id"]
    satisfied = item_id in state.inventory
    item_name = state.items.get(item_id, {}).get("name", item_id)
    return ConditionResult(
        condition_type="item_in_inventory",
        satisfied=satisfied,
        detail=(
            f"Carrying '{item_name}'" if satisfied
            else f"Not carrying '{item_name}'"
        ),
    )


def check_item_used_on_target(state: "WorldState", condition: dict) -> ConditionResult:
    item_id = condition["item_id"]
    target_id = condition["target_id"]
    satisfied = (item_id, target_id) in state.used_items

    item_name = state.items.get(item_id, {}).get("name", item_id)
    # target may be item or NPC
    if target_id in state.items:
        target_name = state.items[target_id].get("name", target_id)
    elif target_id in state.npcs:
        target_name = state.npcs[target_id].get("name", target_id)
    else:
        target_name = target_id

    return ConditionResult(
        condition_type="item_used_on_target",
        satisfied=satisfied,
        detail=(
            f"Used '{item_name}' on '{target_name}'" if satisfied
            else f"'{item_name}' has not been used on '{target_name}'"
        ),
    )


def check_npc_state(state: "WorldState", condition: dict) -> ConditionResult:
    npc_id = condition["npc_id"]
    required_state = condition["state"]

    if npc_id not in state.npcs:
        return ConditionResult(
            condition_type="npc_state",
            satisfied=False,
            detail=f"NPC '{npc_id}' not found in scenario",
        )

    npc = state.npcs[npc_id]
    current_state = npc.get("current_state", npc.get("initial_state", ""))
    satisfied = current_state == required_state
    npc_name = npc.get("name", npc_id)

    return ConditionResult(
        condition_type="npc_state",
        satisfied=satisfied,
        detail=(
            f"{npc_name} is in state '{current_state}' (required: '{required_state}')"
            if not satisfied
            else f"{npc_name} is in state '{required_state}'"
        ),
    )


def check_event_triggered(state: "WorldState", condition: dict) -> ConditionResult:
    event_id = condition["event_id"]
    satisfied = event_id in state.triggered_events
    event_desc = state.events.get(event_id, {}).get("description", event_id)

    return ConditionResult(
        condition_type="event_triggered",
        satisfied=satisfied,
        detail=(
            f"Event '{event_id}' has occurred" if satisfied
            else f"Event '{event_id}' has not occurred ({event_desc})"
        ),
    )


# ---------------------------------------------------------------------------
# Condition dispatcher
# ---------------------------------------------------------------------------

CONDITION_CHECKERS = {
    "agent_at_location": check_agent_at_location,
    "item_in_inventory": check_item_in_inventory,
    "item_used_on_target": check_item_used_on_target,
    "npc_state": check_npc_state,
    "event_triggered": check_event_triggered,
}


def evaluate_condition(state: "WorldState", condition: dict) -> ConditionResult:
    ctype = condition.get("type")
    checker = CONDITION_CHECKERS.get(ctype)
    if not checker:
        return ConditionResult(
            condition_type=ctype or "unknown",
            satisfied=False,
            detail=f"Unknown condition type '{ctype}'",
        )
    return checker(state, condition)


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class GoalVerifier:
    """
    Verifies win conditions for a loaded scenario.

    By default all conditions must be satisfied (AND logic).
    Pass logic="OR" to win when any single condition is met.
    """

    def __init__(self, scenario: dict, logic: str = "AND"):
        self.goal = scenario.get("goal", {})
        self.conditions: list[dict] = self.goal.get("conditions", [])
        self.logic = logic.upper()

    def verify(self, state: "WorldState") -> VerificationResult:
        results = [evaluate_condition(state, cond) for cond in self.conditions]
        satisfied_count = sum(1 for r in results if r.satisfied)
        total = len(results)

        if self.logic == "OR":
            won = satisfied_count > 0
        else:  # AND
            won = satisfied_count == total

        return VerificationResult(
            won=won,
            logic=self.logic,
            condition_results=results,
            satisfied_count=satisfied_count,
            total_count=total,
        )

    def check_single(self, state: "WorldState", condition_index: int) -> ConditionResult:
        """Check one condition by index."""
        if condition_index >= len(self.conditions):
            return ConditionResult("unknown", False, f"No condition at index {condition_index}")
        return evaluate_condition(state, self.conditions[condition_index])

    def progress_report(self, state: "WorldState") -> str:
        result = self.verify(state)
        return result.summary()

    def hint_for_agent(self, state: "WorldState") -> str:
        """Return a plain-language hint about the next unsatisfied condition."""
        result = self.verify(state)
        if result.won:
            return "All goal conditions are satisfied."
        unsatisfied = result.unsatisfied()
        if not unsatisfied:
            return "All conditions satisfied."
        # return the first unsatisfied condition as a nudge
        first = unsatisfied[0]
        return f"Not yet complete: {first.detail}"
