"""
Agent LLM — drives the PLAN → THINK → ACT → OBSERVE → REFLECT → FINISH loop.

Each cycle:
  PLAN    — high-level strategy review (every N steps or when plan is stale)
  THINK   — immediate reasoning about current situation
  ACT     — select and call a tool
  OBSERVE — receive and interpret the tool result
  REFLECT — update memory, beliefs, and plan
  FINISH  — declare success or failure if conditions are met or agent is stuck
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from engine.memory import MemorySystem
from engine.tools import execute_tool, ToolResult, TOOL_REGISTRY
from engine.goal_verifier import GoalVerifier
from engine.world_engine import WorldEngine, WorldState


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    step_number: int
    phase: str
    content: str
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentRunResult:
    success: bool
    won: bool
    total_steps: int
    finish_reason: str
    steps: list[AgentStep]
    final_verification: Any  # VerificationResult


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS = """
Available tools (call exactly one per ACT phase):

move(direction)
  Move in a direction: north, south, east, west, up, down, in, out.

examine(target)
  Examine an item, NPC, or type "room" to look around.

take(item_name)
  Pick up an item from the current room.

use(item_name, target_name)
  Use an inventory item on a target item or NPC.

talk(npc_name, message)
  Speak to an NPC in the current room.

think(thought)
  Record internal reasoning. No world effect.

finish(reason)
  Declare the goal complete or that you cannot continue.
"""

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent agent navigating an interactive fiction world.
Your goal: {goal_description}

{tool_descriptions}

Each step you will receive the current world state and must respond with EXACTLY this format
(all four sections, in order, no extra text before or after):

THINK: <2-3 sentences reasoning about your immediate situation and what to do next>
ACT: {{"tool": "tool_name", "args": {{"param1": "value1"}}}}
OBSERVE: <1-2 sentences predicting what will happen>
REFLECT: <1-2 sentences on what this means for your overall plan>

Rules:
- The ACT line must be a single valid JSON object and nothing else.
- Explore systematically. Examine readable items for clues before using them.
- Talk to NPCs — they often hold essential information.
- Do not repeat the same failed action. If blocked, try a different approach.
- When the goal is fully complete, call finish() with a clear reason.
"""

PLAN_PROMPT = """PLAN: Review your overall strategy given the current situation.
What is your high-level plan and what is the single most important next objective? (2-4 sentences)"""


class Agent:
    def __init__(
        self,
        world_engine: WorldEngine,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.7,
        max_steps: int = 60,
        plan_interval: int = 8,
        context_window: int = 5,  # number of recent steps kept verbatim
    ):
        self.world_engine = world_engine
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps
        self.plan_interval = plan_interval
        self.context_window = context_window
        self._client: anthropic.Anthropic | None = None
        self.memory = MemorySystem()
        self.verifier: GoalVerifier | None = None
        self.steps: list[AgentStep] = []
        self._system_prompt: str = ""
        # Rolling window storage: list of step-buckets.
        # Each bucket is a list of {role, content} dicts produced during that step.
        self._step_buckets: list[list[dict]] = []
        # Intro message stored separately (always included)
        self._intro_msg: str = ""
        # Repeat action detection: last 5 (tool_name, frozenset(args)) tuples
        self._recent_actions: list[tuple] = []

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, state: WorldState) -> AgentRunResult:
        self.verifier = GoalVerifier(
            {"goal": state.goal, "conditions": state.goal.get("conditions", [])}
        )
        self.verifier = GoalVerifier(state.__dict__ | {"goal": state.goal})
        # rebuild with the actual scenario dict structure
        self.verifier = _make_verifier(state)

        # init system prompt
        system_msg = SYSTEM_PROMPT_TEMPLATE.format(
            goal_description=state.goal.get("description", "Complete the scenario."),
            tool_descriptions=TOOL_DESCRIPTIONS,
        )
        self._system_prompt = system_msg
        self._step_buckets = []
        self._recent_actions = []

        # describe starting room
        intro = self.world_engine.describe_current_room(state)
        self._intro_msg = (
            f"Scenario: {state.scenario_name}\n{state.scenario_description}\n\n"
            f"You begin here:\n{intro}\n\n"
            f"Your goal: {state.goal.get('description', '')}"
        )
        self.memory.working.step_number = 0

        for step in range(1, self.max_steps + 1):
            self.memory.working.step_number = step
            snap = state.snapshot()
            self.memory.update_working(snap)

            # --- objective check before agent acts ---
            verification = self.verifier.verify(state)
            if verification.won:
                return self._finish(
                    state, step, "All goal conditions objectively verified as complete.", won=True
                )

            # Each step writes into its own bucket
            bucket: list[dict] = []

            # --- build context injection ---
            context = self._build_context_message(state, step, verification)
            bucket.append({"role": "user", "content": context})

            # --- PLAN (separate API call every plan_interval steps) ---
            if step == 1 or (step % self.plan_interval == 0):
                plan_text = self._call_plan(state, bucket)
                self.memory.working.current_plan = plan_text
                self._record_step(step, "PLAN", plan_text)

            # --- THINK / ACT / OBSERVE(predicted) / REFLECT — single API call ---
            parsed = self._call_think_act_reflect(state, bucket)
            think_text        = parsed["think"]
            act_response      = parsed["act_raw"]
            predicted_observe = parsed["observe"]
            reflect_text      = parsed["reflect"]

            tool_name, tool_args = self._parse_tool_call(act_response)

            self._record_step(step, "THINK", think_text)
            self._record_step(step, "ACT", act_response, tool_name=tool_name, tool_args=tool_args)

            # --- execute tool (real world) ---
            tool_result = self._execute_tool(state, tool_name, tool_args)

            # handle NPC dialogue requiring World LLM
            if tool_result.metadata.get("requires_llm"):
                npc_id = tool_result.metadata["npc_id"]
                message = tool_result.metadata["message"]
                tool_result.observation = self.world_engine.generate_npc_dialogue(
                    state, npc_id, message
                )

            # OBSERVE: use real world result (discard LLM prediction)
            real_observe = _truncate_observation(tool_result.observation)
            tool_result.observation = real_observe

            # --- repeat action detection ---
            action_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            self._recent_actions.append(action_key)
            if len(self._recent_actions) > 5:
                self._recent_actions.pop(0)
            if len(self._recent_actions) >= 3 and all(
                a == action_key for a in self._recent_actions[-3:]
            ):
                real_observe += (
                    "\n\n[WARNING] You have repeated the same action 3 times in a row. "
                    "Please try a completely different approach or explore elsewhere."
                )

            self._record_step(
                step, "OBSERVE", real_observe,
                tool_name=tool_name, tool_args=tool_args, tool_result=tool_result,
            )

            # Complete the bucket with assistant turn + real observation
            bucket.append({
                "role": "assistant",
                "content": (
                    f"THINK: {think_text}\n"
                    f"ACT: {act_response}\n"
                    f"OBSERVE: {predicted_observe}\n"
                    f"REFLECT: {reflect_text}"
                ),
            })
            bucket.append({
                "role": "user",
                "content": f"[World] Actual result of {tool_name}: {real_observe}",
            })

            self._step_buckets.append(bucket)
            self._record_step(step, "REFLECT", reflect_text)

            # record in memory
            self.memory.record_action(
                tool=tool_name,
                tool_args=tool_args,
                observation=real_observe,
                success=tool_result.success,
            )
            self._auto_infer_facts(state, tool_name, tool_args, tool_result)

            # check for finish tool
            if tool_name == "finish":
                final_verification = self.verifier.verify(state)
                return AgentRunResult(
                    success=True,
                    won=final_verification.won,
                    total_steps=step,
                    finish_reason=tool_args.get("reason", reflect_text),
                    steps=self.steps,
                    final_verification=final_verification,
                )

        # max steps reached
        final_verification = self.verifier.verify(state)
        return AgentRunResult(
            success=False,
            won=final_verification.won,
            total_steps=self.max_steps,
            finish_reason="Maximum steps reached without completing the goal.",
            steps=self.steps,
            final_verification=final_verification,
        )

    # ------------------------------------------------------------------
    # Phase calls
    # ------------------------------------------------------------------

    def _build_messages(self, current_bucket: list[dict]) -> list[dict]:
        """
        Assemble the messages list for an API call using the rolling window:
          [intro]  +  [summary of old steps]  +  [last N step-buckets]  +  [current bucket so far]
        Total verbatim steps never exceeds context_window.
        """
        messages: list[dict] = [{"role": "user", "content": self._intro_msg}]

        old_buckets = self._step_buckets[: max(0, len(self._step_buckets) - self.context_window)]
        recent_buckets = self._step_buckets[max(0, len(self._step_buckets) - self.context_window) :]

        if old_buckets:
            # build summary from the last known WorldState (pulled from memory)
            summary = _summarise_old_steps(old_buckets)
            messages.append({"role": "user", "content": summary})
            # need a minimal assistant ack so the alternation rule is satisfied
            messages.append({"role": "assistant", "content": "Understood. Continuing from summary."})

        for b in recent_buckets:
            messages.extend(b)

        messages.extend(current_bucket)
        return messages

    def _call_plan(self, state: WorldState, bucket: list[dict]) -> str:
        """Separate API call for PLAN phase (every plan_interval steps)."""
        bucket.append({"role": "user", "content": PLAN_PROMPT})
        response = self.client.messages.create(
            model=self.model,
            system=self._system_prompt,
            messages=self._build_messages(bucket),
            temperature=self.temperature,
            max_tokens=300,
        )
        reply = response.content[0].text.strip()
        bucket.append({"role": "assistant", "content": reply})
        return reply

    def _call_think_act_reflect(self, state: WorldState, bucket: list[dict]) -> dict:
        """
        Single API call that returns THINK / ACT / OBSERVE / REFLECT together.
        Returns a dict with keys: think, act_raw, observe, reflect.
        """
        prompt = (
            "Now respond with all four sections in order:\n"
            "THINK: ...\n"
            "ACT: {\"tool\": \"...\", \"args\": {...}}\n"
            "OBSERVE: ...\n"
            "REFLECT: ..."
        )
        bucket.append({"role": "user", "content": prompt})
        response = self.client.messages.create(
            model=self.model,
            system=self._system_prompt,
            messages=self._build_messages(bucket),
            temperature=self.temperature,
            max_tokens=600,
        )
        raw = response.content[0].text.strip()
        # bucket gets the assistant turn appended by run() after real OBSERVE is known
        return _parse_combined_response(raw)

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def _build_context_message(
        self, state: WorldState, step: int, verification: Any
    ) -> str:
        snap = state.snapshot()
        room = snap["current_room_data"]
        lines = [
            f"--- Step {step} ---",
            f"Location: {room.get('name', state.agent_location)}",
            f"Inventory: {', '.join(snap['inventory_names']) or 'empty'}",
            f"Visible items: {', '.join(snap['visible_item_names']) or 'none'}",
            f"Visible NPCs: {', '.join(snap['visible_npc_names']) or 'none'}",
            f"Exits: {', '.join(snap['available_exits']) or 'none'}",
            "",
            "Goal progress:",
            verification.summary(),
        ]
        # inject relevant semantic facts
        key_facts = self.memory.semantic.search("")
        if key_facts:
            lines.append("\nKnown facts:")
            for fact in key_facts[:8]:
                lines.append(f"  - {fact.key}: {fact.value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool parsing and execution
    # ------------------------------------------------------------------

    def _parse_tool_call(self, act_text: str) -> tuple[str, dict]:
        """Extract tool name and args from ACT response JSON."""
        # find the outermost {...} block, handling nested braces
        parsed = _extract_json_object(act_text)
        if parsed is not None:
            tool_name = parsed.get("tool", "think").lower()
            args = parsed.get("args", {})
            if tool_name == "think" and "thought" not in args:
                args["thought"] = act_text
            return tool_name, args

        # fallback: try to parse "tool: args" style
        lines = act_text.strip().split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                tool_candidate = parts[0].strip().lower()
                if tool_candidate in TOOL_REGISTRY:
                    return tool_candidate, {"thought": parts[1].strip()}

        # last resort
        return "think", {"thought": act_text}

    def _execute_tool(
        self, state: WorldState, tool_name: str, tool_args: dict
    ) -> ToolResult:
        return execute_tool(state, tool_name, **tool_args)

    # ------------------------------------------------------------------
    # Memory inference
    # ------------------------------------------------------------------

    def _auto_infer_facts(
        self,
        state: WorldState,
        tool_name: str,
        tool_args: dict,
        result: ToolResult,
    ) -> None:
        step = self.memory.working.step_number
        obs = result.observation.lower()

        if tool_name == "take" and result.success:
            item_name = tool_args.get("item_name", "")
            self.memory.infer_fact(
                f"have_{item_name.replace(' ', '_')}",
                True,
                source="observation",
                confidence=1.0,
            )

        if tool_name == "examine" and result.success:
            # extract clues from examination text
            if "key" in obs and "locked" in obs:
                target = tool_args.get("target", "")
                self.memory.infer_fact(
                    f"clue_from_{target.replace(' ', '_')}",
                    result.observation[:200],
                    source="item_read",
                    confidence=0.9,
                )
            if "provides_clue" in str(result.metadata) or any(
                kw in obs for kw in ("must", "need", "require", "code", "secret", "hidden")
            ):
                target = tool_args.get("target", "unknown")
                self.memory.infer_fact(
                    f"clue_{target.replace(' ', '_')}_{step}",
                    result.observation[:300],
                    source="item_read",
                    confidence=0.85,
                )

        if tool_name == "move" and result.success:
            self.memory.infer_fact(
                "current_room", state.agent_location, source="observation", confidence=1.0
            )

        if tool_name == "use" and result.success:
            item = tool_args.get("item_name", "")
            target = tool_args.get("target_name", "")
            self.memory.infer_fact(
                f"used_{item.replace(' ', '_')}_on_{target.replace(' ', '_')}",
                True,
                source="observation",
                confidence=1.0,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_step(
        self,
        step: int,
        phase: str,
        content: str,
        tool_name: str = "",
        tool_args: dict | None = None,
        tool_result: ToolResult | None = None,
    ) -> None:
        self.steps.append(AgentStep(
            step_number=step,
            phase=phase,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_result=tool_result,
        ))

    def _finish(
        self, state: WorldState, step: int, reason: str, won: bool
    ) -> AgentRunResult:
        final_verification = self.verifier.verify(state)
        self._record_step(step, "FINISH", reason)
        return AgentRunResult(
            success=True,
            won=won,
            total_steps=step,
            finish_reason=reason,
            steps=self.steps,
            final_verification=final_verification,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verifier(state: WorldState) -> GoalVerifier:
    """Build a GoalVerifier from a live WorldState."""
    pseudo_scenario = {
        "goal": state.goal,
        "rooms": state.rooms,
        "items": state.items,
        "npcs": state.npcs,
        "events": state.events,
    }
    return GoalVerifier(pseudo_scenario)


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={repr(v)}" for k, v in args.items())


def _summarise_old_steps(buckets: list[list[dict]]) -> str:
    """
    Build a compact summary of completed steps from their conversation buckets.
    Extracts: locations visited, items acquired, tool actions taken, key observations.
    No LLM call — pure text extraction.
    """
    locations: list[str] = []
    items_taken: list[str] = []
    actions: list[str] = []
    observations: list[str] = []

    for bucket in buckets:
        for msg in bucket:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # extract location from context headers
            if role == "user" and content.startswith("--- Step"):
                for line in content.splitlines():
                    if line.startswith("Location:"):
                        loc = line.replace("Location:", "").strip()
                        if not locations or locations[-1] != loc:
                            locations.append(loc)

            # extract ACT lines from assistant turns
            if role == "assistant":
                for line in content.splitlines():
                    if line.upper().startswith("ACT:"):
                        act_text = line[4:].strip()
                        obj = _extract_json_object(act_text)
                        if obj:
                            tool = obj.get("tool", "")
                            args = obj.get("args", {})
                            if tool == "take":
                                item = args.get("item_name", "")
                                if item:
                                    items_taken.append(item)
                            if tool not in ("think",):
                                arg_str = ", ".join(f"{v}" for v in args.values())
                                actions.append(f"{tool}({arg_str})")

            # extract real world results
            if role == "user" and content.startswith("[World] Actual result of"):
                # keep only short observations
                result_text = content.split(":", 1)[-1].strip()
                if len(result_text) < 200:
                    observations.append(result_text)

    parts = [f"已完成行动摘要 ({len(buckets)} steps):"]
    if locations:
        parts.append(f"  位置变化: {' → '.join(dict.fromkeys(locations))}")
    if items_taken:
        parts.append(f"  已收集物品: {', '.join(dict.fromkeys(items_taken))}")
    if actions:
        recent_actions = actions[-10:]  # last 10 actions only
        parts.append(f"  最近行动: {'; '.join(recent_actions)}")
    if observations:
        parts.append(f"  关键观察: {observations[-1]}" if observations else "")

    return "\n".join(p for p in parts if p)


def _parse_combined_response(raw: str) -> dict:
    """
    Parse a combined THINK/ACT/OBSERVE/REFLECT response into its four parts.
    Each section starts with its label on its own token; content runs until the next label.
    Falls back gracefully if sections are missing.
    """
    import re as _re
    sections = {"think": "", "act_raw": "", "observe": "", "reflect": ""}

    # Split on section headers (case-insensitive, allow optional colon/space)
    pattern = _re.compile(r'^(THINK|ACT|OBSERVE|REFLECT)\s*:\s*', _re.IGNORECASE | _re.MULTILINE)
    parts = pattern.split(raw)
    # parts: [pre, label, content, label, content, ...]
    it = iter(parts[1:])  # skip pre-text
    for label, content in zip(it, it):
        key = label.upper()
        text = content.strip()
        if key == "THINK":
            sections["think"] = text
        elif key == "ACT":
            # ACT content may have trailing text after the JSON — extract just the JSON
            sections["act_raw"] = text
        elif key == "OBSERVE":
            sections["observe"] = text
        elif key == "REFLECT":
            sections["reflect"] = text

    # fallback: if ACT not found, try to pull a JSON object from the whole raw text
    if not sections["act_raw"]:
        obj = _extract_json_object(raw)
        if obj:
            sections["act_raw"] = json.dumps(obj)

    # ensure think has something for memory
    if not sections["think"]:
        sections["think"] = raw[:200]

    return sections


def _extract_json_object(text: str) -> dict | None:
    """
    Find and parse the first complete JSON object in text, handling nested braces.
    Strips markdown code fences (```json ... ```) before parsing.
    Returns the parsed dict, or None if no valid object is found.
    """
    # strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


_OBS_MAX_CHARS = 800


def _truncate_observation(obs: str) -> str:
    """Truncate observation to _OBS_MAX_CHARS, preserving leading lines (location/exits/inventory)."""
    if len(obs) <= _OBS_MAX_CHARS:
        return obs
    cut = obs[: _OBS_MAX_CHARS].rsplit("\n", 1)[0]
    return cut + "\n[truncated]"
