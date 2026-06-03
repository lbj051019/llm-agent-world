# LLM Agent in a Virtual World

An interactive fiction engine where a Claude-powered agent navigates text-based scenarios by reasoning through a structured **PLAN → THINK → ACT → OBSERVE → REFLECT** loop, using tool calls to manipulate world state that is independently verified by deterministic code.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│  scenario selection · per-step display · run logging            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │           Agent LLM               │
          │  (claude-sonnet-4-5 · agent.py)   │
          │                                   │
          │  PLAN ──► THINK ──► ACT           │
          │              │        │           │
          │           REFLECT ◄─ OBSERVE      │
          └──────┬────────────────┬───────────┘
                 │                │
     ┌───────────▼──────┐  ┌──────▼──────────────┐
     │   MemorySystem   │  │   Tool Dispatcher    │
     │   memory.py      │  │   tools.py           │
     │                  │  │                      │
     │  WorkingMemory   │  │  move · examine      │
     │  EpisodicMemory  │  │  take · use          │
     │  SemanticMemory  │  │  talk · think        │
     └──────────────────┘  │  finish              │
                           └──────┬───────────────┘
                                  │ mutates
                    ┌─────────────▼──────────────────┐
                    │         WorldEngine             │
                    │         world_engine.py         │
                    │                                 │
                    │  WorldState (single source of   │
                    │  truth: rooms, items, NPCs,     │
                    │  inventory, triggered events)   │
                    │                                 │
                    │  Event trigger system           │
                    │  NPC dialogue (World LLM)       │
                    └─────────────┬──────────────────┘
                                  │ verified by
                    ┌─────────────▼──────────────────┐
                    │        GoalVerifier             │
                    │        goal_verifier.py         │
                    │  5 condition types · AND / OR   │
                    └────────────────────────────────┘
```

---

## Installation

**Requirements:** Python 3.11+, an Anthropic API key.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/llm-agent-world.git
cd llm-agent-world

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
$env:ANTHROPIC_API_KEY = "sk-ant-..."  # Windows PowerShell
```

---

## Running

```bash
# List available scenarios
python main.py --list

# Run a specific scenario (interactive display)
python main.py --scenario haunted_library
python main.py --scenario space_station
python main.py --scenario medieval_castle

# Custom model / step limit
python main.py --scenario space_station --agent-model claude-sonnet-4-5 --steps 80

# Skip scenario validation (faster start)
python main.py --scenario haunted_library --skip-validation

# Run the full test suite (62 tests, no API calls)
python main.py --test
```

**Terminal output example:**

```
════════════════════════════════════════════════════════════════════════
  ◈  LLM AGENT IN A VIRTUAL WORLD
  Scenario : The Haunted Library
  Goal     : Burn the cursed tome, free the ghost, and escape the manor.
════════════════════════════════════════════════════════════════════════

  ────────────────────────────────────────────────────────────────────
  Step   1   📍 Entrance Hall   🎒 empty
  PLAN      Explore the manor systematically. The cursed tome is the key…
  THINK     I am in the entrance hall. I should examine the room first…
  ACT       {"tool": "examine", "args": {"target": "room"}}
  OBSERVE   Entrance Hall — dusty, cobwebbed. Exits: north, east.
  REFLECT   I need to find the library. Will head north next.
```

---

## Scenarios

| Scenario | Name | Challenge |
|---|---|---|
| `haunted_library` | The Haunted Library | Locate a cursed tome hidden across 6 rooms, perform a ritual to free a trapped ghost, then escape — all clues are buried in item descriptions and NPC dialogue. |
| `space_station` | Station Erebus | Restore emergency systems aboard a dying space station, neutralise a rogue AI (SABLE) using a purge code hidden inside a locked specimen case, and reach the airlock. |
| `medieval_castle` | The Siege of Ironhold | Infiltrate a castle as a spy, forge a pardon document by combining three separate items across multiple rooms, steal treason evidence, and escape through a hidden sewer passage. |

---

## Results

Best successful run per scenario:

| Scenario | Outcome | Steps | Wall Time |
|---|---|---|---|
| Haunted Library | ✅ Won | 21 / 60 | ~34 min\* |
| Station Erebus | ✅ Won | 27 / 60 | ~3 min |
| Siege of Ironhold | ✅ Won | 35 / 60 | ~3 min |

> \* The haunted_library run predates the rolling-context-window optimisation and single-API-call-per-step refactor; later runs on all scenarios average 2–4 min.

---

## Design Decisions

### 1. Structured Tool Use instead of Free-Form Natural Language

The agent does not emit prose commands. Every ACT is a strict JSON object dispatched through a typed tool registry:

```json
{"tool": "use", "args": {"item_name": "Brass Key", "target_name": "north"}}
```

This eliminates parsing ambiguity. The tool layer performs fuzzy name matching (case-insensitive, token-overlap fallback) so minor LLM phrasing variation never causes a silent no-op. Failure messages include candidate suggestions so the agent can self-correct immediately:

```
You don't have 'brass key' in your inventory. Inventory: Iron Torch, Ritual Candle.
```

### 2. Observations Are Extracted from World State, Not Generated

The agent's predicted `OBSERVE` section is **discarded**. The real observation is produced by deterministic Python code reading `WorldState` directly:

```
[World] Actual result of move: You move north.

**Great Hall** — a vast stone chamber lit by guttering torches.
Items visible: War Banner, Iron Torch.
Exits: south, east (locked).
```

Observations are always factually accurate. The LLM's prediction is kept only in the conversation history as a reasoning scaffold — it never enters the world state.

### 3. Dual-Layer Goal Verification

Goal completion is checked on two independent paths:

- **Agent layer** — the agent calls `finish(reason)` when it *believes* it has succeeded.
- **Code layer** — `GoalVerifier` evaluates all goal conditions against live `WorldState` on every step, before the agent acts.

If the code confirms all conditions are met, the run ends with `won=True` regardless of what the agent believes. If the agent calls `finish()` but conditions are unmet, `won=False` is recorded. This prevents both false positives (agent hallucinates success) and false negatives (agent gives up prematurely while the world is already in a winning state).

The five verifiable condition types are: `agent_at_location`, `item_in_inventory`, `item_used_on_target`, `npc_state`, and `event_triggered`.

### 4. Rolling Context Window with Code-Generated Summaries

Keeping the full conversation history causes quadratic token-cost growth. The solution retains only the **5 most recent step-buckets** verbatim and replaces older steps with a compact summary produced by pure text extraction — no additional LLM call:

```
已完成行动摘要 (12 steps):
  位置变化: Entrance Hall → Library → Restricted Stacks → Ritual Chamber
  已收集物品: Ritual Candle, Cursed Tome, Silver Key
  最近行动: examine(tome); take(candle); use(candle, brazier)
  关键观察: The brazier flares. The tome begins to burn.
```

Token cost per step stays constant regardless of run length, making 60-step runs practical at standard API rates.

---

## Project Structure

```
llm-agent-world/
├── main.py                   # CLI entry point, terminal display, run orchestration
│
├── engine/
│   ├── agent.py              # Agent LLM: PLAN/THINK/ACT/OBSERVE/REFLECT loop
│   ├── world_engine.py       # WorldState, event system, NPC dialogue (World LLM)
│   ├── tools.py              # 7 tool implementations + fuzzy name matching
│   ├── memory.py             # WorkingMemory, EpisodicMemory, SemanticMemory
│   └── goal_verifier.py      # Deterministic goal condition checker (5 types)
│
├── scenarios/
│   ├── schema.md             # JSON scenario format specification
│   ├── validator.py          # Static validator: BFS reachability, ref checks, cycle detection
│   ├── haunted_library.json  # Gothic mystery — 6 rooms, 9 items, 1 NPC
│   ├── space_station.json    # Sci-fi survival — 7 rooms, 12 items, 1 NPC
│   └── medieval_castle.json  # Spy thriller — 10 rooms, 13 items, 3 NPCs
│
├── tests/
│   ├── test_tools.py         # 15 tests — all 7 tools, locked exits, item matching
│   ├── test_goal_verifier.py # 11 tests — all 5 condition types, AND/OR logic
│   ├── test_memory.py        # 6 tests — compression, confidence weighting
│   ├── test_world_engine.py  # 7 tests — scenario loading, events, NPC state changes
│   └── test_hardening.py     # 24 tests — JSON parsing, truncation, repeat detection
│
├── logs/                     # Per-run .log (debug) and .json (structured summary)
└── requirements.txt          # anthropic>=0.25.0
```

**Test suite:** 62 tests, zero API calls, < 1 second total runtime.

```bash
$ python main.py --test
....................................................................(62 tests)
----------------------------------------------------------------------
Ran 62 tests in 0.003s  OK
```

---

## Known Limitations

**Multi-step item combination is hard-coded.**
The forge-a-pardon puzzle in `medieval_castle` requires combining three items in sequence. The combination logic lives in `tools.py` (`_apply_use_side_effects`). Adding new compound interactions requires modifying engine code rather than just the scenario JSON.

**NPC dialogue history is in-memory only.**
Each `WorldEngine` instance holds per-NPC conversation history at runtime. There is no persistence layer; a resumed run loses all prior NPC dialogue context.

**Single-agent, sequential exploration.**
There is no delegation, sub-agent spawning, or parallel pathfinding. The agent must navigate one room at a time, which can cause redundant backtracking in large scenarios.

**No explicit spatial map.**
The agent relies entirely on in-context recall for room connectivity. A graph-based spatial memory would allow direct pathfinding and significantly reduce wasted steps in late-game navigation.

---

## Future Improvements

- **Scenario DSL** — replace hand-crafted JSON with a higher-level format; auto-generate item combination rules from declarative scenario data rather than hard-coded engine logic.
- **Persistent save/load** — serialise `WorldState` + agent memory to disk so runs can be interrupted and resumed.
- **Graph-based spatial memory** — build an explicit room graph as the agent explores and expose it as a queryable semantic fact.
- **Automated benchmark runner** — headless batch mode that measures win rate, mean steps-to-win, and total token cost across all scenarios over multiple runs.
- **Richer event system** — support conditional effects (`if inventory contains X`) and timed triggers (fire after N steps) declared entirely in JSON.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (Agent reasoning) | Claude claude-sonnet-4-5 via Anthropic Python SDK |
| LLM (NPC dialogue) | Claude claude-sonnet-4-5, separate conversation history per NPC |
| Runtime | Python 3.11, stdlib only — no LangChain, no vector DB |
| Tests | `unittest` — 62 tests, 0 API calls |
