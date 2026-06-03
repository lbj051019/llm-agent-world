"""
Main entry point — CLI for running LLM Agent in a Virtual World.

Usage:
  python main.py                          # interactive scenario selection
  python main.py --scenario haunted_library
  python main.py --scenario space_station --steps 80 --agent-model claude-sonnet-4-5
  python main.py --scenario medieval_castle --world-model claude-sonnet-4-5 --verbose
  python main.py --list                   # list available scenarios
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.agent import Agent, AgentRunResult
from engine.world_engine import WorldEngine
from scenarios.validator import validate_file


# ---------------------------------------------------------------------------
# Logging — file only, never pollute terminal with DEBUG noise
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(scenario_id: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{scenario_id}_{timestamp}.log"

    # suppress noisy third-party loggers (httpx, anthropic, openai)
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger = logging.getLogger("agent_world")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger, log_file


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "blue":   "\033[94m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "magenta":"\033[95m",
    "red":    "\033[91m",
    "white":  "\033[97m",
    "gray":   "\033[90m",
}

PHASE_COLOR = {
    "PLAN":    C["blue"],
    "THINK":   C["cyan"],
    "ACT":     C["green"],
    "OBSERVE": C["yellow"],
    "REFLECT": C["magenta"],
    "FINISH":  C["red"],
}

STEP_WIDTH = 72


def _wrap(text: str, max_chars: int = 140) -> str:
    """Trim to max_chars, break at word boundary, add ellipsis."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def print_banner(scenario_name: str, goal: str) -> None:
    print()
    print(C["bold"] + "═" * STEP_WIDTH + C["reset"])
    print(C["bold"] + f"  ◈  LLM AGENT IN A VIRTUAL WORLD" + C["reset"])
    print(f"  Scenario : {C['white']}{scenario_name}{C['reset']}")
    print(f"  Goal     : {C['dim']}{_wrap(goal, 100)}{C['reset']}")
    print(C["bold"] + "═" * STEP_WIDTH + C["reset"])
    print()


def print_step_header(step: int, location: str, inventory: list[str]) -> None:
    inv = ", ".join(inventory) if inventory else "empty"
    loc_str = f"📍 {location}"
    inv_str = f"🎒 {inv}"
    print(C["gray"] + "─" * STEP_WIDTH + C["reset"])
    print(f"  {C['bold']}Step {step:3d}{C['reset']}   {C['dim']}{loc_str}   {inv_str}{C['reset']}")


def print_phase(phase: str, content: str) -> None:
    color = PHASE_COLOR.get(phase, C["white"])
    label = f"{color}{phase:8s}{C['reset']}"
    # show up to 2 lines (split at sentence boundaries if possible)
    lines = _split_two_lines(content, 120)
    print(f"  {label}  {lines[0]}")
    if len(lines) > 1:
        print(f"  {' ' * 10}  {C['dim']}{lines[1]}{C['reset']}")


def _split_two_lines(text: str, width: int) -> list[str]:
    text = text.strip().replace("\n", " ")
    if len(text) <= width:
        return [text]
    # try to break at sentence end within first `width` chars
    chunk = text[:width]
    for sep in (". ", "! ", "? ", "; "):
        idx = chunk.rfind(sep)
        if idx > width // 2:
            return [text[:idx + 1].strip(), _wrap(text[idx + 1:].strip(), width)]
    # fall back to word break
    cut = chunk.rsplit(" ", 1)[0]
    return [cut, _wrap(text[len(cut):].strip(), width)]


def print_result(result: AgentRunResult, elapsed: float) -> None:
    print()
    print(C["bold"] + "═" * STEP_WIDTH + C["reset"])
    if result.won:
        print(f"  {C['green']}{C['bold']}✓  GOAL ACHIEVED{C['reset']}")
    else:
        print(f"  {C['red']}{C['bold']}✗  GOAL NOT ACHIEVED{C['reset']}")
    print(f"  Steps   : {result.total_steps}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Reason  : {_wrap(result.finish_reason, 100)}")
    print(C["bold"] + "─" * STEP_WIDTH + C["reset"])
    print(f"\n{C['bold']}Goal conditions:{C['reset']}")
    for cr in result.final_verification.condition_results:
        mark = f"{C['green']}✓{C['reset']}" if cr.satisfied else f"{C['red']}✗{C['reset']}"
        print(f"  {mark}  {cr.detail}")
    print()


# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------

def select_scenario_interactive(engine: WorldEngine, scenarios_dir: Path) -> str:
    scenarios = engine.get_scenario_list(scenarios_dir)
    if not scenarios:
        print("No scenario files found in:", scenarios_dir)
        sys.exit(1)
    print("\nAvailable scenarios:\n")
    for i, s in enumerate(scenarios, 1):
        print(f"  [{i}] {C['white']}{s['name']}{C['reset']}")
        print(f"      {C['dim']}{s['description'][:100]}…{C['reset']}\n")
    while True:
        choice = input("Select scenario (number or id): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                return scenarios[idx]["path"]
        for s in scenarios:
            if choice in (s["id"], s["name"]):
                return s["path"]
        print("Invalid selection. Try again.")


# ---------------------------------------------------------------------------
# Run session
# ---------------------------------------------------------------------------

def run_session(
    scenario_path: str,
    agent_model: str = "claude-sonnet-4-5",
    world_model: str = "claude-sonnet-4-5",
    max_steps: int = 60,
    save_log: bool = True,
) -> AgentRunResult:

    engine = WorldEngine(model=world_model)
    state = engine.load_scenario(scenario_path)
    logger, log_file = setup_logging(state.scenario_id)

    print_banner(state.scenario_name, state.goal.get("description", ""))

    agent = Agent(world_engine=engine, model=agent_model, max_steps=max_steps)

    # ---- per-step display state ----
    _step_state: dict = {"current_step": 0, "header_printed": False}

    original_record = agent._record_step

    def recording_hook(step: int, phase: str, content: str, **kwargs):
        original_record(step, phase, content, **kwargs)
        logger.debug("[Step %d] %s: %s", step, phase, content)

        # print step header once per new step number
        if step != _step_state["current_step"]:
            _step_state["current_step"] = step
            snap = state.snapshot()
            print_step_header(
                step,
                snap["current_room_data"].get("name", state.agent_location),
                snap["inventory_names"],
            )

        # only print these phases; skip FINISH here (printed in print_result)
        if phase in ("PLAN", "THINK", "ACT", "OBSERVE", "REFLECT"):
            print_phase(phase, content)

    agent._record_step = recording_hook  # type: ignore[method-assign]

    start_time = time.time()
    try:
        result = agent.run(state)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Agent run failed: %s", e)
        print(f"\n{C['red']}Error: {e}{C['reset']}")
        raise

    elapsed = time.time() - start_time
    print_result(result, elapsed)

    if save_log:
        _save_run_log(result, state, log_file, elapsed)
        print(f"{C['dim']}Log saved → {log_file}{C['reset']}\n")

    return result


def _save_run_log(result: AgentRunResult, state, log_file: Path, elapsed: float) -> None:
    summary = {
        "scenario": state.scenario_id,
        "won": result.won,
        "total_steps": result.total_steps,
        "elapsed_seconds": round(elapsed, 1),
        "finish_reason": result.finish_reason,
        "goal_conditions": result.final_verification.summary(),
        "steps": [
            {
                "step": s.step_number,
                "phase": s.phase,
                "content": s.content[:300],
                "tool": s.tool_name,
                "args": s.tool_args,
            }
            for s in result.steps
        ],
    }
    json_log = log_file.with_suffix(".json")
    with open(json_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Agent in a Virtual World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario", "-s",
        help="Scenario id or path to JSON")
    parser.add_argument("--list", "-l", action="store_true",
        help="List available scenarios and exit")
    parser.add_argument("--steps", type=int, default=60,
        help="Maximum agent steps (default: 60)")
    parser.add_argument("--agent-model", default="claude-sonnet-4-5",
        help="Anthropic model for the Agent LLM")
    parser.add_argument("--world-model", default="claude-sonnet-4-5",
        help="Anthropic model for the World LLM / NPC dialogue")
    parser.add_argument("--no-log", action="store_true",
        help="Skip saving JSON log")
    parser.add_argument("--skip-validation", action="store_true",
        help="Skip scenario validation before run")
    parser.add_argument("--test", action="store_true",
        help="Run the test suite and exit")
    return parser


def resolve_scenario_path(scenario_arg: str, scenarios_dir: Path) -> str:
    if os.path.isfile(scenario_arg):
        return scenario_arg
    candidate = scenarios_dir / f"{scenario_arg}.json"
    if candidate.exists():
        return str(candidate)
    for p in scenarios_dir.glob("*.json"):
        if scenario_arg.lower() in p.stem.lower():
            return str(p)
    print(f"Scenario '{scenario_arg}' not found.")
    sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scenarios_dir = Path(__file__).parent / "scenarios"
    engine = WorldEngine()

    if args.list:
        scenarios = engine.get_scenario_list(scenarios_dir)
        print("\nAvailable scenarios:\n")
        for s in scenarios:
            print(f"  {s['id']:25s}  {s['name']}")
            print(f"  {' ' * 25}  {C['dim']}{s['description'][:80]}…{C['reset']}\n")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    if args.test:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover(str(Path(__file__).parent / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.scenario:
        scenario_path = resolve_scenario_path(args.scenario, scenarios_dir)
    else:
        scenario_path = select_scenario_interactive(engine, scenarios_dir)

    if not args.skip_validation:
        report = validate_file(scenario_path)
        if report.errors:
            print(f"\n{C['red']}Scenario validation failed:{C['reset']}")
            for e in report.errors:
                print(f"  {C['red']}✗{C['reset']} {e}")
            sys.exit(1)
        if report.warnings:
            print(f"{C['yellow']}Scenario warnings:{C['reset']}")
            for w in report.warnings:
                print(f"  {C['yellow']}⚠{C['reset']} {w}")

    run_session(
        scenario_path=scenario_path,
        agent_model=args.agent_model,
        world_model=args.world_model,
        max_steps=args.steps,
        save_log=not args.no_log,
    )


if __name__ == "__main__":
    main()
