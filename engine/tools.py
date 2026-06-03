"""
Tool execution layer. Each tool function receives the current WorldState and
returns a ToolResult(observation, success, state_changed).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.world_engine import WorldState


@dataclass
class ToolResult:
    observation: str
    success: bool
    state_changed: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------

def tool_move(state: "WorldState", direction: str) -> ToolResult:
    """Move the agent in a compass direction or up/down/in/out."""
    direction = direction.lower().strip()
    current_room = state.rooms[state.agent_location]
    exits = current_room.get("exits", {})

    if direction not in exits:
        available = ", ".join(exits.keys()) or "none"
        return ToolResult(
            observation=f"You cannot go {direction} from here. Available exits: {available}.",
            success=False,
        )

    exit_data = exits[direction]

    if exit_data.get("hidden", False) and not state.revealed_exits.get(
        f"{state.agent_location}:{direction}", False
    ):
        return ToolResult(
            observation=f"There is no obvious exit to the {direction}.",
            success=False,
        )

    if exit_data.get("locked", False):
        key_item = exit_data.get("key_item")
        if key_item and key_item in state.inventory:
            # auto-use key to unlock
            exit_data["locked"] = False
            observation = (
                f"You use the {state.items[key_item]['name']} to unlock the passage, "
                f"then move {direction}. "
            )
        else:
            lock_hint = (
                f" It requires: {state.items[key_item]['name']}."
                if key_item and key_item in state.items
                else ""
            )
            return ToolResult(
                observation=f"The way {direction} is locked.{lock_hint}",
                success=False,
            )
    else:
        observation = f"You move {direction}. "

    destination_id = exit_data["to"]
    state.agent_location = destination_id
    dest_room = state.rooms[destination_id]
    observation += f"\n\n**{dest_room['name']}**\n{dest_room['description']}"

    # list visible items and NPCs
    room_items = [
        state.items[i]["name"]
        for i in dest_room.get("items", [])
        if i in state.items and i in state.room_items.get(destination_id, [])
    ]
    room_npcs = [
        state.npcs[n]["name"]
        for n in dest_room.get("npcs", [])
        if n in state.npcs
    ]
    if room_items:
        observation += f"\n\nItems visible: {', '.join(room_items)}."
    if room_npcs:
        observation += f"\nPresent here: {', '.join(room_npcs)}."

    # fire room_entered events
    state.fire_event_trigger("room_entered", room_id=destination_id)

    return ToolResult(observation=observation, success=True, state_changed=True)


# ---------------------------------------------------------------------------
# examine
# ---------------------------------------------------------------------------

def tool_examine(state: "WorldState", target: str) -> ToolResult:
    """Examine an item, NPC, room feature, or the current room."""
    target_lower = target.lower().strip()

    # examine current room
    if target_lower in ("room", "here", "surroundings", "around"):
        room = state.rooms[state.agent_location]
        obs = f"**{room['name']}**\n{room['description']}"
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
        exits_here = []
        for direction, exit_data in room.get("exits", {}).items():
            if not exit_data.get("hidden", False) or state.revealed_exits.get(
                f"{state.agent_location}:{direction}", False
            ):
                locked_str = " (locked)" if exit_data.get("locked") else ""
                exits_here.append(f"{direction}{locked_str}")
        if items_here:
            obs += f"\n\nItems: {', '.join(items_here)}."
        if npcs_here:
            obs += f"\nPresent: {', '.join(npcs_here)}."
        if exits_here:
            obs += f"\nExits: {', '.join(exits_here)}."
        return ToolResult(observation=obs, success=True)

    # check inventory first, then current room items, then NPCs
    item_id = _find_item_id(state, target_lower)
    if item_id:
        item = state.items[item_id]
        return ToolResult(
            observation=f"**{item['name']}**: {item['description']}",
            success=True,
        )

    npc_id = _find_npc_id(state, target_lower)
    if npc_id:
        npc = state.npcs[npc_id]
        return ToolResult(
            observation=f"**{npc['name']}**: {npc['description']}",
            success=True,
        )

    return ToolResult(
        observation=f"You don't see anything called '{target}' here.",
        success=False,
    )


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------

def tool_take(state: "WorldState", item_name: str) -> ToolResult:
    """Pick up an item from the current room."""
    item_id = _find_item_id_in_room(state, item_name.lower().strip())
    if not item_id:
        candidates = _suggest_items(state.items, state.room_items.get(state.agent_location, []))
        return ToolResult(
            observation=f"There is no '{item_name}' here to take. Items here: {candidates}.",
            success=False,
        )

    item = state.items[item_id]
    if not item.get("takeable", False):
        return ToolResult(
            observation=f"The {item['name']} cannot be taken.",
            success=False,
        )

    state.room_items[state.agent_location].remove(item_id)
    state.inventory.add(item_id)

    return ToolResult(
        observation=f"You take the {item['name']}.",
        success=True,
        state_changed=True,
    )


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------

def tool_use(state: "WorldState", item_name: str, target_name: str) -> ToolResult:
    """Use an inventory item on a target (item, NPC, fixture, or locked exit direction)."""
    item_id = _find_item_id_in_inventory(state, item_name.lower().strip())
    if not item_id:
        inv_candidates = _suggest_items(state.items, list(state.inventory))
        return ToolResult(
            observation=f"You don't have '{item_name}' in your inventory. Inventory: {inv_candidates}.",
            success=False,
        )

    item = state.items[item_id]

    # --- exit-unlock shortcut ---
    # Allow using a key item directly on a direction / door description.
    # Matches if this item is the key_item for any locked exit in the current room.
    exit_unlock = _try_unlock_exit(state, item_id, target_name.lower().strip())
    if exit_unlock is not None:
        return exit_unlock

    if not item.get("usable", False):
        return ToolResult(
            observation=f"The {item['name']} doesn't seem usable that way.",
            success=False,
        )

    # resolve target
    target_lower = target_name.lower().strip()
    valid_targets = item.get("use_targets", [])

    # find target id by name or id match
    target_id = _resolve_use_target(state, target_lower, valid_targets)

    if target_id not in valid_targets:
        if valid_targets:
            return ToolResult(
                observation=(
                    f"Using the {item['name']} on '{target_name}' has no effect. "
                    f"It might work on something else."
                ),
                success=False,
            )
        return ToolResult(
            observation=f"The {item['name']} can't be used on '{target_name}'.",
            success=False,
        )

    # execute use
    effect_text = item.get("use_effect", f"You use the {item['name']}.")
    state_changed = False

    # special compound-item creation: blank_pardon + seal → forged_pardon
    _apply_use_side_effects(state, item_id, target_id)

    # fire item_used events
    state.fire_event_trigger("item_used", item_id=item_id)

    # record this use for goal verification
    state.used_items.add((item_id, target_id))

    return ToolResult(
        observation=effect_text,
        success=True,
        state_changed=True,
        metadata={"item_id": item_id, "target_id": target_id},
    )


def _apply_use_side_effects(state: "WorldState", item_id: str, target_id: str) -> None:
    """Handle composite item creation and state mutations triggered by use."""
    # forging the pardon: seal used on blank_pardon OR blank_pardon used on seal
    if item_id == "chancellors_seal" and target_id == "blank_pardon":
        _create_forged_pardon(state)
    elif item_id == "blank_pardon" and target_id in ("chancellors_seal", "ink_and_quill"):
        # only fully forge when both ink and seal have been applied
        if _pardon_ready(state, item_id):
            _create_forged_pardon(state)
    # circuit board assembly
    elif item_id in ("circuit_board", "emergency_toolkit") and target_id in (
        "emergency_toolkit",
        "circuit_board",
    ):
        state.item_flags["circuit_board_assembled"] = True
    # cryo case opened with pry bar → reveal purge code item
    elif item_id == "pry_bar" and target_id == "cryo_specimen_case":
        if "sable_purge_code" not in state.inventory and "sable_purge_code" not in state.room_items.get(
            state.agent_location, []
        ):
            state.room_items.setdefault(state.agent_location, []).append("sable_purge_code")


def _pardon_ready(state: "WorldState", blank_id: str) -> bool:
    has_ink = "ink_and_quill" in state.inventory or state.item_flags.get("pardon_inked")
    has_seal = "chancellors_seal" in state.inventory or state.item_flags.get("pardon_sealed")
    return has_ink and has_seal


def _create_forged_pardon(state: "WorldState") -> None:
    if "forged_pardon" not in state.inventory:
        if "blank_pardon" in state.inventory:
            state.inventory.discard("blank_pardon")
        state.inventory.add("forged_pardon")


# ---------------------------------------------------------------------------
# talk
# ---------------------------------------------------------------------------

def tool_talk(state: "WorldState", npc_name: str, message: str) -> ToolResult:
    """Initiate or continue dialogue with an NPC in the current room."""
    npc_id = _find_npc_id(state, npc_name.lower().strip())
    if not npc_id:
        return ToolResult(
            observation=f"There is no one called '{npc_name}' here.",
            success=False,
        )

    npc = state.npcs[npc_id]
    room = state.rooms[state.agent_location]
    if npc_id not in room.get("npcs", []):
        return ToolResult(
            observation=f"{npc['name']} is not in this room.",
            success=False,
        )

    # Return structured data for WorldEngine to generate LLM dialogue
    return ToolResult(
        observation=f"[TALK:{npc_id}] {message}",
        success=True,
        state_changed=False,
        metadata={"npc_id": npc_id, "message": message, "requires_llm": True},
    )


# ---------------------------------------------------------------------------
# think
# ---------------------------------------------------------------------------

def tool_think(state: "WorldState", thought: str = "") -> ToolResult:
    """Agent internal monologue — no world effect, aids memory and planning."""
    return ToolResult(
        observation=f"[Internal] {thought}",
        success=True,
        state_changed=False,
        metadata={"internal": True},
    )


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------

def tool_finish(state: "WorldState", reason: str) -> ToolResult:
    """Agent declares it has completed the goal or cannot continue."""
    return ToolResult(
        observation=f"[FINISH] {reason}",
        success=True,
        state_changed=False,
        metadata={"finish": True, "reason": reason},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "move": tool_move,
    "examine": tool_examine,
    "take": tool_take,
    "use": tool_use,
    "talk": tool_talk,
    "think": tool_think,
    "finish": tool_finish,
}


def execute_tool(state: "WorldState", tool_name: str, **kwargs) -> ToolResult:
    """Dispatch a tool call by name."""
    tool_name = tool_name.lower().strip()
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(
            observation=f"Unknown tool '{tool_name}'. Available: {', '.join(TOOL_REGISTRY)}.",
            success=False,
        )
    return TOOL_REGISTRY[tool_name](state, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_item_id(state: "WorldState", query: str) -> str | None:
    """Find item by name or id in inventory or current room."""
    candidates = list(state.inventory) + state.room_items.get(state.agent_location, [])
    return _match_item(state.items, candidates, query)


def _find_item_id_in_room(state: "WorldState", query: str) -> str | None:
    candidates = state.room_items.get(state.agent_location, [])
    return _match_item(state.items, candidates, query)


def _find_item_id_in_inventory(state: "WorldState", query: str) -> str | None:
    return _match_item(state.items, list(state.inventory), query)


def _match_item(items_dict: dict, candidates: list, query: str) -> str | None:
    query = query.lower()
    for item_id in candidates:
        if item_id not in items_dict:
            continue
        name_lower = items_dict[item_id]["name"].lower()
        if query == item_id or query in name_lower or name_lower in query:
            return item_id
    # partial token match
    query_tokens = set(query.split())
    for item_id in candidates:
        if item_id not in items_dict:
            continue
        name_tokens = set(items_dict[item_id]["name"].lower().split())
        if query_tokens & name_tokens:
            return item_id
    return None


def _suggest_items(items_dict: dict, candidates: list) -> str:
    """Return a comma-separated list of item names for failure messages."""
    names = [items_dict[i]["name"] for i in candidates if i in items_dict]
    return ", ".join(names) if names else "none"


def _find_npc_id(state: "WorldState", query: str) -> str | None:
    query = query.lower()
    room = state.rooms[state.agent_location]
    for npc_id in room.get("npcs", []):
        if npc_id not in state.npcs:
            continue
        npc = state.npcs[npc_id]
        name_lower = npc["name"].lower()
        if query == npc_id or query in name_lower or name_lower in query:
            return npc_id
    return None


def _try_unlock_exit(
    state: "WorldState", item_id: str, target_query: str
) -> "ToolResult | None":
    """
    If item_id is the key_item for a locked exit in the current room, and
    target_query loosely refers to that exit (direction name, 'door', 'gate',
    'lock', exit description keywords), unlock it and return a ToolResult.
    Returns None if this item isn't a key for any exit here.
    """
    current_room = state.rooms[state.agent_location]
    for direction, exit_data in current_room.get("exits", {}).items():
        if exit_data.get("key_item") != item_id:
            continue
        # this item IS the key for this exit — check if target loosely matches
        desc_words = exit_data.get("description", "").lower().split()
        aliases = {direction, "door", "gate", "lock", "passage", "exit",
                   "north", "south", "east", "west", "up", "down"} | set(desc_words)
        if target_query in aliases or any(target_query in w for w in aliases):
            if exit_data.get("locked", False):
                exit_data["locked"] = False
                item_name = state.items[item_id]["name"]
                use_effect = state.items[item_id].get("use_effect", "")
                obs = use_effect if use_effect else f"You use the {item_name} to unlock the {direction} exit."
                return ToolResult(observation=obs, success=True, state_changed=True)
            else:
                return ToolResult(
                    observation=f"The {direction} exit is already unlocked.",
                    success=True,
                )
    return None


def _resolve_use_target(
    state: "WorldState", target_lower: str, valid_targets: list
) -> str:
    """Resolve a fuzzy target name to a canonical target id."""
    # direct id match
    if target_lower in valid_targets:
        return target_lower
    # name match against items
    for t_id in valid_targets:
        if t_id in state.items and target_lower in state.items[t_id]["name"].lower():
            return t_id
        if t_id in state.npcs and target_lower in state.npcs[t_id]["name"].lower():
            return t_id
    # partial match
    for t_id in valid_targets:
        if target_lower in t_id:
            return t_id
    return target_lower
