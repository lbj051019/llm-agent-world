# Scenario JSON Schema

All scenario files must conform to this specification.

---

## Top-Level Structure

```json
{
  "id":          "string   — unique scenario identifier, snake_case",
  "name":        "string   — display name",
  "description": "string   — one-paragraph flavor text shown to player at start",
  "start_room":  "string   — must match a key in rooms{}",
  "goal":        "Goal     — see Goal Object",
  "rooms":       "{ [room_id: string]: Room }",
  "items":       "{ [item_id: string]: Item }",
  "npcs":        "{ [npc_id: string]: NPC   }",
  "events":      "{ [event_id: string]: Event } — optional"
}
```

---

## Goal Object

```json
{
  "description": "string          — natural-language goal shown to agent",
  "conditions":  "Condition[]     — ALL must be satisfied to win"
}
```

### Condition Types

| type | required fields | description |
|------|----------------|-------------|
| `agent_at_location` | `room_id` | Agent must be in the specified room |
| `item_in_inventory` | `item_id` | Agent must carry the item |
| `item_used_on_target` | `item_id`, `target_id` | Item must have been used on target (item or NPC) |
| `npc_state` | `npc_id`, `state` | NPC's `current_state` must equal `state` |
| `event_triggered` | `event_id` | Named event must have fired at least once |

**Condition Object**:
```json
{
  "type":      "string  — one of the five types above",
  "room_id":   "string  — (agent_at_location)",
  "item_id":   "string  — (item_in_inventory | item_used_on_target)",
  "target_id": "string  — item_id or npc_id (item_used_on_target)",
  "npc_id":    "string  — (npc_state)",
  "state":     "string  — (npc_state)",
  "event_id":  "string  — (event_triggered)"
}
```

---

## Room Object

```json
{
  "id":          "string          — matches its key in rooms{}",
  "name":        "string          — short display name",
  "description": "string          — sensory description given on entry or examine",
  "exits":       "{ [direction: string]: Exit }",
  "items":       "string[]        — item_ids present in this room at start",
  "npcs":        "string[]        — npc_ids present in this room at start",
  "properties":  "{ [key]: any }  — optional freeform flags (dark, locked, etc.)"
}
```

### Exit Object

```json
{
  "to":          "string   — destination room_id",
  "locked":      "boolean  — default false; agent must unlock before using",
  "key_item":    "string   — item_id required to unlock (omit if not locked)",
  "hidden":      "boolean  — default false; only revealed after examine/event",
  "description": "string   — optional one-liner describing the passage"
}
```

**Valid directions**: `north`, `south`, `east`, `west`, `up`, `down`, `in`, `out`

---

## Item Object

```json
{
  "id":          "string          — matches its key in items{}",
  "name":        "string          — display name",
  "description": "string          — text returned by examine",
  "takeable":    "boolean         — can agent add to inventory",
  "usable":      "boolean         — can agent invoke use tool on it",
  "use_targets": "string[]        — valid target ids for use action (items or npcs)",
  "use_effect":  "string          — narrative result when used correctly",
  "properties":  "{ [key]: any }  — optional flags (readable, fragile, lit, etc.)"
}
```

---

## NPC Object

```json
{
  "id":            "string    — matches its key in npcs{}",
  "name":          "string    — display name",
  "description":   "string    — text returned by examine",
  "persona":       "string    — personality/background prompt injected into World LLM",
  "initial_state": "string    — starting value of current_state",
  "current_state": "string    — mutable; updated by events or item_used_on_target",
  "dialogue": {
    "greeting":    "string    — first-contact line",
    "default":     "string    — fallback when World LLM has no specific response",
    "states":      "{ [state: string]: string } — state-specific response hints"
  },
  "gives_item":    "string    — item_id NPC hands over when current_state == trigger_state (optional)",
  "trigger_state": "string    — the state that causes gives_item (optional)"
}
```

---

## Event Object

```json
{
  "id":          "string        — matches its key in events{}",
  "description": "string        — what happens narratively",
  "trigger":     "EventTrigger  — what causes the event",
  "effects":     "EventEffect[] — changes applied when fired"
}
```

### EventTrigger Object

```json
{
  "type":    "string  — item_used | room_entered | npc_state_reached",
  "item_id": "string  — (item_used)",
  "room_id": "string  — (room_entered)",
  "npc_id":  "string  — (npc_state_reached)",
  "state":   "string  — (npc_state_reached)"
}
```

### EventEffect Object

```json
{
  "type":       "string  — unlock_exit | reveal_item | change_npc_state | add_item_to_room",
  "room_id":    "string",
  "direction":  "string  — (unlock_exit)",
  "item_id":    "string  — (reveal_item | add_item_to_room)",
  "npc_id":     "string  — (change_npc_state)",
  "new_state":  "string  — (change_npc_state)"
}
```

---

## Minimal Example

```json
{
  "id": "demo",
  "name": "The Locked Room",
  "description": "You wake in a dim stone chamber. A door to the north is locked.",
  "start_room": "chamber",
  "goal": {
    "description": "Escape the chamber.",
    "conditions": [
      { "type": "agent_at_location", "room_id": "corridor" }
    ]
  },
  "rooms": {
    "chamber": {
      "id": "chamber",
      "name": "Stone Chamber",
      "description": "Damp walls. A rusty key glints on the floor.",
      "exits": {
        "north": { "to": "corridor", "locked": true, "key_item": "rusty_key" }
      },
      "items": ["rusty_key"],
      "npcs": [],
      "properties": {}
    },
    "corridor": {
      "id": "corridor",
      "name": "Dark Corridor",
      "description": "Freedom. The corridor stretches into darkness.",
      "exits": {
        "south": { "to": "chamber", "locked": false }
      },
      "items": [],
      "npcs": [],
      "properties": {}
    }
  },
  "items": {
    "rusty_key": {
      "id": "rusty_key",
      "name": "Rusty Key",
      "description": "An old iron key, spotted with rust.",
      "takeable": true,
      "usable": true,
      "use_targets": ["north_door"],
      "use_effect": "The lock turns with a grinding screech. The door swings open.",
      "properties": {}
    }
  },
  "npcs": {},
  "events": {}
}
```
