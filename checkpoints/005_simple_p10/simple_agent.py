"""Standalone probe-then-p10 experimental bot.

During the first PROBE_HANDS hands it raises pot whenever legal. Showdown
opponent holdings from those hands train one frozen integer p10 rank cutoff
from uniform random five-card boards. Later streets compare an opponent's
p10 possible final rank with that same cutoff. This intentionally simple model
does not learn action frequencies or predict future raises.
"""
from __future__ import annotations

import hashlib
import importlib.util
import atexit
import json
import time
import os
from pathlib import Path
import sys

import numpy as np

from action_values import state_from
from batch_eval import draw, score
from evaluator import CATEGORY_NAMES, category_of
from game import Action, AgentConfig, HandResult, Observation, Street

PROBE_HANDS = max(0, int(os.environ.get("PROBE_HANDS", "100")))
TRAIN_BOARDS_PER_SHOWDOWN = max(1, int(os.environ.get("TRAIN_BOARDS_PER_SHOWDOWN", "256")))
FEATURE_RUNOUTS = max(1, int(os.environ.get("FEATURE_RUNOUTS", "16")))
PAYOFF_SAMPLES = max(1, int(os.environ.get("PAYOFF_SAMPLES", "32")))
RANGE_CANDIDATES = max(1, int(os.environ.get("RANGE_CANDIDATES", "32")))
REFRESH_ATTEMPTS = max(1, int(os.environ.get("REFRESH_ATTEMPTS", "2")))


def _seed(tag: str, *parts: object) -> int:
    data = (tag + ":" + ":".join(map(str, parts))).encode("ascii")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _visible_prefix(board, street):
    n = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[Street(street)]
    return tuple(board[:n])


def _rank_priority(board, runouts):
    """Fixed random card order for a public prefix, independent of candidates."""
    key = tuple(sorted(map(int, board)))
    seed = _seed("potential-worlds-v1", *key)
    return np.random.default_rng(seed).random((max(1, int(runouts)), 36))


def empirical_p10(values, axis=-1):
    """Select observed nearest-rank p10; packed integer scores are not interpolated."""
    values = np.asarray(values)
    n = values.shape[axis]
    if n < 1:
        raise ValueError("p10 needs at least one observed value")
    index = max(0, int(np.ceil(.10 * n)) - 1)
    return np.partition(values, index, axis=axis).take(index, axis=axis)


def potential_p10(hands, board, runouts=FEATURE_RUNOUTS, cache=None):
    """Empirical p10 possible final Omaha score per holding, using public cards only.

    At river the feature is the exact score. Else each holding gets the same
    deterministic random card-priority worlds; only its own cards and public
    prefix are blocked. Candidate order and hero cards cannot affect results.
    """
    hands = np.asarray(hands, dtype=np.int16).reshape(-1, 5)
    board = tuple(map(int, board))
    value_key = (tuple(sorted(board)), int(runouts))
    value_cache = (cache.setdefault("_values", {}) if cache is not None else None)
    hand_keys = [tuple(sorted(map(int, hand))) for hand in hands]
    if value_cache is not None:
        known = value_cache.setdefault(value_key, {})
        if all(key in known for key in hand_keys):
            return np.asarray([known[key] for key in hand_keys], dtype=np.int32)
        missing_keys = list(dict.fromkeys(key for key in hand_keys if key not in known))
        missing_hands = np.asarray(missing_keys, dtype=np.int16).reshape(-1, 5)
        calculated = potential_p10(missing_hands, board, runouts, None)
        known.update(zip(missing_keys, map(int, calculated)))
        return np.asarray([known[key] for key in hand_keys], dtype=np.int32)
    if len(board) == 5:
        return np.asarray(score(hands, np.broadcast_to(board, (len(hands), 5))), dtype=np.int32)
    if len(board) not in (0, 3, 4):
        raise ValueError("board prefix must contain 0, 3, 4, or 5 cards")
    if not len(hands):
        return np.empty(0, dtype=np.int32)
    key = tuple(sorted(board))
    priorities = (cache or {}).get((key, int(runouts)))
    if priorities is None:
        priorities = _rank_priority(board, runouts)
        if cache is not None:
            cache[(key, int(runouts))] = priorities
    else:
        priorities = priorities[:max(1, int(runouts))]
    R = len(priorities)
    blocked = np.zeros((len(hands), 36), dtype=bool)
    blocked[np.arange(len(hands))[:, None], hands] = True
    if board:
        blocked[:, np.asarray(board, dtype=np.int16)] = True
    ranks = np.broadcast_to(np.argsort(np.argsort(priorities, axis=1), axis=1),
                           (len(hands), R, 36)).copy()
    ranks[blocked[:, None, :].repeat(R, axis=1)] = 36
    missing = 5 - len(board)
    chosen = np.argpartition(ranks, missing - 1, axis=2)[:, :, :missing]
    runouts_cards = np.take_along_axis(np.broadcast_to(np.arange(36), ranks.shape), chosen, axis=2)
    boards = np.concatenate((np.broadcast_to(np.asarray(board, dtype=np.int16),
                                              (len(hands), R, len(board))), runouts_cards), axis=2)
    repeated_hands = np.broadcast_to(hands[:, None, :], (len(hands), R, 5)).reshape(-1, 5)
    scores = score(repeated_hands, boards.reshape(-1, 5)).reshape(len(hands), R)
    return empirical_p10(scores, axis=1).astype(np.int32)


def _legal_hands(hero, board, count, seed):
    excluded = np.asarray(tuple(hero) + tuple(board), dtype=np.int16)
    rng = np.random.default_rng(seed)
    rows = max(count, 1)
    cards = draw(rng, excluded, 5, rows)
    return cards.astype(np.int16)


def _history_raise_prefixes(obs):
    events = []
    seen_prefixes = set()
    for idx, event in enumerate(obs.action_history):
        if event.is_me or event.action not in (Action.RAISE_HALF, Action.RAISE_POT):
            continue
        prefix = _visible_prefix(obs.board, event.street)
        if prefix not in seen_prefixes:
            events.append((idx, prefix))
            seen_prefixes.add(prefix)
    return events


def _replay_range(obs, memory, cutoff, candidate_count):
    key = (obs.hand_index, tuple(obs.hole_cards))
    state = memory.setdefault("hand_ranges", {}).get(key)
    board = tuple(obs.board)
    if state is None or state["cutoff_version"] != memory["cutoff_version"]:
        state = {"cutoff_version": memory["cutoff_version"], "hands": None,
                 "prefixes": (), "refreshes": 0, "broad_fallback": False,
                 "reported_refreshes": 0, "reported_fallback": False,
                 "candidate_count": candidate_count}
        memory["hand_ranges"][key] = state
    events = tuple(_history_raise_prefixes(obs))
    # Refresh/replay the entire range when the public board or history changes.
    if (state["hands"] is None or state["prefixes"] != (board, events)
            or state["candidate_count"] != candidate_count):
        state["candidate_count"] = candidate_count
        state["broad_fallback"] = False
        state["reported_fallback"] = False
        state["hands"] = _legal_hands(obs.hole_cards, board, candidate_count,
                                      _seed("range", obs.hand_index, *obs.hole_cards, *board,
                                            memory["cutoff_version"]))
        if cutoff is not None:
            for _, prefix in events:
                p10_scores = potential_p10(state["hands"], prefix, FEATURE_RUNOUTS,
                                           memory["feature_cache"])
                state["hands"] = state["hands"][p10_scores >= cutoff]
        state["prefixes"] = (board, events)
        if not len(state["hands"]):
            # Bounded independent proposal refresh, with all recorded evidence replayed.
            for attempt in range(REFRESH_ATTEMPTS):
                state["refreshes"] += 1
                candidates = _legal_hands(obs.hole_cards, board, candidate_count,
                    _seed("refresh", obs.hand_index, *obs.hole_cards, *board,
                          memory["cutoff_version"], attempt))
                kept = candidates
                if cutoff is not None:
                    for _, prefix in events:
                        p10_scores = potential_p10(kept, prefix, FEATURE_RUNOUTS,
                                                   memory["feature_cache"])
                        kept = kept[p10_scores >= cutoff]
                if len(kept):
                    state["hands"] = kept
                    break
            if not len(state["hands"]):
                # Explicit broad prior fallback; absence of sampled hands is not a fold.
                state["broad_fallback"] = True
                state["hands"] = _legal_hands(obs.hole_cards, board, candidate_count,
                    _seed("broad", obs.hand_index, *obs.hole_cards, *board))
    return state


def new_half(config: AgentConfig, memory: dict):
    memory.clear()
    memory.update({"config": config, "seen_results": set(), "rank_pool": [],
                   "cutoff": None, "cutoff_version": 0, "probe_frozen": False,
                   "feature_cache": {}, "hand_ranges": {}, "stats": {
                       "probe_actions": 0, "trained_showdowns": 0, "range_refreshes": 0,
                       "broad_fallbacks": 0, "decisions": 0, "probe_decisions": 0,
                       "postprobe_decisions": 0, "elapsed_seconds": 0.0,
                       "max_decision_seconds": 0.0, "work_candidates": 0}},
                   diagnostics_printed=False)


def on_hand_end(result: HandResult, memory: dict):
    started = time.perf_counter()
    idx = int(result.hand_index)
    if idx in memory["seen_results"]:
        if idx + 1 >= memory["config"].num_hands:
            _print_half_diagnostics(memory)
        return
    memory["seen_results"].add(idx)
    if idx >= PROBE_HANDS or result.opp_hole is None:
        _record_timing(memory, started, "training")
        if idx + 1 >= memory["config"].num_hands:
            _print_half_diagnostics(memory)
        return
    # Actual showdown board is intentionally ignored; only opp_hole is excluded.
    hole = tuple(sorted(map(int, result.opp_hole)))
    rng = np.random.default_rng(_seed("training", idx, *hole))
    boards = draw(rng, np.asarray(hole, dtype=np.int16), 5,
                  TRAIN_BOARDS_PER_SHOWDOWN).astype(np.int16)
    scores = score(np.broadcast_to(np.asarray(hole, dtype=np.int16), boards.shape), boards)
    memory["rank_pool"].extend(map(int, scores))
    pool = np.asarray(memory["rank_pool"], dtype=np.int32)
    memory["cutoff"] = int(empirical_p10(pool))
    memory["cutoff_version"] += 1
    memory["stats"]["trained_showdowns"] += 1
    _record_timing(memory, started, "training")
    if idx + 1 >= memory["config"].num_hands:
        _print_half_diagnostics(memory)


def _record_timing(memory, started, phase):
    elapsed = time.perf_counter() - started
    memory["stats"]["elapsed_seconds"] += elapsed
    memory["stats"]["max_" + phase + "_seconds"] = max(
        memory["stats"].get("max_" + phase + "_seconds", 0.0), elapsed)


def _candidate_budget(memory):
    cfg = memory["config"]
    target = max(.2, float(cfg.time_bank_seconds) * .50)
    hand_index = int(memory.get("budget_hand_index", 0))
    remaining_hands = max(1, cfg.num_hands - hand_index)
    postprobe_hands = max(1, hand_index - PROBE_HANDS + 1)
    observed = memory["stats"]["postprobe_decisions"] / postprobe_hands
    decisions_per_hand = max(2.0, min(5.0, observed if observed else 3.5))
    allowance = max(0., target - memory["stats"]["elapsed_seconds"]) / (remaining_hands * decisions_per_hand)
    if allowance >= .003:
        return RANGE_CANDIDATES
    if allowance >= .0012:
        return max(12, RANGE_CANDIDATES // 2)
    return max(8, RANGE_CANDIDATES // 4)


def _action_ev(root, action, wins, fold_below):
    """Per-particle EV relative to this decision (prior investments are sunk)."""
    wins = np.asarray(wins, dtype=float)
    if action == Action.FOLD:
        return np.zeros(len(wins), dtype=float)
    after_hero, cost = root.apply(0, int(action))
    if action == Action.CHECK_CALL or not after_hero.stacks[1]:
        return wins * after_hero.pot - cost
    called, _opponent_cost = after_hero.apply(1, int(Action.CHECK_CALL))
    call_ev = wins * called.pot - cost
    fold_ev = np.full(len(wins), root.pot, dtype=float)
    return np.where(fold_below, fold_ev, call_ev)


def _payoff(obs, hands, memory):
    hands = np.asarray(hands, dtype=np.int16).reshape(-1, 5)
    board = tuple(obs.board)
    hero = np.asarray(obs.hole_cards, dtype=np.int16)
    missing = 5 - len(board)
    if missing:
        excluded = np.concatenate((np.broadcast_to(hero, hands.shape), hands,
                                   np.broadcast_to(np.asarray(board, dtype=np.int16),
                                                   (len(hands), len(board)))), axis=1)
        runouts = draw(np.random.default_rng(_seed("payoff", obs.hand_index, obs.street,
                                                   *hero, *board)), excluded, missing, len(hands))
        final_boards = np.concatenate((np.broadcast_to(np.asarray(board, dtype=np.int16),
                                                       (len(hands), len(board))), runouts), axis=1)
    else:
        final_boards = np.broadcast_to(np.asarray(board, dtype=np.int16), hands.shape)
    hero_scores = score(np.broadcast_to(hero, hands.shape), final_boards)
    opp_scores = score(hands, final_boards)
    wins = (hero_scores > opp_scores).astype(float) + .5 * (hero_scores == opp_scores)
    return wins


def choose_action(obs: Observation, memory: dict):
    started = time.perf_counter()
    if obs.hand_index < PROBE_HANDS:
        memory["stats"]["probe_actions"] += 1
        memory["stats"]["probe_decisions"] += 1
        action = Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
        _record_timing(memory, started, "decision")
        return action
    memory["probe_frozen"] = True
    memory["stats"]["decisions"] += 1
    memory["stats"]["postprobe_decisions"] += 1
    memory["budget_hand_index"] = obs.hand_index
    if memory.get("active_hand") != obs.hand_index:
        memory["feature_cache"].clear()
        memory["active_hand"] = obs.hand_index
    cutoff = memory["cutoff"]
    candidate_count = _candidate_budget(memory)
    memory["active_candidate_count"] = candidate_count
    memory["stats"]["work_candidates"] += candidate_count
    range_state = _replay_range(obs, memory, cutoff, candidate_count)
    memory["stats"]["range_refreshes"] += range_state["refreshes"] - range_state["reported_refreshes"]
    range_state["reported_refreshes"] = range_state["refreshes"]
    if range_state["broad_fallback"] and not range_state["reported_fallback"]:
        memory["stats"]["broad_fallbacks"] += 1
        range_state["reported_fallback"] = True
    hands = range_state["hands"]
    if len(hands) > PAYOFF_SAMPLES:
        hands = hands[:PAYOFF_SAMPLES]
    wins = _payoff(obs, hands, memory)
    root = state_from(obs, memory["config"])
    if cutoff is None or range_state["broad_fallback"]:
        fold_below = np.zeros(len(hands), dtype=bool)
        p10_scores = None
    else:
        p10_scores = potential_p10(hands, obs.board, FEATURE_RUNOUTS, memory["feature_cache"])
        fold_below = p10_scores < cutoff
    values = {}
    for action in obs.legal_actions:
        values[action] = float(np.mean(_action_ev(root, action, wins, fold_below)))
    action = max(obs.legal_actions, key=lambda a: values.get(a, -np.inf))
    _record_timing(memory, started, "decision")
    return action


def diagnostics(memory):
    cutoff = memory.get("cutoff")
    category = None if cutoff is None else CATEGORY_NAMES[category_of(cutoff)]
    return {"percentile": 0.10, "percentile_method": "empirical_nearest_rank",
            "probe_hands": PROBE_HANDS, "train_boards_per_showdown": TRAIN_BOARDS_PER_SHOWDOWN,
            "feature_runouts": FEATURE_RUNOUTS, "payoff_samples": PAYOFF_SAMPLES,
            "range_candidates": RANGE_CANDIDATES, "cutoff_score": cutoff,
            "cutoff_category": category, "pooled_rank_samples": len(memory.get("rank_pool", ())),
            **memory.get("stats", {})}


def _print_half_diagnostics(memory):
    if os.environ.get("BOT_DIAGNOSTICS") and not memory.get("diagnostics_printed"):
        print("BOT_DIAGNOSTICS " + json.dumps(diagnostics(memory), sort_keys=True),
              file=sys.stderr, flush=True)
        memory["diagnostics_printed"] = True


def _load_http_shell():
    """Load an isolated copy of agent.py's HTTP routes and replace callbacks."""
    path = Path(__file__).resolve().with_name("agent.py")
    name = f"_simple_probe_http_{id(sys.modules[__name__])}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.new_half = new_half
    module.choose_action = choose_action
    module.on_hand_end = on_hand_end
    module.PARTICIPANT = "ExperimentalSimpleProbe"
    module.AGENT_NAME = "ProbeP10Cutoff"
    return module


_http = _load_http_shell()
app = _http.app
MEMORY = _http.MEMORY
TOKEN = _http.TOKEN


def _print_diagnostics():
    if MEMORY and os.environ.get("BOT_DIAGNOSTICS"):
        _print_half_diagnostics(MEMORY)


atexit.register(_print_diagnostics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")),
                log_level="info", access_log=False)
