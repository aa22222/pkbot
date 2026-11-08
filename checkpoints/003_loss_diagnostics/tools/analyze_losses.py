"""Replay duplicate-match logs and summarize paired hand/scenario outcomes.

Usage:
  python tools/analyze_losses.py match107.json match109.json \
      --output loss-analysis.json --top 12
  python tools/analyze_losses.py match107.json --true-equity

The first-divergence result is an association with the realized paired delta,
not a causal estimate of the value of that first action. Any true-hand equity
is offline hindsight and was unavailable to the bot at the decision.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batch_eval import draw, hand_index, score
from evaluator import CATEGORY_NAMES, card_to_str, category_of, str_to_card
from game import Action, BOARD_CARDS_VISIBLE, Game, Street, STARTING_STACK, make_deal

PRE = np.load(ROOT / "data" / "preflop.npy", allow_pickle=False)
PRE_SORTED = np.sort(PRE)
BB = 2
LARGE_COMMIT_CHIPS = STARTING_STACK // 2
STREET_ORDER = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}


def cards(values):
    return tuple(str_to_card(value) for value in values)


def street_name(street):
    return Street(int(street)).name


def hand_category(hole, board):
    if len(board) < 3:
        return None
    value = int(score(np.asarray(hole, dtype=int), np.asarray(board, dtype=int))[0])
    category = int(category_of(value))
    return {"name": CATEGORY_NAMES[category], "index": category, "score": value,
            "pair_context": pair_context(hole, board)}


def pair_context(hole, board):
    hole_ranks = [card // 4 for card in hole]
    board_ranks = [card // 4 for card in board]
    hole_pairs = sum(hole_ranks.count(rank) >= 2 for rank in set(hole_ranks))
    board_pairs = sum(board_ranks.count(rank) >= 2 for rank in set(board_ranks))
    source = ("hole_and_board" if hole_pairs and board_pairs else "hole_pair" if hole_pairs
              else "board_pair" if board_pairs else "no_pair_source")
    return {"hole_pair_count": int(hole_pairs), "board_pair_count": int(board_pairs),
            "pair_presence_context": source,
            "description": "rank-pair presence only; does not claim those cards form the best Omaha hand"}


def preflop_strength(hole):
    value = float(PRE[hand_index(np.asarray(hole, dtype=int))])
    quantile = float(np.searchsorted(PRE_SORTED, value, side="right") / len(PRE_SORTED))
    bucket = min(9, int(quantile * 10))
    return {"equity": value, "quantile": quantile,
            "decile": f"Q{bucket * 10:02d}-{(bucket + 1) * 10:02d}", "decile_index": bucket}


def action_role(action, owed):
    if action in (Action.RAISE_HALF, Action.RAISE_POT):
        return "raiser"
    if action == Action.FOLD:
        return "folder"
    return "caller" if owed > 0 else "checker"


def _agent_for_seat(half, seat, names):
    agent1_seat = 0 if half == 1 else 1
    return "agent1" if seat == agent1_seat else "agent2"


def replay_hand(match_seed, half, log, names, true_equity=False, equity_cache=None):
    deal_index = int(log["deal_index"])
    button = int(log["button_seat"])
    deal = make_deal(str(match_seed), deal_index)
    logged_holes = tuple(cards(row) for row in log["hole_cards"])
    logged_board = cards(log["board"])
    if logged_holes != deal.hole_cards or logged_board != deal.board:
        raise ValueError(f"seed/deal replay mismatch at half {half}, deal {deal_index}")
    game = Game(deal, button, hand_index=deal_index)
    initial_blinds = tuple(game.contributed)
    investments = [defaultdict(int), defaultdict(int)]
    for seat in (0, 1):
        investments[seat]["PREFLOP"] = initial_blinds[seat]
    per_seat_events = [[], []]
    all_events = []
    commitment_events = [[], []]

    for action_index, record in enumerate(log["actions"]):
        seat = int(record["seat"])
        if game.is_terminal or game.to_act != seat:
            raise ValueError(f"replay turn mismatch at half {half}, deal {deal_index}, action {action_index}")
        action = Action[record["action"]]
        if action not in game.legal_actions():
            raise ValueError(f"illegal logged action at half {half}, deal {deal_index}, action {action_index}")
        if record["street"] != game.street.name:
            raise ValueError(f"replay street mismatch at half {half}, deal {deal_index}, action {action_index}")

        visible = BOARD_CARDS_VISIBLE[game.street]
        public = tuple(game.deal.board[:visible])
        owed = game._to_call(seat)
        pot_before = game.pot
        stack_before = game.stacks[seat]
        invested_before = game.contributed[seat]
        role = action_role(action, owed)
        price = owed / (pot_before + owed) if owed else 0.
        category = hand_category(game.deal.hole_cards[seat], public)
        event = {
            "action_index": action_index,
            "seat": seat,
            "agent": _agent_for_seat(half, seat, names),
            "street": game.street.name,
            "action": action.name,
            "role": role,
            "amount": int(record["amount"]),
            "to_call": int(owed),
            "price": float(price),
            "pot_before": int(pot_before),
            "stack_before": int(stack_before),
            "raises_before": int(game.raises_this_street),
            "board_visible": list(map(card_to_str, public)),
            "category_at_action": category,
        }
        game.apply(action)
        actual_amount = game.contributed[seat] - invested_before
        if actual_amount != int(record["amount"]):
            raise ValueError(f"replay chip mismatch at half {half}, deal {deal_index}, action {action_index}")
        investments[seat][event["street"]] += actual_amount
        event["amount"] = int(actual_amount)
        event["stack_after"] = int(game.stacks[seat])
        event["all_in"] = game.stacks[seat] == 0
        event["contributed_after"] = int(game.contributed[seat])
        all_events.append(event)
        per_seat_events[seat].append(event)
        crossing_large = invested_before < LARGE_COMMIT_CHIPS <= game.contributed[seat]
        if crossing_large and not any(item["kind"] == "large_commitment" for item in commitment_events[seat]):
            commitment_events[seat].append({"kind": "large_commitment", **event})
        if event["all_in"] and not any(item["kind"] == "all_in" for item in commitment_events[seat]):
            commitment_events[seat].append({"kind": "all_in", **event})

    if not game.is_terminal:
        raise ValueError(f"logged hand did not terminate at half {half}, deal {deal_index}")
    result = game.result()
    if tuple(map(int, log["nets"])) != tuple(result.nets):
        raise ValueError(f"replayed net mismatch at half {half}, deal {deal_index}")
    revealed = cards(log["board_revealed"])
    if tuple(result.board_revealed) != revealed:
        raise ValueError(f"replayed visible-board mismatch at half {half}, deal {deal_index}")
    terminal_street = all_events[-1]["street"] if all_events else "PREFLOP"

    seats = []
    for seat in (0, 1):
        hole = deal.hole_cards[seat]
        start_strength = preflop_strength(hole)
        net = int(result.nets[seat])
        street_name_max = max(investments[seat], key=lambda name: investments[seat][name])
        street_events = [event for event in per_seat_events[seat]
                         if event["street"] == street_name_max]
        largest_event = street_events[-1] if street_events else None
        largest_category = largest_event["category_at_action"] if largest_event else None
        final_category = hand_category(hole, revealed)
        commits = commitment_events[seat]
        for commit in commits:
            commit["hand_net_bb"] = net / BB
        if true_equity:
            for commit in commits:
                commit["true_hand_equity_hindsight"] = true_hand_equity(
                    hole, deal.hole_cards[1 - seat], cards(commit["board_visible"]),
                    (deal_index * 2 + seat), equity_cache)
        seats.append({
            "seat": seat,
            "agent": _agent_for_seat(half, seat, names),
            "position": "button" if seat == button else "big_blind",
            "hole_cards": list(map(card_to_str, hole)),
            "net_chips": net,
            "net_bb": net / BB,
            "terminal_street": terminal_street,
            "fold_vs_showdown": "showdown" if result.went_to_showdown else "fold",
            "initial_preflop": start_strength,
            "final_hand_category_visible_at_end": final_category,
            "largest_investment_by_street_chips": dict(investments[seat]),
            "largest_investment_street": street_name_max,
            "largest_investment_action": largest_event["action"] if largest_event else None,
            "largest_investment_action_role": largest_event["role"] if largest_event else None,
            "largest_investment_category_at_decision": largest_category,
            "commitment_events": commits,
            "actions": per_seat_events[seat],
        })
    return {
        "half": int(half), "deal_index": deal_index, "button_seat": button,
        "hole_cards": [list(map(card_to_str, row)) for row in deal.hole_cards],
        "board_full_hindsight": list(map(card_to_str, deal.board)),
        "board_revealed": list(map(card_to_str, revealed)),
        "terminal_street": terminal_street,
        "fold_vs_showdown": "showdown" if result.went_to_showdown else "fold",
        "pot": int(result.pot), "went_to_showdown": bool(result.went_to_showdown),
        "seats": seats, "actions": all_events,
    }


def true_hand_equity(hero, villain, board, deal_seed, cache=None):
    """Offline hindsight equity at a commitment point; never a policy feature."""
    key = (tuple(sorted(hero)), tuple(sorted(villain)), tuple(sorted(board)))
    if cache is not None and key in cache:
        return cache[key]
    excluded = set(hero) | set(villain) | set(board)
    deck = [card for card in range(36) if card not in excluded]
    if len(board) == 5:
        runouts = np.zeros((1, 0), dtype=int)
        method = "exact_river"
    elif len(board) == 4:
        runouts = np.asarray(list(combinations(deck, 1)), dtype=int)
        method = "exact_turn_river"
    elif len(board) == 3:
        runouts = np.asarray(list(combinations(deck, 2)), dtype=int)
        method = "exact_flop_runout"
    elif len(board) == 0:
        rng = np.random.default_rng(0xB17D + int(deal_seed))
        runouts = draw(rng, np.asarray(tuple(hero) + tuple(villain)), 5, 4096)
        method = "4096_preflop_runouts"
    else:
        raise ValueError(f"unsupported commitment board length {len(board)}")
    n = len(runouts)
    complete = np.concatenate((np.broadcast_to(np.asarray(board, dtype=int), (n, len(board))),
                               runouts), axis=1)
    hero_score = score(np.asarray(hero, dtype=int), complete)
    villain_score = score(np.asarray(villain, dtype=int), complete)
    equity = float(np.mean((hero_score > villain_score) + .5 * (hero_score == villain_score)))
    value = {"equity": equity, "method": method, "runouts": n,
             "hindsight_only": True}
    if cache is not None:
        cache[key] = value
    return value


def aggregate(rows, value_key="net_bb", weight_key=None):
    if not rows:
        return {"count": 0, "net_bb": 0., "mean_bb": None, "loss_rate": None}
    weights = np.asarray([r.get(weight_key, 1.) if weight_key else 1. for r in rows], dtype=float)
    values = np.asarray([r[value_key] for r in rows], dtype=float)
    total_weight = float(weights.sum())
    return {
        "count": len(rows), "effective_count": total_weight,
        "net_bb": float(np.dot(weights, values)),
        "gross_won_bb": float(np.dot(weights, np.maximum(values, 0))),
        "gross_lost_bb": float(np.dot(weights, np.minimum(values, 0))),
        "mean_bb": float(np.dot(weights, values) / total_weight) if total_weight else None,
        "loss_rate": float(np.dot(weights, values < 0) / total_weight) if total_weight else None,
        "win_rate": float(np.dot(weights, values > 0) / total_weight) if total_weight else None,
        "draw_rate": float(np.dot(weights, values == 0) / total_weight) if total_weight else None,
    }


def group_stats(rows, key_fn, weight_key=None, value_key="net_bb"):
    groups = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is None:
            key = "unavailable"
        groups[str(key)].append(row)
    return {key: aggregate(value, value_key=value_key, weight_key=weight_key)
            for key, value in sorted(groups.items())}


def _scenario(row):
    return "|".join((row["initial_preflop"]["decile"], row["terminal_street"],
                     row["fold_vs_showdown"], row["largest_investment_street"],
                     row["largest_investment_category_at_decision"]["name"]
                     if row["largest_investment_category_at_decision"] else "no_made_category",
                     ",".join(f"{c['kind']}:{c['role']}" for c in row["commitment_events"])
                     or "no_large_commit"))


def _first_street_action_key(row, street):
    event = next((item for item in row["actions"] if item["street"] == street), None)
    if event is None:
        return "no_action_on_street"
    category = event["category_at_action"] or {}
    pair_context = category.get("pair_context") or {}
    return "|".join((event["action"], category.get("name", "no_made_category"),
                     pair_context.get("pair_presence_context", "unknown")))


def _first_divergence(trace1, trace2):
    actions1, actions2 = trace1["actions"], trace2["actions"]
    for index, (left, right) in enumerate(zip(actions1, actions2)):
        if left["seat"] != right["seat"]:
            raise ValueError("duplicate histories diverged in actor before action divergence")
        if left["action"] != right["action"] or left["amount"] != right["amount"]:
            for state_key in ("street", "board_visible", "pot_before", "to_call", "raises_before"):
                if left[state_key] != right[state_key]:
                    raise ValueError(f"paired first divergence is not at the same public state ({state_key})")
            left_agent = left["agent"]
            right_agent = right["agent"]
            if left_agent == right_agent:
                raise ValueError("same agent label in paired first-divergence rows")
            first = left if left_agent == "agent1" else right
            second = right if left_agent == "agent1" else left
            return {
                "action_index": index,
                "actor_seat": left["seat"],
                "actor_position": "button" if left["seat"] == trace1["button_seat"] else "big_blind",
                "street": left["street"],
                "public_board": left["board_visible"],
                "hand_category_at_decision": left["category_at_action"],
                "actor_hole_cards": trace1["hole_cards"][left["seat"]],
                "actor_initial_preflop": preflop_strength(
                    cards(trace1["hole_cards"][left["seat"]])),
                "pot_before": left["pot_before"],
                "to_call": left["to_call"],
                "price": left["price"],
                "raises_before": left["raises_before"],
                "agent1_action": first["action"],
                "agent1_role": first["role"],
                "agent1_amount": first["amount"],
                "agent2_action": second["action"],
                "agent2_role": second["role"],
                "agent2_amount": second["amount"],
            }
    if len(actions1) != len(actions2):
        return {"action_index": min(len(actions1), len(actions2)),
                "kind": "one_history_terminated_after_common_prefix"}
    return None


def _divergence_key(divergence):
    if divergence is None:
        return "no_action_divergence"
    return "|".join((divergence.get("street", "unknown"),
                     divergence.get("actor_position", "unknown"),
                     (divergence.get("actor_initial_preflop") or {}).get("decile", "unknown"),
                     (divergence.get("hand_category_at_decision") or {}).get("name", "no_made_category"),
                     ((divergence.get("hand_category_at_decision") or {}).get("pair_context") or {}).get("pair_presence_context", "unknown"),
                     "opening" if divergence.get("action_index") == 0 else "later",
                     divergence.get("agent1_action", "none"),
                     divergence.get("agent2_action", "none")))


def _paired_match(data, ours="agent1", true_equity=False):
    if data.get("kind") != "match_result" or not data.get("logs"):
        raise ValueError("input must be a judge match JSON containing --logs")
    seed = str(data["match_seed"])
    names = {"agent1": data.get("agent1_name", "agent1"),
             "agent2": data.get("agent2_name", "agent2")}
    if ours not in names:
        raise ValueError("--ours must be agent1 or agent2")
    equity_cache = {} if true_equity else None
    traces = {1: {}, 2: {}}
    for half_key, logs in data["logs"].items():
        half = int(half_key.replace("half", ""))
        for log in logs:
            trace = replay_hand(seed, half, log, names, true_equity, equity_cache)
            traces[half][trace["deal_index"]] = trace
    if set(traces[1]) != set(traces[2]):
        raise ValueError("duplicate halves do not contain the same deal indices")

    per_agent = {agent: [] for agent in names}
    paired_deals = []
    paired_hand_rows = []
    divergence_rows = []
    for deal_index in sorted(traces[1]):
        first, second = traces[1][deal_index], traces[2][deal_index]
        if first["hole_cards"] != second["hole_cards"] or first["board_full_hindsight"] != second["board_full_hindsight"]:
            raise ValueError(f"duplicate cards differ for deal {deal_index}")
        divergence = _first_divergence(first, second)
        seat_pairs = []
        for seat in (0, 1):
            h1 = first["seats"][seat]
            h2 = second["seats"][seat]
            by_agent = {h1["agent"]: h1, h2["agent"]: h2}
            our_row, other_row = by_agent[ours], by_agent["agent2" if ours == "agent1" else "agent1"]
            delta = our_row["net_bb"] - other_row["net_bb"]
            per_agent[ours].append(our_row)
            per_agent["agent2" if ours == "agent1" else "agent1"].append(other_row)
            paired_hand_rows.append({
                "deal_index": deal_index, "seat": seat,
                "position": "button" if seat == first["button_seat"] else "big_blind",
                "initial_preflop_decile": our_row["initial_preflop"]["decile"],
                "our_terminal_street": our_row["terminal_street"],
                "other_terminal_street": other_row["terminal_street"],
                "our_fold_vs_showdown": our_row["fold_vs_showdown"],
                "other_fold_vs_showdown": other_row["fold_vs_showdown"],
                "our_final_category": (our_row["final_hand_category_visible_at_end"] or {}).get("name"),
                "other_final_category": (other_row["final_hand_category_visible_at_end"] or {}).get("name"),
                "our_final_pair_presence_context": ((our_row["final_hand_category_visible_at_end"] or {}).get("pair_context") or {}).get("pair_presence_context"),
                "other_final_pair_presence_context": ((other_row["final_hand_category_visible_at_end"] or {}).get("pair_context") or {}).get("pair_presence_context"),
                "our_largest_investment_street": our_row["largest_investment_street"],
                "our_largest_investment_action": our_row["largest_investment_action"],
                "our_largest_investment_role": our_row["largest_investment_action_role"],
                "our_largest_investment_category": (our_row["largest_investment_category_at_decision"] or {}).get("name"),
                "other_largest_investment_street": other_row["largest_investment_street"],
                "other_largest_investment_action": other_row["largest_investment_action"],
                "other_largest_investment_role": other_row["largest_investment_action_role"],
                "other_largest_investment_category": (other_row["largest_investment_category_at_decision"] or {}).get("name"),
                "delta_bb": delta, "contribution_weight": .5,
                "scenario": _scenario(our_row) + " vs " + _scenario(other_row),
            })
            seat_pairs.append({
                "seat": seat, "position": "button" if seat == first["button_seat"] else "big_blind",
                "hole_cards": our_row["hole_cards"],
                "our_net_bb": our_row["net_bb"], "other_net_bb": other_row["net_bb"],
                "delta_bb": delta, "our_terminal_street": our_row["terminal_street"],
                "other_terminal_street": other_row["terminal_street"],
                "our_fold_vs_showdown": our_row["fold_vs_showdown"],
                "other_fold_vs_showdown": other_row["fold_vs_showdown"],
                "our_initial_preflop": our_row["initial_preflop"],
                "other_initial_preflop": other_row["initial_preflop"],
                "our_final_category_visible_at_end": our_row["final_hand_category_visible_at_end"],
                "other_final_category_visible_at_end": other_row["final_hand_category_visible_at_end"],
                "our_largest_investment_street": our_row["largest_investment_street"],
                "our_largest_investment_action": our_row["largest_investment_action"],
                "our_largest_investment_role": our_row["largest_investment_action_role"],
                "our_largest_investment_category_at_decision": our_row["largest_investment_category_at_decision"],
                "other_largest_investment_street": other_row["largest_investment_street"],
                "other_largest_investment_action": other_row["largest_investment_action"],
                "other_largest_investment_role": other_row["largest_investment_action_role"],
                "other_largest_investment_category_at_decision": other_row["largest_investment_category_at_decision"],
                "our_commitment_events": our_row["commitment_events"],
                "other_commitment_events": other_row["commitment_events"],
            })
        # Zero-sum duplicate scoring makes both same-seat hand deltas equal.
        if abs(seat_pairs[0]["delta_bb"] - seat_pairs[1]["delta_bb"]) > 1e-9:
            raise ValueError(f"same-deal seat deltas disagree for deal {deal_index}")
        delta_bb = .5 * (seat_pairs[0]["delta_bb"] + seat_pairs[1]["delta_bb"])
        divergence_row = {"deal_index": deal_index, "delta_bb": delta_bb,
                          "divergence": divergence}
        divergence_rows.append(divergence_row)
        paired_deals.append({
            "deal_index": deal_index, "delta_bb": delta_bb,
            "seat_pairs": seat_pairs, "first_divergence": divergence,
            "interpretation": "realized paired outcome associated with first divergence; not causal",
            "cards": {"seat0_hole": first["hole_cards"][0], "seat1_hole": first["hole_cards"][1],
                      "board_revealed_half1": first["board_revealed"],
                      "board_revealed_half2": second["board_revealed"],
                      "board_full_hindsight": first["board_full_hindsight"]},
            "half1_actions": first["actions"], "half2_actions": second["actions"],
        })

    marginal_features = {
        "terminal_street": lambda r: r["terminal_street"],
        "fold_vs_showdown": lambda r: r["fold_vs_showdown"],
        "initial_preflop_quantile": lambda r: r["initial_preflop"]["decile"],
        "final_hand_category_visible_at_end": lambda r: (r["final_hand_category_visible_at_end"] or {}).get("name"),
        "largest_investment_street_category_and_pair_source": lambda r: r["largest_investment_street"] + "|" +
            ((r["largest_investment_category_at_decision"] or {}).get("name", "no_made_category")) + "|" +
            (((r["largest_investment_category_at_decision"] or {}).get("pair_context") or {}).get("pair_presence_context", "unknown")),
        "largest_investment_action": lambda r: "|".join((r["largest_investment_street"],
            r["largest_investment_action"] or "none", r["largest_investment_action_role"] or "none",
            ((r["largest_investment_category_at_decision"] or {}).get("name", "no_made_category")))),
        "first_flop_action": lambda r: _first_street_action_key(r, "FLOP"),
        "commitment_kind_action_role": lambda r: ",".join(c["kind"] + ":" + c["role"]
                                                        for c in r["commitment_events"]) or "none",
    }
    agent_summary = {}
    for agent, rows in per_agent.items():
        agent_summary[agent] = {
            "display_name": names[agent],
            "overall": aggregate(rows),
            "by_scenario": {name: group_stats(rows, fn) for name, fn in marginal_features.items()},
            "commitment_events": [event for row in rows for event in row["commitment_events"]],
        }
        agent_summary[agent]["commitment_event_outcomes"] = group_stats(
            agent_summary[agent]["commitment_events"],
            lambda c: "|".join((c["kind"], c["street"], c["role"], c["action"],
                                (c["category_at_action"] or {}).get("name", "no_made_category"))),
            value_key="hand_net_bb")

    paired_by_scenario = group_stats(paired_hand_rows, lambda r: r["scenario"], "contribution_weight", "delta_bb")
    paired_by_initial_quantile = group_stats(paired_hand_rows, lambda r: r["initial_preflop_decile"],
                                             "contribution_weight", "delta_bb")
    paired_by_terminal_path = group_stats(paired_hand_rows, lambda r: "|".join((
        r["initial_preflop_decile"], r["our_terminal_street"], r["our_fold_vs_showdown"],
        r["other_terminal_street"], r["other_fold_vs_showdown"])), "contribution_weight", "delta_bb")
    divergence_contributions = group_stats(
        divergence_rows, lambda r: _divergence_key(r["divergence"]), value_key="delta_bb")
    sorted_negative = sorted(paired_deals, key=lambda row: row["delta_bb"])
    paired_delta = np.asarray([row["delta_bb"] for row in paired_deals], dtype=float)
    pair_summary = {
        "unique_deals": len(paired_deals),
        "mean_delta_bb_per_deal": float(paired_delta.mean()) if len(paired_delta) else None,
        "net_delta_bb": float(paired_delta.sum()),
        "standard_error_bb": float(paired_delta.std(ddof=1) / np.sqrt(len(paired_delta)))
            if len(paired_delta) > 1 else None,
        "per_initial_hand_quantile": paired_by_initial_quantile,
        "per_terminal_path": paired_by_terminal_path,
        "per_combined_hand_scenario": paired_by_scenario,
        "first_divergence_realized_delta_totals_not_causal": divergence_contributions,
        "all_paired_deals": paired_deals,
        "largest_negative_paired_cases": sorted_negative,
    }
    judge_paired = np.asarray(data.get("paired_bb", []), dtype=float)
    expected_paired = judge_paired if ours == "agent1" else -judge_paired
    vector_matches = len(expected_paired) == len(paired_delta) and np.allclose(
        paired_delta, expected_paired, rtol=0., atol=1e-12)
    judge_mean = float(data["mean_bb_per_deal"]) * (1 if ours == "agent1" else -1)
    computed_mean = float(paired_delta.mean()) if len(paired_delta) else None
    return {
        "match_seed": seed,
        "num_deals": len(paired_deals),
        "agent_names": names,
        "ours": ours,
        "net_bb_per_hand_including_both_halves": {
            agent: agent_summary[agent]["overall"] for agent in names
        },
        "per_agent_scenario_summaries": agent_summary,
        "paired_same_deal_same_seat": pair_summary,
        "paired_delta_computed_mean_bb": computed_mean,
        "judge_paired_mean_bb_from_json": judge_mean,
        "paired_delta_matches_judge_mean_bb": judge_mean,
        "paired_delta_matches_judge_vector": bool(vector_matches),
        "hindsight_equity": {
            "enabled": bool(true_equity),
            "interpretation": "offline true-hole equity at first large-commit and all-in actions; not policy information",
            "large_commit_threshold_chips": LARGE_COMMIT_CHIPS,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", nargs="+", type=Path, help="judge match result JSON files with --logs")
    parser.add_argument("--ours", choices=("agent1", "agent2"), default="agent1")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--true-equity", action="store_true",
                        help="calculate offline hindsight equity at large-commit/all-in decisions")
    parser.add_argument("--output", type=Path, help="write full analysis JSON (default: beside first input)")
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    analyses = []
    for path in args.matches:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        analysis = _paired_match(data, ours=args.ours, true_equity=args.true_equity)
        analysis["source"] = str(path)
        analysis["paired_same_deal_same_seat"]["largest_negative_paired_cases"] = \
            analysis["paired_same_deal_same_seat"]["largest_negative_paired_cases"][:args.top]
        analyses.append(analysis)
    report = {"analysis": "duplicate-deal forensic comparison",
              "ours": args.ours,
              "matches": analyses,
              "limitations": [
                  "Realized paired deltas are noisy outcome evidence, not action-value estimates.",
                  "First-divergence grouped outcomes are associations, not causal action effects.",
                  "Final hand categories for folds use only cards actually revealed by the judge.",
                  "Full board and actual opponent cards appear only in explicitly hindsight-labeled forensic fields.",
              ]}
    output = args.output or (args.matches[0].parent / "loss-analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    concise = [{"seed": row["match_seed"], "deals": row["num_deals"],
                "paired_mean_bb": row["paired_same_deal_same_seat"]["mean_delta_bb_per_deal"],
                "judge_mean_bb": row["paired_delta_matches_judge_mean_bb"],
                "paired_se": row["paired_same_deal_same_seat"]["standard_error_bb"],
                "output": str(output)} for row in analyses]
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
