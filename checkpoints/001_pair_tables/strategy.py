"""Range-aware Monte Carlo with one betting response and bounded computation.

All learned state and caches reset every half. Preflop data is generated offline.
"""
import os
import time
from pathlib import Path
import numpy as np
from game import Action
from batch_eval import score, draw, hand_index
from pair_eval import BoardPairCache, score_holdings
from opponent_model import (history_events, init_model, frequencies, learn_event,
                            selective_weight, policy_components, policy, posterior)
from action_values import values, state_from, response_event

PRE = np.load(Path(__file__).parent / 'data' / 'preflop.npy', allow_pickle=False)
PRE_SORTED = np.sort(PRE)


def new_half(config, memory):
    memory.update(config=config, model=init_model(), rng=np.random.default_rng(81723),
                  hand=-1, seen=set(), events=[], snapshots={}, hands_played=0,
                  used=0., decisions=0, samples=0, ess_total=0., us_per_sample=20.,
                  uncertain=0, cheap_decisions=0, pair_cache=BoardPairCache(),
                  last_obs=None, last_action=None, evidence=None)


def strengths(hands, board, rng, repetitions=4):
    """Expected showdown outcomes against independent random legal hands.

    These draws are separate from action-value runouts; no future card leakage.
    River scores give exact made-hand ordering. Preflop uses the offline table.
    """
    if not board:
        return PRE[hand_index(hands)]
    if len(board) == 5:
        return score(hands, board).astype(float)
    n = len(hands)
    repeated = np.repeat(hands, repetitions, axis=0)
    public = np.broadcast_to(board, (len(repeated), len(board)))
    excluded = np.concatenate((repeated, public), axis=1)
    cards = draw(rng, excluded, 10 - len(board), len(repeated))
    complete = np.concatenate((public, cards[:, 5:]), axis=1)
    mine, other = score(repeated, complete), score(cards[:, :5], complete)
    return ((mine > other) + .5 * (mine == other)).reshape(n, repetitions).mean(axis=1)


def observe(obs, memory):
    if memory['hand'] != obs.hand_index:
        memory['pair_cache'].clear(reset_stats=False)
        memory.update(hand=obs.hand_index, seen=set(), events=[], snapshots={}, evidence=None)
    for event in history_events(obs, memory['config']):
        if event.index not in memory['seen']:
            memory['snapshots'][event.index] = frequencies(memory['model'], event).copy()
            memory['events'].append(event)
            memory['seen'].add(event.index)
            learn_event(memory['model'], event)


def sample(obs, memory, n, cheap=False, emergency=False):
    rng = memory['rng']
    memory['samples'] += n
    excluded = np.array(tuple(obs.hole_cards) + tuple(obs.board))
    cards = draw(rng, excluded, 10 - len(obs.board), n)
    hands = cards[:, :5]
    board = np.concatenate((np.broadcast_to(obs.board, (n, len(obs.board))), cards[:, 5:]), axis=1).astype(int)
    river_table = (memory['pair_cache'].get(obs.board)
                   if len(obs.board) == 5 and n >= 128 else None)
    if river_table is None:
        mine, theirs = score(obs.hole_cards, board), score(hands, board)
    else:
        mine = score_holdings(obs.hole_cards, river_table)
        theirs = score_holdings(hands, river_table)
    wins = (mine > theirs) + .5 * (mine == theirs)
    strength_cache = {}
    # Actions expected with probability > 1-1/n convey less than one particle's
    # worth of exclusion; avoid repeatedly evaluating those near-certain actions.
    events = [] if emergency else [e for e in memory['events']
              if memory['snapshots'][e.index][e.action] < 1 - 1 / n]
    if emergency:
        current_strength = np.zeros(n)
    elif river_table is not None:
        current_strength = theirs.astype(float)
    elif cheap:
        current_strength = PRE[hand_index(hands)] if not obs.board else score(hands, obs.board).astype(float)
    else:
        current_strength = strengths(hands, tuple(obs.board), rng)
    strength_cache[int(obs.street)] = current_strength
    for street in {e.street for e in events} - {int(obs.street)}:
        visible = 0 if street == 0 else street + 2
        public = tuple(obs.board[:visible])
        strength_cache[street] = (PRE[hand_index(hands)] if not public else score(hands, public).astype(float)) if cheap else strengths(hands, public, rng)
    sel = selective_weight(memory['model'])
    weights = np.full(n, 1 / n)
    log_sel = np.zeros(n)
    refs = []
    for event in events:
        freq = memory['snapshots'][event.index]
        strength = strength_cache[event.street]
        selective, independent = policy_components(strength, freq)
        likelihood = sel * selective[:, event.action] + (1 - sel) * independent[:, event.action]
        weights = posterior(weights, likelihood)
        # Save full-history normalizer for showdown learning (selection correction).
        log_sel += np.log(np.maximum(selective[:, event.action], np.finfo(float).tiny))
        refs.append((event, freq, strength.copy()))
    ess = 1 / np.dot(weights, weights)
    # Finite-sample reliability: smoothly retain the broad range when only a few
    # particles carry weight. No zero-sample forced folds or hard range deletion.
    reliability = ess / (ess + np.sqrt(n))
    if events:
        weights = reliability * weights + (1 - reliability) / n
    memory['evidence'] = None if emergency else (refs, log_sel, strength_cache[int(obs.street)].copy(), cheap)
    return wins, strength_cache[int(obs.street)], weights, ess


def action_means(obs, outcomes, wins, weights, has_history):
    means = {a: float(np.dot(weights, result)) for a, result in outcomes.items()}
    if not obs.board and not has_history:
        # The offline equity is an independent, much larger preflop estimate.
        # Use it as a control variate for the shared showdown outcomes.
        estimated = float(np.dot(weights, wins))
        variance = float(np.dot(weights, (wins - estimated) ** 2))
        preflop_equity = float(PRE[hand_index(np.array([obs.hole_cards]))][0])
        if variance > 0:
            for action, outcome in outcomes.items():
                coefficient = np.dot(weights, (outcome - means[action]) * (wins - estimated)) / variance
                means[action] += float(coefficient * (preflop_equity - estimated))
    return means


def fast_decision(obs, memory):
    """Low-cost reserve mode: no history filtering or response-rank simulation."""
    memory['evidence'] = None
    if not obs.board:
        equity = float(PRE[hand_index(np.array([obs.hole_cards]))][0])
    else:
        n = 16
        cards = draw(memory['rng'], np.array(tuple(obs.hole_cards) + tuple(obs.board)), 10 - len(obs.board), n)
        board = np.concatenate((np.broadcast_to(obs.board, (n, len(obs.board))), cards[:, 5:]), axis=1)
        mine, theirs = score(obs.hole_cards, board), score(cards[:, :5], board)
        equity = float(((mine > theirs) + .5 * (mine == theirs)).mean())
        memory['samples'] += n
    outcomes = values(obs, memory['config'], np.array([equity]), np.zeros(1),
                      np.ones(1), memory['model'], 0.)
    return max(outcomes, key=lambda a: outcomes[a][0])


def choose_action(obs, memory):
    start = time.perf_counter()
    observe(obs, memory)
    cfg = memory['config']
    decisions_per_hand = max(2., memory['decisions'] / max(1, memory['hands_played']))
    reserve = .4 * cfg.time_bank_seconds
    left = max(0., reserve - memory['used'])
    allowance = min(.006, left / (max(1, cfg.num_hands - obs.hand_index) * decisions_per_hand))
    # Hard bounded work even if the budget estimate is exhausted.
    cheap = allowance < .0012
    emergency = memory['used'] > .4 * cfg.time_bank_seconds
    n = 16 if emergency else 48 if cheap else max(24, min(128, int(allowance * 1e6 / max(1., memory['us_per_sample']))))
    if cheap:
        memory['cheap_decisions'] += 1
    try:
        if emergency:
            best = fast_decision(obs, memory)
            memory['last_obs'], memory['last_action'] = obs, best
            return best
        wins, strength, weights, ess = sample(obs, memory, n, cheap, emergency)
        selectivity = 0. if emergency else selective_weight(memory['model'])
        outcomes = values(obs, cfg, wins, strength, weights, memory['model'], selectivity)
        means = action_means(obs, outcomes, wins, weights, bool(memory['events']))
        best = max(means, key=means.get)
        ordered = sorted(means, key=means.get, reverse=True)
        if len(ordered) > 1:
            delta = outcomes[ordered[0]] - outcomes[ordered[1]]
            mean = np.dot(weights, delta)
            se = np.sqrt(np.dot(weights, (delta - mean) ** 2) / max(1., ess))
            if mean < 2 * se:
                memory['uncertain'] += 1
                # One extra batch only when measured time leaves room.
                if not cheap and time.perf_counter() - start < allowance / 2 and n < 128:
                    wins, strength, weights, ess = sample(obs, memory, min(128, n * 2))
                    outcomes = values(obs, cfg, wins, strength, weights, memory['model'], selective_weight(memory['model']))
                    means = action_means(obs, outcomes, wins, weights, bool(memory['events']))
                    best = max(means, key=means.get)
        memory['ess_total'] += ess
        memory['last_obs'], memory['last_action'] = obs, best
        return best
    finally:
        elapsed = time.perf_counter() - start
        memory['used'] += elapsed
        memory['decisions'] += 1
        if not cheap:
            memory['us_per_sample'] = .9 * memory['us_per_sample'] + .1 * elapsed * 1e6 / n


def learn_showdown(result, memory, terminal):
    evidence = memory.get('evidence')
    if not result.opp_hole or evidence is None:
        return
    refs, log_sel, current_strength, cheap = evidence
    if terminal is not None:
        event, freq = terminal
        selective, _ = policy_components(current_strength, freq)
        log_sel = log_sel + np.log(np.maximum(selective[:, event.action], np.finfo(float).tiny))
        refs = refs + [(event, freq, current_strength)]
    if not refs:
        return
    # Compare the revealed hand to each hypothesis CONDITIONED on the observed
    # action sequence. The sampled normalizer accounts for reaching showdown.
    actual_log = 0.
    cache = {}
    for event, freq, reference in refs:
        if event.street not in cache:
            visible = 0 if event.street == 0 else event.street + 2
            public = tuple(result.board[:visible])
            if cheap:
                cache[event.street] = PRE[hand_index(np.array([result.opp_hole]))][0] if not public else score(result.opp_hole, public)[0]
            else:
                cache[event.street] = strengths(np.array([result.opp_hole]), public, memory['rng'])[0]
        selective, _ = policy_components(np.array([cache[event.street]]), freq, reference=reference)
        actual_log += np.log(max(selective[-1, event.action], np.finfo(float).tiny))
    peak = log_sel.max()
    log_normalizer = peak + np.log(np.exp(log_sel - peak).mean())
    ratio = np.exp(np.clip(actual_log - log_normalizer, -30, 30))
    mixture = selective_weight(memory['model'])
    responsibility = mixture * ratio / (mixture * ratio + 1 - mixture)
    memory['model']['selective_evidence'] += responsibility
    memory['model']['independent_evidence'] += 1 - responsibility


def on_hand_end(result, memory):
    start = time.perf_counter()
    terminal = None
    obs, action = memory['last_obs'], memory['last_action']
    if obs is not None and obs.hand_index == result.hand_index:
        state, _ = state_from(obs, memory['config']).apply(0, int(action))
        # Only infer actions that are determined by our final action and outcome.
        inferred = None
        if not result.went_to_showdown and result.winner == 'me' and action in (Action.RAISE_HALF, Action.RAISE_POT):
            inferred = 0
        elif result.went_to_showdown and action in (Action.RAISE_HALF, Action.RAISE_POT):
            inferred = 1
        elif result.went_to_showdown and action == Action.CHECK_CALL and obs.to_call == 0 and not obs.is_button:
            inferred = 1  # final free check, opponent checked behind
        if inferred is not None:
            event = response_event(state, obs.street, len(obs.action_history) + 1, inferred)
            freq = frequencies(memory['model'], event).copy()
            learn_event(memory['model'], event)
            terminal = (event, freq)
        if memory['used'] < .35 * memory['config'].time_bank_seconds:
            learn_showdown(result, memory, terminal)
    memory['hands_played'] += 1
    memory['used'] += time.perf_counter() - start
    if os.environ.get('BOT_DIAGNOSTICS') and memory['hands_played'] == memory['config'].num_hands:
        import sys
        n = max(1, memory['decisions'])
        pc = memory['pair_cache']
        print(f"strategy: decisions={n} samples/decision={memory['samples']/n:.1f} "
              f"ESS={memory['ess_total']/n:.1f} internal_seconds={memory['used']:.3f} "
              f"selectivity={selective_weight(memory['model']):.3f} cheap={memory['cheap_decisions']} "
              f"pair_builds={pc.builds} pair_hits={pc.hits} pair_misses={pc.misses} "
              f"pair_cache_size={pc.size}", file=sys.stderr)
