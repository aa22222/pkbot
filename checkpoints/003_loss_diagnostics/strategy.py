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
from belief import (BeliefSnapshot, capture_record, decision_weights, resample_indices,
                    selective_action_likelihood)
from opponent_model import (history_events, init_model, frequencies, learn_event,
                            selective_weight)
from action_values import values, state_from, response_event

PRE = np.load(Path(__file__).parent / 'data' / 'preflop.npy', allow_pickle=False)
PRE_SORTED = np.sort(PRE)
PRE_REFERENCE = PRE_SORTED[np.linspace(0, len(PRE_SORTED) - 1, 1024, dtype=int)]


def new_half(config, memory):
    memory.update(config=config, model=init_model(), rng=np.random.default_rng(81723),
                  hand=-1, seen=set(), events=[], snapshots={}, hands_played=0,
                  used=0., decisions=0, samples=0, ess_total=0., us_per_sample=20.,
                  uncertain=0, cheap_decisions=0, pair_cache=BoardPairCache(),
                  belief=None, likelihood_records=[], reference_cache={}, terminal_seen=set(),
                  samples_proposed=0, samples_retained=0, distinct_total=0.,
                  reserve_decisions=0, full_decisions=0, belief_reuses=0,
                  range_refreshes=0, range_refresh_reasons={}, collision_removals=0,
                  event_suppressions=0, broad_fallback_decisions=0,
                  cached_fallback_decisions=0,
                  broad_fallback_reason=None, showdown_normalizer_skips=0,
                  refreshed_revision=-1, last_obs=None, last_action=None, evidence=None)


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
        memory.update(hand=obs.hand_index, seen=set(), events=[], snapshots={}, evidence=None,
                      belief=None, likelihood_records=[], reference_cache={}, refreshed_revision=-1)
    belief = memory.get('belief')
    if belief is not None:
        removed = belief.filter_board(obs.board)
        memory['collision_removals'] += removed
        if removed:
            memory['refreshed_revision'] = -1
    fresh_records = []
    for event in history_events(obs, memory['config']):
        if event.index not in memory['seen']:
            freq = frequencies(memory['model'], event).copy()
            mixture = selective_weight(memory['model'])
            record = capture_record(event, freq, mixture, obs.board, memory['reference_cache'])
            memory['snapshots'][event.index] = freq.copy()
            memory['events'].append(event)
            memory['likelihood_records'].append(record)
            fresh_records.append(record)
            memory['seen'].add(event.index)
            learn_event(memory['model'], event)
        else:
            memory['event_suppressions'] += 1
    if belief is not None:
        for record in fresh_records:
            if belief.apply_record(record, PRE, PRE_REFERENCE):
                memory['refreshed_revision'] = -1


def _propose_hands(obs, memory, count):
    excluded = np.array(tuple(obs.hole_cards) + tuple(obs.board), dtype=int)
    return draw(memory['rng'], excluded, 5, count)


def ensure_belief(obs, memory, proposal_count, force_refresh=False):
    """Create, update, or wholly refresh a legal uniform proposal batch."""
    snapshot = memory.get('belief')
    starting_revision = snapshot.revision if snapshot is not None else None
    if snapshot is not None:
        removed = snapshot.filter_board(obs.board)
        memory['collision_removals'] += removed
        if removed:
            memory['refreshed_revision'] = -1
        for record in memory['likelihood_records']:
            if snapshot.apply_record(record, PRE, PRE_REFERENCE):
                memory['refreshed_revision'] = -1
    else:
        proposed = _propose_hands(obs, memory, proposal_count)
        memory['samples_proposed'] += len(proposed)
        snapshot = BeliefSnapshot.from_proposal(
            obs.hand_index, obs.hole_cards, obs.board, proposed,
            memory['likelihood_records'], PRE, PRE_REFERENCE)
        memory['belief'] = snapshot
        memory['refreshed_revision'] = -1

    low_support = snapshot.distinct_count < min(8, max(4, len(snapshot.opponent_hands) // 8))
    empty = not len(snapshot.opponent_hands) or not np.any(snapshot.weights)
    if starting_revision is not None and snapshot.revision == starting_revision:
        memory['belief_reuses'] += 1
    refresh_needed = force_refresh or empty or (low_support and memory['refreshed_revision'] != snapshot.revision)
    if refresh_needed:
        reason = 'forced' if force_refresh else 'empty_support' if empty else 'low_distinct_support'
        memory['range_refresh_reasons'][reason] = memory['range_refresh_reasons'].get(reason, 0) + 1
        count = max(int(proposal_count), 48)
        previous_snapshot = snapshot
        proposed = _propose_hands(obs, memory, count)
        memory['samples_proposed'] += len(proposed)
        replacement = BeliefSnapshot.from_proposal(
            obs.hand_index, obs.hole_cards, obs.board, proposed,
            memory['likelihood_records'], PRE, PRE_REFERENCE)
        memory['range_refreshes'] += 1
        memory['belief'] = snapshot = replacement
        memory['refreshed_revision'] = snapshot.revision
        if not len(snapshot.opponent_hands) or not np.any(snapshot.weights):
            cached_legal = np.array([
                not (set(map(int, hand)) & (set(obs.hole_cards) | set(obs.board)))
                for hand in previous_snapshot.opponent_hands], dtype=bool)
            if np.any(cached_legal) and np.any(previous_snapshot.weights[cached_legal]):
                previous_snapshot.opponent_hands = previous_snapshot.opponent_hands[cached_legal]
                previous_snapshot.weights = previous_snapshot.weights[cached_legal]
                previous_snapshot.weights /= previous_snapshot.weights.sum()
                previous_snapshot.board = tuple(obs.board)
                memory['belief'] = snapshot = previous_snapshot
                memory['cached_fallback_decisions'] += 1
                memory['refreshed_revision'] = snapshot.revision
            else:
                # Last resort is an explicit broad legal range, clearly diagnosed.
                broad = _propose_hands(obs, memory, count)
                memory['samples_proposed'] += len(broad)
                snapshot = BeliefSnapshot.from_proposal(
                    obs.hand_index, obs.hole_cards, obs.board, broad, (), PRE, PRE_REFERENCE)
                memory['belief'] = snapshot
                memory['broad_fallback_decisions'] += 1
                memory['broad_fallback_reason'] = 'refreshed proposal had no legal weighted support'
                memory['refreshed_revision'] = snapshot.revision
    return snapshot


def sample(obs, memory, n, cheap=False, emergency=False):
    rng = memory['rng']
    snapshot = ensure_belief(obs, memory, n)
    use_full_belief = len(obs.board) == 5
    raw_weights = decision_weights(snapshot, bool(snapshot.likelihood_records))
    raw_ess = snapshot.effective_sample_size
    if use_full_belief or not emergency or len(snapshot.opponent_hands) <= n:
        hands = snapshot.opponent_hands
        weights = raw_weights
    else:
        indices = resample_indices(raw_weights, n, rng)
        hands = snapshot.opponent_hands[indices]
        weights = np.full(len(hands), 1 / len(hands), dtype=float)
    count = len(hands)
    if use_full_belief:
        complete_board = np.broadcast_to(np.asarray(obs.board, dtype=int), (count, 5))
    else:
        known = np.array(tuple(obs.hole_cards) + tuple(obs.board), dtype=int)
        excluded = np.concatenate((hands, np.broadcast_to(known, (count, len(known)))), axis=1)
        runouts = draw(rng, excluded, 5 - len(obs.board), count)
        complete_board = np.concatenate((np.broadcast_to(obs.board, (count, len(obs.board))),
                                         runouts), axis=1).astype(int)
    river_table = (memory['pair_cache'].get(obs.board)
                   if use_full_belief and count >= 128 else None)
    if river_table is None:
        mine, theirs = score(obs.hole_cards, complete_board), score(hands, complete_board)
    else:
        mine = score_holdings(obs.hole_cards, river_table)
        theirs = score_holdings(hands, river_table)
    wins = (mine > theirs) + .5 * (mine == theirs)
    if use_full_belief:
        current_strength = theirs.astype(float)
    elif cheap:
        current_strength = PRE[hand_index(hands)] if not obs.board else score(hands, obs.board).astype(float)
    else:
        current_strength = strengths(hands, tuple(obs.board), rng)
    distinct_sampled = len(np.unique(np.sort(hands, axis=1), axis=0)) if count else 0
    decision_ess = 1 / np.dot(weights, weights) if len(weights) else 0.
    ess = min(raw_ess, distinct_sampled, decision_ess) if count else 0.
    memory['samples'] += count
    memory['samples_retained'] += count
    memory['distinct_total'] += distinct_sampled
    memory['evidence'] = (snapshot, tuple(snapshot.likelihood_records), cheap)
    return wins, current_strength, weights, ess


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
    """Reserve path: resample from the retained posterior and keep its belief."""
    memory['reserve_decisions'] += 1
    wins, strength, weights, ess = sample(obs, memory, 16, cheap=True, emergency=True)
    memory['ess_total'] += ess
    outcomes = values(obs, memory['config'], wins, strength, weights,
                      memory['model'], selective_weight(memory['model']))
    means = action_means(obs, outcomes, wins, weights, bool(memory['events']))
    return max(means, key=means.get)


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
    else:
        memory['full_decisions'] += 1
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
                if not cheap and len(obs.board) < 5 and time.perf_counter() - start < allowance / 2 and n < 128:
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


def learn_showdown(result, memory, terminal_record=None):
    evidence = memory.get('evidence')
    if not result.opp_hole or evidence is None:
        return
    snapshot, observed_records, _ = evidence
    records = list(observed_records)
    if terminal_record is not None:
        records.append(terminal_record)
    if not records:
        return
    # The prior normalizer uses uniform proposal rows, never posterior or
    # reliability-blended weights. Filter cards revealed after our last action.
    board = tuple(map(int, result.board))
    proposals = snapshot.opponent_hands
    legal = np.array([not (set(map(int, hand)) & set(board)) for hand in proposals], dtype=bool)
    proposals = proposals[legal]
    if not len(proposals):
        memory['showdown_normalizer_skips'] += 1
        return
    log_sel = np.zeros(len(proposals), dtype=float)
    for record in records:
        p = selective_action_likelihood(proposals, record, PRE, PRE_REFERENCE)
        log_sel += np.log(p)
    actual_hand = np.asarray(result.opp_hole, dtype=int).reshape(1, 5)
    actual_log = 0.
    for record in records:
        p = selective_action_likelihood(actual_hand, record, PRE, PRE_REFERENCE)
        actual_log += np.log(p[0])
    peak = log_sel.max()
    log_normalizer = peak + np.log(np.exp(log_sel - peak).mean())
    ratio = np.exp(np.clip(actual_log - log_normalizer, -30, 30))
    mixture = selective_weight(memory['model'])
    responsibility = mixture * ratio / (mixture * ratio + 1 - mixture)
    memory['model']['selective_evidence'] += responsibility
    memory['model']['independent_evidence'] += 1 - responsibility


def on_hand_end(result, memory):
    if result.hand_index in memory['terminal_seen']:
        return
    memory['terminal_seen'].add(result.hand_index)
    start = time.perf_counter()
    terminal_record = None
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
            mixture = selective_weight(memory['model'])
            terminal_record = capture_record(event, freq, mixture, obs.board,
                                             memory['reference_cache'])
            memory['likelihood_records'].append(terminal_record)
            if memory['belief'] is not None:
                memory['belief'].apply_record(terminal_record, PRE, PRE_REFERENCE)
            learn_event(memory['model'], event)
        if memory['used'] < .35 * memory['config'].time_bank_seconds:
            learn_showdown(result, memory, terminal_record)
    memory['hands_played'] += 1
    memory['used'] += time.perf_counter() - start
    if os.environ.get('BOT_DIAGNOSTICS') and memory['hands_played'] == memory['config'].num_hands:
        import sys
        n = max(1, memory['decisions'])
        pc = memory['pair_cache']
        print(f"strategy: decisions={n} samples/decision={memory['samples']/n:.1f} "
              f"ESS={memory['ess_total']/n:.1f} internal_seconds={memory['used']:.3f} "
              f"selectivity={selective_weight(memory['model']):.3f} full={memory['full_decisions']} "
              f"cheap={memory['cheap_decisions']} range_reserve={memory['reserve_decisions']} "
              f"belief_reuse={memory['belief_reuses']} proposed={memory['samples_proposed']} "
              f"retained={memory['samples_retained']} distinct_avg="
              f"{memory['distinct_total']/max(1,memory['decisions']):.1f} "
              f"refreshes={memory['range_refreshes']} collisions={memory['collision_removals']} "
              f"event_suppressed={memory['event_suppressions']} broad_fallback="
              f"{memory['broad_fallback_decisions']}({memory['broad_fallback_reason']}) "
              f"cached_fallback={memory['cached_fallback_decisions']} "
              f"showdown_norm_skips={memory['showdown_normalizer_skips']} "
              f"pair_builds={pc.builds} pair_hits={pc.hits} pair_misses={pc.misses} "
              f"pair_cache_size={pc.size}", file=sys.stderr)
