from itertools import combinations

import numpy as np
import pytest

from belief import (BeliefSnapshot, capture_record, decision_weights,
                    record_likelihood, resample_indices)
from game import Action, AgentConfig, Game
from opponent_model import Event, init_model
from strategy import (PRE, PRE_REFERENCE, ensure_belief, new_half, observe,
                      sample, fast_decision)
import strategy as strategy_module


def _event(index=0, street=1, action=int(Action.RAISE_POT)):
    return Event(index, street, action, True, .3, (0, 1, 2, 3))


def _legal_rows(hero, board, count, seed=34):
    excluded = np.array(tuple(hero) + tuple(board), dtype=int)
    from batch_eval import draw
    return draw(np.random.default_rng(seed), excluded, 5, count)


def _at_street(street, seed="belief-state"):
    game = Game(__import__('game').make_deal(seed, 0), 0)
    while not game.is_terminal:
        if int(game.street) == street and game.to_act == 0:
            return game, game.observation(0)
        game.apply(Action.CHECK_CALL)
    raise AssertionError("could not reach requested street")


def test_snapshot_action_likelihood_is_applied_once_and_keeps_raw_posterior():
    hero = (0, 5, 10, 15, 20)
    board = (2, 7, 12)
    hands = _legal_rows(hero, board, 24)
    record = capture_record(_event(), np.full(4, .25), .6, board)
    snapshot = BeliefSnapshot.from_proposal(3, hero, board, hands, (), PRE, PRE_REFERENCE)
    assert snapshot.apply_record(record, PRE, PRE_REFERENCE)
    after_first = snapshot.weights.copy()
    assert not snapshot.apply_record(record, PRE, PRE_REFERENCE)
    assert np.array_equal(snapshot.weights, after_first)
    assert np.isclose(snapshot.weights.sum(), 1.)
    assert np.any(np.abs(snapshot.weights - 1 / len(snapshot.weights)) > 1e-6)


def test_new_board_filters_collisions_and_renormalizes_weights():
    hero = (0, 5, 10, 15, 20)
    flop = (2, 7, 12)
    turn = 24
    excluded = set(hero) | set(flop) | {turn}
    def hand_with_turn():
        rest = [c for c in range(36) if c not in excluded][:4]
        return [turn, *rest]
    without_turn = [c for c in range(36) if c not in (set(hero) | set(flop) | {turn})][:5]
    rows = np.array([hand_with_turn(), without_turn, hand_with_turn(), without_turn])
    snapshot = BeliefSnapshot(0, hero, flop, rows, np.array([.1, .2, .3, .4]))
    removed = snapshot.filter_board(flop + (turn,))
    assert removed == 2
    assert len(snapshot.opponent_hands) == 2
    assert np.allclose(snapshot.weights, [.2 / .6, .4 / .6])
    assert all(turn not in hand for hand in snapshot.opponent_hands)


def test_refresh_batch_posterior_matches_full_enumeration_on_tiny_support():
    hero = (0, 5, 10, 15, 20)
    board = (2, 7, 12)
    available = [c for c in range(36) if c not in set(hero) | set(board)][:8]
    exact_support = np.asarray(list(combinations(available, 5)), dtype=int)
    record = capture_record(_event(action=int(Action.CHECK_CALL)),
                           np.array([.4, .3, .2, .1]), .7, board)
    prior = np.full(len(exact_support), 1 / len(exact_support))
    exact_likelihood = record_likelihood(exact_support, record, PRE, PRE_REFERENCE)
    expected = prior * exact_likelihood
    expected /= expected.sum()
    rng = np.random.default_rng(811)
    proposals = exact_support[rng.integers(0, len(exact_support), size=8192)]
    refreshed = BeliefSnapshot.from_proposal(0, hero, board, proposals, [record], PRE, PRE_REFERENCE)
    estimated = np.zeros(len(exact_support))
    lookup = {tuple(hand): i for i, hand in enumerate(exact_support)}
    for hand, weight in zip(refreshed.opponent_hands, refreshed.weights):
        estimated[lookup[tuple(hand)]] += weight
    # Independent uniform proposal noise is under one percentage point here.
    assert np.max(np.abs(estimated - expected)) < .01


def test_frozen_reference_is_public_prefix_only_and_particle_invariant():
    hero = (0, 5, 10, 15, 20)
    board_a = (2, 7, 12, 24, 30)
    board_b = (2, 7, 12, 24, 31)
    turn_event = _event(street=2)
    first = capture_record(turn_event, np.full(4, .25), .55, board_a)
    second = capture_record(turn_event, np.full(4, .25), .55, board_b)
    assert first.board == second.board == board_a[:4]
    assert first.reference == second.reference
    hand = _legal_rows(hero, first.board, 1, seed=91)
    expected = record_likelihood(hand, first, PRE, PRE_REFERENCE)[0]
    unrelated = _legal_rows(hero, first.board, 50, seed=177)
    combined = np.concatenate((unrelated, hand), axis=0)
    likelihoods = record_likelihood(combined, first, PRE, PRE_REFERENCE)
    assert likelihoods[-1] == expected
    assert record_likelihood(combined[::-1], first, PRE, PRE_REFERENCE)[0] == expected
    assert first.mixture == .55


def test_preflop_likelihood_requires_the_frozen_sorted_reference():
    hero = (0, 5, 10, 15, 20)
    hand = _legal_rows(hero, (), 1, seed=21)
    record = capture_record(_event(street=0), np.full(4, .25), .5, ())
    with pytest.raises(ValueError, match="preflop_reference"):
        record_likelihood(hand, record, PRE)


def test_reserve_resampling_retains_narrow_posterior_support():
    hero = (0, 5, 10, 15, 20)
    board = (2, 7, 12)
    rows = _legal_rows(hero, board, 2, seed=515)
    hands = np.repeat(rows, 10, axis=0)
    weights = np.array([.09] * 10 + [.01] * 10)
    snapshot = BeliefSnapshot(0, hero, board, hands, weights)
    decision = decision_weights(snapshot, has_history=True)
    sampled = resample_indices(decision, 4096, np.random.default_rng(3))
    chosen = snapshot.opponent_hands[sampled]
    a, b = map(tuple, rows)
    share_a = np.mean([tuple(hand) == a for hand in chosen])
    assert set(map(tuple, chosen)) == {a, b}
    assert share_a > .7


def test_fast_decision_uses_stored_narrow_holdings_without_mutating_them(monkeypatch):
    game, obs = _at_street(1, seed="narrow-reserve")
    memory = {}
    new_half(AgentConfig(), memory)
    memory['hand'] = obs.hand_index
    rows = _legal_rows(obs.hole_cards, obs.board, 32, seed=515)
    weights = np.full(len(rows), .2 / (len(rows) - 1))
    weights[0] = .8
    snapshot = BeliefSnapshot(obs.hand_index, tuple(obs.hole_cards), tuple(obs.board),
                              rows, weights)
    memory['belief'] = snapshot
    memory['refreshed_revision'] = snapshot.revision
    before = snapshot.weights.copy()
    scored = []
    reference_score = strategy_module.score

    def capture_hands(hands, boards):
        array = np.asarray(hands)
        if array.ndim == 2 and array.shape[1] == 5:
            scored.append(array.copy())
        return reference_score(hands, boards)

    monkeypatch.setattr(strategy_module, "score", capture_hands)
    action = fast_decision(obs, memory)
    assert action in obs.legal_actions
    assert scored
    sampled = scored[0]
    assert all(any(np.array_equal(row, candidate) for candidate in rows) for row in sampled)
    assert np.array_equal(snapshot.weights, before)
    assert memory['range_refreshes'] == 0
    assert memory['reserve_decisions'] == 1


def test_zero_support_after_new_card_triggers_full_refresh():
    game, obs = _at_street(2)
    flop = tuple(obs.board[:3])
    turn = obs.board[3]
    excluded = set(obs.hole_cards) | set(flop) | {turn}
    collision_hands = []
    for start in range(36):
        candidate = [turn] + [c for c in range(36) if c not in excluded and c != start][:4]
        if len(set(candidate)) == 5 and not (set(candidate) & (set(obs.hole_cards) | set(flop))):
            collision_hands.append(candidate)
        if len(collision_hands) == 8:
            break
    memory = {}
    new_half(AgentConfig(), memory)
    memory['hand'] = obs.hand_index
    memory['belief'] = BeliefSnapshot(obs.hand_index, tuple(obs.hole_cards), flop,
                                      np.asarray(collision_hands), np.full(8, 1 / 8))
    memory['refreshed_revision'] = -1
    refreshed = ensure_belief(obs, memory, 16)
    assert memory['range_refreshes'] == 1
    assert len(refreshed.opponent_hands) >= 48
    assert np.isclose(refreshed.weights.sum(), 1.)
    assert all(not (set(hand) & (set(obs.hole_cards) | set(obs.board)))
               for hand in refreshed.opponent_hands)
    assert memory['broad_fallback_decisions'] == 0


def test_normal_and_reserve_river_inputs_match_for_same_snapshot():
    game, obs = _at_street(3, seed="river-agreement")
    memory = {}
    new_half(AgentConfig(), memory)
    memory['hand'] = obs.hand_index
    rows = _legal_rows(obs.hole_cards, obs.board, 32, seed=551)
    memory['belief'] = BeliefSnapshot.from_proposal(obs.hand_index, obs.hole_cards,
                                                   obs.board, rows, (), PRE, PRE_REFERENCE)
    memory['refreshed_revision'] = memory['belief'].revision
    normal = sample(obs, memory, 32, cheap=False)
    reserve = sample(obs, memory, 16, cheap=True, emergency=True)
    for left, right in zip(normal, reserve):
        assert np.array_equal(left, right)


def test_new_hand_and_new_half_clear_belief_state():
    memory = {}
    new_half(AgentConfig(), memory)
    game, obs = _at_street(1, seed="reset-a")
    observe(obs, memory)
    ensure_belief(obs, memory, 16)
    assert memory['belief'] is not None
    next_game = Game(__import__('game').make_deal("reset-b", 1), 0, hand_index=1)
    observe(next_game.observation(0), memory)
    assert memory['belief'] is None
    new_half(AgentConfig(), memory)
    assert memory['belief'] is None
    assert memory['likelihood_records'] == []

