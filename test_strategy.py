from itertools import combinations
import numpy as np
import pytest
from evaluator import score_five, warmup, omaha_score, cards_from_str
from game import Action, AgentConfig, Game, make_deal
from batch_eval import score, draw, hand_index
from opponent_model import (init_model, history_events, frequencies, learn_event,
                            policy_components, policy, posterior, Event)
from action_values import state_from, values
from strategy import new_half, observe, ensure_belief, choose_action, on_hand_end, PRE


def test_batch_matches_reference():
    warmup()
    cards = draw(np.random.default_rng(9), np.empty((1, 0), int), 10, 300)
    for n in (3, 4, 5):
        expected = [max(score_five(h + b) for h in combinations(row[:5], 2)
                        for b in combinations(row[5:5+n], 3)) for row in cards]
        assert np.array_equal(score(cards[:, :5], cards[:, 5:5+n]), expected)


def test_draws_are_legal_and_uniform():
    excluded = np.array([0, 1, 2, 3, 4])
    cards = draw(np.random.default_rng(2), excluded, 10, 20000)
    assert (cards >= 5).all()
    assert all(len(set(row)) == 10 for row in cards)
    counts = np.bincount(cards[:, 0], minlength=36)[5:]
    assert counts.min() > 500 and counts.max() < 800


def test_preflop_suit_symmetry():
    hands = np.array([[0, 1, 7, 22, 35], [4, 8, 12, 16, 20]])
    perm = np.array([3, 2, 0, 1])
    other = (hands // 4) * 4 + perm[hands % 4]
    assert np.array_equal(PRE[hand_index(hands)], PRE[hand_index(other)])
    assert np.isfinite(PRE).all() and (PRE >= 0).all() and (PRE <= 1).all()


def test_model_calibration_and_ties():
    freq = np.array([.25, .25, .25, .25])
    for strength in (np.arange(20), np.zeros(20), np.repeat(np.arange(5), 4)):
        sel, ind = policy_components(strength, freq)
        assert np.allclose(sel.sum(axis=1), 1)
        assert np.allclose(sel.mean(axis=0), freq)
        assert np.allclose(policy(strength, freq, 0), ind)
        assert np.allclose(policy(strength, freq, 1), sel)
    m = init_model()
    e = Event(0, 1, 1, True, .3, (0, 1))
    assert np.array_equal(frequencies(m, e)[2:], [0, 0])
    for _ in range(100):
        learn_event(m, e)
    assert frequencies(m, e)[1] > .98


def test_revealed_hand_uses_same_smoothed_reference():
    reference = np.array([.1, .2, .2, .9])
    frequency = np.array([.25, .25, .25, .25])
    original, _ = policy_components(reference, frequency)
    revealed, _ = policy_components(reference, frequency, reference=reference)
    assert np.array_equal(original, revealed)
    extreme, _ = policy_components(np.array([-1., 2.]), frequency, reference=reference)
    assert (extreme > 0).all()
    assert np.allclose(extreme.sum(axis=1), 1.)


def test_history_opportunities_and_dedup():
    memory = {}; new_half(AgentConfig(), memory)
    game = Game(make_deal('history', 0), 0)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs = game.observation(0)
    events = history_events(obs, memory['config'])
    assert len(events) == 1 and not events[0].facing
    observe(obs, memory)
    frozen = memory['likelihood_records'][0]
    assert frozen.mixture == .5
    assert not np.array_equal(frozen.frequency, frequencies(memory['model'], events[0]))
    ensure_belief(obs, memory, 48)
    frozen_weights = memory['belief'].weights.copy()
    observe(obs, memory); ensure_belief(obs, memory, 48)
    assert np.array_equal(memory['belief'].weights, frozen_weights)
    assert memory['model']['global_counts'][False].sum() == 1
    game.apply(Action.RAISE_POT); game.apply(Action.CHECK_CALL)
    observe(game.observation(0), memory)
    assert memory['model']['global_counts'][True][1] == 1


def test_public_betting_matches_engine():
    rng = np.random.default_rng(11)
    for i in range(100):
        game = Game(make_deal('geometry', i), i % 2)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            state = state_from(obs, AgentConfig())
            assert set(state.legal(0)) == set(map(int, obs.legal_actions))
            for action in obs.legal_actions:
                if action == Action.FOLD:
                    continue
                after, cost = state.apply(0, int(action))
                if action in obs.raise_to:
                    assert after.bets[0] == obs.raise_to[action]
                clone = game.clone(); clone.apply(action)
                assert after.pot == clone.pot
                assert after.stacks[0] == clone.stacks[seat]
            game.apply(obs.legal_actions[int(rng.integers(len(obs.legal_actions)))])


def test_posterior_and_no_hidden_hand_response():
    prior = np.array([.5, .5])
    assert np.allclose(posterior(prior, np.array([.1, .9])), [.1, .9])
    game = Game(make_deal('response', 0), 0)
    obs = game.observation(0)
    model = init_model()
    wins = np.array([0., 1.])
    strengths = np.array([0., 1.])
    result = values(obs, AgentConfig(), wins, strengths, prior, model, 0.)
    assert set(result) == set(obs.legal_actions)
    assert all(np.isfinite(x).all() for x in result.values())
    # Degenerate all-win/all-loss samples still obey incremental chip bounds.
    for outcome in result.values():
        assert outcome.min() >= -obs.my_stack
        assert outcome.max() <= obs.pot + obs.opp_stack


def test_terminal_call_counted_once():
    memory = {}; new_half(AgentConfig(), memory)
    game = Game(make_deal('terminal', 0), 0)
    while not game.is_terminal:
        if game.to_act == 0:
            obs = game.observation(0)
            observe(obs, memory)
            action = Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
            memory['last_obs'], memory['last_action'] = obs, action
        else:
            action = Action.CHECK_CALL
        game.apply(action)
    before = sum(v.sum() for v in memory['model']['global_counts'].values())
    on_hand_end(game.hand_result(0), memory)
    after = sum(v.sum() for v in memory['model']['global_counts'].values())
    assert after == before + 1
    on_hand_end(game.hand_result(0), memory)
    assert sum(v.sum() for v in memory['model']['global_counts'].values()) == after
    memory.clear(); new_half(AgentConfig(), memory)
    assert not memory['model']['counts']


def test_delayed_terminal_fold_updates_saved_belief_once():
    memory = {}; new_half(AgentConfig(), memory)
    game = Game(make_deal('terminal-fold', 0), 0)
    obs = game.observation(0)
    observe(obs, memory)
    ensure_belief(obs, memory, 48)
    memory['last_obs'], memory['last_action'] = obs, Action.RAISE_POT
    game.apply(Action.RAISE_POT)
    game.apply(Action.FOLD)
    result = game.hand_result(0)
    before_count = sum(v.sum() for v in memory['model']['global_counts'].values())
    on_hand_end(result, memory)
    after_count = sum(v.sum() for v in memory['model']['global_counts'].values())
    assert after_count == before_count + 1
    terminal = memory['likelihood_records'][-1]
    assert terminal.event.action == int(Action.FOLD)
    assert terminal.event.index in memory['belief'].processed_event_ids
    weights = memory['belief'].weights.copy()
    on_hand_end(result, memory)
    assert sum(v.sum() for v in memory['model']['global_counts'].values()) == after_count
    assert np.array_equal(memory['belief'].weights, weights)


def test_play_under_small_budget():
    memory = {}; new_half(AgentConfig(time_bank_seconds=.01), memory)
    for i in range(10):
        game = Game(make_deal('small-budget', i), i % 2, i)
        while not game.is_terminal:
            if game.to_act == 0:
                action = choose_action(game.observation(0), memory)
            else:
                action = Action.CHECK_CALL
            assert action in game.legal_actions()
            game.apply(action)
        on_hand_end(game.hand_result(0), memory)


def test_reserve_mode_retains_legal_range_across_board_reveal():
    memory = {}; new_half(AgentConfig(), memory)
    memory['used'] = 4.
    game = Game(make_deal('reserve', 0), 0)
    action = choose_action(game.observation(0), memory)
    assert action in game.legal_actions()
    assert memory['belief'] is not None and memory['evidence'] is not None
    assert np.isclose(memory['belief'].weights.sum(), 1.)
    assert all(not (set(row) & set(game.deal.hole_cards[0])) for row in memory['belief'].opponent_hands)
    game.apply(Action.CHECK_CALL); game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)  # opponent checks the flop; our seat acts
    obs = game.observation(game.to_act)
    action = choose_action(obs, memory)
    assert action in obs.legal_actions
    belief = memory['belief']
    assert belief.board == tuple(obs.board)
    assert np.isclose(belief.weights.sum(), 1.)
    assert all(not (set(row) & (set(obs.hole_cards) | set(obs.board)))
               for row in belief.opponent_hands)
    assert memory['reserve_decisions'] == 2
    assert memory['samples_proposed'] >= 16
