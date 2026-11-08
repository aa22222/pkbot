import numpy as np
from itertools import combinations

import simple_agent as bot
from batch_eval import score
from game import Action, AgentConfig, HandResult, Observation, Street
from action_values import Betting
from evaluator import omaha_score


def fresh():
    memory = {}
    bot.new_half(AgentConfig(), memory)
    return memory


def observation(hand=100, street=Street.PREFLOP, board=(), actions=(), legal=None):
    return Observation(hand, street, (0, 5, 10, 15, 20), tuple(board), 3, 0, 1, 2,
        79, 78, True, tuple(legal or (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)),
        {Action.RAISE_HALF: 4, Action.RAISE_POT: 6}, 4, 0, tuple(actions), 2)


def test_probe_pressure_and_call_when_raise_is_unavailable(monkeypatch):
    monkeypatch.setattr(bot, "PROBE_HANDS", 100)
    memory = fresh()
    assert bot.choose_action(observation(99), memory) == Action.RAISE_POT
    assert bot.choose_action(observation(0, legal=(Action.CHECK_CALL,)), memory) == Action.CHECK_CALL


def test_training_ignores_actual_board_deduplicates_and_skips_late(monkeypatch):
    monkeypatch.setattr(bot, "TRAIN_BOARDS_PER_SHOWDOWN", 10)
    hole = (1, 7, 14, 21, 29)
    choices = np.array(list(combinations([c for c in range(36) if c not in hole], 5)))
    generated = choices[np.random.default_rng(0).choice(len(choices), 10, replace=False)]
    def fake_draw(rng, excluded, count, rows):
        assert set(np.asarray(excluded).flat) in ({1, 7, 14, 21, 29}, {0, 9, 13, 22, 31})
        assert count == 5
        return generated.copy() if rows == 10 else second_boards.copy()
    monkeypatch.setattr(bot, "draw", fake_draw)
    m1, m2 = fresh(), fresh()
    a = HandResult(4, 0, 8, True, (0, 2, 4, 6, 8), hole, "split")
    b = HandResult(4, 0, 8, True, (3, 5, 9, 13, 17), hole, "split")
    bot.on_hand_end(a, m1)
    bot.on_hand_end(a, m1)
    bot.on_hand_end(b, m2)
    assert m1["rank_pool"] == m2["rank_pool"]
    assert m1["cutoff"] == m2["cutoff"]
    assert len(m1["rank_pool"]) == 10
    expected = score(np.broadcast_to(np.asarray(hole), generated.shape), generated)
    assert m1["rank_pool"] == list(map(int, expected))
    second_hole = (0, 9, 13, 22, 31)
    choices2 = np.array(list(combinations([c for c in range(36) if c not in second_hole], 5)))
    second_boards = choices2[np.random.default_rng(0).choice(len(choices2), 6, replace=False)]
    monkeypatch.setattr(bot, "TRAIN_BOARDS_PER_SHOWDOWN", 6)
    bot.on_hand_end(HandResult(5, 0, 8, True, (1, 2, 3, 4, 5), second_hole, "split"), m1)
    official = [omaha_score(hole, row) for row in generated]
    official += [omaha_score(second_hole, row) for row in second_boards]
    assert sorted(m1["rank_pool"]) == sorted(official)
    assert m1["cutoff"] == sorted(official)[3]
    assert m1["cutoff"] != sorted(official)[1]
    assert m1["cutoff"] != sorted(official)[len(official) // 2]
    bot.on_hand_end(HandResult(100, 0, 0, True, (), hole, "opp"), m1)
    assert len(m1["rank_pool"]) == 16


def test_delayed_probe_showdown_trains_after_probe_boundary(monkeypatch):
    monkeypatch.setattr(bot, "PROBE_HANDS", 2)
    monkeypatch.setattr(bot, "TRAIN_BOARDS_PER_SHOWDOWN", 4)
    memory = fresh()
    bot.choose_action(observation(2), memory)
    bot.on_hand_end(HandResult(1, 0, 8, True, (0, 2, 4, 6, 8), (1, 7, 14, 21, 29), "split"), memory)
    assert memory["probe_frozen"]
    assert len(memory["rank_pool"]) == 4


def test_potential_feature_independent_of_candidate_order_and_exact_on_river():
    hands = np.array([[0, 4, 8, 12, 16], [1, 5, 9, 13, 17]], dtype=np.int16)
    board = (20, 21, 22, 23, 24)
    expected = score(hands, np.broadcast_to(board, (2, 5)))
    assert np.array_equal(bot.potential_p25(hands, board), expected)
    prefix = (20, 21, 22)
    whole = bot.potential_p25(hands, prefix, 7)
    rev = bot.potential_p25(hands[::-1], prefix, 7)[::-1]
    assert np.array_equal(whole, rev)


def test_potential_value_cache_separates_runout_counts():
    hands = np.array([[0, 4, 8, 12, 16], [1, 5, 9, 13, 17]], dtype=np.int16)
    board = (20, 21, 22)
    cache = {}
    cached_one = bot.potential_p25(hands, board, 1, cache)
    cached_sixteen = bot.potential_p25(hands, board, 16, cache)
    uncached_sixteen = bot.potential_p25(hands, board, 16)
    assert np.array_equal(cached_one, bot.potential_p25(hands, board, 1))
    assert np.array_equal(cached_sixteen, uncached_sixteen)


def test_empirical_p25_nearest_rank_boundaries_and_axis():
    assert bot.empirical_p25(np.array([9])) == 9
    assert bot.empirical_p25(np.array([50, 10, 40, 20, 30])) == 20
    values = np.array([np.arange(16, 0, -1), np.arange(1, 17)])
    assert bot.empirical_p25(values, axis=1).tolist() == [4, 4]


def test_public_prefix_feature_selects_p25_not_p10_or_median(monkeypatch):
    hand = np.array([[0, 4, 8, 12, 16]], dtype=np.int16)
    def indexed_scores(hands, boards):
        return np.arange(len(hands), dtype=np.int32) * 100
    monkeypatch.setattr(bot, "score", indexed_scores)
    observed = bot.potential_p25(hand, (20, 21, 22, 23), 16)
    assert observed.tolist() == [300]


def test_public_prefix_visibility_is_based_on_action_street():
    obs = observation(street=Street.TURN, board=(20, 21, 22, 23))
    assert bot._visible_prefix(obs.board, Street.PREFLOP) == ()
    assert bot._visible_prefix(obs.board, Street.FLOP) == (20, 21, 22)
    assert bot._visible_prefix(obs.board, Street.TURN) == (20, 21, 22, 23)


def test_range_observation_keeps_legal_cards_and_bounded_broad_fallback(monkeypatch):
    monkeypatch.setattr(bot, "RANGE_CANDIDATES", 8)
    monkeypatch.setattr(bot, "REFRESH_ATTEMPTS", 1)
    memory = fresh()
    memory["cutoff"] = 10**9
    memory["cutoff_version"] += 1
    from game import ObservedAction
    obs = observation(board=(25, 26, 27), street=Street.FLOP,
        actions=(ObservedAction(Street.FLOP, False, Action.RAISE_POT, 5, 5),))
    state = bot._replay_range(obs, memory, memory["cutoff"], 8)
    assert state["broad_fallback"] is True
    assert state["refreshes"] == 1
    assert len(state["hands"]) > 0
    assert not set(obs.hole_cards) & set(state["hands"].flat)
    assert not set(obs.board) & set(state["hands"].flat)


def test_range_filter_uses_one_cutoff_for_observed_raise(monkeypatch):
    monkeypatch.setattr(bot, "RANGE_CANDIDATES", 16)
    monkeypatch.setattr(bot, "potential_p25",
        lambda hands, *args, **kwargs: np.asarray(hands)[:, 0].astype(np.int32))
    from game import ObservedAction
    memory = fresh()
    memory["cutoff"] = 18
    obs = observation(board=(25, 26, 27), street=Street.FLOP,
        actions=(ObservedAction(Street.FLOP, False, Action.RAISE_POT, 5, 5),))
    state = bot._replay_range(obs, memory, memory["cutoff"], 16)
    assert len(state["hands"]) > 0
    assert np.all(state["hands"][:, 0] >= 18)


def test_range_fallback_state_recovers_on_new_public_street(monkeypatch):
    monkeypatch.setattr(bot, "RANGE_CANDIDATES", 8)
    monkeypatch.setattr(bot, "REFRESH_ATTEMPTS", 1)
    calls = {"n": 0}
    def feature(hands, *args, **kwargs):
        calls["n"] += 1
        return np.zeros(len(hands), dtype=np.int32) if calls["n"] <= 2 else np.full(len(hands), 100)
    monkeypatch.setattr(bot, "potential_p25", feature)
    from game import ObservedAction
    memory = fresh()
    memory["cutoff"] = 50
    flop = observation(board=(25, 26, 27), street=Street.FLOP,
        actions=(ObservedAction(Street.FLOP, False, Action.RAISE_POT, 5, 5),))
    first = bot._replay_range(flop, memory, 50, 8)
    assert first["broad_fallback"]
    turn = observation(board=(25, 26, 27, 28), street=Street.TURN,
        actions=(ObservedAction(Street.FLOP, False, Action.RAISE_POT, 5, 5),))
    second = bot._replay_range(turn, memory, 50, 8)
    assert not second["broad_fallback"]


def test_raise_ev_respects_fold_pot_and_call_payout_bounds():
    root = Betting(10, (0, 0), (80, 80), 0, 2, 2, 4)
    raise_action = Action.RAISE_POT
    assert np.all(bot._action_ev(root, raise_action, np.ones(3), np.zeros(3, dtype=bool)) == 20)
    assert np.all(bot._action_ev(root, raise_action, np.zeros(3), np.zeros(3, dtype=bool)) == -10)
    assert np.all(bot._action_ev(root, raise_action, np.zeros(3), np.ones(3, dtype=bool)) == 10)


def test_adaptive_budget_does_not_rise_when_more_decisions_use_same_hand_count(monkeypatch):
    monkeypatch.setattr(bot, "PROBE_HANDS", 0)
    memory = fresh()
    memory["config"] = AgentConfig(num_hands=20, time_bank_seconds=8)
    memory["budget_hand_index"] = 10
    memory["stats"]["postprobe_decisions"] = 20
    first = bot._candidate_budget(memory)
    memory["stats"]["postprobe_decisions"] = 40
    assert bot._candidate_budget(memory) <= first


def test_raise_history_is_filtered_only_when_cutoff_exists():
    from game import ObservedAction
    actions = (ObservedAction(Street.FLOP, False, Action.RAISE_POT, 5, 5),)
    obs = observation(street=Street.TURN, board=(25, 26, 27, 28), actions=actions)
    assert bot._history_raise_prefixes(obs) == [(0, (25, 26, 27))]


def test_simple_http_shell_is_isolated_and_exposes_routes():
    assert bot.app is bot._http.app
    assert bot._http.new_half is bot.new_half
    assert bot._http.choose_action is bot.choose_action
    assert bot._http.on_hand_end is bot.on_hand_end
    assert bot._http.AGENT_NAME == "ProbeP25Cutoff"


def test_new_half_resets_training_and_no_showdown_does_not_change_cutoff():
    memory = fresh()
    hole = (1, 7, 14, 21, 29)
    bot.on_hand_end(HandResult(2, -4, 8, False, (0, 2, 4), None, "opp"), memory)
    assert memory["cutoff"] is None
    assert memory["rank_pool"] == []
    bot.on_hand_end(HandResult(3, 0, 8, True, (0, 2, 4, 6, 8), hole, "split"), memory)
    previous = memory["cutoff"]
    bot.on_hand_end(HandResult(4, 4, 8, False, (0, 2, 4), None, "me"), memory)
    assert memory["cutoff"] == previous
    bot.new_half(AgentConfig(), memory)
    assert memory["cutoff"] is None and memory["rank_pool"] == []
    assert memory["seen_results"] == set()


def test_simple_http_routes_call_probe_callbacks():
    from fastapi.testclient import TestClient
    client = TestClient(bot.app)
    started = client.post("/new-half", json={"config": AgentConfig().to_dict()})
    assert started.status_code == 200
    obs = observation(hand=0)
    response = client.post("/act", json={"observation": obs.to_dict(), "hand_results": []})
    assert response.status_code == 200
    assert response.json()["action"] == "RAISE_POT"


def test_simple_http_can_play_three_postprobe_hands(monkeypatch):
    from fastapi.testclient import TestClient
    from game import Game, make_deal
    monkeypatch.setattr(bot, "PROBE_HANDS", 0)
    monkeypatch.setattr(bot, "RANGE_CANDIDATES", 8)
    monkeypatch.setattr(bot, "PAYOFF_SAMPLES", 8)
    monkeypatch.setattr(bot, "FEATURE_RUNOUTS", 4)
    client = TestClient(bot.app)
    assert client.post("/new-half", json={"config": AgentConfig(num_hands=10).to_dict()}).status_code == 200
    for hand in range(3):
        game = Game(make_deal("http-simple", hand), hand % 2, hand)
        pending = []
        turns = 0
        while not game.is_terminal and turns < 80:
            seat = game.to_act
            if seat == 0:
                obs = game.observation(0)
                reply = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending})
                pending = []
                assert reply.status_code == 200, reply.text
                action = Action[reply.json()["action"]]
                assert action in obs.legal_actions
            else:
                action = Action.CHECK_CALL
            game.apply(action)
            turns += 1
        assert game.is_terminal

