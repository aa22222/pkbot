from itertools import combinations

import numpy as np
import pytest

from evaluator import (FLUSH, FULL_HOUSE, STRAIGHT, THREE_OF_A_KIND,
                       category_of, omaha_score, score_five)
from pair_eval import BoardPairCache, build_pair_table, exact_uniform_river, score_holdings


def test_pair_table_symmetry_legality_and_collision_detection():
    board = [0, 5, 11, 18, 27]
    table = build_pair_table(board)
    assert table.dtype == np.int32
    assert np.array_equal(table, table.T)
    assert np.all(np.diag(table) == -1)
    assert np.all(table[board] == -1)
    with pytest.raises(ValueError, match="invalid pair-table"):
        score_holdings([0, 1, 2, 3, 4], table)
    with pytest.raises(ValueError, match="distinct"):
        build_pair_table([0, 0, 1])


@pytest.mark.parametrize("board", [
    [0, 5, 11], [0, 5, 11, 18], [0, 5, 11, 18, 27],
])
def test_pair_scores_match_official_evaluator_on_generated_holdings(board):
    rng = np.random.default_rng(39281 + len(board))
    pool = np.array([c for c in range(36) if c not in board])
    hands = np.array([rng.choice(pool, 5, replace=False) for _ in range(3334)])
    table = build_pair_table(board)
    got = score_holdings(hands, table)
    expected = np.array([max(score_five([*(h[i] for i in pair), *(board[i] for i in triple)])
                            for pair in combinations(range(5), 2)
                            for triple in combinations(range(len(board)), 3))
                         for h in hands])
    assert np.array_equal(got, expected)


def test_pair_lookup_equals_official_on_full_board_and_two_hole_rule():
    board = [1, 7, 14, 22, 31]
    hand = [0, 4, 12, 20, 35]
    table = build_pair_table(board)
    assert score_holdings(hand, table) == omaha_score(hand, board)
    # At least one legal holding whose tempting three-hole/ two-board result is
    # stronger than every exactly-two-hole/three-board result.
    rng = np.random.default_rng(991)
    pool = [c for c in range(36) if c not in set(board)]
    for _ in range(200):
        h = rng.choice(pool, 5, replace=False).tolist()
        exact = omaha_score(h, board)
        unrestricted = max(score_five([*(h[i] for i in p), *(board[i] for i in t)])
                           for p in combinations(range(5), 3)
                           for t in combinations(range(5), 2))
        if unrestricted > exact:
            assert score_holdings(h, table) < unrestricted
            break
    else:
        pytest.fail("could not construct explicit two-hole-card restriction case")


def test_short_deck_rule_edges_and_tie_scores():
    # Flush-over-full-house on the same board, using exact Omaha pair choices.
    board = [0, 1, 4, 5, 8]
    hero = [16, 24, 10, 18, 27]
    villain = [2, 6, 12, 20, 28]
    pair_table = build_pair_table(board)
    hero_score = score_holdings(hero, pair_table)
    villain_pair_score = pair_table[2, 6]
    assert category_of(hero_score) == FLUSH
    assert category_of(villain_pair_score) == FULL_HOUSE
    assert hero_score > villain_pair_score

    # Trips beat a five-card straight in short-deck ranking.
    straight_board = [0, 5, 10, 15, 16]
    trips_hand = [2, 3, 20, 24, 28]
    straight_table = build_pair_table(straight_board)
    trips_score = score_holdings(trips_hand, straight_table)
    straight_score = score_five([2, 17, 5, 10, 15])
    assert category_of(trips_score) == THREE_OF_A_KIND
    assert category_of(trips_score) > category_of(straight_score) == STRAIGHT

    wheel = score_five([32, 1, 6, 11, 12])  # A-6-7-8-9
    six_to_ten = score_five([0, 5, 10, 15, 16])
    assert category_of(wheel) == category_of(six_to_ten) == STRAIGHT
    assert wheel < six_to_ten
    assert score_five([0, 5, 10, 15, 20]) == score_five([1, 4, 11, 14, 21])


def test_cache_uses_order_independent_key_and_clear_resets_counters():
    cache = BoardPairCache()
    first = cache.get([1, 3, 5, 7, 9])
    assert cache.get([9, 7, 5, 3, 1]) is first
    assert (cache.builds, cache.hits, cache.misses, cache.size) == (1, 1, 1, 1)
    cache.clear()
    assert (cache.builds, cache.hits, cache.misses, cache.size) == (0, 0, 0, 0)


def test_exact_uniform_river_oracle_counts_all_legal_holdings_and_is_suit_invariant():
    hero = [0, 5, 10, 16, 21]
    board = [2, 7, 12, 19, 25]
    result = exact_uniform_river(hero, board, chunk_size=2048)
    assert sum(result[:3]) == result[3] == 65780
    assert result[4] == (result[0] + .5 * result[1]) / result[3]
    suit_map = {s: (s + 1) % 4 for s in range(4)}
    relabel = lambda cards: [(c // 4) * 4 + suit_map[c % 4] for c in cards]
    changed = exact_uniform_river(relabel(hero), relabel(board), chunk_size=2048)
    assert changed == pytest.approx(result)
