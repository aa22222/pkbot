"""Batched exact short-deck Omaha scores, checked against evaluator.py.

Rank multisets use base-five digits (at most four cards of any rank).
Suit matching selects the flush lookup. No strategy heuristics live here.
"""
from itertools import combinations, combinations_with_replacement
from math import comb
import numpy as np
from evaluator import score_five

RANK_KEYS = np.array([5 ** (c // 4) for c in range(36)], dtype=np.int32)
PLAIN = np.zeros(5 ** 9, dtype=np.int32)
FLUSH = np.zeros_like(PLAIN)
for ranks in combinations_with_replacement(range(9), 5):
    counts = {r: ranks.count(r) for r in ranks}
    if max(counts.values()) > 4:
        continue
    used = {}
    cards = []
    for r in ranks:
        suit = used.get(r, 0)
        cards.append(4 * r + suit)
        used[r] = suit + 1
    if len(counts) == 5:
        cards[0] += 1  # nonflush representative
    key = sum(5 ** r for r in ranks)
    PLAIN[key] = score_five(cards)
    if len(counts) == 5:
        FLUSH[key] = score_five([4 * r for r in ranks])

PAIRS = np.array(list(combinations(range(5), 2)))
TRIPLES = {n: np.array(list(combinations(range(n), 3))) for n in (3, 4, 5)}
CHOOSE = np.array([[comb(n, k) for k in range(6)] for n in range(37)])


def hand_index(hands):
    hands = np.sort(np.asarray(hands), axis=-1)
    return CHOOSE[hands, np.arange(1, 6)].sum(axis=-1)


def parts(cards, indices, mismatch):
    selected = np.asarray(cards)[..., indices]
    keys = RANK_KEYS[selected].sum(axis=-1)
    suits = selected % 4
    same = (suits == suits[..., :1]).all(axis=-1)
    return keys, np.where(same, suits[..., 0], mismatch)


def score(hands, boards):
    """Broadcast rows of five hole cards against rows of 3--5 board cards."""
    hands, boards = np.atleast_2d(hands), np.atleast_2d(boards)
    hk, hs = parts(hands, PAIRS, -1)
    bk, bs = parts(boards, TRIPLES[boards.shape[-1]], -2)
    keys = hk[..., :, None] + bk[..., None, :]
    flush = hs[..., :, None] == bs[..., None, :]
    values = np.where(flush, FLUSH[keys], PLAIN[keys])
    return values.max(axis=(-2, -1))


def draw(rng, excluded, count, rows):
    """Uniform ordered draws without replacement, per row."""
    excluded = np.broadcast_to(np.atleast_2d(excluded), (rows, len(np.atleast_2d(excluded)[0])))
    keys = rng.random((rows, 36))
    np.put_along_axis(keys, excluded, 2., axis=1)
    return np.argsort(keys, axis=1)[:, :count]
