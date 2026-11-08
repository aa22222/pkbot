"""Exact fixed-board Omaha pair scoring and an offline uniform river oracle."""
from collections import OrderedDict
from itertools import combinations
import math

import numpy as np

import batch_eval

INVALID = -1
NUM_CARDS = 36


def _cards(values, lengths, name):
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1 or len(arr) not in lengths:
        raise ValueError(f"{name} must contain {', '.join(map(str, lengths))} cards")
    if np.any((arr < 0) | (arr >= NUM_CARDS)) or len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} must contain distinct card ids in 0..35")
    return arr


def build_pair_table(board) -> np.ndarray:
    """Return symmetric int32 scores for legal hole pairs on a 3--5 card board."""
    board = _cards(board, (3, 4, 5), "board")
    table = np.full((NUM_CARDS, NUM_CARDS), INVALID, dtype=np.int32)
    board_set = set(map(int, board))
    pair_ids = np.asarray(list(combinations([c for c in range(NUM_CARDS) if c not in board_set], 2)), dtype=np.int64)
    pair_keys, pair_suits = batch_eval.parts(pair_ids, np.array([[0, 1]]), -1)
    pair_keys, pair_suits = pair_keys[:, 0], pair_suits[:, 0]
    board_keys, board_suits = batch_eval.parts(board, batch_eval.TRIPLES[len(board)], -2)
    keys = pair_keys[:, None] + board_keys[None, :]
    flush = pair_suits[:, None] == board_suits[None, :]
    vals = np.where(flush, batch_eval.FLUSH[keys], batch_eval.PLAIN[keys])
    best = vals.max(axis=1)
    table[pair_ids[:, 0], pair_ids[:, 1]] = best
    table[pair_ids[:, 1], pair_ids[:, 0]] = best
    return table


def score_holdings(hands, pair_table) -> np.ndarray:
    """Score (N,5) Omaha holdings from ten pair-table lookups each."""
    hands = np.asarray(hands, dtype=np.int64)
    single = hands.ndim == 1
    hands = np.atleast_2d(hands)
    if hands.ndim != 2 or hands.shape[1] != 5 or np.any((hands < 0) | (hands >= NUM_CARDS)):
        raise ValueError("hands must have shape (N,5) with card ids in 0..35")
    if np.any(np.diff(np.sort(hands, axis=1), axis=1) == 0):
        raise ValueError("each holding must contain five distinct cards")
    table = np.asarray(pair_table)
    if table.shape != (NUM_CARDS, NUM_CARDS):
        raise ValueError("pair_table must have shape (36,36)")
    pairs = np.asarray(list(combinations(range(5), 2)))
    scores = table[hands[:, pairs[:, 0]], hands[:, pairs[:, 1]]]
    if np.any(scores < 0):
        raise ValueError("holding contains a board card or invalid pair-table entry")
    result = scores.max(axis=1)
    return result[0] if single else result


class BoardPairCache:
    """Small LRU cache keyed by the set of cards in a public board."""
    def __init__(self, max_size=4):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._tables = OrderedDict()
        self.builds = self.hits = self.misses = 0

    def get(self, board):
        board = _cards(board, (3, 4, 5), "board")
        key = tuple(sorted(map(int, board)))
        if key in self._tables:
            self.hits += 1
            self._tables.move_to_end(key)
            return self._tables[key]
        self.misses += 1
        table = build_pair_table(board)
        self.builds += 1
        self._tables[key] = table
        if len(self._tables) > self.max_size:
            self._tables.popitem(last=False)
        return table

    def clear(self, reset_stats=True):
        self._tables.clear()
        if reset_stats:
            self.builds = self.hits = self.misses = 0

    @property
    def size(self):
        return len(self._tables)


def exact_uniform_river(hero, board, chunk_size=4096):
    """Enumerate all legal uniform five-card opponent holdings on the river.

    Returns (win_count, tie_count, loss_count, total, equity). This is not a
    tight-range estimate; opponent holdings are uniform over legal combinations.
    """
    hero = _cards(hero, (5,), "hero")
    board = _cards(board, (5,), "board")
    if set(hero) & set(board):
        raise ValueError("hero and board cards must not overlap")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    table = build_pair_table(board)
    hero_score = int(score_holdings(hero, table))
    deck = [c for c in range(NUM_CARDS) if c not in set(hero) | set(board)]
    wins = ties = losses = 0
    batch = []
    def consume(rows):
        nonlocal wins, ties, losses
        values = score_holdings(np.asarray(rows, dtype=np.int64), table)
        wins += int(np.count_nonzero(values < hero_score))
        ties += int(np.count_nonzero(values == hero_score))
        losses += int(np.count_nonzero(values > hero_score))
    for hand in combinations(deck, 5):
        batch.append(hand)
        if len(batch) == chunk_size:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)
    total = math.comb(26, 5)
    assert wins + ties + losses == total
    return wins, ties, losses, total, (wins + .5 * ties) / total
