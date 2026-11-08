"""Persistent legal opponent-hand beliefs and frozen history likelihoods.

Historical filtering uses deterministic features: preflop equity from the
offline table, and best made-hand score on the public board prefix for each
postflop event. This intentionally differs from the normal decision path's
four-runout strength estimate. The event reference sample is deterministic,
public-board-only, and never excludes hero cards.
"""
from dataclasses import dataclass, field

import numpy as np

from batch_eval import draw, hand_index, score
from opponent_model import Event, policy_components


REFERENCE_HANDS = 256
_REFERENCE_SEED = 0x51A7E


@dataclass(frozen=True)
class LikelihoodRecord:
    event: Event
    frequency: tuple[float, float, float, float]
    mixture: float
    board: tuple[int, ...]
    feature_kind: str
    reference: tuple[float, ...] | None


def event_board(board, street):
    """Return only the public prefix visible when this event occurred."""
    visible = 0 if int(street) == 0 else int(street) + 2
    prefix = tuple(map(int, board[:visible]))
    if len(prefix) != visible:
        raise ValueError("observation does not contain the event's public board prefix")
    return prefix


def _reference_seed(board):
    value = _REFERENCE_SEED
    for card in sorted(board):
        value = ((value * 1664525) + card + 1013904223) & 0xFFFFFFFF
    return value


def capture_record(event, frequency, mixture, board, reference_cache=None):
    """Freeze model context before learning the newly observed event."""
    frequency = np.asarray(frequency, dtype=float)
    if frequency.shape != (4,) or np.any(frequency < 0) or not np.isclose(frequency.sum(), 1):
        raise ValueError("frequency must be a normalized four-action distribution")
    public = event_board(board, event.street)
    if not public:
        return LikelihoodRecord(event, tuple(map(float, frequency)), float(mixture), (),
                                "preflop_equity", None)
    # The reference depends only on public cards. Candidate holdings later
    # exclude hero cards, but the policy's rank partition cannot see them.
    if reference_cache is not None and public in reference_cache:
        reference = reference_cache[public]
    else:
        sample = draw(np.random.default_rng(_reference_seed(public)), np.asarray(public),
                      5, REFERENCE_HANDS)
        reference = tuple(sorted(map(float, score(sample, public))))
        if reference_cache is not None:
            reference_cache[public] = reference
    return LikelihoodRecord(event, tuple(map(float, frequency)), float(mixture), public,
                            "made_hand_score", tuple(reference))


def feature_strength(hands, record, preflop_table):
    hands = np.atleast_2d(np.asarray(hands, dtype=np.int64))
    if not record.board:
        return np.asarray(preflop_table[hand_index(hands)], dtype=float)
    return np.asarray(score(hands, record.board), dtype=float)


def record_components(hands, record, preflop_table, preflop_reference=None):
    strengths = feature_strength(hands, record, preflop_table)
    if not record.board and preflop_reference is None:
        raise ValueError("preflop_reference is required for frozen preflop likelihoods")
    reference = preflop_reference if not record.board else np.asarray(record.reference, dtype=float)
    selective, independent = policy_components(
        strengths, np.asarray(record.frequency), reference=reference, reference_sorted=True)
    return selective, independent


def record_likelihood(hands, record, preflop_table, preflop_reference=None):
    selective, independent = record_components(hands, record, preflop_table, preflop_reference)
    action = int(record.event.action)
    result = record.mixture * selective[:, action] + (1 - record.mixture) * independent[:, action]
    return np.maximum(result, np.finfo(float).tiny)


def selective_action_likelihood(hands, record, preflop_table, preflop_reference=None):
    selective, _ = record_components(hands, record, preflop_table, preflop_reference)
    return np.maximum(selective[:, int(record.event.action)], np.finfo(float).tiny)


def normalize(weights):
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if not len(weights) or not np.isfinite(total) or total <= 0:
        return np.zeros_like(weights)
    return weights / total


@dataclass
class BeliefSnapshot:
    hand_index: int
    hero_cards: tuple[int, ...]
    board: tuple[int, ...]
    opponent_hands: np.ndarray
    weights: np.ndarray
    processed_event_ids: set[int] = field(default_factory=set)
    likelihood_records: list[LikelihoodRecord] = field(default_factory=list)
    revision: int = 0

    @classmethod
    def from_proposal(cls, hand_index, hero_cards, board, hands, records,
                      preflop_table, preflop_reference=None):
        hands = np.asarray(hands, dtype=np.int64).reshape(-1, 5).copy()
        legal = np.ones(len(hands), dtype=bool)
        blocked = set(hero_cards) | set(board)
        for row, hand in enumerate(hands):
            legal[row] = len(set(map(int, hand))) == 5 and not (set(map(int, hand)) & blocked)
        hands = hands[legal]
        if len(hands):
            # Uniform proposals may contain duplicate rows. Keep them as equal
            # proposal mass; they are not extra distinct posterior support.
            weights = np.full(len(hands), 1 / len(hands), dtype=float)
        else:
            weights = np.zeros(0, dtype=float)
        snapshot = cls(int(hand_index), tuple(map(int, hero_cards)), tuple(map(int, board)),
                       hands, weights)
        for record in records:
            snapshot.apply_record(record, preflop_table, preflop_reference)
        return snapshot

    def filter_board(self, board):
        board = tuple(map(int, board))
        if board[:len(self.board)] != self.board:
            raise ValueError("public board must extend the saved street prefix")
        added = set(board) - set(self.board)
        if not added:
            return 0
        keep = np.array([not (set(map(int, hand)) & added) for hand in self.opponent_hands],
                        dtype=bool)
        removed = int(len(keep) - np.count_nonzero(keep))
        self.opponent_hands = self.opponent_hands[keep]
        self.weights = normalize(self.weights[keep])
        self.board = board
        self.revision += 1
        return removed

    def apply_record(self, record, preflop_table, preflop_reference=None):
        event_id = int(record.event.index)
        if event_id in self.processed_event_ids:
            return False
        self.processed_event_ids.add(event_id)
        self.likelihood_records.append(record)
        if not len(self.opponent_hands):
            self.revision += 1
            return True
        likelihood = record_likelihood(self.opponent_hands, record, preflop_table,
                                       preflop_reference)
        self.weights = normalize(self.weights * likelihood)
        self.revision += 1
        return True

    @property
    def effective_sample_size(self):
        return float(1 / np.dot(self.weights, self.weights)) if len(self.weights) and np.any(self.weights) else 0.

    @property
    def distinct_count(self):
        return len(np.unique(np.sort(self.opponent_hands, axis=1), axis=0)) if len(self.opponent_hands) else 0


def decision_weights(snapshot, has_history):
    """Apply the old finite-particle reliability blend only to decision use."""
    raw = normalize(snapshot.weights)
    if not has_history or not len(raw):
        return raw
    ess = 1 / np.dot(raw, raw)
    reliability = ess / (ess + np.sqrt(len(raw)))
    return reliability * raw + (1 - reliability) / len(raw)


def resample_indices(weights, count, rng):
    weights = normalize(weights)
    if count <= 0:
        raise ValueError("count must be positive")
    if not len(weights) or not np.any(weights):
        raise ValueError("cannot resample an empty belief")
    return rng.choice(len(weights), size=count, replace=True, p=weights)
