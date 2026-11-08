"""Microbenchmark direct batch scoring versus fresh/reused fixed-board tables."""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_eval import score
from pair_eval import BoardPairCache, score_holdings


def elapsed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    board = np.array([1, 8, 15, 22, 30])
    rng = np.random.default_rng(6221)
    deck = np.array([c for c in range(36) if c not in board])
    print("rows,direct_ms,fresh_table_and_score_ms,reused_table_score_ms")
    for n in (8, 16, 32, 48, 64, 128, 512, 4096):
        hands = np.array([rng.choice(deck, 5, replace=False) for _ in range(n)])
        direct = elapsed(lambda: score(hands, board), args.repeats)
        fresh = elapsed(lambda: score_holdings(hands, BoardPairCache().get(board)), args.repeats)
        cache = BoardPairCache()
        table = cache.get(board)
        reused = elapsed(lambda: score_holdings(hands, table), args.repeats)
        print(f"{n},{direct:.6f},{fresh:.6f},{reused:.6f}")


if __name__ == "__main__":
    main()
