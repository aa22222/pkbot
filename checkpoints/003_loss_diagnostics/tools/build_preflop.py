"""Reproducible offline equity table. Run once; never during server startup."""
import argparse
from itertools import combinations, permutations
from pathlib import Path
import sys
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batch_eval import draw, score, hand_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=64)
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    start = time.perf_counter()
    hands = np.array(list(combinations(range(36), 5)), dtype=np.int16)
    # Pool identical rank/suit structures; all 24 suit relabelings are equivalent.
    canon = np.full(len(hands), np.iinfo(np.int64).max)
    for perm in permutations(range(4)):
        mapped = np.sort((hands // 4) * 4 + np.array(perm)[hands % 4], axis=1)
        code = (mapped * (36 ** np.arange(4, -1, -1))).sum(axis=1)
        canon = np.minimum(canon, code)
    _, classes = np.unique(canon, return_inverse=True)
    values = np.empty(len(hands))
    rng = np.random.default_rng(20261108)
    for lo in range(0, len(hands), 128):
        h = hands[lo:lo + 128]
        own = np.repeat(h, args.samples, axis=0)
        cards = draw(rng, own, 10, len(own))
        a, b = score(own, cards[:, 5:]), score(cards[:, :5], cards[:, 5:])
        values[lo:lo + len(h)] = ((a > b) + .5 * (a == b)).reshape(len(h), -1).mean(axis=1)
        if lo % 32768 == 0:
            print(f'{lo}/{len(hands)} hands', flush=True)
    pooled = np.bincount(classes, weights=values) / np.bincount(classes)
    result = np.empty(len(hands), dtype=np.float32)
    result[hand_index(hands)] = pooled[classes]
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    np.save(dest / 'preflop.npy', result)
    (dest / 'preflop.json').write_text(json.dumps(dict(samples_per_hand=args.samples,
        seed=20261108, hands=len(hands), suit_classes=len(pooled),
        seconds=time.perf_counter() - start), indent=2))
    print(f'Saved {dest / "preflop.npy"}', flush=True)


if __name__ == '__main__':
    main()
