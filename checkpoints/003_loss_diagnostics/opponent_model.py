"""Opportunity counts and uncertain selective/independent action policies."""
from dataclasses import dataclass
import numpy as np
from game import Action, Street


@dataclass(frozen=True)
class Event:
    index: int
    street: int
    action: int
    facing: bool
    price: float
    legal: tuple


def history_events(obs, config):
    bets = [config.small_blind if obs.is_button else config.big_blind,
            config.big_blind if obs.is_button else config.small_blind]
    stacks = [config.starting_stack - b for b in bets]
    pot, street, raises = sum(bets), 0, 0
    result = []
    for index, a in enumerate(obs.action_history):
        if int(a.street) != street:
            street, bets, raises = int(a.street), [0, 0], 0
        seat = 0 if a.is_me else 1
        owed = max(0, bets[1 - seat] - bets[seat])
        legal = [1]
        if owed:
            legal.append(0)
        if raises < config.raise_cap and stacks[seat] > owed and stacks[1 - seat] > 0:
            legal.extend((2, 3))
        if not a.is_me:
            result.append(Event(index, street, int(a.action), bool(owed),
                                owed / (pot + owed) if owed else 0., tuple(legal)))
        stacks[seat] -= a.amount
        bets[seat] = a.street_total
        pot += a.amount
        raises += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
    return result


def key(event):
    # Third-pot call price separates up-to-pot bets from larger effective prices.
    # Quarter-pot call price is a half-pot bet; use exact pot-odds boundaries.
    bucket = 0 if event.price <= .25 else 1 if event.price <= 1 / 3 else 2
    return (event.street != 0, event.facing, bucket)


def init_model():
    return dict(global_counts={}, counts={}, selective_evidence=1., independent_evidence=1.)


def frequencies(model, event):
    mask = np.zeros(4)
    mask[list(event.legal)] = 1
    pooled = model['global_counts'].get(event.facing, np.zeros(4)) + mask
    pooled *= mask
    pooled /= pooled.sum()
    local = model['counts'].get(key(event), np.zeros(4))
    # One pooled pseudo-observation per legal action; no action-specific bias.
    values = (local + len(event.legal) * pooled) * mask
    return values / values.sum()


def learn_event(model, event):
    model['global_counts'].setdefault(event.facing, np.zeros(4))[event.action] += 1
    model['counts'].setdefault(key(event), np.zeros(4))[event.action] += 1


def selective_weight(model):
    return model['selective_evidence'] / (model['selective_evidence'] + model['independent_evidence'])


def policy_components(strengths, frequency, reference=None, reference_sorted=False):
    """Assign low holdings to fold, then call, then raises; share all ties.

    Each action's mean probability exactly matches its observed frequency.
    The independent hypothesis uses the same frequencies for every holding.
    """
    ordered = np.sort(strengths if reference is None else reference) if not reference_sorted \
        else np.asarray(reference)
    n = len(ordered)
    lower = np.searchsorted(ordered, strengths, side='left') / n
    upper = np.searchsorted(ordered, strengths, side='right') / n
    # A revealed strength outside a reference tie group gets one rank interval.
    empty = upper == lower
    lower = np.where(empty, np.maximum(0., lower - .5 / n), lower)
    upper = np.where(empty, np.minimum(1., upper + .5 / n), upper)
    boundaries = np.r_[0., np.cumsum(frequency)]
    overlap = np.maximum(0., np.minimum(upper[:, None], boundaries[1:]) -
                         np.maximum(lower[:, None], boundaries[:-1]))
    selective = overlap / (upper - lower)[:, None]
    # One independent pseudo-observation prevents finite-particle hard zeros.
    selective = (n * selective + frequency) / (n + 1)
    return selective, np.broadcast_to(frequency, selective.shape)


def policy(strengths, frequency, selectivity):
    if selectivity == 0:
        return np.broadcast_to(frequency, (len(strengths), 4))
    selective, independent = policy_components(strengths, frequency)
    return selectivity * selective + (1 - selectivity) * independent


def posterior(prior, likelihood):
    weights = prior * likelihood
    total = weights.sum()
    return weights / total if total > 0 else prior.copy()
