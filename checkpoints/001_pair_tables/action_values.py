"""One opponent response, then our fold/call choice or a check-down leaf.

No leaf assumes knowledge of the sampled opponent hand when choosing actions.
All decisions maximize expectations over the whole conditional range.
"""
from dataclasses import dataclass, replace
import numpy as np
from game import Action
from opponent_model import Event, frequencies, policy


@dataclass(frozen=True)
class Betting:
    pot: int
    bets: tuple
    stacks: tuple
    raises: int
    increment: int
    big_blind: int
    cap: int

    def owed(self, seat):
        return max(0, self.bets[1 - seat] - self.bets[seat])

    def legal(self, seat):
        result = [1]
        if self.owed(seat):
            result.append(0)
        if self.raises < self.cap and self.stacks[seat] > self.owed(seat) and self.stacks[1-seat]:
            result.extend((2, 3))
        return tuple(result)

    def apply(self, seat, action):
        if action == 0:
            return self, 0
        owed = self.owed(seat)
        if action == 1:
            amount = min(owed, self.stacks[seat])
            increment = self.increment
        else:
            max_to = self.bets[1-seat] + self.pot + owed
            target = self.bets[1-seat] + ((self.pot + owed) // 2 if action == 2 else self.pot + owed)
            minimum = self.bets[1-seat] + max(self.increment, self.big_blind)
            target = min(max(target, minimum), max_to, self.bets[seat] + self.stacks[seat])
            amount = target - self.bets[seat]
            increment = target - self.bets[1-seat]
        bets, stacks = list(self.bets), list(self.stacks)
        bets[seat] += amount
        stacks[seat] -= amount
        return replace(self, pot=self.pot + amount, bets=tuple(bets), stacks=tuple(stacks),
                       raises=self.raises + (action in (2, 3)), increment=increment), amount


def state_from(obs, config):
    return Betting(obs.pot, (obs.my_bet, obs.opp_bet), (obs.my_stack, obs.opp_stack),
                   obs.raises_this_street, obs.min_raise_to - obs.opp_bet,
                   obs.big_blind, config.raise_cap)


def response_event(state, street, index=-1, action=1):
    owed = state.owed(1)
    return Event(index, int(street), action, bool(owed),
                 owed / (state.pot + owed) if owed else 0., state.legal(1))


def values(obs, config, wins, strengths, weights, model, selectivity):
    root = state_from(obs, config)
    result = {}
    equivalent = {}
    opponent_acted = any(not a.is_me and a.street == obs.street for a in obs.action_history)
    for action in obs.legal_actions:
        if action == Action.FOLD:
            result[action] = np.zeros(len(wins))
            continue
        state, cost = root.apply(0, int(action))
        signature = (state, cost, action == Action.CHECK_CALL)
        if signature in equivalent:
            result[action] = equivalent[signature]
            continue
        closes = (action == Action.CHECK_CALL and opponent_acted) or not state.stacks[1]
        if closes:
            result[action] = wins * state.pot - cost
            equivalent[signature] = result[action]
            continue
        event = response_event(state, obs.street)
        probabilities = policy(strengths, frequencies(model, event), selectivity)
        outcome = np.zeros(len(wins))
        for response in event.legal:
            p = probabilities[:, response]
            if response == 0:
                leaf = np.full(len(wins), root.pot, dtype=float)
            else:
                after, _ = state.apply(1, response)
                if response == 1:
                    leaf = wins * after.pot - cost
                else:
                    final, call = after.apply(0, 1)
                    calling = wins * final.pot - cost - call
                    folding = np.full(len(wins), -cost, dtype=float)
                    # Select one response using the whole posterior, not each hidden hand.
                    leaf = calling if np.dot(weights * p, calling - folding) > 0 else folding
            outcome += p * leaf
        result[action] = outcome
        equivalent[signature] = outcome
    return result
