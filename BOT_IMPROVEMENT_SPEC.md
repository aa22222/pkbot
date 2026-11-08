# Poker bot improvements: less wasted computation, better beliefs, better future decisions

**Date:** 2026-11-08  
**Status:** implementation specification, with research ideas separated from committed work  
**Implementation workflow:** Luna implements one stage; the primary agent reviews, tests, compares, and checkpoints it before the next stage.

## 1. What we should optimize

The objective is to make better chip decisions within eight seconds per 1,000-hand half. More simulations are useful only if they answer the right question. A very precise estimate against the wrong opponent range can be worse than a modest estimate against a plausible range.

The recommended direction is:

1. Exploit the exact two-hole-card Omaha rule to avoid repeated evaluation.
2. Keep our inferred opponent range when computation becomes scarce.
3. Spend additional work on consequential decisions, particularly river and turn decisions that can be evaluated accurately.
4. Improve the model of future betting only after we can measure its errors.
5. Move expensive learning and search offline when the runtime benefit justifies it.

Do not add more hand-category thresholds, arbitrary equity bonuses, or a larger collection of finely divided opponent statistics without evidence that they help.

### Current evidence

The starting checkpoint is the code accompanying [the final 12-match validation](baseline-results/20261108-015849-861147/summary.json).

| Opponent | Seed 59: bb/duplicate deal | Seed 71: bb/duplicate deal | Verdicts |
|---|---:|---:|---|
| Always half-pot | +6.203 | +7.035 | Win / win |
| Always pot | +4.044 | +4.675 | Win / win |
| Calling station | +12.680 | +13.284 | Win / win |
| Random | +10.341 | +12.649 | Win / win |
| Previous bot | +1.071 | +2.012 | Draw / win |
| Cloned testpokerbot | -0.538 | -1.537 | Draw / loss |

All 12 completed without judge errors or skipped decisions. The maximum charged time was 6.938 seconds per half. These results establish reliability on that machine and those seeds, not general superiority over the field.

The logs also expose a design problem: against always-half, almost every decision used the cheap path. Against testpokerbot, roughly one third used it. Diagnostics do not currently separate every reserve-mode decision cleanly, so do not treat the printed cheap count as an exact count of uniform-range fallbacks.

### Concrete weaknesses in the current code

- `strategy.fast_decision()` drops the inferred opponent range and sets selectivity to zero. Running low on time therefore changes our belief about the opponent, rather than merely reducing estimation precision.
- `strategy.sample()` draws new holdings and recomputes historical strength information on each decision. The opponent's cards did not change between two raises on the same street.
- Four runouts per candidate holding give a very noisy strength estimate. The cheap path instead uses made-hand strength, which misses draws. These are different approximations, not interchangeable measurements.
- Low effective sample size currently causes blending back toward a uniform range. It confounds insufficient numerical coverage with uncertainty about the opponent's actual behavior.
- The selective action model orders hands into fold, call, half-pot raise, and pot raise. That cannot represent polarized bluff/value betting or strong hands that trap by checking.
- Past-action filtering mixes selective and independent behavior at each action, while showdown learning compares whole sequences under selective versus independent hypotheses. That is an approximate learning scheme, not one fully consistent latent-style model.
- Action values stop after one opponent response and our fold/call choice. They miss future value bets, induced bluffs, future folds, and the option to reraise again.
- The preflop control-variate coefficient is fitted on the same samples it corrects. The resampling trigger uses raw rather than corrected value differences. Both should be measured rather than described as exact uncertainty control.

## 2. Boundaries that every implementation must respect

- Use 36 cards, ranks six through ace.
- Use exactly two hole cards and three board cards.
- Flush beats full house; trips beat straight; A-6-7-8-9 is the lowest straight.
- Keep exact chip accounting, legal raise sizes, four-raise cap, position, and all-in handling.
- Keep `agent.py` in the root and preserve the HTTP contract and `PORT` behavior.
- Reset learned state, caches, and random streams every half. No duplicate-deal memory across halves.
- No runtime network access or persistent writes. The code checkpoint and test tools run outside the match.
- No future card may enter a simulated decision before it is revealed.
- Our simulated response must be chosen over the conditional range. It cannot be chosen separately for each sampled hidden hand.
- An opponent policy may use its hypothetical own cards and public information. It must not see our actual private cards merely because the simulator knows them.
- Keep the reference referee and evaluator unchanged.

## 3. Priorities and implementation order

| Priority | Idea | Why it may help | Principal risk | Status |
|---|---|---|---|---|
| P0 | Better diagnostics and reproducible code checkpoints | Makes subsequent claims auditable | Logging overhead | Implement with Stage A |
| P0 | Board-specific two-card score tables | Reuses the defining Omaha computation | Building tables when there is no reuse | Stage A |
| P0 | Exact river reference evaluator | Provides trustworthy labels and debugging cases | Too expensive to run on every decision | Stage A, offline/test first |
| P0 | Range-preserving reserve mode | Prevents late-half calls against an unjustified random range | Reusing stale or illegally blocked holdings | Stage B |
| P1 | Persistent belief particles with valid rejuvenation | Avoids repeatedly rediscovering the same range | Incorrect proposal weights and fake sample diversity | Later stage after B |
| P1 | River betting solver and exact turn runouts | Addresses meaningful future decisions cheaply relative to full-game search | Hidden-information leakage in branching | Later stage |
| P1 | Consistent opponent-policy ensemble | Represents selective, random, polarized, and trapping behavior | Model overfitting and double-counting evidence | Later stage |
| P2 | Offline-trained continuation-value corrections | Buys future-betting knowledge without runtime rollouts | Learning teacher errors | Research gate |
| P2 | Value-of-computation scheduling | Uses time on decisions where it can change the action | Incorrect uncertainty estimates | Research gate |
| P2 | Controlled probes and robust policy mixtures | Learns useful weaknesses without uncontrolled bluffing | Paying too much for information | Research gate |

**Committed work for this cycle is Stages A and B.** The later sections are concrete follow-on proposals, not permission to implement all speculative machinery in the first patch. Stage A must remain useful if Stage B is rejected.

## 4. Stage A: compute pair scores once per board

### 4.1 The central identity

For a fixed board B, define a score for every legal two-card pair p:

`pair_score_B[p] = max(score_five(p + triple) for triple in combinations(B, 3))`

Then a five-card holding H has Omaha score:

`omaha_score(H, B) = max(pair_score_B[p] for p in combinations(H, 2))`

This is exact. The pair table incorporates the unusual hand rankings automatically through the existing evaluator. The five-card holding requires ten lookups after the table is built.

This does not treat the ten pairs as independent poker hands. It explicitly takes their maximum while preserving all five cards and their removal effects.

### 4.2 Interfaces

Add a module such as `pair_eval.py`:

```python
def build_pair_table(board) -> np.ndarray:
    # Symmetric 36 x 36 int32 table.
    # Legal entries contain exact scores.
    # Diagonal and pairs touching board cards contain an invalid sentinel.

def score_holdings(hands, pair_table) -> np.ndarray:
    # hands: (N, 5), or a single hand promoted to one row.
    # Return the maximum of ten pair lookups per holding.

class BoardPairCache:
    def get(self, board): ...
    def clear(self): ...
```

Reuse `batch_eval.py` rank and suit tables when building pair tables. Do not copy the opponent repository's evaluator or create a second set of poker ranking rules.

Use a sorted board tuple as the cache key; the score depends on the set of currently revealed cards, not their order. Historical action processing must still retain the actual street prefixes: a turn card cannot become part of the flop board simply because a cache key is sorted.

Keep the cache in per-half/per-hand memory. Clear it at hand boundaries or use a small bounded LRU. Three actual public boards per hand are the primary reuse case. Do not fill it with thousands of unique Monte Carlo runouts.

### 4.3 Integrate selectively

Good uses:

- River showdown comparisons against many opponent holdings.
- Cheap-path made-hand rankings on the same public board.
- Repeated decisions after raises on a street.
- Historical board-prefix scoring against a new batch of candidate hands.

Poor default use:

- One unique completed board per Monte Carlo row. Constructing a full pair table per row is likely slower than the current batched evaluator.

Retain the existing batch evaluator for varying-board batches. Benchmark the cost of table construction, not just the lookup loop. If a first-use small batch is slower, use direct scoring on the first access and build on reuse, or select an experimentally justified crossover. The choice is an evaluator scheduling decision, not a strategic hand-strength parameter.

### 4.4 Exact river oracle

With five hero cards and five river board cards known, 26 cards remain possible opponent cards. There are:

- `C(26, 2) = 325` legal opponent pairs.
- `C(26, 5) = 65,780` legal opponent holdings.

Add an **offline/test** oracle:

```python
def exact_uniform_river(hero, board):
    # Return win_count, tie_count, loss_count, total, equity.
    # equity = (win_count + 0.5 * tie_count) / total
```

Enumerate five-card holdings in chunks and score them with the pair table. Do not materialize unbounded intermediates. Do not call this the equity against a tight range: it is exact only for the stated uniform legal-hand distribution.

A local three-board experiment using already-created holding arrays took approximately 0.08–0.17 ms to build the 325-pair table and 5.75–6.12 ms to score all 65,780 holdings. **These figures exclude combination-array construction, server overhead, and startup.** They demonstrate feasibility for an oracle, not that production should run exact enumeration on every river.

Possible future extension: exact weighted river equity by enumerating holdings and applying the same frozen history-likelihood model. It is exact with respect to that specified approximate model, not proof that the model describes the opponent.

### 4.5 A more unusual exact-equity approach: count graph cliques

This is optional research after the enumeration oracle exists.

Make each of the 26 unseen cards a vertex. Connect two vertices when their pair score is strictly below our score. A five-card opponent hand loses precisely when its five vertices form a clique: all ten pairs lose. A second graph uses `pair_score <= hero_score`.

If `L` is the number of size-five cliques in the strict graph and `N` the count in the non-strict graph:

```text
hero wins = L
ties      = N - L
hero loses = C(26, 5) - N
```

Bitset intersections and popcounts may count these efficiently without creating every holding array. Benchmark; do not assume a Python graph routine beats NumPy enumeration. Arbitrary history-dependent hand weights do not reduce to this unweighted clique count.

### 4.6 Stage A acceptance criteria

- Compare pair-table scores to the official evaluator on at least 10,000 generated hands distributed over flop, turn, and river.
- Cover flush-over-full-house, trips-over-straight, wheel straight, ties, and the two-hole-card restriction explicitly.
- Verify table symmetry and rejection/detection of board collisions in the checked interface.
- Exact oracle counts sum to 65,780 and its sampled individual holding scores match the reference.
- Suit relabeling preserves oracle equity and score comparisons.
- Cache clearing prevents cross-half state; board-prefix tests prevent future-card leakage.
- Benchmark fresh-table and repeated-table cases at several batch sizes. Keep the production optimization only where the measured result supports it.
- Existing HTTP and full-match legality tests remain green.
- Do not turn on exact river solving in production in this stage.

## 5. Stage B: preserve beliefs when the time budget gets tight

### 5.1 Why this is more important than another score curve

Suppose earlier bets strongly favor a narrow opponent range. Our current reserve path later forgets those bets and evaluates against random hands. This can produce a cheap but systematically wrong call precisely when the pot is largest.

The replacement rule is:

> Reduce the number of holdings or runouts before discarding the information that made a holding plausible.

Stage B should first solve reserve-mode reuse. It should not simultaneously introduce a new opponent policy, a new sampler, and a new search tree.

### 5.2 Data contract

Retain a compact snapshot after ordinary inference:

```python
@dataclass
class BeliefSnapshot:
    hand_index: int
    hero_cards: tuple[int, ...]
    board: tuple[int, ...]
    opponent_hands: np.ndarray   # (N, 5), legal concrete holdings
    weights: np.ndarray         # normalized posterior weights
    processed_event_ids: tuple
    likelihood_records: tuple   # frozen model context needed for later updates
    # Optional cached preflop and historical fixed-board scores.
```

Keep a consistent, independently testable explanation for each weight. Do not store only a scalar equity: new cards and new bets can change it.

### 5.3 Same board, no new opponent action

Reuse holdings and their weights. Generate fewer legal runouts per holding if needed. On the river, use the pair table; there is no runout randomness left.

Use the existing action-value code with the retained weights. This requires passing the range through the reserve path, not collapsing everything into one unweighted equity value.

### 5.4 New public card

Remove holdings containing the newly revealed card and renormalize surviving weights. Conditional on a particular legal opponent holding, the next card is uniform over its remaining deck; the valid holdings share the same denominator. Filtering collisions therefore performs the required update for the observed board card.

If too few distinct holdings survive, mark the belief as numerically weak. Rebuild a fresh legal proposal batch and replay the full relevant history when the budget permits. **Replace** the old approximation in that refresh; do not append fresh uniform hands to posterior samples and normalize them as though they had the same proposal distribution.

Never use a holding that overlaps hero cards or any currently revealed board card.

### 5.5 New opponent action

Apply its likelihood exactly once using a model snapshot fixed before incorporating that observation into learned frequency counts.

An important prerequisite: a likelihood must be a reproducible function of a candidate hand and public context. Recomputing the definition of “top 30%” on a different surviving particle population changes the target model during filtering.

For Stage B, freeze a reference-strength distribution with each action-likelihood record. Evaluate both existing and newly proposed holdings against that same reference. Preflop can use the precomputed equity distribution. A postflop reference may use a fixed legal public-board sample; record the reference and its construction assumptions. Avoid generating a new random strength score every time the same hand/action pair is revisited.

Keep the distribution used to explain a past action separate from a newly estimated distribution used to predict a future action.

### 5.6 Exhausted numerical support

No observed compatible particles is a sampling failure, not proof that the opponent range is empty.

Fallback order:

1. Refresh a small legal batch and weight it by the frozen history model.
2. If a valid cached current-state estimate exists, use it with explicit uncertainty.
3. Use the existing broad-range emergency path only as a last resort, with a distinct diagnostic reason.

Never force a fold merely because a proposal batch missed a rare range. Never claim a fallback estimate is range-aware if it used the prior instead.

### 5.7 Stage B acceptance criteria

- Same-board reserve decisions retain a synthetic narrow posterior rather than reverting to uniform equity.
- New board cards eliminate all colliding holdings and renormalize correctly.
- New actions are applied once, including delayed terminal calls and folds.
- Resubmitting the same observation does not apply its likelihood again.
- A refresh against a tiny enumerable deck agrees with exact posterior probabilities within its stated sampling error.
- The frozen likelihood of a particular hand does not change when unrelated particles are added, removed, or reordered.
- Rare-range failure exercises the refresh path rather than silently asserting zero equity.
- The normal and reserve paths agree on carefully constructed river states when given the same belief.
- Full halves still complete with no timeouts, skipped decisions, or errors.

## 6. Follow-on idea: maintain a belief instead of repeatedly drawing one

After Stage B, extend the snapshot into a persistent particle filter for the entire hand. A particle is a possible opponent five-card holding with a weight. New cards remove impossible particles; new actions update weights.

The benefit is amortization: a hand can have several decisions, but most historical work only needs to happen once.

When weights concentrate, use systematic resampling followed by legal one-card swaps. A symmetric swap proposal can use a Metropolis acceptance probability:

`accept = min(1, target_likelihood(new_hand) / target_likelihood(old_hand))`

This formula assumes the card proposal is actually symmetric and the hand prior is uniform over legal holdings. Include proposal-density and prior ratios when those assumptions do not hold. A swap without an acceptance correction changes the modeled range.

Store likelihood features deterministically or cache them. Fresh noisy likelihood estimates are not automatically valid inside this acceptance rule.

Track distinct holdings, ancestor diversity, and proposal acceptance in addition to weight ESS. After resampling, equal weights give a deceptively high ESS even if every particle descends from one hand.

Production activation requires an offline tiny-deck posterior test and an ablation showing that persistent inference saves time or reduces decision error. Raw particle count is not the success criterion.

## 7. Follow-on idea: specialize the river and turn

### 7.1 River: solve the actual remaining betting problem

There are no future cards on the river. The action set is tiny, raises are capped, and stacks are shallow. This is a much better place for deeper search than the preflop game.

Implement a public betting tree to the end of the street. At each node:

- Use exact legal amounts and chip accounting.
- Update the opponent range on each observed or hypothetical opponent response.
- At hero nodes, choose one action using the entire conditional belief.
- At opponent nodes, apply the opponent policy per possible opponent holding.
- End at fold, matched river bets, or all-in.

An exact showdown leaf does not make the betting strategy an equilibrium solution. It is a best response to the specified opponent model.

Tests must include hands that profit by inducing a bluff, hands that value-reraise after a raise, and cases where calling is better than raising despite high equity. Compare with exhaustive small-deck trees and verify that no per-hidden-hand action choice has slipped in.

### 7.2 Turn: enumerate rivers before adding noisy full rollouts

With hero's five cards, a candidate opponent's five cards, and four board cards fixed, exactly **22** river cards remain legal. Enumerating these removes runout noise for that candidate holding.

There are **27** publicly unseen cards before conditioning on the opponent's holding. These are not interchangeable counts.

For a public river r and current range weights w(h):

`P(r | belief) = sum_h w(h) * 1[r not in h] / 22`

At the child node, remove holdings containing r and renormalize. Hero chooses a river action once for that public river and conditional range, not independently for each hidden holding.

This directly addresses future foldability and value betting. It is a bounded next-street calculation, not unrestricted Monte Carlo tree search.

Start as an offline oracle and a selectively invoked runtime module. Benchmark the complete 27-public-card branching process and range updates; “22 cards per hand” does not imply trivial tree cost.

## 8. Follow-on idea: a consistent ensemble of opponent policies

The current selective-versus-independent model cannot express several common behaviors. Add a small collection of policy families only after the belief machinery is stable:

- Strength-independent aggression or calling.
- Thin value betting: strong and medium hands bet, weak hands check.
- Polarized betting: strong hands and some weak blocker hands bet; medium hands check/call.
- Passive trapping: some strong hands check and continue against bets.
- Price-sensitive continuation.

Generate a broad offline population within these families; do not assign a special identity to a known GitHub opponent.

Represent uncertainty jointly over cards h and policy hypothesis theta:

`P(h, theta | history) proportional to P(theta) P(h) product_j P_theta(action_j | h, public_state_j)`

Do not switch silently between “one fixed latent style” and “independently redraw a style on every action.” Choose and document one generative model. A fixed policy hypothesis within a hand is the initial recommendation.

Every observed action provides evidence after integrating over hidden holdings, even if no showdown occurs. Revealed cards refine that inference. Showdown-only records are a selected sample; folds must not become fabricated weak-hand labels.

Frequency calibration and model selection must share an explicit update schedule. Do not fit rates to an action and then reuse that same action as an independent likelihood observation without accounting for the update.

Start with fixed offline policy particles and online posterior weights. Add online parameter adjustment only after the fixed-model posterior is calibrated on held-out generated opponents.

### A useful new variable: what the opponent thinks we hold

The opponent's call decision depends on our public range, not just its own hand rank. Track a coarse public hero range from our actions, or use a fixed reference policy to estimate it initially.

Do not substitute our actual private cards into the simulated opponent's equity calculation. The simulator knows those cards; the opponent does not.

This can eventually improve blocker bluffs and thin value bets without category-specific magic numbers. It is more complex than a one-sided range, so it is not part of Stages A or B.

## 9. Follow-on idea: learn the value missing from check-down leaves

The current leaf value is approximately `equity * current_pot`. It ignores all future investments and folds. Rather than perform a large search online, train a small model offline to estimate the correction:

`future_value = check_down_value + learned_residual`

Define future value at a leaf as expected future payout minus **future** investment. Subtract chips already invested in reaching the leaf outside this function. This avoids double-charging a bet.

Suggested inputs:

- Street, position, pot, effective stack-to-pot ratio, and raises remaining.
- Equity against the current range.
- Range concentration and policy uncertainty.
- Distribution of equity across possible next cards, not only its mean.
- Nut potential and vulnerability derived from exact evaluation/sampling.
- Predicted response rates for actual candidate bet sizes.

Begin with a small linear/binned model or tiny CPU model. Model size is not the objective. Train on offline river/turn solver labels first, then on bounded rollout labels from a diverse policy population.

Important controls:

- Split by deal seed, suit-equivalence group, and opponent family. Do not let near-identical canonical states occur in both training and validation.
- Keep all candidate-action labels for a state together in one split.
- Use shared runouts for candidate actions.
- Measure action regret against the teacher, not just value mean-squared error.
- Treat the teacher as an approximate model. Test against opponents outside its training population.
- Preserve exact terminal values. The model must not overwrite known fold payouts or all-in/river showdowns.
- Compare with a zero residual. Drop the model if it cannot beat that baseline at the same runtime budget.

This approach can also train a cheap fallback that preserves the effect of a narrow range. The fallback should be a compressed decision model, not a declaration that the opponent has random cards.

## 10. Follow-on idea: spend time according to decision regret

Equal milliseconds per decision is not optimal. A tiny blind decision and a near-all-in call can have the same estimated equity gap but very different chip consequences.

Use three separate quantities:

1. **Card-sampling uncertainty:** additional legal samples can reduce it.
2. **Opponent-model uncertainty:** more samples from the same wrong model do not remove it.
3. **Continuation-value error:** requires deeper search or a better leaf model.

A small pilot batch estimates the top action values and their paired differences. Spend extra work when plausible changes to those differences can change the action and cost meaningful chips.

Approximate value of computation can be estimated offline from replay cases: how much high-budget decision regret disappears after each extra batch? Use those measurements to allocate online work by street, pot size, range concentration, and model disagreement.

Reuse existing samples when adding a batch. The present implementation replaces the earlier batch when it asks for more work. Avoid paying twice for samples that are then thrown away.

The uncertainty calculation must match the estimator actually used. If a control variate corrects the action means, use corresponding corrected residuals or independent batch estimates when deciding whether another batch is useful. The current raw-difference test does not provide that.

For weighted or resampled particles, do not blindly use `standard deviation / sqrt(number_of_rows)`. Duplicated ancestry and reused samples create dependence. Validate uncertainty estimates with independent replay batches.

## 11. More speculative directions worth testing cheaply

### 11.1 Analytical experts for simple opponents

Against a well-supported always-call or always-pot hypothesis, many branches are deterministic. Offline best-response solvers can produce compact policy/value tables for these models. These are useful training anchors and regression opponents.

Do not hard-switch based on a few raises, and do not assume the opponent's name identifies its strategy. Combine expert recommendations using posterior evidence and retain a robust alternative.

### 11.2 A reusable offline policy library

Train several policies against distinct opponent populations. During a match, select or mix them using observed predictive accuracy and expected value. This can buy adaptation with almost no runtime tree search.

A complete policy must still use only its information set. Choosing a different continuation policy for every hidden-hand sample would recreate strategy fusion.

### 11.3 Controlled information-gathering bets

A bet can reveal whether an opponent is price-sensitive. A small early loss might pay for better decisions later in a 1,000-hand half.

Implement only after we can estimate both immediate chip cost and expected future benefit. Do not add random “exploration” raises without a cost model. Prefer probes when ordinary EVs are already close and substantial uncertainty remains.

### 11.4 Match-aware robustness

Tournament points are based on a confidence interval, so profitable but unnecessarily noisy exploitation can produce extra draws. Evaluate policies by the actual duplicate-match verdict as well as chip EV.

Do not turn this into a rule to fold profitable hands whenever ahead. We do not observe the final paired match score during a half, and blanket risk aversion can destroy the edge. Use offline match-level evaluation to choose the overall policy and the amount of exploitation.

### 11.5 A board-pair graph as a representation

The pair table can be more than an evaluator. It identifies which specific two-card combinations beat us, how many of them our cards block, and how that changes across runouts.

These can become exact, interpretable features for a compact value model. For example, “fraction of legal opponent holdings containing a beating pair” is more useful than simply counting threatening ranks on the board. Pair events are correlated: use enumeration or valid graph counting, not an independence approximation across ten hole-card pairs.

## 12. How to decide which ideas survive

### 12.1 Freeze the experimental setup

Record source checkpoint, asset hashes, Python/NumPy versions, opponent source revisions, seed list, work-budget mode, and whether processes restart between halves.

Existing seeds 7, 19, 31, 43, 59, and 71 have been inspected. They can be regression seeds, but must not be described as untouched final holdouts. Reserve new seeds before looking at candidate results.

Use identical deal seeds when comparing two versions. Also run a fixed-work diagnostic mode so speed-driven differences in sample count do not obscure evaluator equivalence. Production scoring still uses the actual time bank.

### 12.2 Measure action quality separately from match luck

Build an offline decision corpus from reproducible legal game states. Give the production bot only the observation. A high-budget reference evaluator may use the engine to sample hidden cards consistent with that observation; it must not use the actual secret hand as the production decision input.

For each state, compare candidate actions using the same random worlds. Report estimated regret, uncertainty, pot size, street, and model assumptions. A check-down oracle is exact only for its leaf assumptions, not a full-game optimal-play oracle.

### 12.3 Match gates

- Tests and HTTP smoke checks pass.
- No illegal moves, exceptions, forfeits, time-bank exhaustion, or skipped decisions in full matches.
- Report maximum charged half time and internal time separately.
- Run passive opponents as well as aggressive opponents; passive play generates many decisions.
- Compare against the immediate parent checkpoint, the original checkpoint, supplied baselines, and testpokerbot.
- Add at least one unseen generated opponent family before claiming general improvement.
- Use several full duplicate seeds for promotion. One win is not enough to prove a feature useful.

For comparing versions A and B against a fixed opponent, compute per-deal paired differences between their duplicate outcomes. Report the mean difference with uncertainty. Also report per-seed results because opponent learning and shared time budgets create dependence across hands; a simple independent-deal confidence interval is only an approximation for the version comparison. The judge's own interval remains the official tournament verdict.

Do not average confidence-interval endpoints. Do not cherry-pick an opponent subset after inspecting results.

### 12.4 Keep or reject

Keep a correctness-preserving evaluator optimization if it reduces representative cost without changing scores. Keep a strategic change only when the evidence supports its value or explicitly label it experimental. Archive rejected code too, with the rejection reason outside the checkpoint directory.

If a match is too small to establish a gain, say “inconclusive,” not “improved.” Runtime stability can justify a change independently of statistically significant chip gain, but that rationale must be explicit.

## 13. Required diagnostics

Add cheap counters; avoid per-sample logging in production:

- Full, cheap, range-preserving reserve, and broad-range fallback decision counts.
- Pair-table builds, hits, misses, and cache size.
- Samples proposed and retained; effective sample size and distinct holding count.
- Range refreshes, board-collision removals, and repeated-event suppression.
- Time spent sampling, scoring, updating beliefs, and evaluating actions, measured in optional profiling mode.
- Reasons for additional computation and fallback.

Add an opt-in offline decision trace with action means, uncertainty diagnostics, selected action, and public context. Keep traces and reports outside code checkpoints and the submitted image.

## 14. Checkpoints and the Luna implementation loop

### Code-only checkpoints

Create `checkpoints/000_baseline/` before any Stage A strategy edits. Subsequent iterations use monotonically numbered directories such as `001_pair_tables` and `002_range_reserve`.

A checkpoint contains source code, tests, dependency declarations, and necessary execution configuration only. Do not include `.npy` assets, match output, logs, Git internals, bytecode, virtual environments, cloned opponents, or the specification itself.

Store asset hashes, source hashes, parent checkpoint, validation results, status, and rationale separately under `checkpoint-manifests/`. Do not overwrite an existing checkpoint. The large preflop asset may remain shared, but its hash must identify exactly which version was used.

To run a checkpoint, stage its source in an ignored temporary opponent directory and supply the pinned shared asset there. Do not modify the checkpoint to make it runnable. Staging may copy the small asset; it is not part of the code-only checkpoint.

### Per-stage process

1. Primary agent finishes the implementation stage specification and saves the pre-change checkpoint.
2. Hand the stage to a **Luna** agent with explicit allowed files and acceptance criteria.
3. Luna implements and runs focused tests, then reports changed files, known limitations, and exact commands/results.
4. Primary agent reads the code, checks the mathematics and hidden-information boundaries, and independently runs tests.
5. Run benchmarks and duplicate matches sequentially. Do not run CPU-heavy training or profiling concurrently with timed matches.
6. Save a code-only candidate checkpoint and a separate result manifest. Mark accepted, rejected, or experimental.
7. Use the results to select the next bounded stage; hand its specification back to Luna.

If Stage A or B fails its acceptance criteria, repair that stage before adding a new strategic feature. If a research idea fails an ablation, preserve the earlier accepted version rather than accumulating dead experimental branches in the production bot.

## 15. First handoff: exact scope for Luna

### Iteration 1 — Stage A

Implement the pair evaluator/cache, exact uniform river oracle tool, focused correctness/performance tests, optional counters, and checkpoint/staging tools. Integrate fixed-board scoring only where measured reuse supports it. Keep opponent policy, action-value formulas, RNG sampling distribution, and time-bank thresholds otherwise unchanged.

Allowed production edits: `batch_eval.py`, new `pair_eval.py`, narrowly scoped `strategy.py` cache plumbing, and diagnostics. Allowed supporting edits: tools, tests, ignore/configuration files needed to keep checkpoints out of tests and Docker. Do not edit the referee, official evaluator, or external opponents.

Deliver a brief implementation report. Do not claim faster tournament play from a microbenchmark alone. The primary agent will run full comparisons and choose promotion.

### Iteration 2 — Stage B

After Stage A review, implement the belief snapshot and range-preserving reserve path. Freeze likelihood references so reusing or refreshing holdings does not redefine the model. Preserve the current opponent-policy family and one-response action evaluator. Do not add MCMC, a new opponent ensemble, or a deeper betting tree in the same patch.

Allowed production edits: a new `belief.py`, `strategy.py`, and only the likelihood-reference interfaces required in `opponent_model.py`. Add tests specified in Section 5 and diagnostics. Save the candidate code checkpoint even if the primary agent rejects promotion.

After these two iterations, use the observed timing, range-retention tests, and match comparisons to choose between persistent particles and the river betting solver. Do not automatically implement the entire research list.
