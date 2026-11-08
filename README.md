# Short Deck PLO5 Bot Tournament: Participant Bundle

## Current bot implementation

`agent.py` delegates to `strategy.py`. The strategy uses:

- `batch_eval.py`: NumPy evaluation with the official short-deck categories and exact 2+3 Omaha construction.
- `pair_eval.py`: exact fixed-board pair tables, a small board cache, and an offline uniform river oracle over all 65,780 legal opponent holdings. Runtime table construction is restricted to batches large enough to justify its setup cost.
- `data/preflop.npy`: offline equity estimates for all 376,992 starting hands, pooled over suit equivalence. Rebuild with `python tools/build_preflop.py --samples 64`; this is an offline task, not server startup work.
- `opponent_model.py`: opportunity-based fold/call/raise counts, with preflop/postflop and price contexts backed by pooled counts. Both an independent-action hypothesis and a strength-selective hypothesis remain possible. Showdowns update their mixture using a correction for the observed action sequence.
- `belief.py`: retains concrete opponent holdings and raw posterior weights through the hand. Each past action has frozen model context; new board cards remove impossible holdings. Reserve mode samples from the retained range, and refreshes replay the frozen history on a new uniform proposal.
- `action_values.py`: compare legal actions using shared Monte Carlo outcomes, including one opponent response and our fold/call response to a reraise. Response decisions use the whole conditional range, never the particular hidden hand in a sample.

The model conditions calling equity on observed betting history. Exact legal raise amounts, caps and all-ins come from a public-state betting implementation tested against `game.py`. Sampling and all learned state reset between halves.

Important limits: later streets are check-down leaves, not a complete poker solver. Historical likelihoods use deterministic preflop equity or postflop made-hand scores; current action predictions can use four independent runouts. This means historical filtering does not explicitly rank draws. The posterior has finite holding coverage, and decision weights still include the existing low-sample reliability blend. Emergency mode retains the range and reduces nonriver evaluations to at most 16 sampled holdings; rivers use all retained holdings. An extra nonriver estimate redraws runouts for the existing holdings rather than doubling independent holding coverage. Action frequencies have broad price buckets rather than a continuous price model.

Run `python -m pytest -q` for evaluator, opponent-model, betting-accounting and server tests. Run `python tools/play_baselines.py --deals 1000 --seeds 31 43` for duplicate matches. The runner also discovers local clones under `external-bots/*/agent.py`; `--bots testpokerbot` selects one. Set `BOT_DIAGNOSTICS=1` to write sample counts, effective sample sizes and internal time to each agent log. Match JSON files and server logs go under `baseline-results/`.

The previous implementation is preserved locally at `external-bots/previous_bot` for direct comparison. External clones and match artifacts are excluded from the submission image. Docker validation still requires Docker via `tools/check_build.sh`.

`simple_agent.py` is a separate experimental probe bot. During the first 100 hands of each half it chooses the largest legal raise, otherwise calls/checks. For each opponent holding revealed in those hands, it simulates 256 boards from preflop and pools the resulting final hand ranks into one cutoff: the empirical 25th percentile. It uses the observed nearest-rank score at zero-based index `ceil(0.25*n)-1` (clamped to 0), without interpolating packed rank integers; with 16 observations this is the fourth-lowest score. Later, it compares candidate holdings' p25 possible final ranks with that same cutoff on each observed raising street. It uses 16 board completions per candidate feature, exact river ranks, and an adaptive number of candidate holdings. Action values use independent payoff boards, with the opponent folding below the cutoff and calling above it; subsequent streets are check-down estimates. No revealed probe holdings means a uniform range with an all-call response model.

Run the experiment against the bundled baselines and cloned bot with:

```powershell
python tools/play_baselines.py --agent simple_agent.py --deals 1000 --seeds 137 149 --bots baseline_always_half baseline_always_pot baseline_calling_station baseline_random testpokerbot
```

The runner saves per-half cutoff and sampling diagnostics in each `agent.log`. The default runner and tournament entry point still use `agent.py`.

The [improvement specification](BOT_IMPROVEMENT_SPEC.md) explains the next stages and their acceptance criteria. Luna implements each stage; primary review, tests, and duplicate comparisons determine whether it is kept. Immutable code-only snapshots live in `checkpoints/`, with hashes and results separately in `checkpoint-manifests/`. To run one, use `python tools/checkpoints.py stage 001_pair_tables --destination external-bots/checkpoint_example`; staging verifies the source and shared preflop asset hashes. Checkpoints and match reports are excluded from the submitted image.

Your bot is **one file, `agent.py`**. It is a complete FastAPI HTTP web server. You edit one marked section, and the tournament runs the file as-is.

## Files
| File | What it is |
|---|---|
| `agent.py` | **Your bot** with starter code. Your agent's file name **MUST BE** agent.py and it **MUST BE** in the root of your repo (i.e. don't move it) |
| `game.py` | The rules and game state (editing this has no effect on the evaluation platform's match referee's game). |
| `evaluator.py` | Cards and hand ranking helper tool (same as above, but its helpers are useful). |
| `judge_engine.py` | A local judge that referees a match between two running bots and prints the result. |
| `baseline-bots/` | Directory of dummy practice opponents, built from the same template as `agent.py`. |
| `test_agent.py` | Tests for your bot. Add your own at the bottom. |
| `requirements.txt` | Add all dependencies your bot requires here. Keep in mind that your Docker image will be built with this. Also must be in the root of your repo (don't move it). |
| `requirements-dev.txt` | Dependencies for local testing. |
| `.dockerignore` | Files left out of your bot's Docker image. The image size is limited, so list big files your bot doesn't need while playing. |
| `docker/` | The Dockerfiles the tournament builds your bot with (don't edit them; edits have no effect on the real build). |
| `tools/check_build.sh` | Builds your bot exactly like the tournament does and smoke-tests it. **Run this before you submit.** |
| `tools/smoke_test.py` | The quick "does this bot work?" check that `check_build.sh` (and the tournament build) runs inside your image. |

## 1. The game in one paragraph
The variant of poker that your agent must play is *Heads-Up Short Deck 5-card Pot-Limit Omaha*.

* Heads-up: 1v1 Poker against other Datathon participant agents in a tournament.
* Short Deck: 36-card deck (6 through A). Two short-deck rules: a **flush beats a full house**, and **three of a kind beats a straight**.
* 5-card Pot-Limit Omaha (PLO): Your hand is the best five cards made from **exactly 2** of your 5 hole (initially dealt) cards and **exactly 3** board/community cards. Every turn you choose one of four actions: `FOLD`, `CHECK_CALL`, `RAISE_HALF` (half pot) or `RAISE_POT` (full pot). Pot-Limit means your raise must be a function of the current pot.

Each player starts every hand with 80 chips (40 big blinds). The tournament will pit your agent against other Datathon participant submitted agents in a 1v1 format. The winner of a match will be determined after 2000 hands (1000 duplicate deals) between agents.

## 2. Set up
Your agent will initially require certain dependencies to run. Install them as such, you may need to open a venv.
```
pip install -r requirements-dev.txt
```

## 3. Write your bot
Open `agent.py` and edit:
1. `PARTICIPANT` and `AGENT_NAME` (your team and bot names).
2. The three functions between `YOUR CODE HERE` and `END OF YOUR CODE`:

| Function | When it runs | What to do |
|---|---|---|
| `new_half(config, memory)` | Start of each 1000-hand half | Set up anything you want to track |
| `choose_action(obs, memory)` | Every time it is your turn | Return one of `obs.legal_actions` |
| `on_hand_end(result, memory)` | After every hand | Optional: learn from the result |

`obs` holds everything you can see: your cards, the board, the pot, what you must call, the legal actions, and the action history. Every field is listed in the `choose_action` docstring.

Keep anything you want to remember in the `memory` dict. It is wiped at the start of each half. In the tournament, each half also runs in a **brand-new copy of your server**, so nothing at all carries over between halves. Locally, one process can play both halves and only `memory` is wiped (other global variables are **not**). Using `memory` makes your bot behave the same in both places.

## 4. Play a match
Use three terminals:
```
python agent.py                             # your bot, on port 5008
python baseline-bots/baseline_random.py     # an opponent, on port 5009
python judge_engine.py --deals 200          # the judge: both halves, 200 deals each (default 1000)
```
The judge prints a report that ends with `RESULT: AGENT1_WIN`, `RESULT: AGENT2_WIN` or `RESULT: CHOP` (or `RESULT: BOTH_FORFEIT` if both bots forfeited). A bot that forfeits either half loses the match, whatever the chips say, just like in the tournament. If your bot crashes, the report shows your traceback.

Every baseline bot listens on port 5009 by default. To run two at once, give one another port, e.g. `PORT=5010 python baseline-bots/baseline_always_pot.py`, and point the judge at it with `--agent2 5010`.

When you run it yourself, `agent.py` listens on every network interface, so other machines on the same network (e.g. hackathon Wi-Fi) can reach it. Set `AGENT_TOKEN=<some secret>` when you start it if that worries you; the judge then needs the same value in `AGENT1_TOKEN` (or `AGENT2_TOKEN`).

Useful judge options:
- `--agent1 5008 --agent2 5009` picks the bots by port (or a full URL like `http://localhost:5008`).
- `--seed 7` uses different cards. The same seed always gives the same cards. The actual tournament uses a secret, randomly generated seed for every match.
- `--deals 1000` plays a full-length match (about 10 seconds with simple bots).
- `--json out.json --logs` saves the full result and every hand.

**Playing halves separately (exactly like the tournament).** Restart both bots between halves to match the tournament's fresh servers:
```
python judge_engine.py --half 1 --seed 7 --json h1.json   # agent1 in seat 0
# ...restart both bots...
python judge_engine.py --half 2 --seed 7 --json h2.json   # agent1 in seat 1
python judge_engine.py combine h1.json h2.json            # pair the halves -> verdict
```
A single half ends with `RESULT: HALF_DONE`. Its chip count is mostly card luck, so only the combined result decides the match.

## 5. Test and check your bot
First install the testing dependencies:
```
pip install -r requirements-dev.txt
```
Then check your bot for tournament compatibility:
```
python -m pytest -q
```

Then check that it **builds and runs as a Docker image**, exactly like the tournament builds it (needs [Docker](https://docs.docker.com/get-docker/); on Windows use WSL or Git Bash):
```
./tools/check_build.sh
```
It builds the base image and your bot, checks the image size, then starts your bot with match restrictions (no network, read-only files) and plays a few sample hands. It ends with `ALL CHECKS PASSED` or tells you what to fix. The first run downloads PyTorch, so it takes a few minutes.

## 6. Submit
1. Commit and **push** your bot to your **public** GitHub repo.
2. Get the full commit ID: `git rev-parse HEAD` (40 characters), or copy it from the commit's page on GitHub.
3. Submit your team name, repo URL and that commit ID in the form.

The tournament builds **exactly that commit**. Anything you push afterwards is ignored unless you edit your form response with a new commit ID.

**If your build fails** (for example, the smoke test fails or the image is too big), your bot can't play. Run `./tools/check_build.sh` on the exact commit you submit.

## 7. How a match is scored
- **Duplicate format:** a match is two halves of 1000 deals on the **same cards**. In half 1 you sit in one seat; in half 2 you sit in the other seat and get the cards your opponent had. This cancels most of the card luck. Look into duplicate dealing for more information.
- **Each half is played separately.** The tournament runs each half in its own fresh environment, with no shared storage and no network access. You can't use what you saw in one half (like your opponent's cards at showdown) in the other half.
- **Score:** your average result in big blinds per deal over both halves, with a 95% confidence interval.
- **Verdict:** you **win** (1 point) if the whole interval is above 0, **lose** (0 points) if it is below 0, and **chop** (0.5 points each) if it includes 0.

## 8. Rules your bot must follow
- **Time:** **8 seconds of thinking per half** (1000 hands), which is roughly 4–7 ms per decision on average. Network time doesn't count: the first 1 ms of every request is free. A single decision may take at most **1 second**. If your time runs out, your bot plays check-or-fold for the rest of the half.
- **Startup:** your server must be answering within **15 seconds** of starting. Do slow setup (like building lookup tables) at startup rather than during your first decision, but keep it under that limit. `agent.py` already calls `warmup()` at startup to build `omaha_score`'s table; put your own setup next to it. Your server starts once per half.
- **Errors:** crashing, returning something that is not an action, or making an illegal move each count as one error. After **5 errors in a half**, your bot **forfeits** that half, and **a forfeit loses the whole match** (0 points), whatever the chip count. A bot that never answers `/new-half` forfeits too. Every crash inside `on_hand_end()` counts as its own error, and finished hands arrive in batches, so one bug there can forfeit you within a few hands. Run `python -m pytest -q` to catch these. (Running out of time bank is **not** an error: your bot just checks or folds for the rest of the half.)
- **Port:** the tournament sets the `PORT` environment variable. Keep the last lines of `agent.py` that read it, otherwise your agent's port is hardcoded and won't run on our evaluation pipeline.
- **Hand results arrive late:** to save time, the judge delivers finished hands together with your next decision request. `on_hand_end()` still runs once per hand, always before `choose_action()` sees the next hand.
- **Keep the server plumbing working.** If you change the routes or the reply format, the judge can't talk to your bot, and every failed call counts as an error.
- **No outside connections:** your bot has no network access during a match, and files it writes are thrown away after each half.
- **Files:** your bot's files are read-only during a match. Only `/tmp` is writable (512 MB), and it's wiped after each half.
- **Libraries:** any library pip can install is allowed. List them in `requirements.txt`. The only limit is size: **your image may add at most 3 GB on top of the base image.** Some tips:
  - The base image already includes **numpy** and **PyTorch (CPU version)**. List `torch` without a version to reuse it; other versions also work but count toward your 3 GB.
  - **There is no GPU.** If you ask for a different PyTorch version, the build installs the CPU version automatically. Don't use GPU packages (anything with a `[cuda]` extra, `jax[cuda]`, `tensorflow[and-cuda]`): they're huge and useless here.
  - **TensorFlow** works (`tensorflow`, about 2 GB), but takes most of your 3 GB.
  - **FastAPI and uvicorn are fixed** at the versions in `requirements.txt`. Asking for other versions makes the build fail.
  - Big libraries load slowly. Your server must answer within 15 seconds of starting, so check that `import`s plus loading your model fit.
- **Threads:** the base image limits PyTorch/numpy to 1 thread (`OMP_NUM_THREADS=1`), because many bots share the machine. Your bot gets about 1 CPU.
- **Model files:** commit them to your repo normally. Each file must be under GitHub's 100 MB limit. **Git LFS files are not downloaded.**
