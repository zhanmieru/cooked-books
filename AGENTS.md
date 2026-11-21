# CookedBooks: instructions for coding agents

CookedBooks is an RL environment for training AI financial auditors. A generator builds a fictional company's books, injects known errors and records what it changed. An agent inspects the books through five tools and submits findings with evidence. A grader scores them.

It is a hackathon entry (Genesis Hackathon 2026, FinTech track). The jury does not rank our agent. It runs 80 published pass/fail checks against our environment. Every rule below exists because one of those checks depends on it.

Deadline: 22 November 2026, 23:59 Astana time. Code freeze: 22 November, 18:00.

## Read first

1. [docs/jury-checks.md](docs/jury-checks.md): what the jury runs and the exact expected results.
2. [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md): file formats, the reset/step contract, matcher and reward rules.
3. [DESIGN.md](DESIGN.md): the accounting model, the four error types, how ground truth is produced.
4. [docs/TASKS.md](docs/TASKS.md): the task list. It also says what already exists.
5. [PLAN.md](PLAN.md): schedule, team split, risks.

If two files disagree, the order of authority is: the jury's page, then `EVALUATION_CONTRACT.md` and `DESIGN.md`, then `PLAN.md`. Fix the lower one in the same commit.

## Setup and commands

Python 3.11 only. Runtime code uses the standard library only.

```bash
uv venv --python 3.11
uv pip install -r requirements.lock
uv run pytest -q
make verify
```

- `make verify` builds the container and runs everything with no network, 2 CPUs and 4 GB RAM. It needs Docker. On macOS with Colima, run `colima start` first.
- Test dependencies are `pytest` and `hypothesis` only, pinned in `requirements.lock`, with wheels in `vendor/wheels/` so the container builds offline. Every vendored wheel is universal (`py3-none-any`), so it installs on any CPU architecture; `make wheels-check` enforces this. Do not add a dependency that ships compiled code. This is why `jsonschema` is not used (it pulls in `rpds-py`) and why `hypothesis` is pinned to 6.155.7, the last pure-Python release.
- `python -m cookedbooks.verifier <episode_dir>` runs the independent verifier on one episode folder and prints why it is accepted or rejected.
- `python -m cookedbooks.verify.shortcuts --seeds 60` measures how well trivial features predict injected errors. Run it after any change to the generator or an injector.
- `python -m cookedbooks.verify.attacks --seeds 5` runs the reward-attack catalogue on generated episodes.
- `python -m cookedbooks.verify.transform SRC DST --permute --relabel --style int` makes a reordered, relabelled copy of an episode folder.
- `python -m cookedbooks.runner.run --policy NAME --episode DIR --mode closed|train` runs one policy in a separate process. `make baselines` prints the policy table.
- `episodes/` and `trajectories.jsonl` are generated artefacts that are committed. After any change to the generator run `make episodes`; after any change to the environment output, the task text or a policy run `make trajectories`. Tests compare the committed files with a fresh build.

## Hard rules

Breaking any of these loses points or blocks scoring.

**Determinism**
- Money is `int` in minor units. Rates and arithmetic use `Decimal` or `Fraction`. Never `float` for money or rates.
- Never use `hash()`. Seed every `random.Random` from `hashlib.sha256` of the seed and a label.
- Sort sets and dict keys before they reach any output.
- No wall-clock time, `uuid`, or OS randomness in generated data.
- No time-zone database. Time zones are fixed UTC offsets taken from the manifest.

**IDs are opaque strings**
- The jury replaces every ID with an arbitrary string and reorders rows, then loads that folder into our environment.
- Never branch on an ID's prefix, length or format. Never sort by ID to get a meaningful order. Find a record's table by lookup.
- Compare IDs and currency codes exactly. No trimming, no case folding.
- Record order in `list_records` is the row order of the file.

**Isolation**
- In `closed` mode the environment never opens `truth.json` or `provenance.json` and always returns reward `0`.
- Scoring happens only in `judge_adapter.score_episode` (and at `submit` in `train` mode, through the same function).
- `cookedbooks/policies` must not import `cookedbooks.grader`, `cookedbooks.world` or `cookedbooks.verifier`.
- `cookedbooks/verifier` must not import `cookedbooks.world`. The verifier has to reach the same ground truth with independent code.
- Tools look records up by dictionary key. No tool turns a caller's string into a file path.
- Error responses are a fixed code and a short message. No paths, tracebacks, seeds or ground truth.
- The calculator parses with `ast` against a whitelist. Never `eval` or `exec`.

**The environment runs on folders it did not create**
- The loader requires only the fields from section 03 of the case. Our extra fields are optional at load time.
- No integrity checks at load time: no hashes, no seed comparison, no verifier run.
- `score_episode` is a generic function of `truth.json`, the public files, `policy.json` and the findings. It does not re-check whether an issue makes accounting sense.
- No minimum episode size. A five-record episode must load.

**Scoring details that are easy to get wrong**
- Default amount tolerance is 0.
- Check structure and limits before normalising sets: 11 evidence IDs with repeats is still a structural error.
- `True` is an `int` in Python. Reject `bool` for `claimed_amount_minor` explicitly.
- An unknown `type` string is a false positive, not a structural error.
- Empty truth and empty findings: precision 0, recall 0, F1 1.

**Data**
- Everything is fictional. No real companies, people, bank details or market rates.
- One episode must generate and run inside 60 seconds on 2 CPUs.

## Workflow

- Pick a task in `docs/TASKS.md`, put your name and `в работе` on it, and mark it `готово` when its check passes.
- Write the test first when the behaviour is listed in `docs/jury-checks.md`. Test names carry the jury's ID, for example `test_C09_calculator_decimal` or `test_M10_no_greedy_matching`.
- When a test or `make verify` exposes a real defect, record it in `docs/failures.md` with the measured number before you fix it, then add the number after the fix. The jury scores this.
- A change to a file format or a rule goes into `EVALUATION_CONTRACT.md` or `DESIGN.md` in the same commit as the code.
- Keep commits small and runnable.

## Git

- `origin` is GitLab. Push only there.
- Never push to GitHub and never call the GitHub API. One teammate updates the GitHub mirror by hand, see [docs/git-workflow.md](docs/git-workflow.md).
- Commit messages: English, imperative mood, no AI co-author trailers.
- Never commit `.venv/`, `reports/`, `.local/` or grader logs.

## Language

- Code, identifiers, comments and commit messages: English.
- `README.md`, `DESIGN.md`, `EVALUATION_CONTRACT.md` and everything in `docs/`: Russian, because the case and the jury are Russian-language. Use plain words and short sentences.
