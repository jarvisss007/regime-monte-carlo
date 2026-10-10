# regime-monte-carlo

## Cloud sessions: pull requests only (rule set 2026-10-10)

The firm runs on Anupam's Mac; the cloud only sends pull requests. If this session is running in the cloud (claude.ai/code, a fresh
checkout with no `~/command-center` or `~/stock-radar` beside this repo):

- Work on a branch and open a pull request. Never push to `main`.
- Never run this repo's writers, scheduled jobs or agents, and never edit books, ledgers, forecasts, journals, state JSON or data files:
  their real state lives only on the Mac (much of it is gitignored), so a cloud copy is stale and partial.
- Never edit `command-center/council/issues.json` and never run `resolver.py`: the register is written on the Mac only.
- Code, tests and docs only. In the pull request, say what you could not verify without the Mac's data.
- A Mac session reviews every cloud pull request before it is merged.

