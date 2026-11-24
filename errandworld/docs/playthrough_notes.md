# Human playthrough notes

FP2 §4.4 asks for about 10 scenarios to be played by team members through the same text
interface the agents use. The goal is to check that task descriptions and tool results are clear
**before** prompts are written and frozen. Use development-set scenarios only: pilot and test
scenarios must stay unseen.

```bash
python scripts/play.py --list dev
python scripts/play.py dev_L20_D1_003 --trace runs/playthroughs/<name>_dev_L20_D1_003.json
```

Pick a mix of lengths (L10, L20, L30) and disruption levels (D0, D1, D3).

## Log

| Date | Player | Scenario | Outcome | Actions / reference / cap | Notes (anything confusing, ambiguous, or surprising) |
|---|---|---|---|---|---|
| | | | | | |

## Issues found

List each issue with the scenario id and the exact text that was unclear, and whether it was
fixed (with the commit) or deliberately left as is.

-
