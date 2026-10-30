#!/bin/bash
# The ship sequence, shared infrastructure across the repositories (cgm-forecast-conformal
# and nir-glucose-reproduction carry this file verbatim). It exists because the same
# mistake was made three times: a command chain that gated on `tail`'s exit code instead of
# pytest's pushed a failing test. Under `set -e` a failing step ends the run; nothing after
# it happens.
#
#   scripts/ship.sh <commit-message-file>
#
# Runs the data-free test suite and stops on any failure; updates the README's test count;
# commits everything staged by `git add -A` with the message file; pushes; waits for CI and
# stops if it is not green; stamps every decision-log entry whose commit is `pending` with
# the commit hash, commits that, pushes, and waits for CI again.
set -euo pipefail
MSG="${1:?usage: scripts/ship.sh <commit-message-file>}"
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.
PY=.venv/bin/python
# every repository's data root is pointed at nothing, so only the data-free tests run
export OHIO_T1DM_ROOT=/nonexistent BIG_IDEAS_ROOT=/nonexistent PPG_DALIA_ROOT=/nonexistent   # every repo's dataset hidden: the data-free suite
N=$("$PY" -m pytest -q --co 2>&1 | tail -1 | grep -oE "^[0-9]+")
"$PY" -m pytest -q > /tmp/ship_pytest.log 2>&1          # under set -e a failure stops here
tail -1 /tmp/ship_pytest.log
S=$(grep -oE "[0-9]+ skipped" /tmp/ship_pytest.log | grep -oE "[0-9]+" || echo 0)
sed -i '' -E "s/(pytest +# )[0-9]+ tests; [0-9]+ need the dataset/\1$N tests; $S need the dataset/" README.md   # any spacing before the #
git add -A
git -c core.hooksPath=.git/hooks commit -q -F "$MSG"
H=$(git log -1 --format=%h); echo "commit $H"
for i in 1 2 3; do git push -q origin main > /tmp/ship_push.log 2>&1 && { echo "pushed on try $i"; break; } || { [ "$i" -eq 3 ] && { cat /tmp/ship_push.log; exit 1; }; sleep 4; }; done
wait_ci() { for i in $(seq 1 36); do sleep 10; out=$(gh run list --limit 1 --json status,conclusion,headSha --jq '.[] | "\(.headSha[0:7]) \(.status) \(.conclusion)"'); echo "$out" | grep -qE "in_progress|queued" || { echo "CI: $out"; echo "$out" | grep -q "$1 completed success"; return; }; done; echo "CI did not finish in time"; return 1; }
wait_ci "$H"
if grep -q '· `pending`$' docs/decisions.md; then
  TODAY=$(date +%Y-%m-%d)
  sed -i '' "s/^- \*\*Date \/ commit:\*\* $TODAY · \`pending\`$/- **Date \/ commit:** $TODAY · \`$H\`/" docs/decisions.md
  echo "stamped $(grep -c "\`$H\`" docs/decisions.md) entries with $H; $(grep -c '· `pending`$' docs/decisions.md) still pending (other dates)"
  git diff --quiet docs/decisions.md && { echo "no entry dated today to stamp; done"; git log --oneline -2; exit 0; }
  "$PY" -m pytest -q tests/test_decision_log.py > /dev/null
  git add docs/decisions.md
  git -c core.hooksPath=.git/hooks commit -q -m "docs: record commit hash $H in the decision log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
  git push -q origin main 2>/dev/null
  wait_ci "$(git log -1 --format=%h)"
fi
git log --oneline -2
