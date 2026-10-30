"""Week 7 results: the fold configurations reconcile to the published headline, the summaries follow
from the per-subject rows, the budget arithmetic is reproducible, and docs/edge-budget.md and the
README section are exactly what src/edge/write_docs.py generates from results/edge_*.csv."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from src.edge.budget import CAPACITY_MAH, TWO_YEARS_H
from src.edge.write_docs import END, START, edge_doc, readme_section

ROOT = Path(__file__).resolve().parents[1]
TEST_COUNT = re.compile(r"# \d+ tests; \d+ need the dataset")


def _rows(name):
    with open(ROOT / "results" / name) as f:
        return list(csv.DictReader(f))


def test_fold_configurations_reconcile_to_the_published_headline():
    rows = _rows("edge_fold_configs.csv")
    assert len(rows) == 15 and all(r["history"] == "15" for r in rows)
    ref = next(float(r["pooled_mae"]) for r in _rows("prediction_rule.csv") if r["rule"] == "mean,hist=15")
    assert abs(np.mean([float(r["mae_hist15"]) for r in rows]) - ref) < 5e-4
    per = {r["subject"]: float(r["mae"]) for r in _rows("edge_variants_per_subject.csv") if r["variant"] == "reference"}
    assert all(abs(per[r["subject"]] - float(r["mae_hist15"])) < 1e-3 for r in rows)    # the edge pipeline is the frozen one


def test_variant_summary_follows_from_the_subject_rows():
    per = _rows("edge_variants_per_subject.csv")
    for s in _rows("edge_variants_summary.csv"):
        own = {r["subject"]: float(r["mae"]) for r in per if r["variant"] == s["variant"]}
        base = {r["subject"]: float(r["mae"]) for r in per if r["variant"] == "causal_f64"}
        assert len(own) == 15 and abs(np.mean(list(own.values())) - float(s["pooled_mae"])) < 2e-3
        d = np.array([own[k] - base[k] for k in own])
        assert abs(d.mean() - float(s["minus_causal_f64"])) < 2e-3
        assert float(s["minus_causal_f64_ci_lo"]) <= float(s["minus_causal_f64"]) + 1e-9 <= float(s["minus_causal_f64_ci_hi"]) + 2e-9
        assert int(s["subjects_worse_than_causal_f64"]) == int((d > 1e-9).sum())


def test_float32_and_float64_agree_and_the_16_bit_state_filter_is_the_outlier():
    s = {r["variant"]: r for r in _rows("edge_variants_summary.csv")}
    assert abs(float(s["causal_f32"]["pooled_mae"]) - float(s["causal_f64"]["pooled_mae"])) < 0.01
    assert float(s["q15_state16"]["minus_causal_f64_ci_lo"]) > 0                   # clearly worse
    for k in ("q15", "q8", "reference"):
        assert float(s[k]["minus_causal_f64_ci_lo"]) < 0 < float(s[k]["minus_causal_f64_ci_hi"])   # indistinguishable


def test_burst_rows_are_paired_on_the_same_points():
    b = _rows("edge_burst.csv")
    assert len({r["update_points"] for r in b}) == 1
    for r in b[1:]:
        assert int(r["led_on_s"]) == 8 + 2 * (int(r["burst_windows"]) - 1)
        assert float(r["ci_lo"]) > 0                                                     # every cold-start burst is worse than continuous


def test_budget_arithmetic_is_consistent():
    for r in _rows("edge_budget.csv"):
        i_ma = float(r["avg_current_uA"]) / 1e3
        assert abs(float(r["capacity_for_2y_mAh"]) - i_ma * TWO_YEARS_H) <= 1.0
        assert abs(float(r["cr2032_months"]) - CAPACITY_MAH / i_ma / (TWO_YEARS_H / 24)) <= 0.06
        assert 0 < float(r["compute_share_pct"]) < 15
    two_year = [r for r in _rows("edge_budget.csv") if float(r["cr2032_months"]) >= 24]
    assert [(r["led"][:3], r["burst_windows"]) for r in two_year] == [("low", "1")]    # the only configuration that reaches two years


def test_memory_totals():
    m = _rows("edge_memory.csv")
    total = next(r for r in m if r["item"] == "RAM total")
    parts = [r for r in m if r["item"] not in ("RAM total",) and not r["item"].startswith("FFT tables")]
    for k in ("float32", "q15", "q8"):
        assert int(total[k]) == sum(int(r[k]) for r in parts) and int(total[k]) < 256 * 1024


def test_generated_documents_match():
    assert (ROOT / "docs" / "edge-budget.md").read_text() == edge_doc(), "docs/edge-budget.md is stale: python -m src.edge.write_docs"
    readme = (ROOT / "README.md").read_text()
    i, j = readme.index(START), readme.index(END) + len(END)
    assert readme[i:j] == readme_section(), "README Week 7 section is stale: python -m src.edge.write_docs"
