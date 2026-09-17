import copy

import numpy as np
import pytest

from src.data import loader
from src.data.loader import DataValidationError, validate_raw

# ---------------------------------------------------------------------------
# Synthetic: exercise every validation path without the dataset.
# ---------------------------------------------------------------------------

DUR = 20  # seconds -> (20 - 8) / 2 + 1 = 7 labels


def _fake_raw(subject_id="S1"):
    rng = np.random.default_rng(0)
    n = lambda fs: DUR * fs  # noqa: E731
    acc = rng.normal(0, 0.05, (n(32), 3))
    acc[:, 2] += 1.0  # gravity on z, units of g
    return {
        "subject": subject_id,
        "label": np.full(7, 72.0),
        "activity": np.ones((n(4), 1)),
        "rpeaks": np.array([100, 800, 1500]),
        "questionnaire": {"SKIN": 3, "AGE": 30, "Gender": " m"},
        "signal": {
            "wrist": {
                "BVP": rng.normal(0, 40, (n(64), 1)),
                "ACC": acc,
                "EDA": np.ones((n(4), 1)),
                "TEMP": np.full((n(4), 1), 32.0),
            },
            "chest": {
                "ECG": rng.normal(0, 0.3, (n(700), 1)),
                "ACC": rng.normal(0, 0.1, (n(700), 3)),
                "Resp": rng.normal(0, 3, (n(700), 1)),
                "EMG": np.full((n(700), 1), -1.5),
                "EDA": np.zeros((n(700), 1)),
                "Temp": np.full((n(700), 1), -273.15),
            },
        },
    }


def test_valid_fake_passes():
    validate_raw(_fake_raw(), "S1")


def test_missing_top_level_key_raises():
    raw = _fake_raw()
    del raw["rpeaks"]
    with pytest.raises(DataValidationError, match="missing top-level keys"):
        validate_raw(raw, "S1")


def test_label_off_by_one_raises():
    raw = _fake_raw()
    raw["label"] = np.full(6, 72.0)
    with pytest.raises(DataValidationError, match="expected 7"):
        validate_raw(raw, "S1")


def test_channel_duration_mismatch_raises():
    raw = _fake_raw()
    raw["signal"]["wrist"]["BVP"] = raw["signal"]["wrist"]["BVP"][:-64]
    with pytest.raises(DataValidationError, match="different durations"):
        validate_raw(raw, "S1")


@pytest.mark.parametrize("factor, hint", [(64.0, "raw 1/64 g"), (1 / 64, "scaled twice")])
def test_acc_in_wrong_units_raises(factor, hint):
    raw = _fake_raw()
    raw["signal"]["wrist"]["ACC"] = raw["signal"]["wrist"]["ACC"] * factor
    with pytest.raises(DataValidationError, match="expected ~1 g"):
        validate_raw(raw, "S1")


def test_non_constant_dummy_channel_raises():
    raw = _fake_raw()
    raw["signal"]["chest"]["EMG"] = np.random.default_rng(1).normal(size=raw["signal"]["chest"]["EMG"].shape)
    with pytest.raises(DataValidationError, match="dummy data but is not constant"):
        validate_raw(raw, "S1")


def test_subject_mismatch_raises():
    with pytest.raises(DataValidationError, match="declares subject"):
        validate_raw(_fake_raw("S2"), "S1")


def test_duplicate_rpeaks_tolerated_and_counted():
    raw = _fake_raw()
    raw["rpeaks"] = np.array([100, 800, 800, 1500])
    validate_raw(raw, "S1")
    rec = loader._record_from_raw(raw, "S1", loader.Provenance("fake", 0, 0.0, "0", "pickle"))
    assert rec.rpeak_duplicates_removed == 1
    np.testing.assert_array_equal(rec.rpeaks, [100, 800, 1500])


def test_decreasing_rpeaks_raise():
    raw = _fake_raw()
    raw["rpeaks"] = np.array([100, 800, 700])
    with pytest.raises(DataValidationError, match="rpeaks decrease"):
        validate_raw(raw, "S1")


def test_rpeaks_beyond_ecg_raise():
    raw = _fake_raw()
    raw["rpeaks"] = np.array([100, DUR * 700])
    with pytest.raises(DataValidationError, match="outside the ECG length"):
        validate_raw(raw, "S1")


def test_record_drops_dummy_channels():
    raw = _fake_raw()
    prov = loader.Provenance("fake", 0, 0.0, "0", "pickle")
    rec = loader._record_from_raw(copy.deepcopy(raw), "S1", prov)
    assert set(rec.signals) == {"wrist_bvp", "wrist_acc", "wrist_eda", "wrist_temp",
                                "chest_ecg", "chest_acc", "chest_resp"}
    assert rec.signals["wrist_bvp"].data.ndim == 1
    assert rec.signals["wrist_acc"].data.shape[1] == 3


# ---------------------------------------------------------------------------
# Real data: skipped when PPG-DaLiA is not present.
# ---------------------------------------------------------------------------

have_data = (loader.DATA_ROOT / "S1" / "S1.pkl").exists()
needs_data = pytest.mark.skipif(not have_data, reason="PPG-DaLiA not found; see data/README.md")


@pytest.fixture(scope="module")
def s1():
    return loader.load_subject("S1")


@needs_data
def test_s1_pinned_shapes(s1):
    # Spec estimated ~576,000 / ~4,500 from a rounded ~9,000 s. Measured values, pinned exactly:
    assert s1.duration_s == 9212.0
    assert s1["wrist_bvp"].data.shape == (589_568,)
    assert s1["wrist_acc"].data.shape == (294_784, 3)
    assert s1["chest_ecg"].data.shape == (6_448_400,)
    assert s1.activity.shape == (36_848,)
    assert len(s1.label) == 4_603


@needs_data
def test_s1_acc_is_in_g_and_not_rescaled(s1):
    mag = np.linalg.norm(s1["wrist_acc"].data, axis=1)
    assert 0.95 < np.median(mag) < 1.05
    assert np.abs(s1["wrist_acc"].data).max() <= loader.WRIST_ACC_LIMIT_G
    assert 0 < s1.wrist_acc_clipped_fraction < 0.01


@needs_data
def test_s1_dummy_channels_excluded(s1):
    assert {"chest_emg", "chest_eda", "chest_temp"}.isdisjoint(s1.signals)
    assert {"chest_ecg", "chest_acc", "chest_resp"} <= set(s1.signals)


@needs_data
def test_s1_bvp_has_no_dc(s1):
    bvp = s1["wrist_bvp"].data
    assert abs(bvp.mean()) < 0.01 * bvp.std()


@needs_data
def test_s6_truncated():
    s6 = loader.load_subject("S6")
    assert s6.duration_s == 5250.0
    assert len(s6.label) == 2_622
    assert set(np.unique(s6.activity)) == {0, 1, 2, 3, 4, 5}  # no lunch, walking or working
    assert s6.rpeak_duplicates_removed == 3


@needs_data
def test_label_count_matches_duration_all_subjects():
    for rec in loader.load_all():
        assert len(rec.label) == loader.expected_label_count(rec.duration_s), rec.subject_id
        assert rec.rpeak_duplicates_removed == {"S6": 3, "S14": 1}.get(rec.subject_id, 0), rec.subject_id


@needs_data
def test_skin_types_match_cohort():
    skin = {rec.subject_id: rec.skin_type for rec in loader.load_all()}
    assert skin["S15"] == 2
    assert {s for s, t in skin.items() if t == 4} == {"S4", "S9", "S10"}
    assert all(t == 3 for s, t in skin.items() if s not in {"S4", "S9", "S10", "S15"})


@needs_data
def test_cache_round_trip_is_identical(s1):
    again = loader.load_subject("S1")
    assert again.provenance.loaded_from == "cache"
    assert again.provenance.sha256 == s1.provenance.sha256
    for name in s1.signals:
        np.testing.assert_array_equal(again[name].data, s1[name].data)
    np.testing.assert_array_equal(again.label, s1.label)
    np.testing.assert_array_equal(again.activity, s1.activity)
