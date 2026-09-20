import numpy as np
import pytest

from src.data import loader
from src.data.windows import (
    WindowAlignmentError,
    iter_windows,
    n_windows_for,
    window_array,
    window_subject,
)
from tests.test_loader import _fake_raw, needs_data


def _fake_record(subject_id="S1"):
    raw = _fake_raw(subject_id)
    return loader._record_from_raw(raw, subject_id, loader.Provenance("fake", 0, 0.0, "0", "pickle"))


def test_window_count_formula():
    assert n_windows_for(20) == 7          # (20 - 8) / 2 + 1
    assert n_windows_for(9212) == 4603     # S1
    assert n_windows_for(5250) == 2622     # S6


def test_windows_tile_the_signal_exactly():
    rec = _fake_record()
    ws = window_subject(rec)
    bvp = rec["wrist_bvp"].data
    assert ws.bvp.shape == (7, 512)
    assert ws.acc.shape == (7, 256, 3)
    for i in (0, 3, 6):
        np.testing.assert_array_equal(ws.bvp[i], bvp[i * 128:i * 128 + 512])
        np.testing.assert_array_equal(ws.acc[i], rec["wrist_acc"].data[i * 64:i * 64 + 256])
    np.testing.assert_array_equal(ws.start_s, np.arange(7) * 2.0)
    assert ws.start_s[-1] + 8 == rec.duration_s   # last window ends exactly at the record end


def test_label_mismatch_raises():
    rec = _fake_record()
    object.__setattr__(rec, "label", rec.label[:-1])
    with pytest.raises(WindowAlignmentError, match="6 labels"):
        window_subject(rec)


def test_activity_is_window_mode():
    rec = _fake_record()
    act = np.array(rec.activity)
    act[:] = 1
    act[:32] = 7          # all of window 0, part of windows 1-3
    object.__setattr__(rec, "activity", act)
    ws = window_subject(rec)
    assert ws.activity[0] == 7          # 32/32 samples
    assert ws.activity[1] == 7          # 24/32
    assert ws.activity[3] == 1          # 8/32
    assert set(np.unique(ws.activity)) <= {1, 7}


def test_window_array_matches_subject_windows():
    rec = _fake_record()
    ws = window_subject(rec)
    np.testing.assert_array_equal(window_array(rec["wrist_bvp"].data, 64, len(ws)), ws.bvp)


def test_iter_windows_agrees_with_stacked():
    rec = _fake_record()
    ws = window_subject(rec)
    wins = list(iter_windows(rec))
    assert len(wins) == len(ws)
    assert [w.index for w in wins] == list(range(len(ws)))
    np.testing.assert_array_equal(wins[2].bvp, ws.bvp[2])
    assert wins[2].hr == ws.hr[2]
    assert wins[0].skin_type == rec.skin_type


# --- real data ---------------------------------------------------------------

@needs_data
def test_window_count_equals_label_count_every_subject():
    """The highest-value assertion in the repo: one window per label, for all 15 subjects."""
    for rec in loader.load_all():
        ws = window_subject(rec, copy=False)
        assert len(ws) == len(rec.label), rec.subject_id
        assert ws.bvp.shape == (len(rec.label), 512), rec.subject_id
        assert ws.acc.shape == (len(rec.label), 256, 3), rec.subject_id
        # Odd-length records (S2, S3, S5, S7, S15) leave a tail shorter than one shift.
        tail = rec.duration_s - (ws.start_s[-1] + 8)
        assert 0 <= tail < loader.SHIFT_S, (rec.subject_id, tail)
        np.testing.assert_array_equal(ws.hr, rec.label)


@needs_data
def test_s1_windows_align_with_activity_protocol():
    ws = window_subject(loader.load_subject("S1"))
    assert set(np.unique(ws.activity)) <= set(range(9))
    # sitting is the first protocol block, and transients exist between blocks
    assert ws.activity[75] == 1
    assert (ws.activity == 0).any()
