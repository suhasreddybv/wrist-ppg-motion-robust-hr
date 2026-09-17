"""60 s of seated-rest BVP for S1, and whether the E4 signal carries a DC component.

Writes figures/s1_rest_bvp.png and prints the numbers quoted in the README.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, load_subject  # noqa: E402

SITTING = 1
OFFSET_S = 60   # skip the first minute of sitting
LENGTH_S = 60


def main() -> None:
    rec = load_subject("S1")
    bvp = rec["wrist_bvp"]
    first_sit = np.flatnonzero(rec.activity == SITTING)[0] / 4  # activity is 4 Hz
    start = int((first_sit + OFFSET_S) * bvp.fs)
    seg = bvp.data[start:start + LENGTH_S * bvp.fs]
    t = np.arange(len(seg)) / bvp.fs

    print(f"S1 seated rest, {start / bvp.fs:.0f}-{start / bvp.fs + LENGTH_S:.0f} s")
    print(f"  60 s segment: mean {seg.mean():.3f}, SD {seg.std():.2f}, |mean|/SD {abs(seg.mean()) / seg.std():.4f}")
    print(f"  whole record: mean {bvp.data.mean():.3f}, SD {bvp.data.std():.2f}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 5))
    ax[0].plot(t, seg, lw=0.7, color="black")
    ax[0].axhline(seg.mean(), color="tab:red", ls="--", lw=1, label=f"mean = {seg.mean():.2f} (SD {seg.std():.1f})")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_ylabel("BVP (E4 units)")
    ax[0].set_title(f"S1, 60 s seated rest from t = {start / bvp.fs:.0f} s: zero-centred, no DC component")
    ax[0].legend(loc="upper right")
    ten = 10 * bvp.fs
    ax[1].plot(t[:ten], seg[:ten], lw=1, color="black")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("first 10 s")
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "s1_rest_bvp.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
