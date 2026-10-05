# Why decoding, and what it assumes

Reading notes for the alternative method, written for the next person who asks why this
repository has a Viterbi decoder in it.

## The problem it was chosen for

Diagnostic A measures something specific: among the windows where the estimate is wrong by
more than 10 bpm, how often is the *correct* peak still present in the spectrum? The answer
has been 58.8%, then 60.4%, then 65.7% as each new component landed. Every improvement so
far has reduced the size of the error without changing its nature — the right answer is
usually there and the estimator picks something else.

That is a selection problem, and greedy selection is the thing doing the selecting. The
tracker looks at one window, commits to one peak, and moves on. Its only way to undo a
commitment is to reset, which throws away the track entirely.

## The HMM formulation

Treat the session as a hidden state sequence. The hidden state at each window is the true
heart rate; the observation is the spectrum. Then:

- **States.** The candidate heart rates for a window are its top-K spectral peaks. Using
  peaks rather than every frequency bin keeps the state space at 3–6 per window instead of
  ~230.
- **Emission.** How well a candidate explains the observation: here, the log of the peak's
  height normalised by the window's maximum. The tallest peak scores 0, a peak at half the
  height scores −0.69.
- **Transition.** How plausible a change between consecutive windows is: a Gaussian penalty
  on the heart-rate difference, −(Δf)²/2σ². With σ = 6 bpm, a 60 bpm jump costs 50 nats and a
  5 bpm drift costs 0.35.

Viterbi finds the single highest-scoring path through the whole sequence. The important
consequence is that **a window's choice can be revised by what comes after it**: if the next
four windows all point at 80 bpm, a lone 140 bpm peak loses even though it was the tallest
in its own window.

### What it assumes

1. **The Markov property.** The next heart rate depends only on the current one, not on the
   trajectory. Real heart rate has momentum — it trends up during a climb — and the model
   cannot represent that.
2. **A stationary transition law.** σ is one number for the whole session, so the same jump
   is equally implausible during sitting and during stairs. This is wrong in a way that
   matters here: stairs is where heart rate changes fastest *and* where motion is worst.
3. **Conditional independence of observations.** Each window's spectrum is assumed informative
   only about its own state. Overlapping windows (8 s windows at a 2 s shift share 75% of
   their samples) violate this badly — neighbouring spectra are strongly correlated, so the
   decoder sees more agreement than there really is and is overconfident.
4. **The truth is among the candidates.** If the correct rate is not one of the top-K peaks,
   no decoding can find it. Diagnostic A is precisely the measurement of how often that
   assumption holds: about two thirds of the time.

## How it differs from the alternatives

**Kalman filtering** assumes a Gaussian state with linear dynamics and gives a closed-form
recursive update. It is cheaper and naturally causal, and it returns a variance, which is
useful. But the posterior here is not Gaussian — it is multi-modal, with one mode at the
cardiac peak and another at the cadence line, which is the whole difficulty. A Kalman filter
must collapse that to a mean, and the mean of 80 and 140 is 110, a value supported by
nothing. Particle filters or an IMM would handle multi-modality causally, at the cost of
sampling noise. Viterbi keeps the modes as discrete states and never averages them.

**TROIKA** (Zhang et al. 2015) is the classic pipeline: sparse signal reconstruction to
sharpen the spectrum, spectral peak tracking with a verification stage, then interpolation
for windows where tracking fails. Its tracking stage is greedy and nearest-to-previous, like
the tracker in this repository, with hand-built rules for when to trust a peak and when to
coast. The difference is not sophistication in the signal processing but *when the decision
is made*: TROIKA decides per window and patches failures afterwards; decoding defers the
decision until the sequence can be scored as a whole.

**SpaMaPlus** (Reiss et al. 2019), the published baseline here, is the same shape as TROIKA's
tracker — mean filter over recent estimates, nearest peak, reset on repeated jumps.

## The known weakness, and what fixing it costs

Full-session Viterbi is **non-causal**. It uses future windows, so it cannot run on a watch,
and a number produced that way does not belong beside SpaMa's or SpaMaPlus's, which are
causal. Reporting it as if it were comparable would be the kind of mistake this repository
exists to avoid.

The fix is **fixed-lag decoding**: emit the estimate for window *t* once window *t+L* has been
seen. That is causal with a known delay of 2L seconds, and it converges to the full solution
as L grows. The cost is latency, and the repository reports the chosen lag in seconds so a
reader knows what "real time" means. A device updating heart rate every few seconds can
usually afford a short lag; one driving an alarm cannot.

A second weakness is that the decoder is only as good as its candidate list. If masking has
removed the cardiac peak, or it was never prominent enough to be in the top K, decoding
cannot recover it — it will choose the best of the wrong options with great confidence.

## References

- Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-scale
  heart rate estimation with convolutional neural networks. *Sensors*, 19(14), 3079.
- Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA: A general framework for heart rate monitoring
  using wrist-type photoplethysmographic signals during intensive physical exercise.
  *IEEE Transactions on Biomedical Engineering*, 62(2), 522–531.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in
  speech recognition. *Proceedings of the IEEE*, 77(2), 257–286. — the standard reference for
  Viterbi decoding and the three assumptions above.
