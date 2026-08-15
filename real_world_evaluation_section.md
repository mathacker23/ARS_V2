# 9.5 Real-World Experimental Evaluation (Yaman)

The synthetic experiment in Section 9.1 tests whether the mathematical
reconstruction machinery behaves as expected under controlled conditions; it
is not evidence for historical reconstruction. This section reports a first,
limited real-audio evaluation for a single raga, Yaman, and is subject to the
data-quality caveats stated below.

## 9.5.1 Data source and provenance

Recordings were sourced from a public third-party repository (RagaVeda,
https://github.com/soha-18/RagaVeda---Hindustani-Classical-Raga-Identification),
consisting of 42 short audio clips (total duration 44.6 minutes) labeled
"Yaman." **This dataset carries no documented license, artist attribution, or
recording provenance.** It should therefore be treated as an unverified
convenience sample, not as an archival or curated corpus in the sense of
Section 12.5. This stands in contrast to, e.g., the Saraga collection [ref],
which provides artist credit, tonic annotation, and (for some recordings)
expert phrase transcription. Results in this section carry correspondingly
weaker evidentiary status and should not be interpreted as validated
Hindustani transcription.

## 9.5.2 Transcription pipeline

1. **Pitch extraction.** f0 contours were extracted via pYIN (librosa),
   fmin=C2, fmax=C6, hop length 256 samples at 22,050 Hz.
2. **Tonic estimation.** No tonic annotation exists for this dataset. Tonic
   was estimated per recording via the mode of the octave-folded pitch-class
   histogram (voiced frames only), with a heuristic octave correction toward
   the lower end of the recording's pitch range. This is a standard
   computational heuristic, **not expert-verified ground truth.** Estimated
   tonic confidence (peak histogram mass fraction) ranged from 0.03 to 0.23
   across files; 10 of 42 files fell below a confidence threshold of 0.08 and
   were excluded from further processing.
3. **Segmentation and quantization.** Voiced pitch contours were median-filtered
   and segmented into stable regions via change-point detection (threshold:
   100 cents, minimum segment length 4 frames ≈ 46 ms). Each segment's median
   pitch (in cents relative to estimated tonic) was snapped to the nearest
   degree of the Yaman scale: S=0, R=200, G=400, m(tivra)=600, P=700, D=900,
   N=1100 cents — an N=7 alphabet matching the synthetic experiment's alphabet
   size. Consecutive repeated symbols were collapsed to single symbolic
   events.
4. Two files were excluded post-quantization for insufficient length
   (< 10 resulting symbols). **32 of 42 files (76%) yielded usable symbolic
   sequences**, lengths ranging from 10 to 565 symbols (mean 70.9).

This pipeline is a symbolic transcription **derived by the present authors**,
not an expert transcription. It is subject to the same caveats raised in
Section 12.2 (shruti/continuous intonation) and Section 7.3 (pitch tracking
is not audio, and audio is not ground-truth pitch): errors in pitch tracking,
tonic estimation, or segmentation propagate directly into the symbolic
alphabet and are not separable from genuine musical structure in the reported
accuracy numbers below.

## 9.5.3 Experimental protocol

Of 32 usable sequences, 2 shorter than 15 symbols were further excluded,
leaving 30. Sequences were split 80/20 at the sequence level (24 train, 6
held-out test), with the split, and all subsequent masking, generated from a
single master seed (20260731, matching the synthetic experiment's seed for
continuity). For each test sequence, missing positions were sampled uniformly
at random without replacement (missing-at-random, matching Section 8.1's
mechanism) at rates {10%, 30%, 50%}, repeated 20 times per sequence per
condition to obtain variance estimates. The order-k Dirichlet(α=0.5)
posterior-mean transition model (Eq. 1) was trained on the 24 training
sequences for k ∈ {1, 2, 3}. Reconstruction used the exact dynamic-programming
decoder of Section 4.2/Algorithm 2 with **no hard grammar constraints applied**
(F(O,R) reduces to observation-consistency only) — a grammar-constrained pass
is left as future work (Section 9.5.5). The DP implementation was verified
against brute-force exhaustive search on 30 small synthetic test cases
(matching scores in all cases), per the Reproducibility Checklist (Appendix C).

## 9.5.4 Results

| Order | Missingness | Mean accuracy | SD | n evaluations | Contexts seen / possible |
|---|---|---|---|---|---|
| 1 | 10% | 0.293 | 0.237 | 120 | 7 / 7 |
| 1 | 30% | 0.287 | 0.172 | 120 | 7 / 7 |
| 1 | 50% | 0.237 | 0.132 | 120 | 7 / 7 |
| 2 | 10% | 0.470 | 0.253 | 120 | 42 / 49 |
| 2 | 30% | 0.414 | 0.163 | 120 | 42 / 49 |
| 2 | 50% | 0.371 | 0.125 | 120 | 42 / 49 |
| 3 | 10% | 0.445 | 0.289 | 120 | 215 / 343 |
| 3 | 30% | 0.394 | 0.186 | 120 | 215 / 343 |
| 3 | 50% | 0.363 | 0.148 | 120 | 215 / 343 |

Uniform-guess baseline (1/N for N=7): 0.143. All reported means exceed this
baseline, indicating the extracted symbol sequences carry learnable
transition structure — though, given the transcription caveats above, this
demonstrates that *the pipeline's output* is learnable, not that Yaman
performance practice per se has been captured.

Consistent with the synthetic result in Section 9.1 and the discussion in
Section 9.4, accuracy does not increase monotonically with order: k=2
outperforms k=3 at every missingness level, plausibly because only 24
training sequences (mean length ~71 symbols) provide enough evidence for the
49 second-order contexts but not for the 343 third-order contexts — of which
only 215 (63%) were ever observed in training. This is a real, not
synthetic, instance of the exponential-parameter-growth tradeoff described in
Section 5.3 and Proposition 3.4 (d_k = N^k(N−1)): d_1=42, d_2=294, d_3=2058,
while the training corpus here contains roughly 1,700 total transitions.

As expected from Section 9.3, accuracy degrades with increasing missingness
across all orders. Standard deviations are large relative to the mean
differences between orders, especially at 10% missingness — consistent with
the paper's own caution against overinterpreting a single point estimate
(Section 9.1).

## 9.5.5 Limitations specific to this evaluation

- **Data provenance.** As stated in 9.5.1, the audio source lacks
  documented licensing or artist attribution. This evaluation should be
  treated as a feasibility pilot, not a validated archival result, and would
  need to be re-run on a properly licensed, attributed corpus (e.g. Saraga)
  before any claim stronger than "the pipeline runs end-to-end on real
  audio" is made.
- **No expert-verified symbolic ground truth.** Unlike a corpus with manual
  phrase transcription, accuracy here measures self-consistency of the
  model against the *authors' own* automated transcription, not agreement
  with an independent human transcription. This is a materially weaker
  evidentiary standard than an evaluation against expert annotation would
  provide (cf. Section 12.5).
- **Tonic estimation uncertainty.** Confidence scores for automatic tonic
  estimation were low throughout (0.03–0.23); errors here shift every
  subsequent pitch measurement by a constant offset per file and could
  systematically distort the resulting symbol distribution.
- **No grammar constraints tested.** This pass used the unconstrained
  Markov model only; the grammar-sensitivity analysis of Section 11.2 has
  not yet been run on real data.
- **Small, single-raga sample.** 30 sequences from one raga is not sufficient
  to generalize any conclusion beyond "the reconstruction machinery behaves
  as expected on this specific pipeline's output for this specific corpus."

These limitations do not undermine the paper's central mathematical claims
(Section 4.3's optimality theorem is verified computationally regardless of
data source, Section 9.5.3), but they do mean this section should be read as
an implementation feasibility check bridging the synthetic experiment and a
future properly licensed, expert-annotated archival evaluation (Section D.4),
not as a validation of historical or musicological authenticity.
