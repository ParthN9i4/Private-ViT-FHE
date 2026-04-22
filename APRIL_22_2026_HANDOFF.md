# Session Handover — April 22, 2026

This file was generated at the end of a long Claude chat session on
April 22, 2026, capturing facts the assistant was confident about from
the conversation log. It is NOT a curated project history or original
lab documentation. Treat it as a starting point for the next session,
not an authoritative reference. Numbers below were produced by runs
completed during or shortly before this session; re-verify against the
JSONs in `experiment_results/` if anything is load-bearing for the paper.

---

## What happened this session (in order)

1. Strategic reality check on the paper (PPAI@AAAI 2027 target). Literature
   survey confirmed: Powerformer (ACL'25), PowerSoftmax (ICLR'25),
   PolyTransformer (ICML'24), Arion (ePrint'25), and EncFormer (April 2026)
   are the frontier. Our paper does not compete on accuracy or latency;
   it contributes a mechanistic failure analysis.

2. RetinaMNIST validation experiment (`fix2_retinamnist.py`) failed.
   Dataset too small (1,080 training samples) and too imbalanced; teacher
   only reached 34.7% balanced accuracy, no usable KD signal. Null result.
   Decision: switch to BloodMNIST.

3. BloodMNIST validation experiment (`fix2_bloodmnist.py`) completed on
   5 seeds x 3 configs x 150 epochs. Teacher 94.55% +/- 0.27% test_acc.
   Config E was metastable: 4/5 seeds pinned at val_bal = 12.50% for
   60–145 epochs before escaping; 1 seed (44) never escaped. Fix 2
   reached 94.84% +/- 0.21% -- matched teacher with 94x lower variance.
   Config E best-checkpoint reporting inflates its numbers; see
   `reanalyze_bloodmnist.py` for honest per-seed stability signature.

4. Mechanism experiment on BloodMNIST (`investigate_collapse_bloodmnist.py`)
   confirmed the CIFAR-10 finding but with inverted symptoms. On CIFAR-10,
   Config E activations *explode* (326x at layer 5). On BloodMNIST, they
   *collapse to a constant* (drop from 264 at layer 1 to ~6 at layer 2
   and stay there). Gradient norms at layer 5 drop 3 orders of magnitude
   below Config D in both cases. Same root cause (per-feature BatchNorm
   fails to control unbounded polynomial attention), opposite surface
   symptom. Figures in `figures/collapse_mechanism_bloodmnist.png` and
   `figures/collapse_dynamics_bloodmnist.png`.

5. During backup, re-discovered the March 2026 MedMNIST results
   (`results/`, `results_step2/`, ..., `results_step5b_coldstart_kd/`).
   These were NOT central to the paper plan when the session started.
   Six-dataset comparison across six pipeline stages including a
   warm-start vs. cold-start KD comparison. Summary JSONs are in
   `experiment_results/legacy_summaries/`.

## Verified numbers (BloodMNIST, 5 seeds, 150 epochs, seed 42 shared teacher)

| Config           | test_acc mean +/- std | test_bal mean +/- std |
|------------------|-----------------------|-----------------------|
| Teacher          | 94.55% +/- 0.27%      | 93.97% +/- 0.40%      |
| Config E         | 71.47% +/- 30.70%     | 68.82% +/- 32.84%     |
| Fix 2 Normalized | 94.84% +/- 0.21%      | 94.50% +/- 0.35%      |

Config E per-seed test_acc: 73.31, 82.81, 18.24, 90.18, 92.81.
These are best-checkpoint numbers. Most seeds spent most of
training pinned at random chance (val_bal 12.50%). The high mean
masks this; the high std is the honest signal.

## Verified numbers (CIFAR-10, from `verify_fixes.py`, 5 seeds)

| Config           | Mean   | Std    | p vs teacher |
|------------------|--------|--------|--------------|
| Teacher          | 76.54% | 3.86%  | ---          |
| Config E         | 40.08% | 20.33% | ---          |
| Fix 1 Clamped    | 69.33% | 2.59%  | 0.0005       |
| Fix 2 Normalized | 80.47% | 0.59%  | 0.0245       |
| Fix 3 RMSNorm    | 71.35% | 0.84%  | 0.0033       |

## Mechanism figure numbers (BloodMNIST, epoch 10, probe batch)

| Layer | Config D (LN) | Config E (BN) | Fix 2 |
|-------|---------------|---------------|-------|
| 0     | 3072          | 1234          | 8.0   |
| 1     | 1679          | 508           | 17.0  |
| 2     | 1979          | 6.6           | 24.7  |
| 3     | 725           | 4.8           | 22.4  |
| 4     | 1360          | 4.6           | 26.6  |
| 5     | 647           | 5.7           | 17.6  |

Config E layer-5 gradient norm: 10^-3 to 10^-4 (three orders of magnitude
below Config D's 10^-1).

## Open questions parked at end of session

1. **Paper structure (MUST discuss with Vineeth and Srinath).** Three distinct
   findings now exist: (a) interaction collapse + Fix 2, (b) cold-start KD
   principle from six-dataset MedMNIST study, (c) CKKS classification head
   soundness result. Unclear whether this is one paper or two.

2. **Cold-start KD result — might be confounded.** Possible confounders: same
   training epochs? same hyperparameters between step5 and step5b? n=1 per
   dataset (no std)? Did step5 use T^2 scaling while step5b did not? Need
   to read the actual training scripts before trusting the finding.

3. **Config E reporting for the paper.** Current checkpoint selector captures
   best-epoch accuracy, which hides the metastability. Proposed fix: report
   both "best checkpoint" AND "end of training" accuracy, or cut at a fixed
   epoch budget (e.g. 75) to show realistic-deployment accuracy. Not yet
   implemented.

4. **CIFAR-10 mechanism figure needs re-run.** The BloodMNIST version
   (`investigate_collapse_bloodmnist.py`) has more instrumentation than the
   original CIFAR-10 `investigate_collapse.py` (adds Fix 2 as a tracked
   config, adds MLP norms, adds gradient norms). To produce the two-column
   figure for the paper (CIFAR-10 explosion | BloodMNIST collapse), need
   to re-run the instrumented version on CIFAR-10. ~15 min of GPU time.

5. **Full-backbone CKKS encryption.** Classification head works (100% match,
   10^-5 error, `ckks_classification_head.py`). Full-backbone not attempted.
   Current plan: encrypt one full transformer block first (~10 levels),
   then all six blocks (~24–44 levels). May need N=2^15 or 2^16 ring dim.

## Advisor feedback to address

Vineeth flagged "combination novelty" as weak framing (earlier this year).
The mechanistic-finding framing ("reproducible failure mode, mechanistically
explained, direct-measurement evidence on two datasets") is the response.
The paper slide deck was refactored to this framing on April 21 -- see the
Beamer source at `poly_vit_ckks.tex` in outputs (not in repo; regenerate
if needed).

## What next session should probably do first

1. Verify the cold-start KD finding is real -- read step5_kd_poly_gelu.py
   and step5b_coldstart_kd.py side by side, confirm identical
   hyperparameters except for the init strategy.
2. Talk to Vineeth and Srinath about paper structure (one paper or two).
3. If one paper: produce the two-column mechanism figure by re-running
   the instrumented investigate script on CIFAR-10.
4. If two papers: decide which one targets PPAI@AAAI 2027 (Oct/Nov 2026
   deadline) and which one goes to a later main conference.

---
