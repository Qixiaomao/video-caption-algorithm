# Additional paired seeds: locked execution protocol

Registered before new training on 2026-09-08. Scope: complete two additional
paired seeds (124 and 125), four runs, and preserve all outcomes. No test split
evaluation or hyperparameter selection in this stage.

Each seed constructs a fresh VideoCaptionModel after set_seed(seed), using
the same local pretrained GPT-2 and ViT. Frozen component hashes must equal
the audited seed-123 baseline. A and B strict-load the same per-seed initial
state. Both projection and mapper initial hashes must differ across seeds.

For seed s, batch RNG seed is s+333 (457 and 458). Caption permutation RNG for
video v is the existing reference_seed(v)+(s-123). Both arms use the resulting
same complete permutations. Training/dropout RNG seed is s. Validation donors
remain seed 321 and bootstrap/retrieval-control seeds remain 123.

Conditions: train1200, all 48,774 caption indices covered, val100/all 4,290
references; 66 epochs, 9,900 steps, batch8, AdamW lr3e-4 wd0.01. GPT-2 and ViT
backbone frozen; projection and mapper trainable. No AMP, scheduler or early
stopping. A uses caption CE; B uses CE+0.10*InfoNCE, tau0.07, exact-caption
multi-positive masking, normalized mean prefix versus frozen WTE mean.

Reuse hash-verified frozen ViT feature caches from full_msvd_balanced_20260905.
Check a direct train batch against cached features for each seed. Record all
source hashes, component hashes, schedule hashes and pre-forward traces.
Every paired trace field (including RNG) must match. Save epoch5/20/40/66
checkpoints and validation reports. Epoch66 is the sole primary endpoint.
Strict reload and component-hash equality are required for final checkpoints.

Smoke: two batches per arm for each seed, val8, identical paired trace and
initial CE/gradient/RNG checks. Formal runs are sequential, seed124 A then B,
seed125 A then B. A failed semantic gate is a scientific outcome, not a reason
to skip its B arm or the other seed. Nonfinite losses, hash/trace mismatch,
OOM or invalid coverage stop execution for engineering diagnosis.

Retain the existing six InfoNCE comparison criteria (B caption gate, NLL
<=A+0.15, token-F1>=A-0.05, retrieval R@1 and gap > A, R@1 > own shuffled
control), and report A's own caption gate separately. These are operational
criteria, not a statistical non-inferiority test. Report each pair without
selecting the best seed. Broader three-seed statistical synthesis and test
evaluation belong to subsequent stages. paper_ready remains false.
