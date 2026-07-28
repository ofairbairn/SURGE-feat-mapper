# TheMapper — Project Checklist

*Derived from the TheMapper guide. Organized by phase, in the order the guide recommends building.*

## Phase 0 — Wiring & baseline (now, ~a few days)
- [ ] Confirm the SURGE CNN Adapter + MNIST loader feed batches correctly
- [ ] Stand up PCA as the Rung-1 representation (no training needed)
- [ ] Wire a skeleton `run_mapper()`: PCA → k-means/HDBSCAN → a crude UMAP plot
- [ ] **Milestone:** first "the tool exists" moment — MNIST turned into a coloured 2-D map

## Phase 1 — Representation: finish AE/VAE (in progress, ~1 week)
- [ ] Finish the modular Autoencoder (AE) using the CNN encoder via the Adapter
- [ ] Build the VAE on top of that
- [ ] Benchmark reconstruction (MSE/MAE + eyeball comparison of x̂) across PCA vs AE vs VAE
- [ ] Pick the simplest rung with acceptable reconstruction (default assumption: VAE, for a smoother latent)
- [ ] **Deliverable:** representation module + reconstruction comparison table/plot

## Phase 2 — Core MVT: cluster + anomaly + Atlas on MNIST (~1 week)
- [ ] Cluster on `z` using HDBSCAN as the default method
- [ ] Add a quick k-means/GMM confirmer
- [ ] Validate clusters against the 10 known digit labels using ARI / NMI
- [ ] Build anomaly triage: reconstruction error + latent density → ranked list with reasons + human-review warning
- [ ] Sanity-check anomaly detection with an OOD test (held-out digit or Fashion-MNIST)
- [ ] Build the Atlas picture: UMAP/t-SNE of `z`, coloured by cluster and by anomaly score
- [ ] **Deliverable:** `run_mapper(MNIST)` produces the full MVT output — this is the demo
- [ ] Freeze this MNIST run as the known-good reference baseline

## Phase 3 — Data integration: swap in TokaMaker (NSTX-U) (triggered when data arrives, ~1–2 weeks out)
- [ ] Pin down "what is one sample?" *before* the data lands (full 2-D flux map → CNN Adapter; scalar equilibrium parameters → MLP encoder; 1-D radial profiles → stacked-channel CNN or concatenated MLP vector)
- [ ] Write `data/tokamaker_loader.py` behind the same interface as the MNIST loader (same batch shape/normalization contract)
- [ ] Normalize consistently (fit scaling on train data, apply everywhere)
- [ ] Re-run the frozen MNIST pipeline unchanged on TokaMaker
- [ ] Validate against the generated pseudo-ground-truth: do clusters/effective-#modes recover the designed regime families? Do injected off-distribution equilibria top the anomaly list?
- [ ] Produce the physics Atlas + triage list as the headline result
- [ ] *(Optional, can be done early while AE/VAE bakes)* Build a day-0 synthetic sanity toy (`make_blobs` + uniform noise) to test code paths: does Hopkins correctly call noise "random"? Is k = 1 allowed? Does Vendi ≈ 1 on duplicated points?

## Phase 4 — Enhancement layer (after the MVT runs on both datasets)
- [ ] Add cluster stability checks: bootstrap Jaccard + consensus → trust score per cluster
- [ ] Implement Vendi V1 — global Vendi Score + q-profile on `z` (install `vendi-score`, add VS as a column in the k-selection panel, plot the q-profile) — *highest priority, ship first*
- [ ] Implement Vendi V2 — marginal Vendi contribution as an anomaly signal (leave-one-out or rank-1-update loop); wire into the anomaly combiner as Signal 5; validate it separates "duplicated glitch" from "distinct novelty" on a toy example
- [ ] *(If time)* Implement Vendi V3 — within-/between-cluster Vendi as a diversity-based cluster-validity index (flags clusters to split or merge)
- [ ] *(Stretch)* Implement Vendi V4 — Vendi Sampling for diverse Atlas prototypes and diverse anomaly-triage ordering
- [ ] Add full k-selection confirmer panel: GMM/BIC, silhouette, gap statistic, CH, DB — check agreement
- [ ] Add Atlas persistence + Procrustes alignment so retrained latents stay consistently oriented over time
- [ ] Produce a short technical report and demo presentation

## Cross-cutting discipline (apply throughout, not phase-specific)
- [ ] Always cluster and compute Vendi Score on the latent `z` — never on the 2-D UMAP/t-SNE picture
- [ ] Check cluster tendency (Hopkins/VAT) before clustering; stop and report "no structure" if the data is random
- [ ] Prefer HDBSCAN (no forced k) before methods that require guessing k
- [ ] If a k must be chosen, confirm agreement across multiple indices and allow k = 1
- [ ] Treat anomaly output as a ranked list with reasons, never a binary yes/no verdict
- [ ] Always benchmark a fancier method against the PCA baseline before adopting it
- [ ] Keep a human-review step — the tool triages, it never issues an automatic "new physics" verdict

## Reference: full common-mistakes checklist (Section 13 of the guide)
- [ ] MVT running end-to-end before adding depth
- [ ] Features standardized first
- [ ] Correct encoder/adapter chosen (CNN for images/2-D fields, MLP for vectors)
- [ ] Clustering happens in `z`, not on the UMAP/t-SNE plot
- [ ] Cluster tendency checked before clustering
- [ ] HDBSCAN tried before forcing a `k`
- [ ] If `k` was picked, multiple indices agree and k = 1 was allowed
- [ ] Cluster stability checked under resampling
- [ ] Anomalies ranked with reasons, not a binary verdict
- [ ] Fancy model compared to the PCA baseline
- [ ] UMAP/t-SNE distances treated as meaningless
- [ ] Vendi Score computed on `z`, not the 2D plot
- [ ] Tool validated on MNIST (known answer) before trusting it on physics data
- [ ] Tool defers to the scientist for "bug vs. new physics" calls

---
*Note: WEST RF and M3D-C1 datasets are explicitly out of scope for now and deferred to future work.*
