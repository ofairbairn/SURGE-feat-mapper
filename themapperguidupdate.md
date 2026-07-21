# TheMapper — A Student's Guide to Automated Feature Discovery & Anomaly Detection

**Part of the SURGE (Surrogate Unified Robust Generation Engine) framework**
**Repository:** https://github.com/S-Villar/SURGE

---

## 0. How to read this document

This guide is written for someone who is **new to representation learning, clustering, and anomaly detection**. Every method is explained in three layers:

- **What it is** — a plain-language description.
- **Why we use it here** — its specific role in TheMapper.
- **Watch out** — the traps and misconceptions.

If you read nothing else, read **Section 2 (the mental model)** and **Section 10 (the recommended pipeline)**. For *what to build and in what order*, read **Section 12 (implementation plan)** and **Section 16 (dataset-integration phase)**. Everything else is the "why" behind those.

Boxes marked **🧭 KEY IDEA** are the load-bearing concepts. Boxes marked **⚠️ PITFALL** are mistakes that are easy to make and expensive to fix.

---

## 1. What is TheMapper trying to do?

Modern scientific facilities (e.g. fusion experiments, telescopes, accelerators) produce **huge, high-dimensional datasets**. Nobody can inspect every shot/experiment by hand. TheMapper is the component of SURGE that:

1. **Compresses** each high-dimensional data sample into a small "fingerprint" (a *latent vector*).
2. **Organizes** these fingerprints into a map — the **"Latent Atlas"** — where similar experiments sit close together.
3. **Discovers groups** (clusters) that may correspond to different physical *operating regimes*.
4. **Flags anomalies** — samples the model cannot explain — for a human scientist to review.

> 🧭 **KEY IDEA — TheMapper does not replace the scientist. It triages.**
> There is no purely automatic way to know whether an unusual data point is *corrupted junk* or *exciting new physics*. TheMapper's job is to **rank and highlight** what deserves human attention, and to say honestly when it finds **no structure at all**.

---

## 2. The mental model: the whole pipeline on one page

```
                        RAW SCIENTIFIC DATA
                        (high-dimensional x)
                                │
              ┌─────────────────┴─────────────────┐
              │   STEP A — REPRESENTATION           │
              │   Learn a compact "fingerprint" z   │
              │   PCA  →  Autoencoder  →  VAE        │
              └─────────────────┬─────────────────┘
                                │  z  (latent vector, e.g. 8–32 numbers)
              ┌─────────────────┴─────────────────┐
              │   STEP B — IS THERE ANY STRUCTURE?  │
              │   Cluster-tendency test (Hopkins/VAT)│
              └───────┬───────────────────┬────────┘
                      │ yes               │ no
                      │                   └──► report "no clustering found"
              ┌───────┴──────────┐
              │ STEP C — CLUSTER │   on z  (NOT on the 2D picture!)
              │ HDBSCAN / GMM /  │
              │ k-means          │
              └───────┬──────────┘
                      │ cluster labels
       ┌──────────────┼───────────────────────────┐
       │              │                            │
┌──────┴──────┐ ┌─────┴───────┐        ┌───────────┴──────────┐
│ STEP D      │ │ STEP E      │        │ STEP F — VISUALIZE   │
│ STABILITY   │ │ ANOMALY     │        │ UMAP / t-SNE of z    │
│ Is cluster  │ │ DETECTION   │        │ *display only*,      │
│ reproducible│ │ recon error │        │ colour by labels     │
│ (bootstrap)?│ │ + latent    │        └──────────────────────┘
└─────────────┘ │ distance    │
                └─────┬───────┘
                      │ ranked list of "please review this"
                      └──► HUMAN SCIENTIST
```

> 🧭 **KEY IDEA — Three separate decisions that people wrongly merge:**
> 1. **WHERE** do you cluster? → in the latent space `z`, **never** on the 2D UMAP/t-SNE picture.
> 2. **WHICH** algorithm? → prefer ones that don't force you to guess the number of groups.
> 3. **HOW MANY** clusters / are they real? → validate by *stability* and *human review*, not a single magic number.

---

## 3. Background concepts (start here if the words are new)

### 3.1 "High-dimensional data"
Each data sample is described by many numbers (a plasma profile might be temperature/density at hundreds of radial positions). A sample with 200 numbers is a **point in 200-dimensional space**. We cannot picture 200 dimensions, and distances behave strangely there (the *curse of dimensionality*): everything becomes almost equally far apart, which confuses clustering. This is *why* we compress first.

### 3.2 "Latent space" / "embedding"
A **latent vector** `z` is a short summary (say 16 numbers) that keeps the *essential* structure of the original 200 numbers. The collection of all `z` is the **latent space**. Think of it as a **compression** that keeps meaning, like describing a face with "round, bearded, glasses" instead of every pixel.

### 3.3 "Unsupervised learning"
We have **no labels** telling us which regime each experiment belongs to. The algorithm must find structure on its own. Clustering and anomaly detection are unsupervised.

### 3.4 "Reconstruction"
If a model can compress `x → z` **and** rebuild `z → x̂` accurately, then `z` captured the important information. The rebuild error `‖x − x̂‖` is our honesty check — and, later, an anomaly signal.

---

## 4. STEP A — Learning the representation (the "fingerprint")

We build a **ladder** of methods from simple to complex. **Always start at the bottom rung** and only climb if the simpler method is not good enough. This keeps you honest and gives you baselines.

### Rung 1 — PCA (Principal Component Analysis)
- **What it is:** A classical, *linear* method that finds the directions of greatest variation in the data and keeps the top few. It rotates the data so the first axis captures the most spread, the second the next most, etc.
- **Why we use it:** It is fast, deterministic, needs no training, and gives an instant baseline latent space and reconstruction error. Often it is *already good enough* for tabular scientific features.
- **Watch out:** It is **linear** — it cannot capture curved/nonlinear structure. If PCA reconstruction is poor, climb the ladder.
- *Refs:* Pearson 1901; Jolliffe 2002.

### Rung 2 — Autoencoder (AE)
- **What it is:** A neural network shaped like an hourglass. The **encoder** squeezes `x` down to a small `z` (the bottleneck); the **decoder** tries to rebuild `x` from `z`. Training adjusts the network to minimize reconstruction error.
- **Why we use it:** It learns **nonlinear** compression, so it can capture structure PCA misses. The bottleneck *is* our latent space.
- **Watch out:** A plain AE latent can be "lumpy" and arbitrarily scaled — distances in it are not always meaningful, and empty gaps between points can appear. That makes clustering and the Atlas harder.
- *Refs:* Hinton & Salakhutdinov 2006; Bengio, Courville & Vincent 2013.

```
   Autoencoder (hourglass):

   x  ─►┌────────┐          ┌────────┐─► x̂
        │ encoder│─►  z  ─► │ decoder│
   200 ─►└────────┘  16     └────────┘─► 200
   dims        (bottleneck)        dims
                (latent)
```

> 🧭 **KEY IDEA — MLP encoder vs. CNN encoder (the "Adapter" choice).** The encoder/decoder shape must match the *shape of one sample*:
> - **Vector / tabular sample** (e.g. scalar equilibrium parameters, a 1-D profile) → a plain **MLP** (fully-connected) encoder.
> - **Image / 2-D field sample** (e.g. an MNIST digit, a 2-D flux map) → a **convolutional (CNN)** encoder, which respects spatial structure and needs far fewer parameters than flattening the image into an MLP.
> In SURGE this choice is handled by the **CNN Adapter** (already implemented) that feeds image-like inputs into the AE/VAE. Whenever you move to a new dataset, the *first* question is "is one sample a vector or a 2-D field?" — that decides which adapter/encoder to use. See Section 16.

### Rung 3 — Variational Autoencoder (VAE)  ← recommended default for the Atlas
- **What it is:** An autoencoder that, instead of mapping `x` to a single point, maps it to a **small probability cloud** (a Gaussian) in latent space. A regularization term (the *KL divergence*) gently pushes all these clouds to overlap into one smooth, well-behaved space.
- **Why we use it:** The smoothness makes the latent space **continuous and distance-meaningful** — ideal for clustering, for building the Atlas, and for aligning versions over time (Procrustes, Section 10). It also gives a **probabilistic** reconstruction score we can reuse for anomaly detection.
- **Watch out:** Needs more data and care than an AE; can "over-smooth" (posterior collapse) and blur fine detail. Compare its reconstruction to the AE and PCA — don't assume fancier is better.
- *Refs:* Kingma & Welling 2014; Rezende, Mohamed & Wierstra 2014.

```
   VAE latent:  each x → a little Gaussian "cloud", not a point.
   KL term pulls all clouds toward a shared, smooth region:

        AE latent (lumpy)          VAE latent (smooth)
        •      •••                   • • • • •
            •         •      →      • • • • • •
          ••    •  •                 • • • • •
        (gaps, odd scale)         (continuous, isotropic)
```

### Rung 4 (stretch) — Disentangled / β-VAE, and Contrastive (SimCLR)
- **β-VAE:** turns up the KL pressure so individual latent dimensions try to line up with **interpretable physical quantities** (e.g. one axis ≈ density). Use only for the "physics-informed latent" stretch goal. *Ref:* Higgins et al. 2017.
- **Contrastive / SimCLR:** learns representations by teaching the model that two augmented views of the same sample should be close. Useful for **sparse/noisy** data where AE/VAE latents are unstable. *Ref:* Chen et al. 2020.

> 🧭 **KEY IDEA — Do you even need a VAE?**
> Maybe not. If PCA reconstruction is good and PCA+clustering finds sensible groups, **use PCA**. Climb the ladder only when a rung's reconstruction/clustering quality is insufficient. But note the project *deliverables* require an AE/VAE anyway (reconstruction error is a core anomaly signal), so build it — just also keep the PCA baseline for comparison.

> ⚠️ **PITFALL — "Deeper is better."** On small or mostly-linear scientific datasets, VAEs can *underperform* PCA and overfit. Always report reconstruction error (MSE/MAE) at each rung side by side.

---

## 5. STEP B — Before clustering: *is there any structure at all?*

Clustering algorithms **always return clusters, even from pure noise.** So we first test whether grouping is even meaningful. This is **cluster tendency**.

### Hopkins statistic
- **What it is:** A number roughly between 0 and 1 that compares how your data is spread versus a uniformly random cloud. **≈0.5 → random (no clusters).** Closer to 1 → real clustering tendency.
- **Why we use it:** A cheap gate. If it says "random," TheMapper should honestly report *"no significant clustering detected"* rather than inventing groups.
- *Refs:* Banerjee & Dave 2004.

### VAT / iVAT (Visual Assessment of cluster Tendency)
- **What it is:** Reorders the pairwise-distance matrix and shows it as an image. Real clusters appear as **dark square blocks along the diagonal**. The number of blocks hints at the number of clusters.
- **Why we use it:** A quick *visual* second opinion, and it suggests a plausible cluster count for later.
- *Refs:* Bezdek & Hathaway 2002.

```
   VAT image:   dark = similar, light = dissimilar

   ██▓░░░░░      3 dark blocks on the diagonal
   ▓██░░░░░   →  ≈ 3 clusters present
   ░░░██▓░░
   ░░░▓██░░
   ░░░░░░██
```

> ⚠️ **PITFALL — Skipping this step.** The single most common clustering mistake is running k-means on noise and "discovering" regimes that don't exist. Don't.

---

## 6. STEP C — Clustering (do it in `z`, never on the 2D picture)

> ⚠️ **PITFALL — Clustering the UMAP/t-SNE plot.**
> t-SNE and UMAP are for *looking*, not for *measuring*. They distort distances, densities, and gaps; they can create fake clusters and hide real ones. **Cluster in the latent space `z` (or PCA space). Use UMAP/t-SNE only to *display* the labels you already computed.** (See Wattenberg et al. 2016; Chari & Pachter 2023.)

### The three algorithms we offer, and when to use each

#### HDBSCAN — the recommended default
- **What it is:** A **density-based** method. It looks for regions where points are packed closely together and calls those clusters; sparse points in between are labeled **noise**. It figures out the **number of clusters by itself** and handles clusters of different shapes and sizes.
- **Why we use it:** (1) You don't have to guess the number of clusters. (2) It **naturally flags outliers as "noise"** — which doubles as a first anomaly signal (Step E). (3) It's built by the same author as UMAP and pairs well with it.
- **Watch out:** Sensitive to its `min_cluster_size` parameter; in very high dimensions run it on the compressed `z`, not raw `x`.
- *Refs:* Campello, Moulavi & Sander 2013; McInnes, Healy & Astels 2017. (Original DBSCAN: Ester et al. 1996.)

```
   Density clustering:  ●●● dense = cluster,   · sparse = noise

        ●●●●              ▲▲▲
       ●●●●●●    ·   ·   ▲▲▲▲▲     ·  ← noise / candidate anomalies
        ●●●●       ·      ▲▲▲
      cluster 1        cluster 2
```

#### Gaussian Mixture Model (GMM) — the probabilistic confirmer
- **What it is:** Assumes the data is a blend of several Gaussian "blobs" and estimates each blob's center, shape, and weight. Every point gets a **probability** of belonging to each blob (a *soft* assignment).
- **Why we use it:** Soft probabilities tell us which points are **ambiguous** (near a boundary) — useful for flagging uncertain samples. It pairs naturally with the VAE (both assume Gaussians). Model selection via **BIC** picks the number of blobs in a principled way.
- **Watch out:** Assumes elliptical blobs; struggles with weird shapes (where HDBSCAN shines).
- *Refs:* see McLachlan & Peel 2000; model selection via Schwarz 1978 (BIC).

#### k-means — the simple, fast baseline
- **What it is:** You tell it a number `k`; it finds `k` centers and assigns each point to the nearest one, iterating until stable.
- **Why we use it:** Fast, familiar, a good sanity baseline. Great when clusters really are round and similar-sized.
- **Watch out:** You **must** pick `k`; assumes round, equal-sized clusters; every point is forced into a cluster (no noise/outlier concept). This is exactly why it's a baseline, **not** the default.
- *Refs:* MacQueen 1967; Lloyd 1982.

### If you must choose `k` (for k-means or GMM): don't trust one number

Compute several indices and look for **agreement**, and always allow the answer **"k = 1" (no clusters):**

| Method | Idea | Note |
|---|---|---|
| **Elbow** (inertia vs k) | look for the "bend" | subjective; automate with **Kneedle** (Satopää 2011) |
| **Silhouette** | cohesion vs separation | intuitive; biased to round clusters (Rousseeuw 1987) |
| **Gap statistic** | compare to a random null | can say "no clusters"; more compute (Tibshirani et al. 2001) |
| **Calinski–Harabasz** | variance ratio | cheap confirmer (1974) |
| **Davies–Bouldin** | cluster similarity | cheap confirmer (1979) |
| **BIC** (for GMM) | penalized likelihood | principled model choice (Schwarz 1978) |
| **Vendi effective #modes** | diversity / effective count | label-free prior on k (see Section 15) |

```
   Elbow plot                    Silhouette
   inertia │•                     score │      •
           │ •                          │   •     •
           │  •___ ← "elbow" = k        │ •          •
           │     •‾•‾•‾•                 │•              •
           └───────────── k             └───────────────── k
                                        pick k at the peak
```

> 🧭 **KEY IDEA — "How many clusters?" is the wrong first question.**
> Ask instead: *Is there structure?* (Step B) → *Can an algorithm find it without me guessing k?* (HDBSCAN) → *Is the result reproducible?* (Step D). The number of clusters becomes an **output**, reported with a confidence, not a knob you twist until it looks nice.

---

## 7. STEP D — Are the clusters *real*? Stability validation

A cluster you found once might be an accident of this particular dataset. **Real clusters reappear** when you slightly perturb the data.

### Bootstrap stability (per-cluster Jaccard)
- **What it is:** Repeatedly take random subsamples of the data, re-run the clustering, and measure how often each cluster **reappears** (via the Jaccard overlap). Stable clusters score high; artifacts score low.
- **Why we use it:** It tells us **which individual clusters to trust** — crucial when a scientist asks "is *this* regime real?"
- *Refs:* Hennig 2007; overview in von Luxburg 2010.

### Consensus clustering
- **What it is:** Cluster many resampled versions, then build a matrix of "how often were points A and B grouped together." Choose the number of clusters that is **most reproducible**.
- **Why we use it:** A robust, defensible way to settle on a cluster count.
- *Refs:* Monti et al. 2003. (Related: prediction strength, Tibshirani & Walther 2005.)

```
   Co-association matrix (consensus):
   dark = "these two points almost always cluster together"

   ██░░░░     Two crisp blocks that survive resampling
   ██░░░░  →  = two trustworthy clusters
   ░░████
   ░░████
```

> 🧭 **KEY IDEA — Reproducibility is the closest thing to "truth" we have** without labels. Report a **stability score per cluster** alongside every result.

---

## 8. STEP E — Anomaly detection (the "please review this" list)

This is where TheMapper earns its keep for scientists. We combine **complementary signals**, because each catches different kinds of "weird."

### Signal 1 — Reconstruction error (needs AE/VAE)
- **What it is:** If the model rebuilds a sample poorly (`‖x − x̂‖` is large), the sample doesn't fit what the model learned as "normal."
- **Why we use it:** Directly answers "the model has never seen anything like this." Great for corrupted measurements **and** genuinely new regimes. With a VAE you can use the **reconstruction probability** (An & Cho 2015), which accounts for uncertainty.

### Signal 2 — Latent-space distance / density
- **What it is:** Points that sit far from every cluster, or in **low-density** latent regions, are suspicious. HDBSCAN gives an **outlier score (GLOSH)**; for GMM use **Mahalanobis distance** to the nearest component.
- **Why we use it:** Catches samples that reconstruct "okay" but land nowhere sensible in the organized map.
- *Refs:* Campello et al. 2015 (GLOSH).

### Signal 3 — Classical outlier detectors (optional confirmers)
- **Isolation Forest** (Liu et al. 2008) and **Local Outlier Factor** (Breunig et al. 2000) on `z` give cheap independent second opinions.

### Signal 4 — Abrupt change among look-alikes (your specific case)
- **What it is:** Samples that resemble their neighbors in most dimensions but show a **sudden jump** in one physical quantity. This is a **novelty** pattern — the point breaks a learned relationship.
- **Why we use it:** This is precisely the "is it a bug or new physics?" situation you described. TheMapper should **not** decide — it should **flag and rank** it for a scientist.

### Signal 5 — Marginal Vendi contribution (diversity-based)
- **What it is:** How much the dataset's *effective diversity* drops when you remove a point. Points that represent a genuinely new "kind" contribute a lot; near-duplicates contribute ≈0. See Section 15.
- **Why we use it:** It separates *"many copies of one glitch"* from *"several genuinely new regimes"* — a distinction the density signals cannot make on their own.

### How to combine them
1. Compute each signal, standardize to a common scale (e.g. percentile rank).
2. Combine into one **anomaly score** (a weighted sum, or "flag if any signal is extreme").
3. Output a **ranked triage list** (top-N most anomalous) plus a plain-language reason ("high reconstruction error + low latent density").
4. Attach a **warning** requesting human supervision — never an automatic "non-physical / new physics" verdict.

> 🧭 **KEY IDEA — Score and rank, don't judge.** A ranked list of the 20 strangest experiments with reasons is far more useful (and honest) than a binary "anomaly: yes/no." Anomaly-detection reviews (Ruff et al. 2021; Pang et al. 2021; Chandola et al. 2009) all frame the output as scores for human adjudication.

```
   Anomaly triage output (what the scientist sees):

   rank  sample   score  reason
   ────────────────────────────────────────────────
    1    #4471    0.98   recon err ↑↑  + latent density ↓↓   ⚠ review
    2    #1188    0.91   HDBSCAN noise + Mahalanobis ↑        ⚠ review
    3    #0902    0.87   abrupt jump in density vs neighbors  ⚠ review
   ...
```

---

## 9. STEP F — Visualization (UMAP / t-SNE): looking, not measuring

- **t-SNE** — great at revealing local neighborhoods and separated blobs; **distorts** global distances and cluster sizes. (van der Maaten & Hinton 2008; Wattenberg et al. 2016.)
- **UMAP** — faster, keeps a bit more global structure; still not distance-faithful. (McInnes et al. 2018.)

**Use them only to *display* the latent space and paint each point by the cluster label / anomaly score you computed in `z`.** If a cluster from Step C also looks coherent in the UMAP picture, that's reassuring — but the picture is never the source of truth.

> ⚠️ **PITFALL — Reading distances off a t-SNE/UMAP plot.** "Cluster A is twice as far from B as from C" is **not** a valid statement about the real data. Sizes and gaps in these plots are largely meaningless.

---

## 10. The recommended pipeline (put it all together)

```
1. PREPROCESS      scale/standardize features; handle missing values
2. REPRESENT       PCA baseline  →  AE  →  VAE  (compare reconstruction!)
                   choose the simplest rung with acceptable reconstruction
2b.DIVERSITY       Vendi Score on z (+ q-profile) → effective #modes, prior on k
3. TENDENCY        Hopkins + VAT on z   → if "random", STOP & report no structure
4. CLUSTER (on z)  HDBSCAN (default)  + GMM/k-means as confirmers
                   if picking k: agree across silhouette/gap/CH/DB/BIC/Vendi, allow k=1
4b.VALIDATE        within/between-cluster Vendi → diversity quality per cluster
5. STABILITY       bootstrap Jaccard per cluster (+ consensus) → keep a score
6. ANOMALY         recon error + latent density/GLOSH (+ IsoForest/LOF)
                   + marginal Vendi contribution → combined score → ranked triage
7. VISUALIZE       UMAP/t-SNE of z, coloured by cluster & anomaly score (display only)
8. HUMAN REVIEW    scientist inspects stable clusters + top anomalies
9. ATLAS + ALIGN   store labels/scores; Procrustes-align latents across retrains;
                   Vendi-sample diverse prototypes for the Atlas
```

### On version-to-version consistency (Procrustes)
When you retrain the model, the latent space can come out **rotated/flipped** even if the structure is the same. **Procrustes analysis** finds the best rotation/scaling to line up the new latent with the old one, so "the same regime stays in the same place" in the Atlas over time. Works best on a **smooth VAE latent**. (Schönemann 1966; Gower 1975.)

---

## 11. Suggested software / module layout

| Purpose | Library |
|---|---|
| AE/VAE training | **PyTorch** |
| PCA, k-means, GMM, silhouette/CH/DB, IsolationForest, LOF | **scikit-learn** |
| HDBSCAN + GLOSH outlier scores | **hdbscan** |
| UMAP | **umap-learn** |
| t-SNE | **scikit-learn** / **openTSNE** |
| Hopkins / VAT | **pyclustertend** |
| Elbow auto-detection | **kneed** |
| Diversity (Vendi Score) | **vendi-score** (github.com/vertaix/Vendi-Score) |
| Graph clustering (optional, Leiden) | **leidenalg** / **scanpy** |
| Image datasets & loaders (MNIST, etc.) | **torchvision** |

**Suggested modules inside TheMapper:**
```
themapper/
├── data/          # loaders: mnist_loader.py, tokamaker_loader.py
├── adapters/      # cnn_adapter.py (image/2-D field → encoder), mlp_adapter.py (vector)
├── represent/     # pca.py, autoencoder.py, vae.py  (+ shared reconstruction metrics)
├── tendency/      # hopkins.py, vat.py
├── cluster/       # hdbscan_.py, gmm.py, kmeans.py, select_k.py
├── stability/     # bootstrap_jaccard.py, consensus.py
├── anomaly/       # recon_error.py, latent_density.py, combine_scores.py
├── diversity/     # vendi.py, vendi_cluster.py, vendi_anomaly.py, vendi_sample.py
├── viz/           # umap_.py, tsne_.py, plots.py
├── align/         # procrustes.py
├── report/        # atlas.py (save/serve the Latent Atlas), triage_report.py
└── pipeline.py    # run_mapper(dataset) → runs the whole thing end-to-end
```
*Already implemented in SURGE:* the **CNN Adapter** (`adapters/`) and an **MNIST loader** (`data/`). New datasets are added as new loaders behind the same interface so the rest of the pipeline is dataset-agnostic (see Section 16).

---

## 12. Implementation plan (build a working tool, then point it at the data)

> **Priority: show the tool working end-to-end, fast.** This plan is organized into **phases that each produce something demonstrable**, not a method-per-week syllabus. The goal is a runnable `run_mapper(dataset)` that goes raw data → latent → clusters → anomaly triage → Atlas picture, first on **MNIST**, then swapped onto the **TokaMaker (NSTX-U)** data the moment it arrives (~1–2 weeks).
>
> **Current state:** ✅ CNN Adapter + ✅ MNIST loader implemented; 🔧 AE/VAE in progress.

### The Minimum Viable Tool (MVT) — this is the definition of "the tool works"
`run_mapper(dataset)` takes a dataset and produces, end-to-end and reproducibly:
1. a trained representation (`z`) with a reconstruction score,
2. cluster labels computed **on `z`**,
3. a ranked **anomaly triage list** with reasons,
4. a **UMAP/t-SNE Atlas picture** coloured by cluster and by anomaly score.

Everything else in this guide (stability, GMM/k-means confirmers, full Vendi suite, Procrustes) is an **enhancement layer** added *after* the MVT runs.

---

### Phase 0 — Wiring & baseline *(now; ~a few days)*
**Goal:** the plumbing runs end-to-end on MNIST with the simplest possible representation.
- Confirm the SURGE **CNN Adapter + MNIST loader** feed batches correctly.
- Stand up **PCA** as the Rung-1 representation (no training) so the *rest* of the pipeline can be built and tested immediately.
- Wire a skeleton `run_mapper()` that does PCA → k-means/HDBSCAN → a UMAP plot, even if crude.
- **Deliverable:** one script/notebook that turns MNIST into a coloured 2-D map. *This is your first "the tool exists" moment.*

### Phase 1 — Representation (finish AE/VAE) *(in progress; ~1 week)*
**Goal:** replace PCA with a learned latent and prove it's better.
- Finish the modular **AE**, then the **VAE**, using the CNN encoder via the Adapter.
- Benchmark reconstruction (MSE/MAE, + eyeball `x̂`) for **PCA vs AE vs VAE**. Pick the simplest rung with acceptable reconstruction (default: VAE for the smooth latent).
- **Deliverable:** representation module + a reconstruction comparison table/plot.

### Phase 2 — Core MVT: cluster + anomaly + Atlas on MNIST *(~1 week)*
**Goal:** the full MVT runs on MNIST and is *demonstrably correct* against known labels.
- **Cluster on `z`:** HDBSCAN as default (+ a quick k-means/GMM confirmer). Validate against the 10 digit labels with **ARI / NMI**.
- **Anomaly triage (MVT slice):** reconstruction error + latent density → ranked list with reasons + human-review warning. Sanity-check with an OOD test (held-out digit or Fashion-MNIST).
- **Atlas picture:** UMAP/t-SNE of `z` coloured by cluster and by anomaly score (display only).
- **Deliverable:** `run_mapper(MNIST)` produces the full MVT output. **This is the demo you show.**

> 🧭 **KEY IDEA — Freeze the MNIST demo as your reference.** Once Phase 2 works on MNIST, you have a *known-good* baseline. When you later swap in TokaMaker, anything that breaks is a data/adapter issue, not a method bug.

### Phase 3 — DATA INTEGRATION: swap in TokaMaker (NSTX-U) *(triggered when the data arrives, ~1–2 weeks out — see Section 16)*
**Goal:** the exact same `run_mapper()` runs on the physics data.
- Add a **`tokamaker_loader.py`** behind the same interface as the MNIST loader.
- Decide **"what is one sample?"** → choose CNN Adapter (2-D flux map) or MLP (parameter vector). See Section 16.4.
- Re-run the frozen pipeline; check that clusters/anomalies recover the **regime families you generated** (your pseudo-ground-truth).
- **Deliverable:** `run_mapper(TokaMaker)` produces clusters + anomaly triage + Atlas on the real data.

### Phase 4 — Enhancement layer *(after the MVT runs on both datasets)*
Add depth **only once the tool demonstrably works** on both datasets:
- **Robustness:** cluster **stability** (bootstrap Jaccard + consensus) → trust scores per cluster.
- **Diversity (Vendi):** add **V1** (effective #modes → prior on k) and **V2** (marginal contribution → anomaly Signal 5). See Section 15. These are high-value, low-code.
- **Confirmers & full k-selection:** GMM/BIC, silhouette/gap/CH/DB agreement panel.
- **Atlas persistence + Procrustes** alignment across retrains; optionally **V3/V4** Vendi.
- **Deliverable:** the enhanced tool + a short technical report and demo presentation.

### One-page phase map
```
PHASE 0  Wiring & baseline (PCA)        → tool runs end-to-end (crude)     [days]
PHASE 1  Representation (AE→VAE)         → learned latent beats PCA         [~1 wk]  (in progress)
PHASE 2  Core MVT on MNIST              → cluster+anomaly+Atlas, validated  [~1 wk]  ★ DEMO
PHASE 3  DATA INTEGRATION: TokaMaker    → same tool on physics data        [when data lands]
PHASE 4  Enhancement (stability, Vendi, → depth & polish                   [after MVT]
         confirmers, Procrustes, Atlas)
```

> 🧭 **KEY IDEA — Benchmark at every step.** Always compare each new (fancier) method to the previous baseline on the *same* reconstruction and clustering metrics. Complexity must earn its place — and it must never break the MVT.

---

## 13. Common mistakes checklist (pin this above your desk)

- [ ] Did I get the **MVT running end-to-end** before adding depth? (Section 12)
- [ ] Did I **standardize** features first?
- [ ] Did I pick the right **encoder/adapter** (CNN for images/2-D fields, MLP for vectors)?
- [ ] Am I clustering in **`z`**, not on the UMAP/t-SNE plot?
- [ ] Did I check **cluster tendency** before clustering?
- [ ] Did I try **HDBSCAN** (no k needed) before forcing a `k`?
- [ ] If I picked `k`, do **several indices agree**, and did I allow **k = 1**?
- [ ] Did I check **cluster stability** under resampling?
- [ ] Are anomalies **ranked with reasons**, not a binary verdict?
- [ ] Did I compare the fancy model to the **PCA baseline**?
- [ ] Am I treating UMAP/t-SNE distances as **meaningless**?
- [ ] Did I compute the **Vendi Score on `z`, not on the 2D plot**? (Section 15)
- [ ] Did I **validate on MNIST (known answer)** before trusting the tool on physics data? (Section 16)
- [ ] Does the tool **defer to the scientist** for "bug vs new physics"?

---

## 14. References (full citations)

**Visualization / embeddings (visualization only)**
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). *How to Use t-SNE Effectively.* Distill. https://doi.org/10.23915/distill.00002
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE.* Journal of Machine Learning Research, 9, 2579–2605.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426.
- Chari, T., & Pachter, L. (2023). *The specious art of single-cell genomics.* PLoS Computational Biology, 19(8), e1011288. https://doi.org/10.1371/journal.pcbi.1011288

**Dimensionality reduction / representation learning**
- Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine, 2(11), 559–572.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with Neural Networks.* Science, 313(5786), 504–507.
- Bengio, Y., Courville, A., & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives.* IEEE TPAMI, 35(8), 1798–1828.
- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR. arXiv:1312.6114.
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models.* Proc. 31st ICML, 1278–1286.
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* Proc. 37th ICML, 1597–1607.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324. (MNIST / CNNs)

**Cluster tendency**
- Banerjee, A., & Dave, R. N. (2004). *Validating clusters using the Hopkins statistic.* IEEE Int. Conf. on Fuzzy Systems, 1, 149–153.
- Bezdek, J. C., & Hathaway, R. J. (2002). *VAT: A tool for visual assessment of (cluster) tendency.* Proc. IJCNN, 2225–2230.
- Hartigan, J. A., & Hartigan, P. M. (1985). *The Dip Test of Unimodality.* The Annals of Statistics, 13(1), 70–84.

**Clustering algorithms**
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proc. 5th Berkeley Symposium, 1, 281–297.
- Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129–137.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN).* Proc. 2nd KDD, 226–231.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN).* PAKDD, LNCS 7819, 160–172.
- McInnes, L., Healy, J., & Astels, S. (2017). *hdbscan: Hierarchical density based clustering.* Journal of Open Source Software, 2(11), 205.
- Campello, R. J. G. B., Moulavi, D., Zimek, A., & Sander, J. (2015). *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection (GLOSH).* ACM TKDD, 10(1), 1–51.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models.* Wiley. (Gaussian Mixture Models)
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities.* Scientific Reports, 9, 5233.

**Choosing k / cluster validity**
- Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). *Estimating the number of clusters in a data set via the gap statistic.* JRSS-B, 63(2), 411–423.
- Caliński, T., & Harabasz, J. (1974). *A dendrite method for cluster analysis.* Communications in Statistics, 3(1), 1–27.
- Davies, D. L., & Bouldin, D. W. (1979). *A Cluster Separation Measure.* IEEE TPAMI, PAMI-1(2), 224–227.
- Schwarz, G. (1978). *Estimating the Dimension of a Model.* The Annals of Statistics, 6(2), 461–464. (BIC)
- Pelleg, D., & Moore, A. (2000). *X-means: Extending K-means with Efficient Estimation of the Number of Clusters.* Proc. 17th ICML, 727–734.
- Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.* 31st ICDCSW, 166–171.

**Stability / consensus validation**
- Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). *Consensus Clustering.* Machine Learning, 52, 91–118.
- Hennig, C. (2007). *Cluster-wise assessment of cluster stability.* Computational Statistics & Data Analysis, 52(1), 258–271.
- Tibshirani, R., & Walther, G. (2005). *Cluster Validation by Prediction Strength.* Journal of Computational and Graphical Statistics, 14(3), 511–528.
- von Luxburg, U. (2010). *Clustering Stability: An Overview.* Foundations and Trends in Machine Learning, 2(3), 235–274.

**Anomaly / novelty detection**
- Ruff, L., Kauffmann, J. R., Vandermeulen, R. A., Montavon, G., Samek, W., Kloft, M., Dietterich, T. G., & Müller, K.-R. (2021). *A Unifying Review of Deep and Shallow Anomaly Detection.* Proceedings of the IEEE, 109(5), 756–795.
- Pang, G., Shen, C., Cao, L., & van den Hengel, A. (2021). *Deep Learning for Anomaly Detection: A Review.* ACM Computing Surveys, 54(2), 1–38.
- Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly Detection: A Survey.* ACM Computing Surveys, 41(3), 1–58.
- An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection using Reconstruction Probability.* SNU Data Mining Center, Tech. Report 2015-2.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* Proc. 8th IEEE ICDM, 413–422.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers.* Proc. ACM SIGMOD, 93–104.

**Diversity (Vendi Score / Vertaix)**
- Friedman, D., & Dieng, A. B. (2023). *The Vendi Score: A Diversity Evaluation Metric for Machine Learning.* Transactions on Machine Learning Research (TMLR). arXiv:2210.02410.
- Pasarkar, A. P., & Dieng, A. B. (2024). *Cousins of the Vendi Score: A Family of Similarity-Based Diversity Metrics for Science and Machine Learning.* AISTATS. arXiv:2310.12952.
- Pasarkar, A. P., Bencomo, G. M., Olsson, S., & Dieng, A. B. (2023). *Vendi Sampling for Molecular Simulations: Diversity and Precision.* arXiv:2310.19952.
- Hill, M. O. (1973). *Diversity and Evenness: A Unifying Notation and Its Consequences.* Ecology, 54(2), 427–432. (background: "effective number" / Hill numbers)
- Code: Vertaix, *Vendi-Score.* https://github.com/vertaix/Vendi-Score

**Deep / joint clustering (optional advanced)**
- Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis (DEC).* Proc. 33rd ICML, 478–487.
- Jiang, Z., Zheng, Y., Tan, H., Tang, B., & Zhou, H. (2017). *Variational Deep Embedding (VaDE).* Proc. IJCAI, 1965–1972.
- Min, E., Guo, X., Liu, Q., Zhang, G., Cui, J., & Long, J. (2018). *A Survey of Clustering with Deep Learning.* IEEE Access, 6, 39501–39514.

**Alignment (Latent Atlas across versions)**
- Schönemann, P. H. (1966). *A generalized solution of the orthogonal Procrustes problem.* Psychometrika, 31(1), 1–10.
- Gower, J. C. (1975). *Generalized Procrustes Analysis.* Psychometrika, 40(1), 33–51.

---

## 15. Diversity with the Vendi Score (Vertaix) — prioritized add-ons

This section adds **diversity-based** capabilities to TheMapper using the **Vendi Score (VS)** from Vertaix. It is written so the intern can add real value *incrementally*, starting with the highest **profit-to-implementation-complexity ratio**. These belong in **Phase 4** (the enhancement layer) — after the MVT runs.

### 15.1 What the Vendi Score is (in plain language)
The Vendi Score answers **"how many *effectively distinct* things are in this set?"** — with **no labels, no reference dataset, and no cluster assignment required.** You only need a way to measure **similarity** between samples.

- Build an `n×n` similarity matrix `K` of your samples (e.g. cosine or RBF on the latent `z`), scale it by `1/n`.
- Its eigenvalues behave like a probability distribution (they sum to 1).
- `VS = exp(Shannon entropy of those eigenvalues)`.

Interpretation — VS is an **effective count**:
- 100 identical samples → VS ≈ **1** (one distinct thing).
- 100 completely different samples → VS ≈ **100**.
- Real data lands in between, e.g. VS ≈ 3.4 = "behaves like ~3–4 distinct populations."

The **q-order Vendi ("Vendi distribution")** is a *family* of these numbers: a knob `q` controls whether rare modes count as much as common ones (like ecology's Hill numbers). Sweeping `q` gives a **diversity profile** curve, richer than a single number. (Friedman & Dieng 2023; Pasarkar & Dieng 2024; background: Hill 1973.)

> 🧭 **KEY IDEA — Clustering asks "which group is each point in?" The Vendi Score asks "how many effectively distinct things are here?"** The second question is often what "how many populations?" really means — and it needs no partition.

```
   VS as an "effective number of populations":

   all identical        two balanced modes       many distinct
   ●●●●●●●●●             ●●●●   ▲▲▲▲               ● ▲ ■ ◆ ✦ ✚ ...
   VS ≈ 1               VS ≈ 2                    VS ≈ many
```

> ⚠️ **PITFALL — same rule as clustering:** compute VS on the **latent `z`** (or raw features with a domain kernel), **never on the 2D UMAP/t-SNE coordinates**. The kernel choice (cosine, RBF bandwidth) affects the number, so **always report the kernel**.

### 15.2 The four Vendi features, ranked by profit / complexity

The table below is the decision aid. **Tier 1 = do these first** (best bang for the buck). Tiers descend in ratio.

| # | Feature | What it gives you (profit) | Effort | Profit/Complexity | Tier |
|---|---|---|---|---|---|
| V1 | **Global Vendi + q-profile on `z`** | Label-free "effective #modes" → an independent vote on *k* and a one-line dataset diversity summary | Very low (a few lines calling `vendi-score`) | ★★★★★ | **1** |
| V2 | **Marginal Vendi contribution as an anomaly signal** | New, *complementary* anomaly score that separates "many copies of one glitch" from "several new regimes" | Low (leave-one-out loop / rank-1 update) | ★★★★☆ | **1** |
| V3 | **Within- / between-cluster Vendi (cluster quality)** | Diversity-based cluster validity: flags clusters that should be split or merged | Low–medium (reuse V1 per cluster) | ★★★☆☆ | **2** |
| V4 | **Vendi Sampling for the Atlas & triage ordering** | Diverse prototypes per cluster; anomaly list ordered to show *varied* cases first, not near-duplicates | Medium (greedy selection loop) | ★★☆☆☆ | **3** |

Rationale for the ranking: **V1 and V2 reuse the exact same one-function VS computation** and each unlocks a distinct, high-value capability (better *k* selection; a genuinely new anomaly signal), so their ratio is highest. V3 is the same computation applied per-cluster — cheap, but its payoff (validity index) overlaps with tools you already have. V4 is the most code and adds polish rather than new detection power, so it is last.

---

### V1 — Global Vendi Score + q-profile (Tier 1, do first)

- **What it is:** One VS number on all of `z`, plus a `VS(q)` curve for `q` in, say, `{0.1, 0.5, 1, 2, 5, ∞}`.
- **Why we use it:** (1) A label-free **effective number of populations** → add it as an extra column in the "how many clusters?" agreement panel (Section 6). If silhouette says k=8 but VS≈3, most of those 8 are near-duplicates. (2) A single, quotable **dataset diversity summary** for reports and for comparing datasets/model versions.
- **How (sketch):**
  ```
  from vendi_score import vendi
  vs = vendi.score_dual(Z)            # or score(K) with your kernel matrix K
  # q-profile:
  profile = {q: vendi.score(K, q=q) for q in [0.1,0.5,1,2,5]}
  ```
- **Deliverable:** `diversity/vendi.py` returning `vs` and the q-profile; a plot of `VS(q)`.

```
   q-profile (Vendi distribution):
   VS(q)│•                     flat curve  → balanced modes
        │ •___                  steep drop → one dominant mode,
        │     •‾‾•‾‾•                        rest are rare
        └────────────── q
       0   1    2    5
```

### V2 — Marginal Vendi contribution as an anomaly signal (Tier 1, do second)

- **What it is:** For each point `i`, `Δᵢ = VS(all) − VS(all without i)`. Large positive `Δᵢ` = removing this point noticeably lowers the effective number of populations → it represents a **distinct kind** on its own.
- **Why we use it:** It is **complementary** to reconstruction error and latent density:
  - Recon error: "the model can't rebuild it."
  - Latent density (GLOSH/Mahalanobis): "it's in a sparse region."
  - **Vendi contribution: "it adds a *new kind* of thing."**
  A cluster of 5 identical corrupted readings looks anomalous by density, but has **low aggregate Vendi contribution** (diverse from normal data, but not from each other) — which is exactly how you tell **"5 copies of one glitch" from "5 genuinely new regimes."** This is your "bug vs. new physics" triage need, made quantitative. Add it as **Signal 5** in Section 8.
- **How (sketch):** naive leave-one-out recomputes VS `n` times (`O(n·n³)` — fine for a few thousand points on `z`); for larger `n` use eigen/rank-1 update tricks or subsample.
  ```
  base = vendi.score(K)
  contrib = [base - vendi.score(delete_row_col(K, i)) for i in range(n)]
  # standardize to percentile rank, feed into anomaly combiner
  ```
- **Deliverable:** `diversity/vendi_anomaly.py` producing a per-point Vendi-contribution score wired into `anomaly/combine_scores.py`.

```
   Marginal Vendi contribution:

   typical point in a dense mode → removing it barely changes VS → Δ ≈ 0
   lone novel point (new kind)   → removing it drops VS          → Δ large
   5 identical glitches          → each removal barely changes VS → Δ small
                                    (they cover each other) ⇒ "copies", not new regimes
```

### V3 — Within-/between-cluster Vendi (Tier 2)

- **What it is:** VS computed *inside* each cluster (internal diversity) and *across* cluster prototypes (separation).
- **Why we use it:** A diversity-based **cluster validity** check that complements silhouette/stability:
  - **High within-cluster VS** → the cluster is probably a merge of sub-populations → candidate to **split**.
  - **Low between-cluster VS** (relative to #clusters) → some clusters are near-duplicates → candidate to **merge**.
- **How:** reuse V1 on each cluster's members and on the set of centroids/medoids.
- **Deliverable:** `diversity/vendi_cluster.py` returning per-cluster internal VS + a between-cluster VS, reported next to the Section 6 indices.

### V4 — Vendi Sampling for the Atlas & triage ordering (Tier 3)

- **What it is:** Greedily pick a subset that **maximizes** the Vendi Score → a maximally diverse, minimally redundant set.
- **Why we use it:** (1) Choose **representative prototypes per cluster** for the Atlas display. (2) **Order the anomaly triage list** so the scientist sees *diverse* anomalies first instead of 20 near-identical ones. (3) Build a **diverse validation/benchmark set** covering all regimes.
- **How:** greedy forward selection adding the point that increases VS the most, until budget reached.
- **Deliverable:** `diversity/vendi_sample.py` used by the Atlas/prototype code and the triage-ordering step.

---

### 15.3 Where these plug into the pipeline (recap)

```
2b. DIVERSITY  V1: VS(z) + q-profile → effective #modes (prior on k, tendency backup)
4b. VALIDATE   V3: within/between-cluster VS → diversity cluster-quality index
6+. ANOMALY    V2: marginal Vendi contribution → Signal 5 in the combiner
7b. ATLAS      V4: Vendi Sampling → diverse prototypes + diverse triage ordering
```

### 15.4 Suggested order for the Vendi work (Phase 4)
1. Install `vendi-score`; implement **V1** on `z`; add VS as a column in the k-selection panel; plot the q-profile. *(highest ratio, ship first)*
2. Implement **V2** (marginal contribution); wire into the anomaly combiner; show on a toy example that it separates "duplicated glitch" from "distinct novelty."
3. If time: **V3** cluster diversity index.
4. Stretch / could-have: **V4** Vendi Sampling for prototypes and triage ordering.

> 🧭 **KEY IDEA — Feed Vendi the same `z` you cluster in, with a documented kernel.** VS is only as meaningful as the embedding and kernel behind it — exactly the same discipline as "cluster in `z`, not on the 2D plot."

---

## 16. Dataset-integration phase (MNIST → TokaMaker NSTX-U)

Only **two datasets** are in scope: **MNIST** (to build and prove the tool) and **TokaMaker (NSTX-U)** (the real target). MNIST is the *test harness*; TokaMaker is the *deliverable dataset*, arriving in ~1–2 weeks. This section is the detailed spec for **Phase 3** of the implementation plan (Section 12).

### 16.1 Two-dataset strategy at a glance

| Dataset | Role | When | Why |
|---|---|---|---|
| **MNIST** | Build + prove the tool (test harness) | now → Phase 2 | Known answer (10 classes), trivial to load, exercises the CNN Adapter + full ladder, lets you *validate* the pipeline before physics |
| **TokaMaker (NSTX-U)** | Real target dataset | Phase 3, when data lands (~1–2 wks) | Your generated set with **controllable pseudo-ground-truth** — you can check the tool recovers the regime families you designed; flux maps reuse the CNN Adapter |

> 🧭 **KEY IDEA — MNIST is the rehearsal; TokaMaker is the show.** Build the whole tool on MNIST because you *know the answer* there. The moment TokaMaker arrives, you should only have to add a loader and re-run — the method is already trusted.
>
> *(WEST RF and M3D-C1 are out of scope for now; they can be revisited as future work once TheMapper is validated on these two datasets.)*

### 16.2 What MNIST is for (build + validate the MVT)
Run the whole MVT (Section 12, Phase 2) on MNIST and confirm each part against ground truth:
1. **Represent:** PCA vs AE vs VAE reconstruction (MSE + eyeball `x̂`). The deep models should clearly win → justifies the VAE.
2. **Cluster on `z`:** HDBSCAN/GMM → does it find ~10 groups? Score against the digit labels with **ARI / NMI**. (4/9/7 often merge — a good "clusters ≠ classes" lesson.)
3. **Anomaly triage:** reconstruction error + latent density → ranked list; OOD sanity test with a **held-out digit** or **Fashion-MNIST** (both should top the list).
4. **Atlas picture:** UMAP/t-SNE of `z` coloured by cluster & anomaly score.

*(Phase-4 Vendi checks, when you get there: V1 effective #modes should land near ~10; V2 should flag the held-out digit as a "new kind" and separate duplicated-glitch from genuine novelty.)*

> ⚠️ **PITFALL — Don't over-fit your intuition to MNIST.** Its clusters are cleaner than physics data will ever be. MNIST proves the *plumbing and logic*, not that the tool will look this tidy on TokaMaker.

### 16.3 The integration checklist (Phase 3, when TokaMaker arrives)
1. **Write `data/tokamaker_loader.py`** behind the **same interface** as the MNIST loader (same batch shape/normalization contract), so `run_mapper()` doesn't change.
2. **Decide "what is one sample?"** (Section 16.4) → pick the CNN Adapter or an MLP encoder.
3. **Normalize consistently** (fit scaling on train, apply to all).
4. **Re-run the frozen MNIST pipeline** unchanged on TokaMaker.
5. **Validate against your generated regimes:** do the clusters / Vendi effective-#modes recover the parameter families you designed? Do injected off-distribution equilibria top the anomaly list?
6. **Produce the physics Atlas** + triage list as the headline result.

### 16.4 The one decision to pin down before TokaMaker: *what is one "sample"?*
This dictates the encoder/adapter — settle it *before* the data lands so integration is fast:
- **A full 2-D flux map / field image** → **CNN Adapter** (same path as MNIST). *Recommended if available* — reuses everything and is the most faithful representation.
- **A vector of scalar equilibrium parameters** (Ip, β, elongation, triangularity, …) → **MLP encoder** (add an `mlp_adapter`).
- **A set of 1-D radial profiles** → either stack as channels (1-D conv / CNN) or concatenate to a vector (MLP).

The same discipline as MNIST applies: **standardize each sample**, and **compute Vendi and cluster in `z`, not on the 2-D plot**.

### 16.5 Optional day-0 sanity toy
A tiny synthetic set (`scikit-learn make_blobs` for clusters + a uniform-noise set for "no structure") lets you verify the *code paths* in an afternoon while AE/VAE are still baking: does Hopkins say "random" on noise? is **k = 1** allowed? does Vendi ≈ 1 on duplicated points? This makes the "is there structure at all?" lesson (Section 5) concrete before any real data.

---

*Document prepared for the SURGE / TheMapper internship. Scope: two datasets — MNIST (test harness) → TokaMaker NSTX-U (target, ~1–2 weeks out). Current state: CNN Adapter + MNIST loader implemented; AE/VAE in progress. Plan: get the Minimum Viable Tool running end-to-end on MNIST (Phases 0–2), then swap in the TokaMaker loader (Phase 3), then add the enhancement layer incl. Vendi V1/V2 (Phase 4). Priority: show the tool working. (WEST RF and M3D-C1 are deferred to future work.)*