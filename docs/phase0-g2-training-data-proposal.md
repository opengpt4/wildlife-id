# Proposal — training data for the G2 regional head

**Status: APPROVED — Route A, funded 2026-11-11.** G2 remains **FAILING**
(75.0% vs the 85% bar) until a head is trained and measured on the frozen
147-photo manifest. Owner decisions recorded in §7; the proposal body below is
kept as the reasoning that was approved rather than rewritten to match the
outcome.

Prepared 2026-11-11. Supply figures measured from the iNat API
(`data/species/hk_candidate_licence_mix.json`); accuracy figures marked *estimate*
are not measurements.

## 1. What G2 needs, and why a head is the candidate fix

G2: top-1 ≥ 85% on the 20 commonest HK species. Zero-shot fp32 reaches 75.0%
(147-photo eval, K=200). The head would be a linear classifier on frozen BioCLIP
features — one encoder pass at inference, unchanged latency, +0.6 MB artifact.

A leave-one-out probe trained on **3 photos per species** scored 61.9%, *worse*
than zero-shot. That is not evidence against heads; it is evidence that 3
examples per class cannot fit a 200-way linear boundary. The question this
proposal prices is: what does a real training set cost, and is it legal to build
one for a paid app?

## 2. Supply — measured, not assumed

All 200 candidate species already have far more than 50 research-grade
observations in HK (median 1,333; the thinnest candidate has 346). **Data
quantity is not the constraint. Licence is.**

Photo-licence mix across the 200 candidate species (observations per licence,
HK research grade):

| Licence | Observations | Share |
|---|---|---|
| CC-BY-NC | 186,829 | 54.1% |
| unlicensed / all rights reserved | 142,959 | 41.4% |
| CC-BY | 10,022 | 2.9% |
| CC-BY-SA + CC-BY-NC-SA | 3,103 | 0.9% |
| CC0 | 2,380 | 0.7% |
| **commercially permissive (CC0 + CC-BY only)** | **12,402** | **3.6%** |

*Footnote on the counts:* each bucket is a separate `species_counts` query for that
licence, so an observation with photos under two licences is counted in both. The
"unlicensed" row is the residual (total − measured buckets) and therefore absorbs
any overlap; treat all percentages as ±2 points. The 3.6% permissive figure is the
one the plan turns on and it is measured directly, not as a residual.

Per-species feasibility if restricted to CC0 + CC-BY:

| Threshold | Species clearing it |
|---|---|
| ≥ 15 permissive photos | 193 / 200 |
| ≥ 20 permissive photos | 180 / 200 |
| ≥ 50 permissive photos | 91 / 200 |

No candidate species has zero permissive photos; the median is 45. So a
**commercially clean** training set is possible for 180–193 of the 200 species,
provided we accept a hard cap of roughly 20–30 photos per species.

## 3. The licence decision — the owner must make this knowingly

This is a legal/product call, not an engineering one, and I am not a lawyer. What
I can say with confidence:

- **CC0 and CC-BY** permit commercial use with attribution. This is the only
  bucket that is unambiguously safe for a paid app.
- **CC-BY-NC / -SA** is 55.0% of the available data and explicitly excludes
  commercial use. Whether *model weights* trained on NC images are themselves an
  adaptation is legally unsettled — some jurisdictions and publishers say yes,
  others treat the weights as non-infringing statistics. Do not bet a paid
  product on the permissive reading without advice.
- **The 41.4% unlicensed bucket is worse, not better**: iNaturalist observers who
  chose no licence reserved all rights, and "it was public on the internet" is not
  a licence to train on it. A pipeline that silently ingests these is the most
  likely source of a future complaint, because it is a default rather than a
  decision.
- Attribution obligations (CC-BY) must be dischargeable in-app: a licences screen
  listing photographers for training data, or a documented decision that
  aggregate statistics do not require per-item attribution.

Options, in ascending cost/risk:

| Route | Data available | Consequence |
|---|---|---|
| **A. Permissive only (CC0 + CC-BY)** | 180/200 species @ ≥20, ~3,900–5,600 photos | Ship a paid app with a clear conscience; 20 species have thin or no clean data and stay zero-shot. |
| **B. Permissive + NC** | 200/200 species, unlimited | Requires either a non-commercial app, or a legal opinion that weights are not derivative. |
| C. All licences incl. unlicensed | 200/200, ~345k obs | Highest risk; I would not build this without a written opinion. |
| D. Own/licensed imagery | none today | Cleanest, most expensive; out of scope for Phase 0. |

**Recommendation: Route A, as a pilot.** It costs hours, is legally clean, and
answers the real question — does a properly-trained head beat zero-shot at all?
If it does, the incremental value of the missing 20 species and the NC data
becomes a measurable product decision rather than a leap of faith.

## 4. Scope — proposed Tier A pilot

| Item | Value |
|---|---|
| Species | 180 (every candidate with ≥ 20 permissive photos) |
| Photos per species | up to 30, capped at available |
| Total photos | ~5,000 (measured: 5,566 at cap 30 over the 180 species) |
| Renderings | iNat **medium** (~500 px, ~300 KB) — we downscale to 224 anyway; `large` would triple the bytes for no signal |
| Download at medium | ~1.5 GB, wall time ~20–40 min at 1–2 images/s, resumable |
| Embedding extraction | 224 px, MPS: **26 ms/image measured** → ~2.5 min for 5,000 + decode overhead, call it 6 min |
| Head training | closed-form ridge / logistic on 5,000 × 512 — seconds |
| Disk in staging | ~1.5 GB photos (gitignored), ~10 MB features |
| Total wall time | under an hour, unattended |

Not included: the 20 thin species, any NC/unlicensed image, any model
architecture change, any server work.

## 5. Expected result (estimate, wide band — not a promise)

With 20–30 examples per class on frozen 512-d features, a linear head over 200
classes should plausibly reach **82–90% top-1** and **82–90% common-20**, versus
77.5% / 75.0% zero-shot today. The pessimistic anchor is the 3-per-class probe at
61.9%; the realistic floor is that the head adds *nothing* and we stay at 77/75.

Two things constrain this more than sample count:

1. **The head can only rearrange the same features.** If BioCLIP's 512-d embedding
   does not separate two congeners, no linear head will. The eval already shows
   these confusions: `Papilio polytes → Papilio xuthus`, `Pseudozizeeria maha →
   Zizina otis`, `Euploea midamus → Euploea core`. A head will fix some of this
   (it can learn regional prior structure), not all.
2. **G2 is 60 photos.** A jump from 75% to 85% on 60 images is 6 photos changing
   side. The gate is a coarse instrument at this sample size; the pilot should
   report the paired per-photo delta, not just the new headline.

**Falsification rule for the pilot:** if Tier A lands below 80% top-1 overall, the
head is not the answer; stop and re-plan the accuracy strategy (better backbone,
two-stage coarse-then-fine classification, or capture UX) rather than buying more
data.

## 6. Train / eval separation I would enforce

The eval set must not be reachable from the training set, and a random split does
not achieve that — near-duplicate sessions and prolific observers leak across a
random split. Rules, all implementable from the observation metadata:

1. Exclude the **147 eval observation ids** by construction.
2. Exclude **every observation by any of the 92 eval-set observers** (observer
   disjoint).
3. Exclude any observation within **±3 days and 500 m** of an eval observation of
   the same species (session/duplicate-image guard).
4. Split the remainder **by observer**, 85% train / 15% validation — never by
   image, never randomly.
5. Freeze the eval manifest's sha256 and record it in every training run; the eval
   set is not regenerated by the training pipeline.
6. Report per-species counts *after* exclusion; a species that drops below 15
   examples is dropped from the head and logged, not silently underweighted.

Expected attrition: the eval observers are 92 of a much larger pool, and the 500 m
guard is narrow, so I expect to lose under 5% of the pool. That estimate gets
replaced by a measured number before any training run.

## 7. Owner decision (2026-11-11)

- **Route A — commercially permissive only (CC0 + CC-BY).** No CC-BY-NC, no
  share-alike, no unlicensed photos, including for the 20 species that fall short.
  Those 20 stay on zero-shot and are documented per species; that is a routing
  design, not a defect.
- **Route A keeps the paid-app option open.** Shipping a paid app on this artifact
  requires no further licence analysis. Choosing Route B or C later would require
  a legal opinion *before* any weight is trained on NC data. This question is
  closed for Phase 0 and must not be re-opened without new information.
- **Tier A funded**: download, train, evaluate, with the conditions in §8.

## 8. Conditions attached to the funded run

1. G2 stays **RED** until measured on the frozen 147-photo manifest, with its
   sha256 asserted. No interim "looks like 85%" claims.
2. Falsification rule stands: if the head lands below 80% top-1, it is not worth
   its complexity; revert to zero-shot and document a lower bar. It will be said
   plainly if that happens.
3. Report head vs zero-shot on **both** the full 147-photo set and the common-20
   subset, paired.
4. Attribution file for the training set: per-photo licence, creator and source
   URL, so what was trained on is provable.

## 9. Original decision request (answered above, kept for the record)

1. **Licence route** — A (permissive), B (permissive + NC, with a legal opinion),
   or C. This gates everything else.
2. **Is the app paid?** If yes, Route A is the only route I would build without
   outside advice.
3. **Fund Tier A now?** ~1 hour unattended, ~1.5 GB staging disk, no device needed.

Also worth knowing before you decide: this machine is not where the TFLite export
can happen (see `docs/phase0-results.md` §3b), so a Linux container/CI is needed
for the shipped artifact regardless. That does not affect the data question.

*(Status at the time this section was written: nothing downloaded, head
untrained, G2 red. Superseded by §7.)*

## 10. Outcome of the funded run (2026-11-11)

Trained and evaluated as scoped: 3,903 photos (Route A), 179 species, observer-split,
hyperparameters chosen on validation only.

**G2 = 78.33% at the validation-selected configuration → FAIL. Zero-shot on the
same photos: 75.00%.** No measurable gain across three seeds. The falsification
rule fired (< 80% top-1): the head does not earn its complexity, and the
recommendation is to ship zero-shot and document 75% as the measured common-20
bar, carrying 85% as an unmet v2 target.

Two corrections to this proposal's own estimates, recorded for honesty:

- **Volume**: the budget said ~5,566 photos; Route A delivered **3,903**, because
  permissive photos cluster around fewer observers than the raw per-species counts
  implied, and observer-diversity rules cap what is usable. Median 20/species, not 30.
- **Expected accuracy**: the proposal estimated 82–90% common-20. The measured
  value is 78%. The pessimistic anchor (61.9%) was closer to the truth than the
  central estimate.

A single configuration (wd = 1e-4) did reach 85.0% common-20; it was not
validation-selected and is reported in `docs/phase0-results.md` §8 as a
methodological trap, not as a result.
