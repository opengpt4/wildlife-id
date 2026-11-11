# Owner slice — 50 real field photos

This folder is **yours to fill**. It is the honest half of the Phase 0 evaluation
set and it is the only slice that predicts what the app will actually experience.

## Why this exists

The automated 150-photo slice comes from iNaturalist "research grade"
observations. Those are curated close-ups: accepted by two identifiers, usually
crisp, well lit, subject filling the frame, shot with a macro or long lens. Your
app will be handed a one-handed phone shot of a butterfly 3 m away, half
occluded, backlit, with the subject covering 5% of a 12 MP frame.

If the model scores 85% on the iNat slice and 55% on yours, the product owner
needs the 55%. **Both numbers get reported; yours decides go/no-go.**

## What to drop in

- 50 photos, ideally **3–5 per species** across 10–20 species, including species
  you consider common. If you only have 20, drop 20 — it still scores, with wider
  error bars, and the pipeline reports them.
- Photos **you took yourself** with the phone you would actually use, or from a
  camera as long as they were not cropped for a field guide.
- Include the bad ones on purpose: distant, shaky, backlit, cluttered,
  partly hidden, wrong light. A file of hero shots measures nothing.
- One subject per photo. If two species are in frame, the file is ambiguous —
  skip it or note the primary subject.
- Any animal photographed in the wild in Hong Kong. Captive/zoo birds are fine
  only if flagged in `notes`; they are a different distribution.

## How to name them

File names are free-form. **Filenames are not trusted for labels** — put the
label in `owner_metadata.csv`, which is the single source of truth:

```csv
file,scientific_name,common_name,taxon_group,observed_on,latitude,longitude,notes
IMG_4821.HEIC,Papilio polytes,Common Mormon Swallowtail,butterfly,2026-10-04,,,shaky, 4m away
IMG_4830.HEIC,Passer montanus,Eurasian Tree Sparrow,bird,2026-10-04,22.31,114.17,backlit against sky
```

- `scientific_name` — the correct answer. If you are not sure of the species,
  leave the row out or use `scientific_name` of the genus and add
  `notes: uncertain`; a wrong label silently poisons the eval set.
- `taxon_group` — `butterfly` or `bird`.
- `observed_on` — `YYYY-MM-DD` if known. Used for the month-prior experiment.
- `latitude` / `longitude` — optional, decimal degrees. Used only for the
  location-prior experiment; never uploaded anywhere.
- `notes` — free text. Distance, conditions, why it is hard.

A template with two example rows is at `owner_metadata.csv.template`.

## Requirements for a usable photo

- Long side ≥ 1024 px (don't send WhatsApp-forwarded images; they are
  recompressed down to ~800 px and will understate the model).
- HEIC, JPEG or PNG. HEIC is fine — the pipeline converts it.
- No edits, no filters, no crops beyond what the phone did by default.

## What happens to them

They are scored with the identical protocol as the iNat slice (384 px centre
crop, closed-set over the 200-species regional list), reported as a **separate
slice**, never averaged with the other one, and never redistributed — this folder
is gitignored in the real repo. Only aggregate numbers leave this machine.

## Scoring a partial set

Partial sets are fine. The scorer reports the count with every number and
computes Wilson confidence intervals, so 20 photos gives a number with an honest
error bar rather than a fake precise one.
