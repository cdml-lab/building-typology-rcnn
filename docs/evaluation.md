# Detector evaluation (checkpoint `fasterrcnnv100-12.pth`)

## Setup
- **Model:** Faster R-CNN, ResNet-101 FPN, trained weights v100-12.
- **Categories:** the 10 analysis categories. References and predictions
  are both mapped through the merge of Plus Shape into H Shape before
  scoring.
- **Tiles:** the 255 validation tiles (`Unseen`), 512 × 512 px at 0.4 m.
  They come from areas selected separately from the training areas
  before tiling. The study filters keep annotated boxes ≥ 1,000 px² and
  tiles with ≥ 10 such boxes.
- **Confidence:** fixed at 0.3.
- **IoU:** 0.5 is the reported setting. The IoU 0.4 column is a
  comparison for this documentation only; in it **both** the NMS threshold
  and the matching threshold were changed to 0.4, because the study code
  uses one threshold for both.

### Metric definitions
- **Detection recall:** the share of annotated building instances whose
  best-overlapping prediction reaches the IoU threshold. Matching is
  many-to-one: each reference takes its highest-IoU prediction.
- **Classification F1:** support-weighted F1 of the reference vs the
  predicted category, computed over the matched references only.
- **Overall weighted F1:** support-weighted F1 over all references, where a
  missed reference counts as the prediction "no building". This combines
  detection and classification.

The validation tiles informed the selection of the training epoch and of
the thresholds, so they are a **validation set**, not an independent test
set.

**For the manuscript:** report the 10-category results at IoU 0.5 and
confidence 0.3, i.e. the GOV rows of the first three metric columns. The
6,440 annotations are **annotated building instances**, not unique
buildings: the validation tiles overlap, and some buildings are observed
in more than one tile. The IoU 0.4 column, the distinct-building estimate,
the 228-tile subset and the boundary-overlap diagnostics below are kept
for this documentation only.

## Results

| imagery | set | tiles | annotated building instances | detection recall % (IoU 0.5) | classification F1 % (IoU 0.5) | overall weighted F1 % (IoU 0.5) | overall weighted F1 % (NMS and matching IoU 0.4) |
|---|---|---|---|---|---|---|---|
| GOV | **all** | 255 | 6,440 | **82.34** | **75.59** | **67.05** | 67.13 |
| GOV | City A | 94 | 2,307 | 74.30 | 58.32 | 49.14 | 49.09 |
| GOV | City B | 77 | 2,135 | 84.22 | 84.49 | 76.57 | 77.90 |
| GOV | City C | 84 | 1,998 | 89.64 | 84.38 | 77.51 | 76.57 |
| Esri | all | 255 | 6,429 | 77.97 | 75.05 | 64.10 | 64.17 |
| Esri | City A | 94 | 2,305 | 70.76 | 62.46 | 50.72 | 49.63 |
| Esri | City B | 77 | 2,133 | 76.75 | 82.92 | 70.70 | 72.65 |
| Esri | City C | 84 | 1,991 | 87.64 | 79.91 | 71.61 | 71.27 |

The other two metrics at IoU 0.4 are in `detector_evaluation.csv` (GOV
all: recall 86.32, classification F1 73.45). The Esri tiles cover the same
validation locations as the GOV tiles, on Esri World Imagery. Their
annotation files differ slightly, which gives slightly different reference
counts after the filters.

**Sample sizes.** 255 tiles per imagery source, holding 6,440 (GOV) and
6,429 (Esri) annotated building instances. The validation tiles overlap one another, so
the same building appears in several tiles. Merging boxes that coincide
on the map (IoU ≥ 0.5) gives about **833 distinct buildings** (GOV:
City A 320, City B 245, City C 268). This is an upper estimate, because a
building cut by a tile edge can survive as a separate partial box. The
annotations carry no building identifiers. Each tile is scored on its own,
so a building counts once per tile that contains it.

## Separation of validation and training areas
The validation tiles come from areas designated separately from the
training areas before tiling. This was checked against the data
(`verification/validation_separation.py`):

| check (evaluated GOV tiles) | City A | City B | City C | all |
|---|---|---|---|---|
| validation references intersecting **any** training box (same source) | 0 | 0 | 0 | 0 |
| validation references matching a training box at map IoU ≥ 0.5 | 0 | 0 | 0 | 0 |
| tiles whose **footprint** intersects a training tile footprint | 0 | 23 | 4 | 27 |
| … of which ≥ 50 % of the tile footprint | 0 | 5 | 0 | 5 |
| footprint overlap area | 0 | 0.059 km² | 0.001 km² | 0.060 km² |
| minimum distance to a training tile | 155 m | 0 m | 0 m | – |

**Result:** no validation building is annotated in a training tile. The
check finds no contradiction of the stated separation at the level of
buildings. Where tile footprints overlap (City B, and marginally City C),
the shared ground lies at the edges of the designated areas and contains
no annotated building of the other split.

### The footprint-separated subset
`buildingtypology evaluate --spatially-separate` (and the earlier
"spatially separate subset") excludes the 27 evaluated tiles whose 512 px
footprint intersects the footprint of any training tile of the same area
and source. It is built from the world files only; it does not use the
designated area polygons or building identities.

That leaves 228 tiles, 5,927 references and about 808 distinct buildings
(City A 94 / 2,307 / 320; City B 54 / 1,683 / 221; City C 80 / 1,937 / 267).
At IoU 0.5 the scores are 82.00 / 74.98 / 66.18 %. This subset is a
sensitivity check on tile extents; it is **not** an independent test set.
Since the building-level check finds no shared buildings, the full 255
tiles are the reference evaluation.
