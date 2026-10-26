# Output schema

## `infer` → detections (one polygon per detection box)

| column | type | meaning |
|---|---|---|
| geometry | Polygon | detection box in the image's CRS |
| network_class_id | int 1–11 | network output (see `classes.md`) |
| network_class | str | network class name (11 classes) |
| class_name | str | analysis class (10 classes; Plus Shape → H Shape) |
| score | float 0–1 | detection confidence (≥ the `--conf` threshold) |

## `assign` → footprints with a building form (interface to the morphology repository)

All input columns are kept; added:

| column | type | meaning |
|---|---|---|
| network_class_id | int 0–11 | 0 = no detection covers ≥ 20 % of the footprint |
| network_class | str | network class name, `None` if unclassified |
| **class_name** | str | **analysis class (10 categories) or `None`** — the column the morphology pipeline reads |
| score | float | score of the assigned detection (NaN if unclassified) |
| coverage | float 0–1 | share of the footprint covered by the assigned box |

Format: any GDAL vector format (GeoPackage recommended); any CRS (the
morphology pipeline reprojects). The morphology pipeline places each
classified footprint in a block by its representative point and computes,
per block, the share of each of the 10 classes among classified buildings
plus the number of classified buildings. Footprints with `class_name` =
`None` are not counted.

Compatibility: schema version **1.0** (this repository v1.0.0); the
morphology repository v1.0.0 expects schema 1.0.
