# Annotation and prediction sample

Six validation tiles (two per study area, national orthophoto source, 512 ×
512 px at 0.4 m) in the dataset layout of `docs/data_format.md`:

| folder | content |
|---|---|
| `images/` | the image tiles (`.jpg`) and their world files (`.jgw`, EPSG:2039) |
| `labels/` | Pascal VOC annotations (raw form codes, `docs/classes.md`) |
| `predictions/` | predictions of `fasterrcnnv100-12.pth`: pixel boxes, network class, analysis class (10 categories), score (confidence ≥ 0.3, NMS 0.5) |
| `overlays/` | annotations (left) and predictions (right), boxes coloured by category |

`manifest_example.csv` lists the tiles. Run the evaluation on the sample
with `buildingtypology evaluate examples/annotations_sample --checkpoint …`.

Imagery: national aerial orthophoto (© the national survey agency), shared
here as a small research sample. The complete annotation set (10,071 tiles)
and the full imagery are not published. They are available on request: open an issue at https://github.com/cdml-lab/building-typology-rcnn/issues (or contact the corresponding author of the article); requests are subject to approval.
