# building-typology-rcnn

Detection and building-form classification of buildings in aerial
orthophotos with a Faster R-CNN (ResNet-101 FPN), and assignment of the
predicted forms to building footprints. The output feeds the block
morphology pipeline [`urban-block-morphology`](https://github.com/cdml-lab/urban-block-morphology)
**v1.0.0** as the 10 building-form shares of a block (output schema 1.0, see
`docs/output_schema.md`).

**Network vs analysis categories.** The network predicts **11** building-form
classes (+ background). For the morphology analysis the rare *Plus Shape*
output is merged into *H Shape*, giving **10** categories. Every output
carries both (`network_class` and `class_name`); see `docs/classes.md`.

## Install

```bash
conda env create -f environment.yml      # or: python -m venv .venv
conda activate building-typology
pip install -e .                          # PyTorch: see the note in environment.yml
```

## Use the trained checkpoint

The checkpoint `fasterrcnnv100-12.pth` (242 MB) is attached to the [v1.0.0 release](https://github.com/cdml-lab/building-typology-rcnn/releases/tag/v1.0.0) as a release asset, not stored in Git. The code checks its SHA-256 on load; the aerial imagery it was trained on is not included
(`docs/checkpoint.md`: file name, version, SHA-256, evaluation configuration).

```bash
# 1. detect and classify (GeoTIFF, 0.3-0.4 m pixels, RGB)
buildingtypology infer ortho.tif --checkpoint fasterrcnnv100-12.pth --out detections.gpkg
#    to reproduce the study's labels add --edge-mode resize (default: pad; docs/method.md)
# 2. label building footprints (any vector format with polygons)
buildingtypology assign detections.gpkg footprints.gpkg --out footprints_typology.gpkg
```

`footprints_typology.gpkg` is the file the morphology repository reads
(`typology_predictions` in its configuration).

## Evaluate, train, document a dataset

```bash
buildingtypology evaluate <annotations_root> --checkpoint fasterrcnnv100-12.pth --out report
buildingtypology evaluate <annotations_root> --checkpoint ... --out report_sep --spatially-separate   # tile-extent sensitivity check, docs/evaluation.md
buildingtypology train <annotations_root> --out my_checkpoint.pth
buildingtypology manifest <annotations_root> --out manifest.csv
```

Dataset layout and annotation format: `docs/data_format.md`. Method
(image preparation, area separation, thresholds, overlapping tiles,
assignment): `docs/method.md`.

## Example

`examples/run_example.py` runs the whole chain (inference, assignment,
output schema check) on a small synthetic GeoTIFF and footprint layer
generated on the fly, so it works without any licensed imagery. With the
checkpoint it also loads and runs the trained network; without it, it uses
an untrained network to exercise the code path only.

```bash
python examples/run_example.py --checkpoint fasterrcnnv100-12.pth
```

## Tests

```bash
pip install pytest && pytest -q
```

## Licence and citation

Code: MIT (`LICENSE`). Checkpoint and annotations: see their own
documentation. Cite with `CITATION.cff`.
