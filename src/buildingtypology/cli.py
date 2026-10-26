"""Command line: buildingtypology {infer,assign,evaluate,train,manifest}."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="buildingtypology")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="detect and classify buildings in a GeoTIFF")
    p.add_argument("image"); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="detections (GeoJSON / GPKG)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tile-nms", type=float, default=0.5)
    p.add_argument("--global-nms", type=float, default=0.2)
    p.add_argument("--edge-mode", choices=["pad", "resize"], default="pad",
                   help="edge windows: pad (default) or resize (as in the study run)")
    p.add_argument("--device", default=None)

    p = sub.add_parser("assign", help="assign detections to building footprints")
    p.add_argument("detections"); p.add_argument("footprints")
    p.add_argument("--out", required=True)
    p.add_argument("--coverage", type=float, default=0.20)

    p = sub.add_parser("evaluate", help="metrics on annotated validation tiles")
    p.add_argument("root"); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="report path (csv + json)")
    p.add_argument("--areas", nargs="*"); p.add_argument("--source", default="GOV")
    p.add_argument("--conf", type=float, default=0.3); p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--classes", choices=["analysis", "network"], default="analysis")
    p.add_argument("--spatially-separate", action="store_true",
                   help="sensitivity check: drop validation tiles whose footprint intersects a training tile footprint")
    p.add_argument("--device", default=None)

    p = sub.add_parser("train", help="train a detector on annotated tiles")
    p.add_argument("root"); p.add_argument("--out", required=True)
    p.add_argument("--areas", nargs="*"); p.add_argument("--epochs", type=int, default=30)

    p = sub.add_parser("manifest", help="write the dataset manifest")
    p.add_argument("root"); p.add_argument("--out", required=True)
    p.add_argument("--alias", default="{}", help='JSON, e.g. {"north": "A"}')
    p.add_argument("--crs", default="")

    a = ap.parse_args(argv)
    if a.cmd == "infer":
        from .infer import predict_raster
        from .model import load_checkpoint
        m, dev = load_checkpoint(a.checkpoint, a.device)
        g = predict_raster(m, dev, a.image, a.conf, a.tile_nms, a.global_nms, a.edge_mode)
        g.to_file(a.out)
        print(f"{len(g)} detections -> {a.out}")
    elif a.cmd == "assign":
        import geopandas as gpd
        from .assign import assign_to_footprints
        out = assign_to_footprints(gpd.read_file(a.footprints), gpd.read_file(a.detections),
                                   a.coverage)
        out.to_file(a.out)
        print(f"{(out['class_name'] != 'None').sum()} of {len(out)} footprints classified -> {a.out}")
    elif a.cmd == "evaluate":
        from .evaluate import evaluate, write_report
        from .model import load_checkpoint
        m, dev = load_checkpoint(a.checkpoint, a.device)
        df = evaluate(m, dev, a.root, a.areas, a.source, "Unseen", a.conf, a.iou, a.classes,
                      a.spatially_separate)
        write_report(df, a.out)
        print(df.round(2).to_string(index=False))
    elif a.cmd == "train":
        from .train import train
        print("best validation overall weighted F1:", train(a.root, a.out, a.areas, a.epochs))
    elif a.cmd == "manifest":
        from .manifest import build_manifest
        df = build_manifest(a.root, json.loads(a.alias), a.crs)
        df.to_csv(a.out, index=False)
        print(f"{len(df)} annotated tiles -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
