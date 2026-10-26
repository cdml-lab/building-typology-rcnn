"""Pascal-VOC tiles -> training / evaluation samples.

Folder layout (one folder per area and imagery source):

    <root>/<area>/<source>/<split>/images/<tile>.jpg   (+ .jgw world file)
    <root>/<area>/<source>/<split>/labels/<tile>.xml   (Pascal VOC)

split: "Data" (training) or "Unseen" (validation). Box filters, as in the
study: boxes smaller than 1000 px^2 are dropped; tiles with fewer than 10
remaining boxes are dropped.
"""
from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import numpy as np

from .classes import raw_to_network

MIN_BOX_AREA_PX = 1000
MIN_BOXES_PER_TILE = 10
TILE_PX = 512


def collect_paths(root, areas=None, sources=("GOV", "ESRI"), split="Data"):
    """Paired (image, label) paths for the given areas, imagery sources and
    split; pairs are matched by file stem."""
    imgs, labs = [], []
    root = Path(root)
    for area in sorted(p for p in root.iterdir() if p.is_dir()):
        if areas and area.name not in areas:
            continue
        for src in sorted(p for p in area.iterdir() if p.is_dir()):
            if sources and not any(s in src.name for s in sources):
                continue
            d = src / split
            im = {Path(p).stem: p for p in glob(str(d / "images" / "*.jpg"))}
            lb = {Path(p).stem: p for p in glob(str(d / "labels" / "*.xml"))}
            for k in sorted(set(im) & set(lb)):
                imgs.append(im[k])
                labs.append(lb[k])
    return imgs, labs


def read_voc(path, min_area=MIN_BOX_AREA_PX):
    """[(network_class_id, (xmin, ymin, xmax, ymax)), ...] with the study filters."""
    out = []
    for obj in ET.parse(path).getroot().iter("object"):
        cid = raw_to_network(int(obj.find("name").text))
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (int(float(bb.find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if x0 == x1 or y0 == y1:
            continue
        if (x1 - x0) * (y1 - y0) < min_area:
            continue
        out.append((cid, (x0, y0, x1, y1)))
    return out


def rotate_boxes(boxes, angle, size=TILE_PX):
    """Axis-aligned envelope of each box rotated about the tile centre; boxes
    leaving the tile are dropped (training augmentation, as in the study)."""
    out = []
    a = -math.radians(angle)
    c = size / 2.0
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for cid, (x0, y0, x1, y1) in boxes:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) - c
        r = pts @ rot.T + c
        nx0, ny0, nx1, ny1 = r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()
        if 0 <= nx0 < size and 0 <= ny0 < size and nx1 <= size and ny1 <= size:
            out.append((cid, (nx0, ny0, nx1, ny1)))
    return out


class TileDataset:
    """torch Dataset of 512-px tiles. With random_rotate, every kept tile is
    added a second time rotated by a random angle in [10, 350] degrees."""

    def __init__(self, img_paths, label_paths, transform=None, random_rotate=False,
                 min_boxes=MIN_BOXES_PER_TILE, seed=None):
        rng = random.Random(seed)
        self.transform = transform
        self.samples = []
        self.dropped_tiles = 0
        for ip, lp in zip(img_paths, label_paths):
            boxes = read_voc(lp)
            if len(boxes) < min_boxes:
                self.dropped_tiles += 1
                continue
            self.samples.append((ip, boxes, 0.0))
            if random_rotate:
                ang = rng.uniform(10, 350)
                rb = rotate_boxes(boxes, ang)
                if len(rb) >= min_boxes:
                    self.samples.append((ip, rb, ang))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torchvision.transforms.functional as TF
        from PIL import Image
        from torchvision import transforms
        path, boxes, ang = self.samples[i]
        img = Image.open(path).convert("RGB")
        if ang:
            rot = TF.rotate(img, ang, expand=True)
            canvas = Image.new("RGB", (TILE_PX, TILE_PX), (0, 0, 0))
            canvas.paste(rot, ((TILE_PX - rot.size[0]) // 2, (TILE_PX - rot.size[1]) // 2))
            img = canvas
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img, boxes


def collate(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch])
    targets = [{"labels": torch.tensor([c for c, _ in bx], dtype=torch.int64),
                "boxes": torch.tensor([list(b) for _, b in bx], dtype=torch.float32)}
               for _, bx in batch]
    return imgs, targets


def world_file_bounds(jgw_path, size=TILE_PX):
    """(xmin, ymin, xmax, ymax) of a tile from its ESRI world file."""
    a, _, _, e, c, f = [float(x) for x in Path(jgw_path).read_text().split()]
    return (c, f + e * size, c + a * size, f)
