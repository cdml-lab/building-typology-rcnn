# Annotation format and dataset layout

```
<root>/<area>/<source>/<split>/images/<tile>.jpg    512 x 512 px RGB
<root>/<area>/<source>/<split>/images/<tile>.jgw    ESRI world file (georeference)
<root>/<area>/<source>/<split>/labels/<tile>.xml    Pascal VOC
```

`source`: GOV | ESRI; `split`: Data (training) | Unseen (validation).

Pascal VOC (one `<object>` per building):

```xml
<annotation>
  <filename>000000000040.jpg</filename>
  <size><width>512</width><height>512</height><depth>3</depth></size>
  <object>
    <name>10</name>                      <!-- raw form code, docs/classes.md -->
    <bndbox><xmin>427.02</xmin><ymin>0.00</ymin><xmax>512.00</xmax><ymax>64.69</ymax></bndbox>
  </object>
</annotation>
```

Pixel coordinates, origin top-left. World file: six lines (pixel size x,
rotation, rotation, −pixel size y, x and y of the upper-left pixel centre).

## Manifest
`buildingtypology manifest <root> --out manifest.csv` writes one row per
annotated tile: area, source, split, annotation and image paths, whether
the image is present, the tile's map bounds (from the world file), the CRS,
box counts before/after the filters and whether the tile enters the
evaluation.
