# Categories

| network id | network class (11 outputs) | analysis class (10) | raw annotation codes |
|---|---|---|---|
| 0 | Background | – (unclassified: `None`) | – |
| 1 | Rectangle Building | Rectangle Building | 1 |
| 2 | Long Rectangle | Long Rectangle | 11, 12, 101, 102 |
| 3 | H Shape | H Shape | 2 |
| 4 | Plus Shape | **H Shape** (merged) | 3 |
| 5 | L Shape | L Shape | 4 |
| 6 | Geometric Building | Geometric Building | 7, 20 |
| 7 | Geometric Shape | Geometric Shape | 5, 6 |
| 8 | Carpet Houses | Carpet Houses | 8, 81 |
| 9 | Private House_A | Private House_A | 9 |
| 10 | Private House_B | Private House_B | 91, 92 |
| 11 | U Shape | U Shape | 10 |

* The network keeps a separate output for *Plus Shape*; the merge happens
  after prediction (`classes.NETWORK_TO_ANALYSIS`). The merge changes the
  study's validation scores by less than 0.5 percentage points.
* Annotations of the study contain a few boxes with code 0 (building
  without a form class, City A only); they count as references in the
  detection metrics and as class 0 in the classification metrics.
* The classes describe building **forms seen from above**, not land use:
  "Private House" is a footprint shape class.
* Column names used by the morphology repository: `rcnn_` + lower-case
  name with `_` (e.g. `rcnn_private_house_a`).
