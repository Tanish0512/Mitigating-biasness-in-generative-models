"""Class counts and bias metrics at labeller v2 thresholds 0.6-0.9 from an apply_labeller.py CSV.

The reference is dataset-0-v2's class mix (6,000 per high class, 300 per low class).

Example::

    python threshold_metrics.py ../results/labels_g1_v2_mit.csv labeller_v2_thresholds_G1v2mit.json
"""

import csv
import json
import sys

from compare_labellers import CLASSES, metrics

labels_csv, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(labels_csv)))
n = {c: 6000 if c in ("aircraft", "car", "ship") else 300 for c in CLASSES}
p = {c: n[c] / sum(n.values()) for c in CLASSES}
res = {}
for t in (0.6, 0.7, 0.8, 0.9):
    lab = [r["label"] if float(r["probability"]) >= t and r["label"] != "background" else "unrecognisable" for r in rows]
    counts = {c: lab.count(c) for c in CLASSES + ["unrecognisable"]}
    res[str(t)] = {"counts": counts, **metrics(lab, p)}
json.dump(res, open(out, "w"), indent=2)
for t, m in res.items():
    print(t, m["counts"], "HDAI %.3f log2DOR %+.2f" % (m["HDAI"], m["log2_DOR"]))
