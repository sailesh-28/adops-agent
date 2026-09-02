"""CSV ingest/export in a platform-export-like shape.

Validation is deterministic: schema, types, non-negativity, referential integrity,
and recomputation of any derived metric the file happens to supply. A file's own
CTR/CPA columns are never trusted -- they are recomputed and mismatches reported.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

from . import metrics as M
from .schema import CSV_COLUMNS

INT_FIELDS = ("period", "impressions", "reach", "clicks")
FLOAT_FIELDS = ("spend", "conversions", "revenue")
DERIVED = ("ctr", "cpc", "cpm", "cvr", "cpa", "roas", "frequency")


def write_csv(rows: List[dict], path: str, include_derived: bool = True) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    cols = list(CSV_COLUMNS) + (list(DERIVED) if include_derived else [])
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            out = {c: r.get(c) for c in CSV_COLUMNS}
            if include_derived:
                d = M.derive(r)
                for c in DERIVED:
                    out[c] = "" if d[c] is None else round(d[c], 6)
            writer.writerow(out)
    return path


def read_csv(path: str) -> Tuple[List[dict], List[str]]:
    """Return (rows, problems). Rows that fail validation are dropped and reported."""
    problems: List[str] = []
    rows: List[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return [], ["missing required columns: %s" % ", ".join(missing)]
        for n, raw in enumerate(reader, start=2):
            row: Dict[str, object] = {}
            bad = False
            for c in CSV_COLUMNS:
                v = (raw.get(c) or "").strip()
                if c in INT_FIELDS:
                    try:
                        row[c] = int(float(v))
                    except ValueError:
                        problems.append("line %d: %s is not a number (%r)" % (n, c, v))
                        bad = True
                elif c in FLOAT_FIELDS:
                    try:
                        row[c] = float(v)
                    except ValueError:
                        problems.append("line %d: %s is not a number (%r)" % (n, c, v))
                        bad = True
                else:
                    if not v:
                        problems.append("line %d: %s is empty" % (n, c))
                        bad = True
                    row[c] = v
            if bad:
                continue
            for c in INT_FIELDS + FLOAT_FIELDS:
                if c != "period" and float(row[c]) < 0:
                    problems.append("line %d: %s is negative (%s)" % (n, c, row[c]))
                    bad = True
            if row.get("clicks", 0) > row.get("impressions", 0):
                problems.append("line %d: clicks (%s) exceed impressions (%s)"
                                % (n, row["clicks"], row["impressions"]))
                bad = True
            if row.get("reach", 0) > row.get("impressions", 0):
                problems.append("line %d: reach (%s) exceeds impressions (%s)"
                                % (n, row["reach"], row["impressions"]))
                bad = True
            if bad:
                continue
            # never trust supplied derived columns -- recompute and report drift
            derived = M.derive(row)
            for c in DERIVED:
                supplied = (raw.get(c) or "").strip()
                if supplied:
                    try:
                        given = float(supplied)
                    except ValueError:
                        continue
                    calc = derived[c]
                    if calc is None:
                        continue
                    if abs(given - calc) > max(0.001, abs(calc) * 0.01):
                        problems.append("line %d: supplied %s=%s disagrees with the "
                                        "recomputed value %.6f (recomputed value used)"
                                        % (n, c, supplied, calc))
            rows.append(row)
    return rows, problems


def summarise(rows: List[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    agg = M.aggregate(rows)
    periods = sorted({r["period"] for r in rows})
    return {
        "rows": len(rows),
        "periods": [periods[0], periods[-1]],
        "campaigns": len({r["campaign_id"] for r in rows}),
        "ad_sets": len({r["ad_set_id"] for r in rows}),
        "creatives": len({r["creative_id"] for r in rows}),
        "spend": round(agg["spend"], 2),
        "conversions": round(agg["conversions"], 1),
        "cpa": round(agg["cpa"], 2) if agg["cpa"] else None,
        "ctr_pct": round(agg["ctr"] * 100, 3) if agg["ctr"] else None,
        "roas": round(agg["roas"], 2) if agg["roas"] else None,
    }
