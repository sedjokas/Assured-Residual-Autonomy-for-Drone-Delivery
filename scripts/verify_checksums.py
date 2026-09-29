#!/usr/bin/env python3
from pathlib import Path
import hashlib, csv, sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/"MANIFEST_SHA256.csv"
if not manifest.exists():
    raise SystemExit("MANIFEST_SHA256.csv not found")
bad=[]
with manifest.open(newline="") as f:
    for row in csv.DictReader(f):
        p=ROOT/row["path"]
        if not p.exists():
            bad.append((row["path"],"missing"));continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=row["sha256"]: bad.append((row["path"],"hash mismatch"))
if bad:
    print("FAIL");[print(*x) for x in bad];sys.exit(1)
print("OK - all manifest hashes match")
