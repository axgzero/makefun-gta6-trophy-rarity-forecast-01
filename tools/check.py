#!/usr/bin/env python3
"""Local checks for the GTA VI trophy forecast entry.

Runs main.py on a directory of *.in case files, then validates every
*.out file against the kit's output contract and, when a reference answer
is supplied, scores it with the competition's pinball loss on the logit
scale. Usage:

    python3 tools/check.py <input_dir> <output_dir> [<answers.json>]

answers.json (optional): {"<case_id>": {"<trophy_id>": <rate>, ...}, ...}
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def logit(pct):
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    answers = {}
    if len(sys.argv) > 3:
        with open(sys.argv[3], "r", encoding="utf-8") as fh:
            answers = json.load(fh)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
               MAKEFUN_OUTPUT_DIR=out_dir)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                          env=env, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    ok = True
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out_path = os.path.join(out_dir, case_id + ".out")
        size = os.path.getsize(out_path)
        assert size <= 65536, "%s over size cap: %d" % (out_path, size)
        with open(out_path, "r", encoding="utf-8") as fh:
            out = json.load(fh)
        assert set(out) == {"trophies"}, "top-level keys wrong: %s" % out.keys()
        rows = out["trophies"]
        assert len(rows) == len(case["trophies"]), (
            "row count %d != trophy count %d" % (len(rows), len(case["trophies"])))
        ids = sorted(r["id"] for r in rows)
        assert ids == sorted(t["id"] for t in case["trophies"]), "ids mismatch"
        for row in rows:
            assert set(row) == {"id", "q"}, "row keys wrong: %s" % row.keys()
            assert len(row["q"]) == 5
            for q in row["q"]:
                assert isinstance(q, (int, float)) and math.isfinite(q)
                assert 0 <= q <= 100, "quantile out of range: %s" % q
            assert row["q"] == sorted(row["q"]), "quantiles not ordered"
        print("%s: valid, %d trophies, %d bytes" % (case_id, len(rows), size))
        if case_id in answers:
            levels = [0.05, 0.25, 0.5, 0.75, 0.95]
            losses, misses, covered = [], [], 0
            by_id = {r["id"]: r["q"] for r in rows}
            for tid, y in answers[case_id].items():
                q = by_id[int(tid)]
                losses += [pinball(y, qq, tau)
                           for qq, tau in zip(q, levels)]
                misses.append(abs(y - q[2]))
                covered += 1 if q[0] <= y <= q[4] else 0
            print("  pinball loss %.6f  typical miss %.2fpp  coverage %.2f"
                  % (sum(losses) / len(losses),
                     sum(misses) / len(misses),
                     covered / len(answers[case_id])))
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
