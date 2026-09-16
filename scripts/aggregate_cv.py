import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/cv_summary.json"))
    args = p.parse_args()
    results = [json.loads(x.read_text()) for x in args.metrics]
    if len({x["fold"] for x in results}) != len(results):
        raise ValueError("Duplicate folds")
    names = [n for n in results[0]["metrics"] if n not in {"evaluated_queries", "queries_without_positive"}]
    report = {"folds": sorted(x["fold"] for x in results), "n_folds": len(results), "metrics": {}}
    for name in names:
        values = [x["metrics"][name] for x in results]
        report["metrics"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "query_weighted_mean": float(
                np.average(values, weights=[x["metrics"]["evaluated_queries"] for x in results])
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
