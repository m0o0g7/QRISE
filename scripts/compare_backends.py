"""Report paired scalar/component differences without requiring equality."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar")
    parser.add_argument("components")
    args = parser.parse_args()
    def load(path):
        data = json.loads(Path(path).read_text())
        return {(r["n"], r["target"], r["seed"], r["policy"]): r for r in data["runs"]}
    old, new = load(args.scalar), load(args.components)
    groups = defaultdict(list)
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        groups[(key[0], key[1], key[3])].append(dict(
            scalar_success=a["success"], component_success=b["success"],
            epr_delta=b["epr_pairs"] - a["epr_pairs"],
            latency_delta=b["latency_slots"] - a["latency_slots"],
            fidelity_delta=(b["fidelity"] - a["fidelity"]) if a["fidelity"] is not None and b["fidelity"] is not None else None))
    output = []
    for (n, target, policy), rows in groups.items():
        record = dict(n=n, target=target, policy=policy, paired_trials=len(rows))
        for metric in rows[0]:
            values = [r[metric] for r in rows if r[metric] is not None]
            record[metric] = sum(values) / len(values) if values else None
        output.append(record)
    path = Path(args.components).parent / "scalar_comparison.json"
    path.write_text(json.dumps(output, indent=2))
    print("Compared {} matched trials; wrote {}".format(len(old.keys() & new.keys()), path))


if __name__ == "__main__":
    main()
