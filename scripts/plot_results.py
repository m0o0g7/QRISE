"""Keep measured aggregates and render recon_Qu-style grouped bar panels."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)
    data = json.loads((root / "results.json").read_text())
    rows, manifest = data["runs"], data["manifest"]
    metrics = []
    for n, target in manifest["cells"]:
        for policy in manifest["policies"]:
            selected = [r for r in rows if (r["n"], r["target"], r["policy"]) == (n, target, policy)]
            if not selected:
                continue
            successful = [r for r in selected if r["success"]]
            fidelity_rows = [r for r in selected if r["fidelity"] is not None]
            metrics.append(dict(n=n, target=target, policy=policy, trials=len(selected),
                                expected_trials=len(manifest["seeds"]),
                                successes=len(successful), success_rate=len(successful) / len(selected),
                                fidelity=mean(fidelity_rows, "fidelity"),
                                fidelity_samples=len(fidelity_rows),
                                successful_fidelity=mean(successful, "fidelity"),
                                epr_pairs=mean(selected, "epr_pairs"), latency_slots=mean(selected, "latency_slots"),
                                successful_epr_pairs=mean(successful, "epr_pairs"),
                                successful_latency_slots=mean(successful, "latency_slots"),
                                attempts_per_success=(sum(r["epr_pairs"] for r in selected) / len(successful)) if successful else None,
                                failures=dict(Counter(r["reason"] for r in selected if not r["success"]))))
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # recon_Qu used three horizontal grouped-bar panels. Keep that visual
    # layout without changing this project's completed-pair fidelity aggregate.
    policies = ("GREEDY", "HSP", "NOPP", "Q-CAST", "Q-RISE")
    colors = ("#e3a52f", "#278334", "#6c267f", "#d7372f", "#263be8")
    sizes = sorted({n for n, target in manifest["cells"] if target == .9})
    targets = sorted({target for n, target in manifest["cells"] if n == 50})
    lookup = {(m["n"], m["target"], m["policy"]): m for m in metrics}
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.grid": True, "grid.alpha": .18})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    bar_width = .15
    panels = (
        ("fidelity", sizes, "Number of Nodes", "End-to-End Fidelity",
         "(a) Fidelity vs. network size"),
        ("epr_pairs", targets, "F_total", "Avg. EPR Pair Usage",
         "(b) EPR vs. fidelity (50 nodes)"),
        ("latency_slots", targets, "F_total", "Avg. Latency (slots)",
         "(c) Latency vs. fidelity (50 nodes)"),
    )
    for axis, (field, values, xlabel, ylabel, title) in zip(axes, panels):
        x = np.arange(len(values))
        finite = []
        for index, (policy, color) in enumerate(zip(policies, colors)):
            heights = []
            for value in values:
                key = (value, .9, policy) if field == "fidelity" else (50, value, policy)
                measured = lookup.get(key, {}).get(field)
                heights.append(np.nan if measured is None else measured)
            finite.extend(value for value in heights if np.isfinite(value))
            bars = axis.bar(x + (index - 2) * bar_width, heights, bar_width,
                            color=color, edgecolor="#333333", linewidth=.3, label=policy)
            # Keep measured F even at low success rates. Hatching
            # indicates incomplete target success, not an invented F value.
            if field == "fidelity":
                for value, bar in zip(values, bars):
                    metric = lookup.get((value, .9, policy), {})
                    if metric.get("success_rate", 1) < 1:
                        bar.set_hatch("///")
        axis.set(xticks=x, xticklabels=[str(v) for v in values],
                 xlabel=xlabel, ylabel=ylabel, title=title)
        axis.set_axisbelow(True)
        # recon_Qu fixed F to .8..1 and EPR to 0..1500. Expand those bounds
        # when measured results fall outside them; never clip failed-target F.
        if field == "fidelity":
            lower = min(.8, np.floor((min(finite) - .02) * 20) / 20) if finite else .8
            axis.set_ylim(max(0, lower), 1)
        elif field == "epr_pairs":
            axis.set_ylim(0, max(1500, max(finite, default=0) * 1.08))
        else:
            axis.set_ylim(0, max(224, max(finite, default=0) * 1.04))
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .925), ncol=5, frameon=False)
    title = ("Q-RISE NetSquid component reconstruction" if manifest["backend"] == "components"
             else "Q-RISE paper-informed mathematical reconstruction")
    # Label calibrated physics and comparator rules on the exported artifact.
    if manifest.get("profile"):
        mode = manifest["config"].get("baseline_mode", "legacy")
        if manifest["backend"] == "scalar":
            mode = "archived +3"
        elif mode == "fixed":
            mode += " r=" + str(manifest["config"]["baseline_fixed_rounds"])
        title += " [{}; {}]".format(manifest["profile"], mode)
    fig.suptitle(title, y=.995, fontsize=11)
    fig.text(.5, .01,
             "{} seeds; F: completed pairs; hatched F bars: success < 100%; costs: all requests; success rates separate".format(len(manifest["seeds"])),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .83))
    # Keep existing README links working, and offer recon_Qu's file names too.
    fig.savefig(root / "comparison.png", dpi=220)
    fig.savefig(root / "comparison.pdf")
    fig.savefig(root / "figure2_reconstruction.png", dpi=220)
    fig.savefig(root / "figure2_reconstruction.svg")
    plt.close(fig)

    # Preserve the previous fourth panel as a separate, matching bar chart.
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(sizes))
    for index, (policy, color) in enumerate(zip(policies, colors)):
        heights = [lookup.get((n, .9, policy), {}).get("success_rate", np.nan) for n in sizes]
        axis.bar(x + (index - 2) * bar_width, heights, bar_width,
                 color=color, edgecolor="#333333", linewidth=.3, label=policy)
    axis.set(xticks=x, xticklabels=[str(n) for n in sizes], xlabel="Number of Nodes",
             ylabel="Target Success Rate", ylim=(0, 1.05), title="Target fidelity = 0.9")
    axis.set_axisbelow(True)
    fig.legend(*axis.get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(root / "success_rate.png", dpi=220)
    fig.savefig(root / "success_rate.pdf")
    plt.close(fig)
    print("Wrote metrics.json, three-panel comparison/figure2_reconstruction, and success_rate plots")


if __name__ == "__main__":
    main()
