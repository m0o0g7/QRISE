"""Run/resume declared physical scenarios; never rescale measured outcomes."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import multiprocessing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from qrise_math.routing import Config
from qrise_math.topology import make_topology
from qrise_components.profiles import profile_settings

POLICIES = ("GREEDY", "HSP", "NOPP", "Q-CAST", "Q-RISE")
SIZES = (20, 30, 40, 50, 60, 70, 80, 90)
TARGETS = (.6, .7, .8, .9)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute_trial(job):
    """Each process owns its independent NetSquid simulator and random stream."""
    backend, n, target, seed, index, policy, config, device, optical = job
    topology = make_topology(n, seed, optical)
    quantum_seed = seed * 100 + index
    if backend == "scalar":
        spec = importlib.util.spec_from_file_location("qrise_math.scalar_reference", ROOT / "reference/scalar_routing.py")
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        result = reference.run(policy, topology, target, np.random.default_rng(quantum_seed), config)
    else:
        from qrise_components.backend import run
        result = run(policy, topology, target, quantum_seed, config, device)
    return dict(n=n, target=target, seed=seed, quantum_seed=quantum_seed, **dataclasses.asdict(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New result directory (default outputs/<profile>)")
    parser.add_argument("--profile", choices=("mechanism", "mechanism_e2e", "fig2", "hardware_only", "recon"), help="Components default: mechanism; scalar: recon")
    parser.add_argument("--seed-start", type=int, default=1701, help="First topology seed; use fresh seeds for independent validation")
    parser.add_argument("--nodes", nargs="+", type=int, choices=SIZES, help="Select a smaller node sweep")
    parser.add_argument("--targets", nargs="+", type=float, help="Targets; with --nodes runs their Cartesian product")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, help="Select policies without changing their seeds")
    parser.add_argument("--baseline-mode", choices=("path_mpr", "fixed"), help="Default path_mpr has no extra rounds")
    parser.add_argument("--baseline-rounds", type=int, help="Required only with --baseline-mode fixed")
    parser.add_argument("--source-fidelity-offset", type=float, help="Common source input offset, not an output adjustment")
    parser.add_argument("--memory-rate-scale", type=float, help="Common multiplier on physical memory noise")
    parser.add_argument("--guard-floor-base", type=float, help="Link-floor base; the slope comes from the chosen profile")
    parser.add_argument("--cost-fidelity-weight", type=float, help="Alpha in the unchanged Eq.(3); nonnegative")
    parser.add_argument("--guard-margin", type=float, help="E2E guard margin for ALL policies")
    parser.add_argument("--baseline-filter", choices=("shared", "unfiltered"), help="Shared F filter is a stronger-baseline control")
    parser.add_argument("--reachability-filter", choices=("on", "off"), help="Reject currently forecast dead ends before generating pairs")
    parser.add_argument("--guard-mode", choices=("equal", "chain"), help="Q-RISE guard ablation")
    parser.add_argument("--lookahead-slots", type=float, help="Known synthetic quality forecast; 0 disables it")
    parser.add_argument("--link-floor", choices=("shared", "none"), help="Common link quality floor or only E2E target")
    parser.add_argument("--degraded-link-fraction", type=float, help="Shared proportion of degraded links; 0 disables heterogeneity")
    parser.add_argument("--quality-amplitude", type=float, help="Synthetic quality drift amplitude")
    parser.add_argument("--quality-period-slots", type=float, help="Quality drift period; does not accelerate geometric motion")
    parser.add_argument("--backend", choices=("components", "scalar"), default="components")
    parser.add_argument("--smoke", action="store_true", help="50 nodes, target .9, one seed")
    parser.add_argument("--workers", type=int, default=1, help="Independent simulator processes")
    parser.add_argument("--seeds", type=int, default=10, help="Topology seeds starting at 1701")
    parser.add_argument("--resume", action="store_true", help="Continue identical manifest from results.json")
    parser.add_argument("--ideal-memory", action="store_true", help="Declared diagnostic: turn storage noise off for ALL policies")
    parser.add_argument("--ideal-swap", action="store_true", help="Declared diagnostic: turn physical swap noise off for ALL policies")
    parser.add_argument("--gate-noise", type=float, default=0.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.seed_start < 0 or (args.targets and any(not 0 < t < 1 for t in args.targets)):
        parser.error("Seeds must be nonnegative and targets must be in (0, 1)")
    if args.smoke and (args.nodes or args.targets):
        parser.error("Use --smoke or --nodes/--targets, not both")
    if args.ideal_memory and args.memory_rate_scale is not None:
        parser.error("Use --ideal-memory or --memory-rate-scale, not both")
    if (args.baseline_mode == "fixed") != (args.baseline_rounds is not None):
        parser.error("Specify --baseline-mode fixed and --baseline-rounds together")
    profile = args.profile or ("recon" if args.backend == "scalar" else "mechanism")
    #v1 The archived scalar engine still contains its historical +3 baseline.
    #v1 Reject new planner/device overrides instead of silently ignoring them.
    overrides = (args.baseline_mode, args.baseline_rounds, args.source_fidelity_offset,
                 args.memory_rate_scale, args.guard_floor_base, args.guard_margin, args.cost_fidelity_weight,
                 args.baseline_filter, args.guard_mode, args.lookahead_slots, args.link_floor, args.reachability_filter,
                 args.degraded_link_fraction, args.quality_amplitude, args.quality_period_slots)
    if args.backend == "scalar" and (profile != "recon" or any(x is not None for x in overrides)
                                     or args.ideal_memory or args.ideal_swap or args.gate_noise):
        parser.error("Archived scalar execution supports only the recon profile without component overrides")
    from math import isfinite
    numeric = [args.gate_noise] + [v for v in (args.source_fidelity_offset, args.memory_rate_scale,
        args.guard_floor_base, args.guard_margin, args.cost_fidelity_weight, args.lookahead_slots,
        args.degraded_link_fraction, args.quality_amplitude, args.quality_period_slots) if v is not None]
    if any(not isfinite(v) for v in numeric) or (args.cost_fidelity_weight is not None and args.cost_fidelity_weight < 0):
        parser.error("Numeric parameters must be finite and the cost weight must be nonnegative")
    if args.link_floor == "none" and args.guard_floor_base is not None:
        parser.error("--link-floor none cannot be combined with --guard-floor-base")
    config, device, optical = profile_settings(profile)
    try:
        config = dataclasses.replace(config, **{k: v for k, v in {
            "baseline_mode": args.baseline_mode, "baseline_fixed_rounds": args.baseline_rounds,
            "qrise_floor_base": args.guard_floor_base, "guard_margin": args.guard_margin,
            "alpha": args.cost_fidelity_weight, "qrise_guard_mode": args.guard_mode,
            "qrise_lookahead_slots": args.lookahead_slots,
            "qrise_reachability_filter": None if args.reachability_filter is None else args.reachability_filter == "on",
            "baseline_fidelity_filter": None if args.baseline_filter is None else args.baseline_filter == "shared",
            "baseline_guard_floor": None if args.link_floor is None else args.link_floor == "shared"}.items() if v is not None})
        device = dataclasses.replace(device,
            memory_rate_scale=(0 if args.ideal_memory else device.memory_rate_scale
                               if args.memory_rate_scale is None else args.memory_rate_scale),
            swap_noise=not args.ideal_swap, gate_depolar_probability=args.gate_noise)
        if args.link_floor == "none":
            config = dataclasses.replace(config, qrise_floor_base=0.0, qrise_floor_target_weight=0.0)
        optical = dataclasses.replace(optical, **{k: v for k, v in {
            "source_fidelity_offset": args.source_fidelity_offset,
            "degraded_link_fraction": args.degraded_link_fraction,
            "phase_fidelity_amplitude": args.quality_amplitude,
            "quality_period_slots": args.quality_period_slots}.items() if v is not None})
    except ValueError as exc:
        parser.error(str(exc))
    seeds = list(range(args.seed_start, args.seed_start + (1 if args.smoke else args.seeds)))
    if args.smoke:
        cells = [[50, .9]]
    elif args.nodes or args.targets:
        cells = [[n, t] for n in sorted(set(args.nodes or [50])) for t in sorted(set(args.targets or [.9]))]
    else:
        cells = [[n, .9] for n in SIZES] + [[50, t] for t in TARGETS[:-1]]
    policies = [p for p in POLICIES if args.policies is None or p in args.policies]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "src").rglob("*.py"))}
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import netsquid as ns
    manifest = dict(reconstruction_version="review_mechanism_v2", backend=args.backend, profile=profile, quantum_components=args.backend == "components",
                    optical=dataclasses.asdict(optical),
                    config=dataclasses.asdict(config), device=dataclasses.asdict(device),
                    policies=policies, seeds=seeds, cells=cells,
                    expected_trials=len(seeds) * len(cells) * len(policies),
                    netsquid_version=ns.__version__, source_sha256=source_hashes, runner_sha256=runner_hash,
                    reference_source_sha256=json.loads((ROOT / "reference/source_sha256.json").read_text()),
                    assumptions=[("Archived scalar: original baseline +3 and clamp 5..10" if args.backend == "scalar"
                                  else "Component baseline mode: {}; no additive round penalty".format(config.baseline_mode)),
                                 "Profiles are reconstruction hypotheses, not recovered poster inputs",
                                 "Mechanism includes a declared shared link-quality floor; mechanism_e2e removes it for every policy",
                                 "Components: no mandatory baseline minimum rounds, no unscaled Gao memory expiry, no synthetic setup delay",
                                 "Quality variation is a known synthetic trace, not a measured orbital ephemeris",
                                 "Baseline filter is a scoped adapter choice; shared-filter control is available",
                                 "Optional forward reachability uses known current forecast, not a precomputed reserved route or perfect future availability",
                                 "Chain prediction uses elapsed time and actual processor memory rate, never measured qubit F as routing input",
                                 "Poster target values are comparison references, never simulator outputs",
                                 "No synthetic Q-CAST recovery EPR or per-policy latency charges",
                                 "Initial Werner F uses the declared source curve, common offset and optional per-link quality offset",
                                 "Planning B_u is link.width; actual storage is finite processor memory",
                                 "Ideal instantaneous bilateral twirling; ideal herald detection",
                                 "Gate durations 1/10/10 us are unpublished device assumptions",
                                 "Physical memory rate = optical.damping_rate_s * device.memory_rate_scale per qubit",
                                 "Planner retains recon one-slot Gao aging at optical.damping_rate_s; it is not the physical noise channel",
                                 "224ms is a whole-request cutoff here; poster link cutoff does not specify this scheduler",
                                 "Swap depolarization calibrated at F=1; physical BSM success assumed 1",
                                 "Q-CAST planning score retains its original .9 per-swap factor"],
                    #v1 Persist the execution schedule so old deferred-swap results
                    #v1 cannot be mistaken for the new incremental component run.
                    execution_schedule=("v1_incremental_swap" if args.backend == "components"
                                        else "recon_scalar_deferred_composition"),
                    aggregation="Report all attempts/time including failures; fidelity absent if no E2E pair; also show success rate")
    output = Path(args.output or ("outputs/scalar_recon" if args.backend == "scalar" else "outputs/" + profile))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "results.json"
    rows = []
    if path.exists():
        if not args.resume:
            parser.error("Output already exists; choose another directory or --resume")
        previous = json.loads(path.read_text())
        if previous["manifest"] != manifest:
            parser.error("Resume requires identical source/config/experiment manifest")
        rows = previous["runs"]
    #v1 Freeze executed inputs so later calibrations can still audit old data.
    snapshot = output / "source_snapshot"
    for relative in source_hashes:
        saved = snapshot / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        if not saved.exists():
            saved.write_bytes((ROOT / relative).read_bytes())
    runner_copy = snapshot / "scripts/run_experiment.py"
    runner_copy.parent.mkdir(parents=True, exist_ok=True)
    if not runner_copy.exists():
        runner_copy.write_bytes(Path(__file__).read_bytes())
    done = {(r["n"], r["target"], r["seed"], r["policy"]) for r in rows}
    start = time.monotonic()
    write_json(path, dict(manifest=manifest, runs=rows))
    jobs = [(args.backend, n, target, seed, index, policy, config, device, optical)
            for n, target in cells for seed in seeds for index, policy in enumerate(POLICIES)
            if policy in policies and (n, target, seed, policy) not in done]
    def save(row):
        rows.append(row)
        write_json(path, dict(manifest=manifest, runs=rows))
        print("{}/{} n={} target={} {}: {} F={} EPR={} slots={:.3f}".format(
            len(rows), manifest["expected_trials"], row["n"], row["target"], row["policy"], row["reason"],
            "-" if row["fidelity"] is None else "{:.5f}".format(row["fidelity"]),
            row["epr_pairs"], row["latency_slots"]), flush=True)
    if not jobs:
        pass
    elif args.workers == 1 and args.backend == "scalar":
        for job in jobs:
            save(execute_trial(job))
    else:
        # Recycle isolated simulators to bound native NetSquid object/cache
        # growth. This changes host execution only, never simulated physics.
        with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=5) as pool:
            for row in pool.imap_unordered(execute_trial, jobs, chunksize=1):
                save(row)
    print("Saved {} trials; executed {} this invocation in {:.1f}s; {}".format(
        len(rows), len(jobs), time.monotonic() - start, path))


if __name__ == "__main__":
    main()
