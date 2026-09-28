"""Audit measured event accounting and frozen source for completed output."""
import hashlib
import json
from pathlib import Path
import argparse
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    data = json.loads((Path(args.directory) / 'results.json').read_text())
    manifest, rows = data['manifest'], data['runs']
    assert len(rows) == manifest['expected_trials'], 'incomplete experiment'
    keys = {(r['n'],r['target'],r['seed'],r['policy']) for r in rows}
    assert len(keys) == len(rows), 'duplicate trial keys'
    source_root = Path(args.directory) / 'source_snapshot'
    if not source_root.exists():
        source_root = ROOT
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source_root / path).read_bytes()).hexdigest() == digest, path
    if 'runner_sha256' in manifest:
        assert hashlib.sha256((source_root / 'scripts/run_experiment.py').read_bytes()).hexdigest() == manifest['runner_sha256']
    for r in rows:
        c = r['counters']
        assert r['epr_pairs'] == c['generation_attempts']
        assert c['generation_attempts'] == c['delivered_pairs'] + c['generation_losses'] + c['in_flight_emissions']
        assert 0 <= c['purification_attempts'] - (c['purification_discarded_pairs'] - c['purification_failures']) <= 1
        assert 0 <= r['elapsed_ns'] <= manifest['config']['cutoff_slots'] * manifest['config']['slot_s'] * 1e9
        assert abs(r['latency_slots'] - r['elapsed_ns'] / (manifest['config']['slot_s'] * 1e9)) < 1e-8
        assert c['peak_memory_positions'] <= 14
        if r['fidelity'] is not None:
            assert r['success'] == (r['fidelity'] >= r['target'])
            assert c['in_flight_emissions'] == 0
            # A restart discards one accumulated pair, even if many links were
            # consumed in forming it. Retain compatibility with legacy counters.
            discarded = c.get('restart_discarded_pairs', r['rollback_count'])
            assert c['delivered_pairs'] - c['purification_discarded_pairs'] - c['swaps'] - discarded == 1
    print('PASS: {} unique trials, frozen source, clock, raw attempts, purification accounting, final pair conservation'.format(len(rows)))


if __name__ == '__main__':
    main()
