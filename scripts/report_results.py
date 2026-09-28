"""Build the experiment report from saved results, never from target numbers."""
import json
import argparse
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]


def avg(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return mean(values) if values else None


def number(value, digits=3):
    return '—' if value is None else format(value, '.{}f'.format(digits))


def selected(rows, n, target, policy):
    return [r for r in rows if (r['n'], r['target'], r['policy']) == (n, target, policy)]


# Build the report exclusively from the selected dataset and its manifest.
def incremental_report(data, output):
    manifest, rows = data['manifest'], data['runs']
    config = manifest['config']
    mode = config.get('baseline_mode', 'legacy_extra3')
    rule = ('경로 최저 예상 F의 MPR, 추가 라운드 없음' if mode == 'path_mpr'
            else '모든 baseline 링크에 고정 {}회'.format(config['baseline_fixed_rounds']) if mode == 'fixed'
            else '역사적 +3·최소5·최대10 규칙')
    lines = ['# 실제 컴포넌트 실행 결과', '',
             '프로필: `{}`. Baseline 정제: {}.'.format(manifest.get('profile', 'legacy'), rule),
             '스케줄: `{}`. 저장 결과 {} / {}회.'.format(
                 manifest['execution_schedule'], len(rows), manifest['expected_trials']),
             '각 링크 준비 후 즉시 BSM·보정을 완료하고 다음 링크를 진행한다. 모든 정책에 같은 실행기를 사용한다.',
             '첫 링크와 누적 얽힘 쌍은 후속 링크를 기다리며 메모리 잡음을 받는다.', '',
             '| 노드 | 목표 | 정책 | 성공/실행 | F 표본 수 | 완성 쌍 평균 F | 생성 시도/요청 | 지연 ms/요청 |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for n, target in manifest['cells']:
        for policy in manifest['policies']:
            group = selected(rows, n, target, policy)
            latency = avg(group, 'latency_slots')
            lines.append('| {} | {} | {} | {}/{} | {} | {} | {} | {} |'.format(
                n, target, policy, sum(r['success'] for r in group), len(group),
                sum(r['fidelity'] is not None for r in group), number(avg(group, 'fidelity'), 5),
                number(avg(group, 'epr_pairs'), 1),
                number(None if latency is None else latency * manifest['config']['slot_s'] * 1000)))
    lines.extend(['', 'F는 목표 미달 완성 쌍을 포함하고, 생성 시도·지연은 실패 요청도 포함한다.',
                  '비용과 목표 충족률을 함께 비교한다. 경로 adapter는 원 논문의 전체 프로토콜 구현이 아니다.',
                  '원문은 점진적 MCSFM을 설명하지만 gate scheduling과 재시작 절차까지 공개하지는 않았다.',
                  '적은 seed의 실행은 동작 점검용이며 Fig.2 경향 재현의 통계적 검증은 별도다.', ''])
    if manifest.get('profile'):
        optical, device = manifest['optical'], manifest['device']
        lines.extend(['## 공개되지 않은 입력 가정', '',
            '- 공통 초기 F: 거리/위상 곡선 + {} + 선언된 링크별 품질 오프셋.'.format(optical['source_fidelity_offset']),
            '- 공통 실제 메모리 depolarization rate: {} /s/큐비트.'.format(optical['damping_rate_s'] * device['memory_rate_scale']),
            '- Eq.(3) alpha={}; baseline 후보의 fidelity 필터={}. 경로 점수 규칙 자체는 유지한다.'.format(config['alpha'], config.get('baseline_fidelity_filter', True)),
            '- Q-RISE guard={}, 예측 구간={} slots. Baseline 공통 링크 floor 적용={}.'.format(
                config.get('qrise_guard_mode', 'equal'), config.get('qrise_lookahead_slots', 0), config.get('baseline_guard_floor', False)),
            '- 링크 floor={} + {} × target; 준비 후 실측 F를 보장하는 사후 판정이 아니라 계획 기준이다.'.format(config['qrise_floor_base'], config['qrise_floor_target_weight']),
            '- 열화 링크 비율={}, 품질 진폭={}, 품질 주기={} slots; 합성 입력이며 실제 LEO 궤도 자료가 아니다.'.format(
                optical.get('degraded_link_fraction', 0), optical.get('phase_fidelity_amplitude', .004), optical.get('quality_period_slots') or optical.get('dynamics_period_slots', 420)),
            '- 링크 관측은 기존 1슬롯 Gao aging을 유지한다. chain 모드의 누적 예측은 실제 processor 잡음률로 대기 노화를 반영한다.',
            '- 현재 예측의 전방 도달 가능성 검사={}. 전체 링크 상태가 알려졌다는 추가 계획 가정이다.'.format(config.get('qrise_reachability_filter', False)),
            '- 전체 요청 cutoff {} slots; 포스터의 링크별 cutoff와 구분해야 한다.'.format(config['cutoff_slots']),
            '- 입력 선택에 활용한 seed1701–1703의 결과는 독립 검증으로 간주하지 않는다.'
                if config.get('qrise_guard_mode') == 'chain' else '- 기존 fig2 보정 seed는 1701/1702, 1801–1803이다. 원저자 설정으로 확인되지 않았다.',
            '- 정제 성공 확인·유한 메모리·홉별 BSM 이벤트는 공통 실행기를 사용한다.', '',
            '## 포스터의 50노드 수치와 직접 비교', '',
            '| 지표 | 포스터의 목표값 | 이번 측정 |', '|---|---:|---:|'])
        qr, qc = selected(rows, 50, .9, 'Q-RISE'), selected(rows, 50, .9, 'Q-CAST')
        for field, label, target in [('fidelity', 'Q-RISE E2E F', .9188),
                                     ('epr_pairs', 'Q-RISE EPR 생성 시도', 213.7),
                                     ('latency_slots', 'Q-RISE 슬롯', 59.7)]:
            lines.append('| {} | {} | {} |'.format(label, target, number(avg(qr, field), 4)))
        for field, label, target in [('epr_pairs', 'Q-CAST 대비 EPR 절감 (%)', 67),
                                     ('latency_slots', 'Q-CAST 대비 지연 절감 (%)', 46)]:
            a, b = avg(qr, field), avg(qc, field)
            value = None if a is None or b is None or b == 0 else 100 * (1 - a / b)
            lines.append('| {} | {} | {} |'.format(label, target, number(value, 2)))
        lines.extend(['', '절감률은 실패 비용을 포함한 요청당 평균의 비율이다. 성공률이 다르면 같은 서비스 품질의 절감으로 해석할 수 없다.',
            '숫자·노드별 실패를 강제로 맞추지 않는다. 성공한 baseline, 목표 미달 및 cutoff도 모두 표에 포함한다.', '',
            '## 마감 실패와 정제 비용 진단', '',
            '| 노드 | 정책 | cutoff/요청 | 정제 단계 종류 | 평균 정제 시도 |',
            '|---:|---|---:|---|---:|'])
        for n, target in manifest['cells']:
            if target != .9:
                continue
            for policy in manifest['policies']:
                group = selected(rows, n, target, policy)
                rounds = sorted({x for r in group for x in r['rounds']})
                attempts = [r['counters']['purification_attempts'] for r in group]
                lines.append('| {} | {} | {}/{} | {} | {} |'.format(n, policy,
                    sum(r['reason'] == 'cutoff' for r in group), len(group), ', '.join(map(str, rounds)),
                    number(mean(attempts) if attempts else None, 1)))
        lines.append('')
    output.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote {} from {} component trials'.format(output, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(ROOT / 'outputs/mechanism'))
    parser.add_argument('--output', help='Defaults to <directory>/REPORT.md')
    args = parser.parse_args()
    data = json.loads((Path(args.directory) / 'results.json').read_text())
    output = Path(args.output) if args.output else Path(args.directory) / 'REPORT.md'
    # Older datasets need their original reporting script and execution model.
    if data['manifest'].get('execution_schedule') != 'v1_incremental_swap':
        parser.error('This report supports incremental component results only')
    incremental_report(data, output)


if __name__ == '__main__':
    main()
