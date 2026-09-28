# 보존한 실험 결과

실험별 `results.json`과 `source_snapshot/`을 함께 보존합니다. 스냅샷은 실제 실행 당시의
코드이므로 현재 소스의 주석 정리와 별개로 변경하지 않습니다. 보고서는 저장된 결과에서 생성합니다.

| 폴더 | 횟수 | Seed | 용도 |
|---|---:|---|---|
| [mechanism_55](mechanism_55/) | 55 | 1701 | 조건별 한 표본의 빠른 비교 |
| [mechanism_550](mechanism_550/) | 550 | 1701–1710 | 조건별 10개 표본의 기본 비교 |
| [review_final_165](review_final_165/) | 165 | 3701–3703 | 마지막 알고리즘 수정 후 새 seed의 독립 검증 |
| [review_final_shared_15](review_final_shared_15/) | 15 | 3701–3703 | 50노드·목표 .9, baseline에도 F 필터 제공 |
| [review_final_e2e_30](review_final_e2e_30/) | 30 | 3701–3703 | 50노드·목표 .6/.9, 모든 정책에서 추가 floor 제거 |

1701–1710 중 일부 seed는 입력 선택에도 사용되었습니다. 550회라는 횟수만으로 독립 검증이라고
해석하지 않습니다. 독립 검증 역시 조건별 3개 표본이므로 통계적 강건성은 제한적입니다.

## 기본 결과: 50노드, 목표 fidelity 0.9

`mechanism_550`에서 각 정책 10회가 모두 목표를 충족했습니다.

| 정책 | 평균 fidelity | 평균 생성 시도 | 평균 지연 (slots) |
|---|---:|---:|---:|
| Q-RISE | 0.91571 | 190.5 | 32.881 |
| Q-CAST adapter | 0.93071 | 545.3 | 80.346 |

이 조건에서 EPR 약 65.1%, 지연 약 59.1% 절감입니다.
[비교 그래프](mechanism_550/comparison.png)와 [성공률](mechanism_550/success_rate.png)을 함께 봅니다.

## 가정의 영향: 새 seed 3701–3703

| 조건 | 목표 | Q-RISE EPR | Q-CAST EPR | EPR 절감 |
|---|---:|---:|---:|---:|
| 기본 | .9 | 192.3 | 662.7 | 71.0% |
| Baseline도 F 필터 | .9 | 192.3 | 329.0 | 41.5% |
| 추가 floor 없음 | .6 | 7.3 | 8.0 | 8.3% |

상대 성능에는 후보 필터와 추가 링크 품질 요구가 영향을 줍니다. 실패 비용도 비용 평균에 포함되며,
완성 쌍 fidelity와 성공률은 별개 지표입니다. 자세한 조건과 실패는 각 폴더의 `REPORT.md`에 있습니다.

## 감사와 재실행

프로젝트 루트에서 실행합니다.

```bash
python scripts/audit_results.py outputs/mechanism_550
python scripts/plot_results.py outputs/mechanism_550
python scripts/report_results.py --directory outputs/mechanism_550
```

현재 소스와 보존된 스냅샷은 주석·문서가 다를 수 있습니다. 감사는 스냅샷을 기준으로 수행합니다.
새 실험은 새 출력 폴더를 사용하며, 기존 실행의 소스 해시를 수정해 resume 검사를 우회하지 않습니다.

새 실험 폴더는 기본적으로 Git에서 제외됩니다. 공개할 데이터만 `.gitignore` 보존 목록에 추가하세요.
