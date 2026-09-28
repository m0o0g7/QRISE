# Q-RISE with NetSquid Components

Q-RISE 포스터의 라우팅·정제 계획을 NetSquid 양자 컴포넌트로 실행하는 연구용 재구현입니다.
경로와 정제 횟수는 수식으로 계획하고, EPR 생성·손실·정제·스와핑·고전 통신은 이벤트로 실행합니다.
최종 fidelity는 양자 상태에서, 비용과 지연은 생성 시도 수와 시뮬레이션 시각에서 구합니다.

원저자의 시뮬레이터를 복제한 구현은 아닙니다. 미공개 장치·위상·예측 조건은 명시적인 가정이며,
GREEDY/HSP/NOPP/Q-CAST는 비교를 위한 단일 경로 adapter입니다.

## 환경 준비

검증 환경은 **Python 3.7.17 / NetSquid 1.1.7**입니다. 공개 Python 패키지 버전은
[requirements-tested.txt](requirements-tested.txt)에 기록되어 있습니다.

NetSquid는 이 저장소에 포함하지 않습니다. 먼저 별도로 제공되는 NetSquid 배포 절차에 따라
호환 환경에 설치한 다음, 프로젝트 루트에서 다음을 실행합니다.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e .
python -c "import netsquid as ns; print(ns.__version__)"
```

이하 명령의 `python`은 NetSquid가 설치된 환경의 Python입니다. 실행 스크립트는 프로젝트의
`src/`를 직접 불러오므로 editable 설치 없이도 실행할 수 있습니다.

## 실험 실행

모든 명령은 프로젝트 루트에서 실행합니다. 출력 폴더는 아직 결과가 없는 이름을 사용합니다.
실험 성공 후에만 그래프를 생성하도록 두 명령을 `&&`로 연결했습니다.

### 55회: 조건별 seed 1개

```bash
python scripts/run_experiment.py --seeds 1 --workers 2 --output outputs/run_55 \
  && python scripts/plot_results.py outputs/run_55
```

### 550회: 조건별 seed 10개

```bash
python scripts/run_experiment.py --seeds 10 --workers 2 --output outputs/run_550 \
  && python scripts/plot_results.py outputs/run_550

python scripts/report_results.py --directory outputs/run_550
python scripts/audit_results.py outputs/run_550
```

기본 조건은 **11개 × 정책 5개 × seed 수**입니다.

- 목표 0.9에서 노드 수 20, 30, 40, 50, 60, 70, 80, 90: 8개 조건
- 50노드에서 목표 0.6, 0.7, 0.8: 3개 조건
- 기본 seed 시작값: 1701. 새로운 표본은 `--seed-start 4701`처럼 지정합니다.

`--workers`는 독립 실험을 실행할 프로세스 수입니다. 시뮬레이션의 장치 성능을 바꾸지 않습니다.
`--nodes`, `--targets`, `--policies`로 실험 범위를 줄일 수 있습니다.

```bash
# 50노드 × 목표 0.9 × 두 정책 × 새 seed 3개 = 6회
python scripts/run_experiment.py \
  --nodes 50 --targets .9 --policies Q-RISE Q-CAST \
  --seeds 3 --seed-start 4701 --workers 2 --output outputs/subset
```

### 중단된 실행 재개

원래 실행 명령에 `--resume`을 추가하면 완료된 실험을 건너뜁니다.
**소스·설정·seed·실험 조건이 동일해야 합니다.** 주석만 바뀌어도 소스 해시가 달라지므로
기존 실행에 이어 붙일 수 없습니다. `--workers`만 변경하는 것은 가능합니다.

## 결과 읽기

실험 결과는 `outputs/` 아래에 저장되며, 이 폴더는 git에 포함하지 않습니다.

| 파일 | 내용 |
|---|---|
| `results.json` | 설정, seed, 소스 해시, 요청별 결과, 이벤트 계수, 링크 기록 |
| `source_snapshot/` | 해당 실험에 사용한 소스와 실행 스크립트 |
| `metrics.json` | 조건·정책별 평균, 표본 수, 성공률, 실패 원인 |
| `comparison.png/pdf` | fidelity·EPR·지연의 세 패널 그래프 |
| `figure2_reconstruction.png/svg` | 동일 비교 그래프의 추가 형식 |
| `success_rate.png/pdf` | 목표 0.9에서 노드 수별 성공률 |
| `REPORT.md` | `report_results.py`로 생성하는 결과·가정 보고서 |

- **Fidelity:** 최종 쌍이 생성된 요청의 평균입니다. 목표 미달 쌍도 포함합니다.
- **EPR:** 손실·정제 실패·재시작을 포함한 실제 소스 생성 시도 수입니다.
- **Latency:** 실제 시뮬레이션 경과시간입니다. 기본 1슬롯 = 1ms입니다.
- **Success:** 제한 시간 내 완료하고 최종 fidelity가 목표 이상인 요청의 비율입니다.
- EPR과 지연 평균에는 실패 요청도 포함합니다. 성공률이 다른 조건의 비용만 비교하면 오해할 수 있습니다.
- Fidelity 막대에는 빗금을 표시하지 않습니다. 성공률은 별도 `success_rate` 그래프에서 확인합니다.

## 실행 구조

```text
scripts/run_experiment.py: execute_trial()
  -> topology.make_topology()
  -> backend.run(): 시뮬레이터 초기화, Trial 시작
     -> Trial.run()
        -> run_qrise() / run_baseline(): 경로·정제 계획
        -> prepare_link()
           -> generate(): QSource, QuantumChannel, herald
           -> purify_once(): 양쪽 CNOT, 측정, 결과 교환
        -> extend_chain(): 새 링크마다 BSM 및 끝점 보정
        -> 완료 메시지, 최종 fidelity, 비용·지연 기록
```

Q-RISE는 매 홉에서 후보를 고르고 정제 횟수를 계산합니다. Baseline은 시작 시 전체 경로와
경로 최저 예상 fidelity에 따른 공통 정제 횟수를 정합니다. **두 방식 모두 같은 물리 실행기로
새 링크 준비 직후 스와핑**합니다. Baseline에 추가 정제 횟수나 정책별 비용 가산은 없습니다.

| 위치 | 역할 |
|---|---|
| [profiles.py](src/qrise_components/profiles.py) | 실험의 공통 장치·광학 입력과 계획 설정 |
| [physics.py](src/qrise_math/physics.py) | 생성 확률, BBPSSW, MCSFM, guard 역산, 저장 열화 예측 |
| [routing.py](src/qrise_math/routing.py) | 후보 필터, 비용식, 도달 가능성, 정제 횟수, baseline 경로 점수 |
| [topology.py](src/qrise_math/topology.py) | layered grid, 링크 폭·메모리, 거리·품질 trace |
| [backend.py](src/qrise_components/backend.py) | 생성·정제·홉별 스와핑·재시작·완료 처리 |
| [components.py](src/qrise_components/components.py) | 실제 source, channel, processor, 잡음 모델 |
| [programs.py](src/qrise_components/programs.py) | 정제·Bell 측정·Pauli 보정 회로 |

## 실험 설정

모든 실험은 [profiles.py](src/qrise_components/profiles.py)의 한 가지 설정을 사용합니다.
숫자는 포스터에서 복원된 장치 사양이 아니라 선언한 가정입니다.

| 항목 | 설정 |
|---|---|
| 초기 fidelity | 0.920 중심 거리·위상 곡선에 공통 +0.005 |
| 열화 링크 | 링크마다 20% 확률, fidelity 오프셋 −0.05~−0.02 |
| 품질 변화 | 위상 항 진폭 0.012, 주기 120슬롯 |
| 메모리 | 큐비트당 depolarizing rate 0.01/s |
| 게이트 | 단일 1µs, CNOT·측정 10µs, 개별 게이트 잡음 0 |
| 스와핑 | BSM 성공률 1 가정, 별도 depolarizing 잡음 적용 |
| 공통 링크 floor | 0.965 + 0.015 × 최종 목표 |
| 계획 | alpha=10, guard margin=0.004, 예측 범위 20슬롯, 누적 chain guard, 도달 가능성 필터 |
| 자원 | 노드당 메모리 10–14위치, 링크당 3–7 source/channel lane |
| 마감 | 모든 정책에 전체 요청 224ms |

## 대조 실험

Baseline의 초기 fidelity 필터는 기본적으로 꺼져 있습니다. 이는 **adapter의 가정**이며,
원 논문 모든 baseline의 성질로 일반화하지 않습니다. 명령줄 옵션으로 가정 하나씩을 바꿔
영향을 구분합니다. 바꾼 값은 `results.json`의 설정에 함께 저장됩니다.

```bash
# Baseline에도 같은 초기 fidelity 필터 적용
python scripts/run_experiment.py \
  --nodes 50 --targets .9 --seeds 3 --seed-start 4701 \
  --baseline-filter shared --output outputs/shared_filter

# 모든 정책에서 추가 링크 품질 하한 제거
python scripts/run_experiment.py \
  --nodes 50 --targets .6 .9 --seeds 3 --seed-start 4701 \
  --link-floor none --output outputs/e2e_only

# 같은 조건에서 메모리 잡음 제거
python scripts/run_experiment.py \
  --nodes 50 --targets .9 --seeds 3 --seed-start 4701 \
  --ideal-memory --output outputs/ideal_memory
```

전체 옵션은 `python scripts/run_experiment.py --help`로 확인합니다.

## 재현 범위와 한계

- 합성 layered grid와 거리·품질 trace를 사용합니다. 실제 위성 궤도나 관측 데이터가 아닙니다.
- 초기 fidelity 곡선, floor, 예측 범위, 비용 가중치와 장치 값은 미공개 항목을 채운 가정입니다.
- 향후 품질 trace와 전체 링크 상태를 안다고 가정합니다. 예측 오차·상태 배포 비용은 미반영입니다.
- 계획의 `B_u`는 링크 폭이며 저장된 EPR 재고가 아닙니다. 실행 메모리는 별도로 제한합니다.
- 링크 관측은 원본의 1슬롯 Gao 감쇠를 유지합니다. 누적 chain 예측과 실제 메모리는
  processor 잡음률을 사용하므로 예측 모델 간 차이가 남아 있습니다.
- 정제 실패·통신·메모리 대기는 실행하지만, 모든 링크를 병렬 준비하는 최적 스케줄은 아닙니다.
- 이상적인 순간 twirling과 검출, 결정적인 BSM 성공을 가정합니다. Herald 전파 지연은 계산합니다.
- 포스터의 링크별 cutoff와 현재 전체 요청 마감은 의미가 다릅니다.
- 단일 요청·단일 경로 adapter 비교이며 다중 요청 혼잡이나 원 논문의 전체 복구 절차는 미구현입니다.

결과는 **선언한 조건에서의 성능 비교**입니다. Q-RISE의 보편적 우위를 입증하지 않습니다. 
특히 낮은 fidelity 여유, 강한 후보 필터의 경로 배제, 예측 오차와 재시작 비용을
함께 평가해야 합니다.

## 검증과 저장소 구성

```bash
python -m pytest -q
python scripts/audit_results.py outputs/run_550
```

테스트는 실제 생성·정제·BSM, 유한 메모리, 마감, 계획 규칙, 품질 예측과 도달 가능성을 검사합니다.
감사 스크립트는 요청 수, 생성·정제·쌍 보존 회계와 시계를 검사합니다.

```text
src/          계획 수식과 NetSquid 실행 코드
scripts/      실험, 그래프, 보고서, 결과 감사
tests/        실행·계획 검증
outputs/      로컬 실험 결과 (git 제외)
```

## 논문

Y. Gao, A. Valera, W. K. G. Seah, *Poster: Q-RISE: Entanglement Routing Optimization
in LEO Satellite Networks*, IEEE SECON, 2026. DOI: `10.1109/SECON68281.2026.11579152`.

계획 로직은 `recon_Qu`의 수학 재구현에서 출발했습니다.
