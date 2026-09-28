# Scalar reference

`scalar_routing.py`는 원본 `recon_Qu` 라우팅·실행 코드의 보관본입니다.
계획 호환성 테스트와 `run_experiment.py --backend scalar`가 사용하므로 삭제하면 안 됩니다.
`source_sha256.json`은 출발점이 된 원본 파일들의 해시 기록이며 외부 파일을 읽는 설정이 아닙니다.

이 엔진에는 baseline 정제 `MPR + 3`, 최소 5·최대 10 제한과 원래 scalar 비용 가정이 남아 있습니다.
현재 component backend는 해당 추가 정제·합성 비용을 적용하지 않습니다. 두 엔진의 차이를
순수한 NetSquid 물리 효과만으로 해석하지 마세요.

이 파일은 원본 비교 기준으로 유지합니다. 현재 실행·계획을 수정할 때는 `src/`를 사용합니다.
