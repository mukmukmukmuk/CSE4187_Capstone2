# 후보 번들 `population-104w-n20000-seed42-none`

쿠폰 미발송 시뮬레이터 데이터로 최종 학습한 번들이다. 52주 모델 4종과 4주 모델(`xgboost_gross_4w`)이 들어 있다.

**상태: 후보.** 전달 번들(`clv.api.HANDOFF_BUNDLE` = `artifacts/population-24m-n20000-seed42`)은 바꾸지 않았다.
아래 열린 문제를 정한 뒤 전달용으로 지정한다. 지금 쓰려면 `model_dir`로 이 디렉터리를 직접 넘긴다.

```python
from clv.api import predict_for_strategy
result = predict_for_strategy(FEATURES, ledger=DATASET, model_dir="clv/artifacts/population-104w-n20000-seed42-none",
                              require_gross_4w=True, require_control_only=True)
```

## 데이터

시뮬레이터 커밋 `d1992c4`(팀 main)의 기본 `population.yaml`이다. 104주(과거 52주 + 미래 52주), 20,000명,
`--coupon-mode none`(전원 미발송)으로 생성했다. 두 seed 모두 쿠폰 발송·사용·할인은 0건이다.

| seed | 용도 | 주문 | `raw_database_sha256` | `dataset_id` 끝부분 |
|---|---|---|---|---|
| 42 | 학습 | 48,178 | `76f255cac9bda22df85e31d5f76247b962421d756f14e746faeed80c8057e264` | `…-104w-seed42-n20000-c1d9799b77664f07-28d` |
| 43 | 학습에 쓰지 않은 평가 | 47,145 | `a0536e354437faaa3901b4954573c4d70d452e5c28b0dc5a51a398590fee0818` | `…-104w-seed43-n20000-16067626f77f9b53-28d` |

```bash
cd simulator
OUT=../results/simulator/main-d1992c4/population-104w-n20000-seed$SEED-none
PYTHONPATH=src python -m cli v4 generate --config src/simulate/v4/config/population.yaml --mode full \
  --n-customers 20000 --weeks 104 --seed $SEED --coupon-mode none --output $OUT/dataset
PYTHONPATH=src python -m cli v4 validate --data $OUT/dataset/raw.sqlite
PYTHONPATH=src python -m cli v4 features --data $OUT/dataset --output $OUT/features
```

## 학습

```bash
python -m clv train --weekly $D42/features --ledger $D42/dataset --require-gross-4w \
  --output results/clv/final-104w-n20000-seed42-none
```

- 기준시점 t0 = 2024-12-30(서울 월요일, 실험 `as_of_at`)이다. 정책은 `control_no_coupon`이다.
  미발송 데이터라 20,000명 전원이 학습 대상이고, 쿠폰을 받아 제외된 고객은 0명이다.
- 고객 분할은 Train 12,000명, Validation 4,000명, Test 4,000명이다. 52주 정답 0원 비율은 68.2%다.
- 52주 최종 모델은 Validation RMSE 기준 `xgboost_two_part`다.
- 4주 구매자는 Train 1,328명, Validation 430명, Test 463명이다. 선택 설정은 `reg:gamma`, 깊이 3,
  `min_child_weight` 5, 나무 41개다.
- 이 디렉터리에는 체크섬 대상 파일, `model_card.json`, `reports/evaluation.json`만 넣었다. 고객별 예측 CSV 같은
  나머지 보고서는 학습 출력 디렉터리에 있다.

## 검증 결과

`reports/verification/`에 원자료가 있다. 신뢰구간은 고객 단위 Bootstrap(2,000회) 95%다.

```bash
python -m clv.scripts.compare_bundles $D43/features $D43/dataset 2024-12-30 \
  clv/artifacts/population-24m-n20000-seed42 results/clv/final-104w-n20000-seed42-none compare_seed43.json
python -m clv.scripts.gross_4w_baselines results/clv/final-104w-n20000-seed42-none \
  results/simulator/main-d1992c4 gross_4w_baselines.json
```

### 52주: 같은 seed 43 고객 20,000명, 기존 전달 번들과 비교

| 최종 모델 | MAE | RMSE | 평균 예측(실제 4,172) | Spearman | 상위 20% 일치율 |
|---|---|---|---|---|---|
| 기존 전달 번들 (`xgboost`) | 5,545 | 11,755 | 3,654 | 0.272 | 0.360 |
| 이 번들 (`xgboost_two_part`) | 5,756 | 11,692 | 4,146 | 0.293 | 0.364 |

- 차이(이 번들 − 기존): MAE +211 [+194, +228], RMSE −63 [−91, −35].
- 이 번들은 평균이 실제와 거의 같다. 기존 번들은 평균을 12% 낮게 잡는다. 정답의 68%가 0원이라 MAE는 낮게 예측할수록
  유리하다. 순위와 RMSE는 이 번들이 조금 낫다. 개선 폭은 작다.
- seed 42 Test 4,000명에서 이 번들의 `xgboost_two_part`는 MAE 5,992, RMSE 12,646, 상위 20% 일치율 0.364다.

### 4주: 고객별 차이를 예측하지 못한다

| seed 43 구매자 2,171명(실제 평균 7,266) | MAE | RMSE | Spearman | 평균 예측 |
|---|---|---|---|---|
| `xgboost_gross_4w` | 5,519 [5,157, 5,895] | 10,411 [8,978, 11,862] | 0.007 | 6,564 |
| 상수: 학습 구매자 평균 | 5,652 | 10,386 | – | 6,925 |
| 상수: 학습 구매자 중앙값 | 5,082 | 10,803 | – | 4,273 |
| 고객의 과거 52주 주문당 총액 | 5,667 | 10,963 | 0.015 | 5,027 |
| Gamma-Gamma 기대 금액 × 학습 배율 | 5,672 | 10,455 | −0.014 | 6,885 |

- seed 42 Test 구매자 463명에서도 Spearman은 −0.045이고, MAE 신뢰구간은 [5,091, 7,149]로 넓다.
- 고객 간 예측 표준편차는 약 480인데 실제 표준편차는 약 10,400이다. 모델은 사실상 상수를 낸다. RMSE는 학습 평균
  상수와 차이가 없다(상수 − 모델: [−53, +4]).
- 고객 자신의 과거 주문 금액도 미래 4주 구매 금액과 순위상관이 없다. 이 시뮬레이터에서 구매 시 금액은 고객 이력과
  무관하게 생성되는 것으로 보인다.

## 열린 문제

1. 고객별 구매 금액 수준이 없는 것이 시뮬레이터의 의도인지 시뮬레이터 담당과 확인한다. 의도라면 4주 값은 상수와 같다.
2. 4주 예측 평균(6,564)이 학습 구매자 평균(6,925)과 seed 43 실제 평균(7,266)보다 낮다. 결합식의 증분 매출과 ROI도
   5~10% 낮게 나온다. 조기 종료 기준인 Validation 구매자 평균이 6,450으로 낮았던 것이 한 원인이다.
3. 위 두 가지를 정한 뒤 `HANDOFF_BUNDLE`을 이 번들로 바꾸고 `clv/api.md`의 수치와 경로를 갱신한다.

`final_strategy.inputs.integrate_inputs`에 이 번들의 seed 43 52주·4주 batch를 넣으면 20,000명 모두 통과한다
(`policy_condition = control_no_coupon`, 52주·4주 종료 시각 검사 포함).
