# 후보 번들 `population-104w-n20000-seed42-none`

쿠폰 미발송 시뮬레이터 데이터로 최종 학습한 번들이다(clv 0.4.0, 번들 형식 3, 학습 코드 커밋 `be97680`).
52주 모델 4종과 4주 조건부 총구매액 후보 4종이 모두 들어 있다.

**상태: 후보.** 전달 번들(`clv.api.HANDOFF_BUNDLE` = `artifacts/population-24m-n20000-seed42`)은 바꾸지 않았다.
지금 쓰려면 `model_dir`로 이 디렉터리를 직접 넘긴다.

```python
from clv.api import predict_for_strategy
result = predict_for_strategy(FEATURES, ledger=DATASET, model_dir="clv/artifacts/population-104w-n20000-seed42-none",
                              require_gross_4w=True, require_control_only=True)
# 다른 4주 후보(민감도 분석·비교용): gross_4w_model="xgboost_gross_4w" 등
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
python -m clv train --weekly $D42/features --ledger $D42/dataset --require-gross-4w --output <새 디렉터리>
```

- 기준시점 t0 = 2024-12-30(서울 월요일, 실험 `as_of_at`)이다. 정책은 `control_no_coupon`이다.
  미발송 데이터라 20,000명 전원이 학습 대상이고, 쿠폰을 받아 제외된 고객은 0명이다.
- 고객 분할은 Train 12,000명, Validation 4,000명, Test 4,000명이다. 52주 정답 0원 비율은 68.2%다.
  52주 최종 모델은 Validation RMSE 기준 `xgboost_two_part`다.
- 4주 구매 고객은 Train 1,328명, Validation 430명, Test 463명이다.
- 이 디렉터리에는 체크섬 대상 파일, `model_card.json`, `reports/evaluation.json`만 넣었다. 고객별 예측 CSV 같은
  나머지 보고서는 학습 출력 디렉터리에 있다.

## 4주 기본 모델 선택

규칙은 [model_spec.md](../../model_spec.md) 3장에 있다. Train + Validation 구매 고객 1,758명(평균 6,808)으로
고객 단위 5겹 교차검증을 3번 반복한 결과다(`registry.json`의 `gross_4w.selection`).

| 후보(단순한 순서) | 교차검증 RMSE | 최선과의 MSE 차이 | 차이의 표준오차 | 1-SE 안 | 교차검증 예측 평균 |
|---|---|---|---|---|---|
| **`mean_gross_4w`** (기본) | **8,875** | 0 (최선) | – | 예 | 6,808 |
| `gamma_gamma_gross_4w` | 8,966 | 1,619,896 | 676,765 | 아니오 | 6,808 |
| `xgboost_gross_4w` | 8,928 | 951,092 | 302,750 | 아니오 | 6,805 |
| `xgboost_gamma_gross_4w` | 8,907 | 566,602 | 225,247 | 아니오 | 6,559 |

학습 구매 고객 평균 상수가 교차검증 오차도 가장 낮다. 나머지 세 후보는 모두 그보다 1 표준오차 넘게 나쁘다.
단순해서가 아니라 성능으로도 1위다. gamma 손실 XGBoost는 교차검증에서도 평균을 낮게 잡는다(6,559 대 6,808).

## 검증 결과

`reports/verification/`에 원자료가 있다. 신뢰구간은 고객 단위 Bootstrap 95%다.

```bash
python -m clv evaluate --weekly $D43/features --ledger $D43/dataset --model-dir <이 번들> --cutoff 2024-12-30 \
  --population all --output evaluate_new_seed43_all.json          # 52주 4종 + 4주 후보 4종
python -m clv.scripts.compare_bundles $D43/features $D43/dataset 2024-12-30 \
  clv/artifacts/population-24m-n20000-seed42 <이 번들> compare_seed43.json   # 52주, 기존 번들과 같은 고객 비교
```

### 4주 후보: 선택에 쓰지 않은 고객

| 후보 | seed 42 Test 463명 MAE | RMSE | seed 43 2,171명 MAE | RMSE | seed 43 Spearman | seed 43 평균 예측 |
|---|---|---|---|---|---|---|
| `mean_gross_4w` | 6,138 | 12,811 [8,604, 16,962] | 5,652 [5,284, 6,014] | 10,386 [8,929, 11,842] | – | 6,925 |
| `gamma_gamma_gross_4w` | 6,167 | 12,871 | 5,672 | 10,455 | −0.014 | 6,885 |
| `xgboost_gross_4w` | 6,128 | 12,819 | 5,647 | 10,387 | 0.032 | 6,918 |
| `xgboost_gamma_gross_4w` | 6,014 | 12,856 | 5,519 | 10,411 | 0.007 | 6,564 |

- 실제 평균은 seed 42 Test 7,710, seed 43 7,266이다. 금액 꼬리가 길어 표본 평균 자체가 흔들린다
  (seed 43 표준오차 약 223).
- 어떤 후보도 고객 간 순위를 맞히지 못한다(Spearman 0 근처). 이 데이터에서 구매 시 금액은 과거 이력과 무관하게
  생성되는 것으로 보인다. 그래서 4주 값은 사실상 모든 고객에게 같은 수준이다.

### 52주: 같은 seed 43 고객 20,000명, 기존 전달 번들과 비교

| 최종 모델 | MAE | RMSE | 평균 예측(실제 4,172) | Spearman | 상위 20% 일치율 |
|---|---|---|---|---|---|
| 기존 전달 번들 (`xgboost`) | 5,545 | 11,755 | 3,654 | 0.272 | 0.360 |
| 이 번들 (`xgboost_two_part`) | 5,756 | 11,692 | 4,146 | 0.293 | 0.364 |

- 차이(이 번들 − 기존): MAE +211 [+194, +228], RMSE −63 [−91, −35].
- 이 번들은 평균이 실제와 거의 같고, 기존 번들은 평균을 12% 낮게 잡는다. 정답의 68%가 0원이라 MAE는 낮게
  예측할수록 유리하다. 순위와 RMSE는 이 번들이 조금 낫다. 개선 폭은 작다.
- seed 42 Test 4,000명에서 이 번들의 `xgboost_two_part`는 MAE 5,992, RMSE 12,646, 상위 20% 일치율 0.364다.

`final_strategy.inputs.integrate_inputs`에 이 번들의 seed 43 52주·4주 batch를 넣으면 20,000명 모두 통과한다
(기본 후보와 `xgboost_gross_4w` 모두, `policy_condition = control_no_coupon`, 52주·4주 종료 시각 검사 포함).

## 열린 문제

1. 고객별 구매 금액 수준이 없는 것이 시뮬레이터의 의도인지 시뮬레이터 담당과 확인한다.
2. 기본 4주 값이 상수라 결합식에서 고객 간 차이는 Uplift와 쿠폰 비용에서만 생긴다. 팀이 알고 쓰도록 공유한다.
3. 위를 정한 뒤 `HANDOFF_BUNDLE`을 이 번들로 바꾸고 `clv/api.md`의 수치와 경로를 갱신한다.
