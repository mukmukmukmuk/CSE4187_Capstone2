# CLV 모델 명세: 52주 CLV와 4주 조건부 총구매액

CLV 담당자와 Uplift 담당자가 합의한 예측 대상·학습 모집단·데이터 계약이다. 구현은 `clv/`에 있고, 전달·추론 방법은
[api.md](api.md), 설치·실행은 [readme.md](readme.md)에 있다.

## 0. 문서의 성격

이 문서는 **팀 내부 합의**다. 공식 프로젝트 요구사항([docs/request.md](../docs/request.md),
[docs/request_specification.md](../docs/request_specification.md))을 바꾸거나 대체하지 않는다.

| 구분 | 내용 | 출처 |
| --- | --- | --- |
| 공식 요구사항 | 고객별 CLV 예측(기간 표현은 12개월), 최소 20,000명·12개월 이상 데이터, 학습·튜닝에 쓰지 않은 고객으로 최종 전략 평가 | 공식 문서 |
| 팀 내부 합의 | CLV 기간은 입력 13주·예측 52주(364일)로 구현, 52주 CLV의 Control-only 학습, 4주(28일) 조건부 총구매액 모델 추가, 두 모델의 같은 고객·같은 `t0`, Uplift와의 결합식, 원천 금액 단위 그대로 사용 | 이 문서 |

CLV 예측 기간은 팀 합의로 **52주(364일)**로 통일했다. 공식 요구사항의 '12개월'과 다르다는 점을 알고 정한 의도된 설계이며,
달력 12개월 방식으로 바꾸지 않는다. `predicted_clv_12m`은 기존 인터페이스와의 호환을 위해 유지하는 열 이름이다.
저장소의 [docs/request.md](../docs/request.md) 7항도 "향후 52주 기준"으로 적혀 있다.

## 1. 두 모델 요약

| 구분 | 52주 CLV | 4주 조건부 총구매액 |
| --- | --- | --- |
| 예측 열 | `predicted_clv_12m` | `predicted_gross_4w_if_purchase` |
| 입력 이력 | `t0` 이전 13주 | `t0` 이전 13주(같은 예측 변수 17개) |
| 예측 구간 | `[t0, t0 + 52주)` | `[t0, t0 + 4주)` = 28일 |
| 학습 모집단 | Control | Control 중 4주 관측이 완전하고 4주 안에 구매한 고객 |
| 금액 기준 | 할인 후 실결제액 | 할인 전 총구매액 |
| 의미 | 쿠폰을 보내지 않았을 때의 장기 고객가치 | 쿠폰을 보내지 않았을 때, 구매한다면 기대되는 단기 구매 규모 |
| 대상 정의 ID | `clv_paid_amount_52w_v1` | `gross_amount_4w_if_purchase_v1` |
| 모델 | Baseline, BG/NBD + Gamma-Gamma, XGBoost, 두 파트 XGBoost 중 Validation RMSE 최소 | 후보 넷을 모두 저장하고 교차검증 1-표준오차 규칙으로 기본 모델을 고른다(아래) |

두 모델은 같은 `customer_id`와 같은 `t0`를 쓴다. 추론은 같은 13주 예측 변수 행 하나에서 두 값을 함께 계산한다.

## 2. 52주 CLV (Control-only)

- 정의: `t0`부터 52주 동안의 결제액 합. 결제액은 `gross_revenue - redemption_discount_cost`(할인 후 실결제액)다.
  쿠폰 발송비(`marketing_send_cost_minor`)는 빼지 않는다. 기존 정의 그대로다.
- 학습 모집단: Control(쿠폰 미발송군)만 Train/Validation/Test로 나눈다. Treatment(쿠폰 발송군)는 학습·모델 선택·평가에
  쓰지 않고 `unused`로 둔다. 그래서 예측값은 "쿠폰을 보내지 않았을 때의 예상 실결제액"이다.
- 52주가 모두 완전 관측된 고객만 정답이 있다. 기존 출력 열과 상위 20% 표시(`is_high_value_customer`)는 그대로다.

## 3. 4주 조건부 총구매액

### 정의

```text
predicted_gross_4w_if_purchase(x) = E[ 4주 할인 전 총구매액 | 4주 안에 1회 이상 구매, X(t0) = x ]   (쿠폰 미발송 조건)
```

### 정답

- 관측 구간은 `[t0, t0 + 4주)`다. 고객별로 그 구간의 주간 `gross_revenue`(할인 전 금액)를 모두 더한다.
- 반복 주문을 포함한다. 할인액과 쿠폰 발송비는 빼지 않는다.
- 내부 열은 다음 세 개다.
  - `actual_purchases_4w`: 4주 안 구매 횟수
  - `actual_gross_4w_if_purchase`: 구매 고객의 총구매액. 구매가 없으면 0이 아니라 결측
  - `gross_4w_window_complete`: 4주가 모두 완전 관측됐는지. 아니면 두 값 모두 결측
- 주문 원장(`raw.sqlite`)을 주면 4주 구매 횟수와 총구매액을 원장의 `gross_minor` 합계와 고객별로 대조한다.
- 52주 정답을 13으로 나누어 쓰지 않는다. 52주 두 파트 모델의 `expected_amount_if_purchase`도 재사용하지 않는다.

### 후보 모델

52주 모델에서 4주 몫을 잘라 쓰지 않는다. 52주 값은 비구매 고객의 0원이 섞인 기댓값이라, 결합식에 넣으면 구매 확률이
두 번 곱해진다. 대신 같은 입력으로 4주 조건부 금액을 직접 학습하는 후보 넷을 모두 번들에 저장한다.

| 후보(단순한 순서) | 내용 | 주문 원장 |
| --- | --- | --- |
| `mean_gross_4w` | Train 구매 고객의 평균 총구매액(상수). 비교 기준 | 필요 없음 |
| `gamma_gamma_gross_4w` | Gamma-Gamma 기대 거래 금액 × Train 배율. 52주 BG/NBD + GG와 같은 모형·보정 기간(13주)을 할인 전 금액으로 적합 | 필요 |
| `xgboost_gross_4w` | XGBoost, 제곱오차 손실 | 필요 없음 |
| `xgboost_gamma_gross_4w` | XGBoost, gamma 손실. 금액이 0인 구매 고객이 있으면 학습하지 않는다(`skipped`) | 필요 없음 |

추론에서는 `gross_4w_model`(CLI `--gross-4w-model`)로 어느 후보든 고를 수 있다. 생략하면 아래 규칙으로 고른 기본
모델이다. 쓴 후보는 4주 batch의 `model_version`(`clv-<버전>+<후보>.<체크섬>`)과 `producer_metadata.gross_4w`에 남는다.

### 기본 모델 선택 규칙

1. **지표: 제곱오차.** 결합식(`uplift_4w × 4주 금액`)에는 기대값이 필요하다. 평균을 맞히는 예측을 가장 좋게 평가하는
   손실이 제곱오차다. MAE는 중앙값을 맞히는 예측을 좋게 평가하므로 쓰지 않는다.
2. **추정: 반복 K겹 교차검증.** Test를 뺀 Train + Validation 구매 고객으로 고객 단위 5겹 교차검증을 3번 반복한다.
   - 겹마다 후보를 처음부터 다시 학습하고, 학습에 쓰지 않은 겹을 예측한다.
   - XGBoost는 겹의 학습 고객 안에서 다시 검증 고객을 떼어 설정과 트리 수를 고른다(중첩 교차검증).
   - 검증 고객 한 묶음(수백 명)만으로 고르면 금액 꼬리가 길어 결과가 잡음에 흔들린다. 그래서 이 방법을 쓴다.
3. **결정: 1-표준오차 규칙.** 평균 제곱오차가 가장 낮은 후보와의 차이가 그 차이의 표준오차 안에 드는 후보 중 가장
   단순한 것을 고른다. 표준오차는 같은 고객끼리 짝지은 오차 차이로 계산한다. 더 복잡한 후보는 잡음보다 확실히 나을
   때만 기본 모델이 된다.
4. **보고: Test.** Test 구매 고객은 선택에 쓰지 않는다. 후보마다 MAE·RMSE·편향·순위상관·실제/예측 평균과 고객 단위
   Bootstrap 95% 구간을 남긴다(`registry.json`의 `gross_4w.candidate_test_metrics`).

교차검증 표(후보별 평균 제곱오차, 최선과의 차이, 표준오차, 규칙 안 여부, 교차검증 예측 평균)는
`registry.json`의 `gross_4w.selection`에 있다. `gross_4w_final_model`을 후보 이름으로 주면 규칙 대신 그 후보를 기본으로 쓴다.

### 학습·평가

- 학습 대상은 Control, 4주 관측 완전, 4주 안 구매 1회 이상을 모두 만족하는 고객이다.
- 52주 모델과 같은 고객 단위 분할(Train/Validation/Test)을 쓴다. 최종 후보는 Train으로 적합한다. XGBoost 후보는
  Validation RMSE로 설정과 트리 수를 고른다.
- 금액이 0인 구매 고객도 정답으로 쓴다. 이때는 0을 다룰 수 없는 `xgboost_gamma_gross_4w`를 학습하지 않고 기록한다.
- 쿠폰 미발송 조건으로 정의한 모델이라 `observed_experiment_mix` 정책에서는 학습하지 않는다(`skipped`).
- 학습 시점 `t0`는 실험 시각(`as_of_at`)이 기본이다. 4주 구간이 쿠폰 유효기간(28일)·Uplift 구간과 같아야 하기 때문이다.
- 평가는 Control Test 고객 중 4주 안에 실제로 구매한 고객으로 한다. 지표는 MAE, RMSE, 평가 고객 수, 실제 평균
  총구매액, 예측 평균 총구매액이다(`reports/evaluation.json`의 `gross_4w`는 기본 모델, `candidates_test`는 후보 전부).
  `python -m clv evaluate`도 평가 대상 구매 고객으로 후보마다 같은 지표를 낸다(`gross_4w.candidates`).
- 배정군 진단(`gross_4w.assignment_diagnostic`): Treatment와 Control의 구매자당 4주 할인 전 총구매액을 비교한다.
  구매 고객 수·비율, 평균·중앙값, 차이와 95% 구간(Welch 정규 근사), 비율, 모델 예측 평균을 남긴다. Treatment는 이
  진단에만 쓰고 4주 모델 학습에는 쓰지 않는다.
  - 모델 예측 평균과 MAE는 적합·선택에 쓰지 않은 구매 고객(Control Test, Treatment 전원)으로만 낸다.
  - `t0`가 실험 시각(`as_of_at`)과 같지 않거나 4주 구간 안에 배정되지 않은 Treatment가 있으면 차이를 내지 않고 이유를
    남긴다. `t0`가 실험 시각보다 앞서 배정 시각이 구간 안에 들어가더라도 4주 구간이 쿠폰 발송 구간과 어긋나므로 비교하지 않는다.

## 4. 학습 모집단 정책

| 정책 | 정의 | 쓰는 곳 |
| --- | --- | --- |
| `control_no_coupon` | `treatment == 0`이고 52주 정답 구간의 관측 주에 받은 쿠폰이 0건인 고객. Treatment와 배정 값이 없는 고객은 `unused` | 새 학습의 기본값 |
| `observed_experiment_mix` | 배정과 무관하게 모든 고객. 정답에 학습 실험의 쿠폰 효과가 섞인다 | 기존 전달 번들(`artifacts/population-24m-n20000-seed42`) |

- 정책은 고객 분할 **전에** 한 번 적용하고(`population.policy_mask`), 52주 모델 4종과 4주 모델이 같은 결과를 쓴다.
- 정책은 학습·선택·평가 대상만 정한다. 최종 추론은 Treatment와 Control을 포함한 전체 고객을 예측한다.
- Control인데 정답 구간에 쿠폰을 받은 고객은 `control_no_coupon`에서 빠지고 `registry.json`의
  `population.control_with_coupon_excluded`에 인원이 남는다. 현재 시뮬레이터 출력에서는 0명이다.
- 정책은 `registry.json`의 `population_policy`와 `model_card.json`의 `policy_condition`에 기록된다. 기존 번들의 정책은
  바꾸지 않는다. 52주 대상 정의 ID는 두 정책에서 같으므로, 소비자는 `policy_condition`으로 정책을 확인한다.

## 5. Uplift·최종 전략과의 연결

- Uplift 담당자는 같은 고객·같은 `t0`의 다음 값을 제공한다.
  - `p0`: 쿠폰을 보내지 않았을 때 4주(28일) 안에 구매할 확률
  - `p1`: 쿠폰을 보냈을 때 그 확률
  - `uplift_4w = p1 - p0`
- 최종 전략은 다음 값을 근사적인 쿠폰 발송 가치로 쓴다.

  ```text
  발송 가치 ≈ uplift_4w × predicted_gross_4w_if_purchase - 예상 쿠폰 비용
  ```

- 이 곱은 쿠폰 때문에 새로 구매한 고객의 4주 총구매액이 Control 구매 고객과 같다고 가정한다. 발송군과 미발송군의
  구매자당 금액은 다를 수 있으므로, 3장의 배정군 진단으로 이 가정을 확인한다.
- `predicted_gross_4w_if_purchase`는 할인 전 금액이다. 할인액은 매출에서 빼지 않았으므로 예상 쿠폰 비용에 넣는다.
  예상 쿠폰 비용의 정의는 최종 전략 담당이 정한다.
- 52주 CLV(할인 후 실결제액, 52주)와 4주 금액(할인 전 총구매액, 4주, 구매 조건부)은 서로 다른 지표다. 더하거나 바로
  비교하지 않는다.

## 6. 데이터 계약과 Adapter

식별·시간 키는 기존 Feature Store 계약을 그대로 쓴다. 키는 `dataset_id`, `customer_id`, `as_of_at`, `reference_week`,
`snapshot_id`이고, `t0 = as_of_at`이며 서울 월요일 00:00이어야 한다.

현재 실제 데이터(시뮬레이터 `main` 84372a7)에서 확인한 원천 열과 내부 이름은 다음과 같다. 원천 이름을 바꿀 때는
`schema.py`의 매핑과 `adapter.py`만 고친다.

| 내부 이름 | 원천 열(`weekly_panel.parquet`) | 용도 |
| --- | --- | --- |
| `gross_revenue` | `gross_revenue_minor` | 4주 할인 전 총구매액 |
| `redemption_discount_cost` | `redemption_discount_cost_minor` | 52주 결제액 = 총매출 - 할인액 |
| `purchases` | `purchases` | 4주 구매 횟수 |
| `coupons` | `coupons_sent` | Control의 쿠폰 미수령 확인 |
| `treatment` | `treatment`(0 = Control). `group`이 있으면 `"control"`과 일치해야 함 | 학습 모집단 정책 |
| `assigned_at`, `as_of_at` | 같은 이름 | 진단(4주 구간 안 배정 여부), 기본 기준시점 |
| 원장 `gross_amount` | `orders.gross_minor` | 4주 정답 원장 대조 |

시뮬레이터에서 아직 확정되지 않았으며 재학습 전에 확인할 항목은 다음과 같다. 새 열은 임의로 가정하지 않았다.

1. 배정 규약: 고객당 한 번 배정, `treatment` 0/1, `group` 표기, `assignment_probability` 0.5 고정. 다중 처치군이나 고객별
   배정 확률이 생기면 Control 정의와 표본 가중을 다시 정한다.
2. Control이 정답 구간에 쿠폰을 받을 수 있는지(현재 0건).
3. 할인 전 총구매액에 반품·취소를 반영하는지와, 반영한다면 차감 시점.
4. 쿠폰 유효기간 28일과 Uplift 구간(4주)이 유지되는지. 바뀌면 `schema.GROSS_WEEKS`를 함께 바꾼다.
5. 실험 시각이 하나(`as_of_at` 단일)인지, 여러 라운드인지.
6. 관측 기간이 `t0` 이후 52주를 모두 포함하는지(현재 24개월).
7. 실험 적격 조건(최소 이력 일수 등). 바뀌면 13주 이력 완전 고객 비율이 달라진다.

금액 단위는 시뮬레이터 원천 단위를 그대로 쓴다(현재 `currency=SOURCE_UNSPECIFIED`, `amount_unit=0.01`). 팀 결정으로
KRW 환산·환율 적용·원화 예산과의 정합성은 CLV 범위에서 제외했고, 필요하면 예산 최적화 모듈의 통합 단계에서 다룬다.
단위 검증(학습·추론 단위 일치, 단위 누락 거절)과 메타데이터 기록은 그대로 유지한다.

## 7. 번들과 재학습 상태

- 기존 전달 번들은 그대로 보존한다(`policy_condition = observed_experiment_mix`, 번들 형식 1, 4주 모델 없음).
  52주 추론은 이전과 같은 값을 내고, 4주 값은 만들지 않는다(결측 + `gross_4w_status = "not_trained"`).
- 새 번들(형식 2)은 정책, 4주 모델의 버전·체크섬·Test 지표를 `model_card.json`의 `policy_condition`과 `gross_4w`에 남긴다.
- 시뮬레이터 `d1992c4`의 쿠폰 미발송 104주 데이터(20,000명, seed 42)로 최종 학습한 후보 번들이
  [`artifacts/population-104w-n20000-seed42-none/`](artifacts/population-104w-n20000-seed42-none/README.md)에 있다.
  데이터 생성 명령, seed 43 평가, 기존 번들과의 같은 고객 비교, 4주 기준선 비교는 그 README에 있다.
- 전달 번들 교체(`HANDOFF_BUNDLE`, `api.md` 수치)는 아직 하지 않았다. 4주 모델이 고객별 차이를 예측하지 못하는 문제와
  평균을 낮게 잡는 문제를 정한 뒤 진행한다.
- 재학습 명령은 다음과 같다. 이미 번들이 있는 디렉터리에는 `--overwrite` 없이 쓰지 않는다. `--require-gross-4w`는
  4주 모델을 학습하지 못하면 아무것도 저장하지 않게 한다.

  ```bash
  # --cutoff를 생략하면 실험 기준시점(as_of_at)이다. 104주 데이터는 2024-12-30이다.
  python -m clv train --weekly <features 디렉터리> --ledger <dataset 디렉터리> \
    --require-gross-4w --output <새 번들 디렉터리>
  ```
