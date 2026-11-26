"""4주 조건부 총구매액 후보 모델과 기본 모델 선택 규칙.

대상은 E[``[t0, t0 + 4주)`` 할인 전 총구매액 | 4주 안에 1회 이상 구매, X(t0)]이다. 후보 넷을 모두 학습해 번들에
저장하고, 그중 하나를 기본 모델(registry ``gross_4w.model``)로 정한다. 추론에서는 이름으로 다른 후보를 고를 수 있다.

후보(단순한 순서, 선택 규칙의 우선순위)
    mean_gross_4w           Train 구매 고객 평균(상수). 비교 기준이다.
    gamma_gamma_gross_4w    Gamma-Gamma 기대 거래 금액 × Train 배율. 52주 BG/NBD + GG와 같은 Gamma-Gamma 모형을
                            할인 전 금액으로 적합한다(주문 원장이 필요하다).
    xgboost_gross_4w        XGBoost, 제곱오차 손실.
    xgboost_gamma_gross_4w  XGBoost, gamma 손실(금액이 0인 구매 고객이 있으면 학습하지 않는다).

기본 모델 선택(``select_one_standard_error``)
    1. 지표: 제곱오차. 결합식에는 기대값(평균)이 필요하고, 평균을 맞히는 예측을 가장 좋게 평가하는 손실이 제곱오차다
       (MAE는 중앙값을 맞히는 예측을 좋게 평가한다).
    2. 추정: Test를 뺀 Train + Validation 구매 고객으로 고객 단위 K겹 교차검증을 R번 반복한다. 각 겹에서 후보를 처음부터
       다시 학습하고(XGBoost는 겹 안에서 다시 나눈 검증 고객으로 설정과 트리 수를 고른다), 학습에 쓰지 않은 겹을 예측한다.
       고객마다 반복 평균 제곱오차를 낸다.
    3. 결정: 1-표준오차 규칙. 평균 제곱오차가 가장 낮은 후보와의 차이가 그 차이의 표준오차(같은 고객끼리 짝지은 오차
       차이의 표준오차) 이내인 후보 중 가장 단순한 것을 고른다. 더 복잡한 후보는 잡음보다 크게 나을 때만 고른다.
    Test 구매 고객은 선택에 쓰지 않고 후보별 지표(고객 단위 Bootstrap 95% 구간 포함)를 보고하는 데만 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import to_cutoff
from .io import read_json, write_json
from .probabilistic import MIN_GG_CUSTOMERS, GammaGamma, rfm_summary
from .xgboost_model import GROSS_GAMMA_NAME, GROSS_NAME, GrossXGBoost4w

MEAN_GROSS = "mean_gross_4w"
GG_GROSS = "gamma_gamma_gross_4w"
XGB_GROSS = GROSS_NAME
XGB_GAMMA_GROSS = GROSS_GAMMA_NAME
# 단순한 순서. 1-표준오차 규칙은 이 순서로 앞선 후보를 고른다.
GROSS_CANDIDATES = (MEAN_GROSS, GG_GROSS, XGB_GROSS, XGB_GAMMA_GROSS)
XGB_OBJECTIVES = {XGB_GROSS: "reg:squarederror", XGB_GAMMA_GROSS: "reg:gamma"}
SELECTION_RULE = ("one-standard-error rule on repeated K-fold out-of-fold squared error over train + validation "
                  "buyers: the simplest candidate whose mean squared error is within one paired standard error of "
                  "the best candidate")


class MeanGross4w:
    """Train 구매 고객의 평균 4주 총구매액을 모든 고객에게 낸다."""

    name = MEAN_GROSS
    needs_ledger = False

    def __init__(self):
        self.value: float | None = None
        self.train_counts: dict = {}

    def fit(self, y) -> "MeanGross4w":
        y = np.asarray(y, dtype=float)
        if not len(y) or not (np.isfinite(y).all() and (y >= 0).all()):
            raise ValueError(f"{self.name} needs finite nonnegative gross amounts of buyers")
        self.value, self.train_counts = float(y.mean()), {"train_buyers": int(len(y))}
        return self

    def predict(self, frame: pd.DataFrame, orders=None, t0=None) -> np.ndarray:
        if self.value is None:
            raise RuntimeError(f"{self.name} model is not fitted")
        return np.full(len(frame), self.value, dtype=float)

    def summary(self) -> dict:
        return {"value": self.value, **self.train_counts}

    def save(self, directory: str | Path) -> Path:
        return write_json(Path(directory) / f"{self.name}.json", {"model": self.name, **self.summary()})

    @classmethod
    def load(cls, directory: str | Path) -> "MeanGross4w":
        meta = read_json(Path(directory) / f"{MEAN_GROSS}.json")
        if meta.get("model") != MEAN_GROSS:
            raise ValueError(f"not a {MEAN_GROSS} file")
        model = cls()
        model.value, model.train_counts = float(meta["value"]), {"train_buyers": meta.get("train_buyers")}
        return model


def _gross_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """``rfm_summary``가 할인 전 금액(``gross_amount``)으로 거래 금액을 집계하도록 바꾼 주문 원장."""
    return orders.assign(paid_amount=orders["gross_amount"])


class GammaGammaGross4w:
    """Gamma-Gamma 기대 거래 금액(할인 전, t0 직전 보정 기간)에 Train 배율을 곱한다.

    Gamma-Gamma 모수는 Train 분할 고객 전체의 보정 기간 할인 전 거래로 적합한다(52주 BG/NBD + GG와 같은 보정 기간과
    거래 단위). 보정 기간에 반복 거래가 없는 고객은 모집단 평균 거래 금액을 쓴다. 배율은 Train 구매 고객의
    (4주 총구매액 합 / 기대 거래 금액 합)으로, 4주 안의 구매 횟수와 거래 단위 차이를 한 숫자로 맞춘다.
    """

    name = GG_GROSS
    needs_ledger = True

    def __init__(self, calibration: str = "feature_window", period: str = "D", penalizer: float = 0.0):
        self.calibration, self.period, self.penalizer = calibration, period, float(penalizer)
        self.gamma_gamma = GammaGamma()
        self.scale: float | None = None
        self.diagnostics: dict = {}

    def fit_value_model(self, orders: pd.DataFrame, t0, customers) -> "GammaGammaGross4w":
        """``customers``의 보정 기간 할인 전 반복 거래로 Gamma-Gamma 모수를 적합한다(정답을 쓰지 않는다)."""
        summary = rfm_summary(_gross_orders(orders), t0, self.calibration, self.period).reindex(pd.Index(customers))
        repeat = summary.loc[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
        if len(repeat) < MIN_GG_CUSTOMERS:
            raise ValueError(f"{self.name} needs at least {MIN_GG_CUSTOMERS} repeat customers with positive spend")
        info = self.gamma_gamma.fit(repeat["frequency"], repeat["monetary_value"], self.penalizer)
        population_mean = self.gamma_gamma.population_mean()
        if not (info.get("converged") and np.isfinite(population_mean)):
            raise ValueError(f"{self.name}: Gamma-Gamma did not converge to a finite mean (q <= 1 or failed fit)")
        self.diagnostics = {"params": self.gamma_gamma.params, "converged": bool(info.get("converged")),
                            "repeat_customers": int(len(repeat)), "population_mean_value": population_mean}
        return self

    def expected_value(self, orders: pd.DataFrame, t0, customer_ids) -> np.ndarray:
        summary = rfm_summary(_gross_orders(orders), t0, self.calibration, self.period).reindex(pd.Index(customer_ids))
        values = self.gamma_gamma.expected_value(summary["frequency"].fillna(0).to_numpy(), summary["monetary_value"])
        return np.asarray(values, dtype=float)

    def fit_scale(self, orders: pd.DataFrame, t0, customer_ids, y) -> "GammaGammaGross4w":
        y = np.asarray(y, dtype=float)
        values = self.expected_value(orders, t0, customer_ids)
        if not len(y) or not np.isfinite(values).all() or values.sum() <= 0:
            raise ValueError(f"{self.name}: expected transaction values are not positive and finite")
        self.scale = float(y.sum() / values.sum())
        self.diagnostics["scale"], self.diagnostics["train_buyers"] = self.scale, int(len(y))
        return self

    def predict(self, frame: pd.DataFrame, orders=None, t0=None) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError(f"{self.name} model is not fitted")
        if orders is None:
            raise ValueError(f"{self.name} needs the order ledger (raw.sqlite)")
        return np.clip(self.expected_value(orders, to_cutoff(t0), frame["customer_id"]) * self.scale, 0.0, None)

    def summary(self) -> dict:
        return {"calibration": self.calibration, "transaction_period": self.period, "penalizer": self.penalizer,
                **self.diagnostics}

    def save(self, directory: str | Path) -> Path:
        return write_json(Path(directory) / f"{self.name}.json", {"model": self.name, **self.summary()})

    @classmethod
    def load(cls, directory: str | Path) -> "GammaGammaGross4w":
        meta = read_json(Path(directory) / f"{GG_GROSS}.json")
        if meta.get("model") != GG_GROSS:
            raise ValueError(f"not a {GG_GROSS} file")
        model = cls(meta["calibration"], meta["transaction_period"], meta["penalizer"])
        model.gamma_gamma.params, model.scale = meta["params"], float(meta["scale"])
        model.diagnostics = {key: value for key, value in meta.items() if key not in (
            "model", "calibration", "transaction_period", "penalizer")}
        return model


def load_gross_model(models_dir: str | Path, name: str):
    """번들 ``models/`` 아래의 4주 후보 하나를 불러온다."""
    models_dir = Path(models_dir)
    if name == MEAN_GROSS:
        return MeanGross4w.load(models_dir)
    if name == GG_GROSS:
        return GammaGammaGross4w.load(models_dir)
    if name in XGB_OBJECTIVES:
        return GrossXGBoost4w.load(models_dir / name)
    raise ValueError(f"unknown 4-week model {name!r}; choose one of {GROSS_CANDIDATES}")


def save_gross_model(model, models_dir: str | Path) -> None:
    models_dir = Path(models_dir)
    model.save(models_dir / model.name if model.name in XGB_OBJECTIVES else models_dir)


def score_gross(model, frame: pd.DataFrame, orders=None, t0=None) -> np.ndarray:
    """``frame`` 고객 순서대로 4주 조건부 총구매액을 낸다(0 이상, 유한값이 아니면 오류)."""
    if model.name in XGB_OBJECTIVES:
        values = model.predict(frame)
    else:
        values = model.predict(frame, orders, t0)
    values = np.asarray(values, dtype=float)
    if not (np.isfinite(values).all() and (values >= 0).all()):
        raise ValueError(f"{model.name} predictions must be finite and nonnegative")
    return values


def fit_candidate(name: str, config, train: pd.DataFrame, validation: pd.DataFrame | None, label: str,
                  orders=None, t0=None, value_model: GammaGammaGross4w | None = None):
    """후보 하나를 ``train`` 구매 고객으로 학습한다. XGBoost는 ``validation``으로 설정과 트리 수를 고른다.

    ``gamma_gamma_gross_4w``는 정답 없이 적합한 ``value_model``(Gamma-Gamma 모수)을 받아 배율만 ``train``으로 맞춘다.
    """
    y = train[label].to_numpy(dtype=float)
    if name == MEAN_GROSS:
        return MeanGross4w().fit(y)
    if name == GG_GROSS:
        if value_model is None:
            raise ValueError(f"{GG_GROSS} needs a fitted Gamma-Gamma value model")
        model = GammaGammaGross4w(value_model.calibration, value_model.period, value_model.penalizer)
        model.gamma_gamma.params, model.diagnostics = dict(value_model.gamma_gamma.params), dict(value_model.diagnostics)
        return model.fit_scale(orders, t0, train["customer_id"], y)
    if name in XGB_OBJECTIVES:
        grid = {**{key: list(value) for key, value in config.gross_4w_grid.items()}, "objective": [XGB_OBJECTIVES[name]]}
        return GrossXGBoost4w(config.seed, grid, config.xgb_fixed, name=name).fit(
            train, y, validation, validation[label].to_numpy(dtype=float))
    raise ValueError(f"unknown 4-week model {name!r}")


def _folds(n: int, folds: int, seed: int) -> np.ndarray:
    """고객 ``n``명을 ``folds``겹으로 섞어 나눈 겹 번호."""
    order = np.random.default_rng(seed).permutation(n)
    assignment = np.empty(n, dtype=int)
    assignment[order] = np.arange(n) % folds
    return assignment


def cross_validated_losses(names, config, buyers: pd.DataFrame, label: str, orders=None, t0=None,
                           value_model: GammaGammaGross4w | None = None) -> tuple[dict, dict, dict]:
    """반복 K겹 교차검증의 고객별 평균 제곱오차를 후보마다 낸다.

    각 겹의 학습 고객에서 다시 Validation 비율(``split_fractions``의 검증/(학습+검증))만큼을 떼어 XGBoost의 설정과
    트리 수를 고른다(중첩 교차검증). 한 겹에서라도 학습에 실패한 후보는 선택에서 빼고 사유를 남긴다.

    Returns:
        (후보별 고객 손실 배열, 후보별 교차검증 예측 평균, 실패 사유).
    """
    folds, repeats = config.gross_4w_cv_folds, config.gross_4w_cv_repeats
    y = buyers[label].to_numpy(dtype=float)
    inner_share = config.split_fractions[1] / (config.split_fractions[0] + config.split_fractions[1])
    squared = {name: np.zeros((repeats, len(buyers))) for name in names}
    predicted = {name: np.zeros((repeats, len(buyers))) for name in names}
    failures = {}
    for repeat in range(repeats):
        assignment = _folds(len(buyers), folds, config.seed + repeat)
        for fold in range(folds):
            held = assignment == fold
            fit_part = buyers.loc[~held]
            inner = np.random.default_rng(config.seed + 1000 * (repeat + 1) + fold).random(len(fit_part)) < inner_share
            for name in names:
                if name in failures:
                    continue
                try:
                    if name in XGB_OBJECTIVES:
                        model = fit_candidate(name, config, fit_part.loc[~inner], fit_part.loc[inner], label)
                    else:
                        model = fit_candidate(name, config, fit_part, None, label, orders, t0, value_model)
                    values = score_gross(model, buyers.loc[held], orders, t0)
                except (ValueError, RuntimeError) as error:
                    failures[name] = f"cross-validation fit failed (repeat {repeat}, fold {fold}): {error}"
                    continue
                squared[name][repeat, held] = (values - y[held]) ** 2
                predicted[name][repeat, held] = values
    losses = {name: squared[name].mean(axis=0) for name in names if name not in failures}
    means = {name: float(predicted[name].mean()) for name in losses}
    return losses, means, failures


def select_one_standard_error(losses: dict, order=GROSS_CANDIDATES) -> tuple[str, list[dict]]:
    """1-표준오차 규칙으로 기본 후보를 고른다.

    ``losses``는 후보별 같은 고객 순서의 손실 배열이다. 평균 손실이 가장 낮은 후보(best)와 각 후보의 고객별 손실 차이
    평균이 그 차이의 표준오차 이하이면 규칙 안에 든다. 규칙 안에 든 후보 중 ``order``에서 가장 앞선(단순한) 것을 고른다.

    Returns:
        (고른 후보, 후보별 표: cv_mse, cv_rmse, mse_minus_best, se_of_difference, within_one_se, selected).
    """
    names = [name for name in order if name in losses]
    if not names:
        raise ValueError("no 4-week candidate has cross-validated losses")
    mse = {name: float(np.mean(losses[name])) for name in names}
    best = min(names, key=lambda name: (mse[name], names.index(name)))
    rows = []
    for name in names:
        difference = np.asarray(losses[name], dtype=float) - np.asarray(losses[best], dtype=float)
        se = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0.0
        rows.append({"model": name, "cv_mse": mse[name], "cv_rmse": float(np.sqrt(mse[name])),
                     "mse_minus_best": float(difference.mean()), "se_of_difference": se,
                     "within_one_se": bool(difference.mean() <= se), "best": name == best})
    chosen = next(row["model"] for row in rows if row["within_one_se"])
    for row in rows:
        row["selected"] = row["model"] == chosen
    return chosen, rows


def bootstrap_metrics(actual, predicted, draws: int, seed: int) -> dict:
    """MAE, RMSE, 예측 평균의 고객 단위 Bootstrap 95% 구간(백분위수)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if draws <= 0 or len(actual) < 2:
        return {}
    index = np.random.default_rng(seed).integers(0, len(actual), size=(draws, len(actual)))
    error = predicted - actual
    interval = lambda values: [float(value) for value in np.percentile(values, [2.5, 97.5])]
    return {"draws": int(draws), "mae_ci95": interval(np.abs(error)[index].mean(axis=1)),
            "rmse_ci95": interval(np.sqrt((error ** 2)[index].mean(axis=1))),
            "mean_actual_ci95": interval(actual[index].mean(axis=1)),
            "mean_predicted_ci95": interval(predicted[index].mean(axis=1))}
