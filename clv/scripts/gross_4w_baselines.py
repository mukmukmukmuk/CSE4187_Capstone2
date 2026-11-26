"""Analysis only: 4-week conditional gross — xgboost_gross_4w vs simple baselines, customer bootstrap 95% CIs.

usage: python -m clv.scripts.gross_4w_baselines <bundle> <sim_root> <out_json>
Baselines use only seed-42 train buyers (constants, GG scale) and each customer's pre-t0 orders.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clv import pipeline
from clv.dataset import build_dataset
from clv.io import jsonable, read_json
from clv.probabilistic import rfm_summary
from clv.schema import GROSS_LABEL_COLUMN, GROSS_COMPLETE_COLUMN

B, SEED = 2000, 20261126
bundle, sim, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
registry = read_json(bundle / "registry.json")
t0 = pd.Timestamp(registry["reference_week"])
gross_model = pipeline._models_for(bundle, ["xgboost_gross_4w"])["xgboost_gross_4w"]
prob = pipeline._models_for(bundle, ["bgnbd_gg"])["bgnbd_gg"]
saved = pd.read_csv(bundle / "reports" / "gross_4w_labelled_predictions.csv", dtype={"customer_id": str})


def history(orders, ids, weeks):
    rows = orders.loc[orders["order_at"].lt(t0) & orders["order_at"].ge(t0 - pd.Timedelta(weeks=weeks))]
    return rows.groupby("customer_id")["gross_amount"].mean().reindex(ids).to_numpy(float)


def gg_value(orders, ids):
    s = rfm_summary(orders, t0, prob.calibration, prob.period).reindex(ids)
    x = s["frequency"].fillna(0).to_numpy()
    return prob.gamma_gamma.expected_value(x, s["monetary_value"])


def buyers_of(seed):
    d = sim / f"population-104w-n20000-seed{seed}-none"
    panel, orders, _ = pipeline.prepare_inputs(d / "features", d / "dataset")
    frame = build_dataset(panel, t0, orders)
    frame = frame.loc[frame[GROSS_COMPLETE_COLUMN].astype(bool) & frame[GROSS_LABEL_COLUMN].notna()].reset_index(drop=True)
    return frame, orders


def metrics(actual, preds, ref="xgboost_gross_4w"):
    rng = np.random.default_rng(SEED)
    n = len(actual)
    idx = rng.integers(0, n, size=(B, n))
    res = {"n": n, "mean_actual": float(actual.mean()), "median_actual": float(np.median(actual))}
    err = {k: np.asarray(v, float) - actual for k, v in preds.items()}
    for k, e in err.items():
        res[k] = {"mae": float(np.abs(e).mean()), "mae_ci95": np.percentile(np.abs(e)[idx].mean(1), [2.5, 97.5]).tolist(),
                  "rmse": float(np.sqrt((e ** 2).mean())),
                  "rmse_ci95": np.percentile(np.sqrt((e ** 2)[idx].mean(1)), [2.5, 97.5]).tolist(),
                  "spearman": float(pd.Series(actual).corr(pd.Series(np.asarray(preds[k], float)), method="spearman"))
                  if np.std(preds[k]) > 0 else None,
                  "mean_pred": float(np.mean(preds[k]))}
        if k != ref:
            dm = np.abs(e)[idx].mean(1) - np.abs(err[ref])[idx].mean(1)
            dr = np.sqrt((e ** 2)[idx].mean(1)) - np.sqrt((err[ref] ** 2)[idx].mean(1))
            res[k][f"mae_minus_{ref}_ci95"] = np.percentile(dm, [2.5, 97.5]).tolist()
            res[k][f"rmse_minus_{ref}_ci95"] = np.percentile(dr, [2.5, 97.5]).tolist()
    return res


# Fit the baselines on seed-42 train buyers only.
f42, o42 = buyers_of(42)
split = saved.set_index("customer_id")["split"]
f42["split"] = f42["customer_id"].map(split)
train = f42.loc[f42["split"].eq("train")]
y_train = train[GROSS_LABEL_COLUMN].to_numpy(float)
c_mean, c_median = float(y_train.mean()), float(np.median(y_train))
gg_train = gg_value(o42, train["customer_id"])
gg_scale = float(y_train.sum() / gg_train.sum())


def predictions(frame, orders):
    ids = frame["customer_id"]
    h13, h52 = history(orders, ids, 13), history(orders, ids, 52)
    return {"xgboost_gross_4w": gross_model.predict(frame),
            "const_train_mean": np.full(len(frame), c_mean),
            "const_train_median": np.full(len(frame), c_median),
            "hist13w_mean_order_gross_else_mean": np.where(np.isfinite(h13), h13, c_mean),
            "hist52w_mean_order_gross_else_mean": np.where(np.isfinite(h52), h52, c_mean),
            "gg_value_x_train_scale": gg_value(orders, ids) * gg_scale}


test42 = f42.loc[f42["split"].eq("test")].reset_index(drop=True)
f43, o43 = buyers_of(43)
report = {"t0": t0.isoformat(), "fit": {"train_buyers": len(train), "const_mean": c_mean, "const_median": c_median,
                                        "gg_scale": gg_scale},
          "seed42_test_buyers": metrics(test42[GROSS_LABEL_COLUMN].to_numpy(float), predictions(test42, o42)),
          "seed43_unseen_buyers": metrics(f43[GROSS_LABEL_COLUMN].to_numpy(float), predictions(f43, o43))}
out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
for part in ("seed42_test_buyers", "seed43_unseen_buyers"):
    r = report[part]
    print("==", part, "n", r["n"], "mean", round(r["mean_actual"]), "median", round(r["median_actual"]))
    for k, v in r.items():
        if isinstance(v, dict):
            print(f"  {k:38s} MAE {v['mae']:8.0f} {[round(x) for x in v['mae_ci95']]}  RMSE {v['rmse']:8.0f} "
                  f"{[round(x) for x in v['rmse_ci95']]}  rho {v['spearman'] if v['spearman'] is None else round(v['spearman'], 3)}"
                  f"  dMAE {[round(x) for x in v.get('mae_minus_xgboost_gross_4w_ci95', [])]}"
                  f"  dRMSE {[round(x) for x in v.get('rmse_minus_xgboost_gross_4w_ci95', [])]}")
