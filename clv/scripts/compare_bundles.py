"""Compare two CLV bundles on the same unseen customers, with customer-bootstrap 95% CIs.

usage: python -m clv.scripts.compare_bundles <weekly> <ledger> <cutoff> <old_bundle> <new_bundle> <out_json>
Also reports bootstrap CIs for the new bundle's 4-week metrics (its saved test buyers and these unseen buyers).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clv import pipeline
from clv.io import read_json, jsonable
from clv.schema import LABEL_COLUMN, GROSS_LABEL_COLUMN, GROSS_COMPLETE_COLUMN

B, SEED = 2000, 20261126
weekly, ledger, cutoff, old_dir, new_dir, out = sys.argv[1:7]
panel, orders, _ = pipeline.prepare_inputs(weekly, ledger)


def boot(actual, preds: dict, pairs=()):
    """MAE/RMSE point + percentile CI per model, and paired differences for (a, b) pairs (b - a)."""
    rng = np.random.default_rng(SEED)
    n = len(actual)
    idx = rng.integers(0, n, size=(B, n))
    res = {}
    err = {k: np.asarray(v, float) - actual for k, v in preds.items()}
    for k, e in err.items():
        mae_b = np.abs(e)[idx].mean(1)
        rmse_b = np.sqrt((e ** 2)[idx].mean(1))
        res[k] = {"mae": float(np.abs(e).mean()), "mae_ci95": np.percentile(mae_b, [2.5, 97.5]).tolist(),
                  "rmse": float(np.sqrt((e ** 2).mean())), "rmse_ci95": np.percentile(rmse_b, [2.5, 97.5]).tolist(),
                  "mean_pred": float(np.mean(preds[k]))}
    for a, b in pairs:
        dm = np.abs(err[b])[idx].mean(1) - np.abs(err[a])[idx].mean(1)
        dr = np.sqrt((err[b] ** 2)[idx].mean(1)) - np.sqrt((err[a] ** 2)[idx].mean(1))
        res[f"{b} - {a}"] = {"mae_diff": res[b]["mae"] - res[a]["mae"], "mae_diff_ci95": np.percentile(dm, [2.5, 97.5]).tolist(),
                             "rmse_diff": res[b]["rmse"] - res[a]["rmse"], "rmse_diff_ci95": np.percentile(dr, [2.5, 97.5]).tolist()}
    return {"n": n, "mean_actual": float(np.mean(actual)), **res}


frames, finals = {}, {}
for tag, bundle in (("old", Path(old_dir)), ("new", Path(new_dir))):
    registry = read_json(bundle / "registry.json")
    config, t0, customers = pipeline._labelled_frame(panel, orders, registry, cutoff, "all", bundle)
    labelled = customers.loc[customers["label_window_complete"].astype(bool) & customers["policy_eligible"].astype(bool)
                             & customers["training_population"].astype(bool)].reset_index(drop=True)
    scored = labelled[["customer_id", LABEL_COLUMN]].copy()
    for name, model in pipeline._models_for(bundle, registry["trained_models"]).items():
        scored[f"{tag}:{name}"] = pipeline._score(model, labelled, orders, t0)
    frames[tag], finals[tag] = scored.set_index("customer_id"), f"{tag}:{registry['final_model']}"
    if tag == "new":
        new_registry, new_customers = registry, customers

old, new = frames["old"], frames["new"]
same = old.index.sort_values().equals(new.index.sort_values())
joined = old.join(new.drop(columns=[LABEL_COLUMN]), how="inner")
assert np.allclose(joined[LABEL_COLUMN], new.loc[joined.index, LABEL_COLUMN])
models = [c for c in joined.columns if ":" in c]
pairs = [(finals["old"], finals["new"])] + [(f"old:{m}", f"new:{m}") for m in
                                            ("baseline", "bgnbd_gg", "xgboost", "xgboost_two_part")
                                            if f"old:{m}" in joined and f"new:{m}" in joined]
report = {"weekly": weekly, "cutoff": cutoff, "same_customers": bool(same), "customers_old": len(old),
          "customers_new": len(new), "final_models": finals, "bootstrap": {"B": B, "seed": SEED, "unit": "customer"},
          "clv_52w": boot(joined[LABEL_COLUMN].to_numpy(float), {m: joined[m] for m in models}, pairs)}

# 4-week model of the new bundle: unseen buyers here, and its own saved test buyers.
if new_registry["gross_4w"]["status"] != "trained":
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    sys.exit("new bundle has no trained 4-week model")
buyers = new_customers.loc[new_customers[GROSS_COMPLETE_COLUMN].astype(bool)].reset_index(drop=True)
pred = pipeline._models_for(Path(new_dir), ["xgboost_gross_4w"])["xgboost_gross_4w"].predict(buyers)
mask = buyers[GROSS_LABEL_COLUMN].notna() & buyers["training_population"].astype(bool)
report["gross_4w_unseen"] = boot(buyers.loc[mask, GROSS_LABEL_COLUMN].to_numpy(float), {"xgboost_gross_4w": pred[mask.to_numpy()]})
saved = pd.read_csv(Path(new_dir) / "reports" / "gross_4w_labelled_predictions.csv", dtype={"customer_id": str})
test = saved.loc[saved["split"].eq("test") & saved[GROSS_LABEL_COLUMN].notna()]
report["gross_4w_saved_test"] = boot(test[GROSS_LABEL_COLUMN].to_numpy(float),
                                     {"xgboost_gross_4w": test["predicted_gross_4w_if_purchase"].to_numpy(float)})
Path(out).write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
