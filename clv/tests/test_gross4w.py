"""4주 후보 선택 규칙(1-표준오차)과 단순 후보의 저장·불러오기."""
import numpy as np
import pandas as pd
import pytest

from clv.gross4w import (
    GROSS_CANDIDATES, MEAN_GROSS, XGB_GROSS, MeanGross4w, bootstrap_metrics, select_one_standard_error,
)


def test_one_standard_error_rule_prefers_the_simpler_candidate_within_noise():
    rng = np.random.default_rng(0)
    base = rng.gamma(1.0, 100.0, size=2_000)
    # 복잡한 후보가 평균은 조금 낫지만 고객별 차이가 잡음 수준이면 단순한 후보를 고른다.
    noisy = {MEAN_GROSS: base, XGB_GROSS: base - 0.5 + rng.normal(0, 50, size=base.size)}
    chosen, rows = select_one_standard_error(noisy)
    table = {row["model"]: row for row in rows}
    assert table[XGB_GROSS]["best"] and table[MEAN_GROSS]["within_one_se"] and chosen == MEAN_GROSS
    # 잡음보다 확실히 나으면 복잡한 후보를 고른다.
    better = {MEAN_GROSS: base, XGB_GROSS: base - 20 + rng.normal(0, 5, size=base.size)}
    chosen, rows = select_one_standard_error(better)
    assert chosen == XGB_GROSS and not {row["model"]: row for row in rows}[MEAN_GROSS]["within_one_se"]
    assert [row["model"] for row in rows] == [name for name in GROSS_CANDIDATES if name in better]
    with pytest.raises(ValueError):
        select_one_standard_error({})


def test_mean_candidate_round_trip_and_bootstrap(tmp_path):
    model = MeanGross4w().fit([100.0, 300.0])
    model.save(tmp_path)
    loaded = MeanGross4w.load(tmp_path)
    assert loaded.predict(pd.DataFrame({"customer_id": ["a", "b", "c"]})).tolist() == [200.0] * 3
    with pytest.raises(ValueError):
        MeanGross4w().fit([1.0, -1.0])
    interval = bootstrap_metrics([100.0, 200.0, 400.0], [200.0, 200.0, 200.0], 500, 0)
    assert interval["draws"] == 500 and interval["mae_ci95"][0] <= interval["mae_ci95"][1]
    assert bootstrap_metrics([1.0, 2.0], [1.0, 2.0], 0, 0) == {}
