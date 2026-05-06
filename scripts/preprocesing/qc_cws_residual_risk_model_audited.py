
"""Residual-risk modeling utilities for the CWS/OWS quality-control pipeline."""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


KEY_COLS = ["date", "station_id", "network"]


@dataclass
class ResidualRiskConfig:
    """Configuration for split-safe residual-risk modeling."""

    city: str = "project_id"



    input_path: str | None = None
    output_dir: str = "outputs/residual_risk"
    run_label: str | None = None
    overwrite_existing_run: bool = False



    reference_method: str = "auto"
    calibration_mode: str = "time_train"
    source_reference_run_label: str | None = None
    source_reference_manifest_path: str | None = None


    split_strategy: str = "time"
    valid_start: str = "2021-10-01"
    test_start: str = "2021-11-01"
    valid_frac: float = 0.15
    test_frac: float = 0.15
    random_state: int = 42


    target_mode: str = "auto"
    target_col: str = "target_ref_risk"
    target_band_col: str = "target_ref_risk_band_z"


    station_min_train_rows: int = 24
    high_solar_quantile: float = 0.75
    low_wind_quantile: float = 0.25


    feature_modes: str = "context_history,context_static_met"
    include_qc_flags: bool = False
    include_coordinates: bool = False
    drop_numeric_landcover_codes: bool = True


    run_catboost: bool = True
    iterations: int = 1200
    learning_rate: float = 0.04
    depth: int = 8
    l2_leaf_reg: float = 8.0
    auto_class_weights: str | None = "Balanced"
    early_stopping_rounds: int = 100
    thread_count: int | None = None
    used_ram_limit: str | None = "42gb"
    use_gpu: bool = False
    verbose: int = 100
    max_train_rows: int | None = None
    max_valid_rows: int | None = None



    fixed_low_risk_prob: float = 0.05
    fixed_high_risk_prob: float = 0.50
    quantile_low: float = 0.50
    quantile_high: float = 0.95


    retention_fractions: tuple[float, ...] = tuple(np.round(np.linspace(0.50, 0.99, 20), 3))


def ensure_utc_datetime(obj) -> pd.Series:
    return pd.to_datetime(obj, utc=True, errors="coerce")


def clean_station_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().astype(str)


def normalize_network_name(value: object) -> str:
    s = str(value).strip()
    low = s.lower()
    if "wunderground" in low:
        return "Wunderground"
    if "netatmo" in low:
        return "Netatmo"
    if low in {"ows", "official", "official_weather_station"} or "ows" in low:
        return "OWS"
    return s


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="float")
        elif pd.api.types.is_integer_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="integer")
        elif x[col].dtype == "object":
            nunique = x[col].nunique(dropna=True)
            if 0 < nunique <= min(500, max(20, len(x) // 50)):
                try:
                    x[col] = x[col].astype("category")
                except Exception:
                    pass
    return x


def save_dataframe_auto(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as e:
            warnings.warn(f"Parquet save failed ({e}); falling back to pickle.")
            path = path.with_suffix(".pkl")
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return path
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    path = path.with_suffix(".pkl")
    df.to_pickle(path)
    return path


def load_dataframe_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def _first_existing(cols: Iterable[str], candidates: Sequence[str]) -> str | None:
    colset = set(map(str, cols))
    for c in candidates:
        if c in colset:
            return c
    return None


def _safe_numeric(s: pd.Series | np.ndarray | list) -> pd.Series:
    return pd.to_numeric(pd.Series(s), errors="coerce")


def robust_mad_sigma(values, floor: float = 1e-6) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(arr)
    if not np.isfinite(sig) or sig <= 0:
        sig = floor
    return float(max(sig, floor))


def _safe_slope(x, y, min_n: int = 20) -> float:
    xx = pd.to_numeric(pd.Series(x), errors="coerce").astype("float64")
    yy = pd.to_numeric(pd.Series(y), errors="coerce").astype("float64")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < min_n:
        return np.nan
    xv = xx[mask].to_numpy()
    yv = yy[mask].to_numpy()
    var = np.var(xv)
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(np.cov(xv, yv, ddof=0)[0, 1] / var)


def _as_string_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("Missing").astype(str)


def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize core columns and add basic temporal fields."""
    x = df.copy()
    missing = [c for c in ["date", "station_id", "network"] if c not in x.columns]
    if missing:
        raise KeyError(f"Input reference table is missing required columns: {missing}")

    x["date"] = ensure_utc_datetime(x["date"])
    x["station_id"] = clean_station_id(x["station_id"])
    x["network"] = x["network"].map(normalize_network_name)

    x["hour"] = x["date"].dt.hour.astype("Int8")
    x["month"] = x["date"].dt.month.astype("Int8")
    x["dayofyear"] = x["date"].dt.dayofyear.astype("Int16")
    x["dow"] = x["date"].dt.dayofweek.astype("Int8")
    x["is_weekend"] = (x["dow"] >= 5).astype("Int8")
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"].astype(float) / 24).astype("float32")
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"].astype(float) / 24).astype("float32")
    x["doy_sin"] = np.sin(2 * np.pi * x["dayofyear"].astype(float) / 365.25).astype("float32")
    x["doy_cos"] = np.cos(2 * np.pi * x["dayofyear"].astype(float) / 365.25).astype("float32")

    if {"u10_ms", "v10_ms"}.issubset(x.columns) and "wind_speed" not in x.columns:
        x["wind_speed"] = np.sqrt(pd.to_numeric(x["u10_ms"], errors="coerce") ** 2 + pd.to_numeric(x["v10_ms"], errors="coerce") ** 2)

    if "ssrd_wm2" in x.columns and "is_daylight" not in x.columns:
        x["is_daylight"] = (pd.to_numeric(x["ssrd_wm2"], errors="coerce") > 5).astype("Int8")
    elif "is_daylight" not in x.columns:
        x["is_daylight"] = x["hour"].between(8, 18).astype("Int8")

    if "tp_mm" in x.columns and "is_rain" not in x.columns:
        x["is_rain"] = (pd.to_numeric(x["tp_mm"], errors="coerce") > 0).astype("Int8")

    if "cws_ref_z" in x.columns and "abs_cws_ref_z" not in x.columns:
        x["abs_cws_ref_z"] = pd.to_numeric(x["cws_ref_z"], errors="coerce").abs()

    return downcast_dataframe(x)


def make_time_split(
    df: pd.DataFrame,
    valid_start: str,
    test_start: str,
    split_col: str = "split",
) -> pd.DataFrame:
    x = df.copy()
    vs = pd.to_datetime(valid_start, utc=True)
    ts = pd.to_datetime(test_start, utc=True)
    x[split_col] = "train"
    x.loc[x["date"] >= vs, split_col] = "valid"
    x.loc[x["date"] >= ts, split_col] = "test"
    return x


def make_station_holdout_split(
    df: pd.DataFrame,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    if valid_frac < 0 or test_frac < 0 or valid_frac + test_frac >= 1:
        raise ValueError("valid_frac and test_frac must be nonnegative and sum to < 1.")

    x = df.copy()
    groups = x[["network", "station_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(groups))
    n_test = int(round(len(groups) * test_frac))
    n_valid = int(round(len(groups) * valid_frac))

    split = np.array(["train"] * len(groups), dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_valid]] = "valid"
    groups[split_col] = split

    return x.merge(groups, on=["network", "station_id"], how="left", validate="many_to_one")


def assign_splits(df: pd.DataFrame, cfg: ResidualRiskConfig, split_col: str = "split") -> pd.DataFrame:
    if cfg.split_strategy == "time":
        return make_time_split(df, cfg.valid_start, cfg.test_start, split_col=split_col)
    if cfg.split_strategy == "station_holdout":
        return make_station_holdout_split(
            df,
            valid_frac=cfg.valid_frac,
            test_frac=cfg.test_frac,
            random_state=cfg.random_state,
            split_col=split_col,
        )
    raise ValueError("split_strategy must be 'time' or 'station_holdout'")


def make_high_confidence_target(
    df: pd.DataFrame,
    band_col: str = "target_ref_risk_band_z",
    out_col: str = "target_ref_risk_hiconf",
) -> pd.DataFrame:
    """Create a binary target that leaves the ambiguous band unlabeled."""
    x = df.copy()
    if band_col not in x.columns:
        x[out_col] = np.nan
        return x

    band = x[band_col].astype("string")
    x[out_col] = np.where(
        band == "high_confidence_high_risk",
        1,
        np.where(band == "high_confidence_low_risk", 0, np.nan),
    )
    x[out_col] = pd.to_numeric(x[out_col], errors="coerce")
    return x


def resolve_training_target(df: pd.DataFrame, cfg: ResidualRiskConfig) -> tuple[pd.DataFrame, str]:
    """Create/choose the classifier target column."""
    x = df.copy()
    mode = cfg.target_mode

    if mode == "auto":
        if cfg.target_band_col in x.columns:
            mode = "high_confidence_bands"
        else:
            mode = "hard"

    if mode == "high_confidence_bands":
        x = make_high_confidence_target(x, band_col=cfg.target_band_col)
        target_col = "target_ref_risk_hiconf"
    elif mode == "hard":
        if cfg.target_col not in x.columns:
            raise KeyError(f"Hard target column {cfg.target_col!r} not found.")
        target_col = cfg.target_col
        x[target_col] = pd.to_numeric(x[target_col], errors="coerce")
    else:
        raise ValueError("target_mode must be 'auto', 'hard', or 'high_confidence_bands'")

    x["is_model_labeled"] = x[target_col].isin([0, 1]).astype("int8")
    return downcast_dataframe(x), target_col


def _match_fraction_feature(df: pd.DataFrame, family: str) -> str | None:
    """Find local OWS match-fraction columns for LC/LCZ."""
    if family.upper() == "LCZ":
        candidates = [
            "ref_frac_same_LCZ_point_lg",
            "ref_frac_same_LCZ_buffer_lg",
            "ref_frac_same_LCZ_point",
            "ref_frac_same_LCZ_buffer",
            "ref_frac_same_same_LCZ_point_lg",
            "ref_frac_same_same_LCZ_buffer_lg",
            "ref_frac_same_same_LCZ_point",
            "ref_frac_same_same_LCZ_buffer",
        ]
    else:
        candidates = [
            "ref_frac_same_LC_point_lg",
            "ref_frac_same_LC_buffer_lg",
            "ref_frac_same_LC_point",
            "ref_frac_same_LC_buffer",
            "ref_frac_same_same_LC_point_lg",
            "ref_frac_same_same_LC_buffer_lg",
            "ref_frac_same_same_LC_point",
            "ref_frac_same_same_LC_buffer",
        ]
    return _first_existing(df.columns, candidates)


def add_environmental_support_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add local reference-difficulty and urban-context support features.

    These features are designed to be known without using the CWS residual label:
    OWS density/distance/spread, LC/LCZ mismatch, elevation/building deltas,
    satellite missingness/age, and simple radiation/NDVI/LST interactions.
    """
    x = df.copy()


    if "ref_local_n" in x.columns:
        n = pd.to_numeric(x["ref_local_n"], errors="coerce")
        x["ref_support_low_density"] = (n < 3).astype("Int8")
        x["ref_support_density_component"] = ((6 - n.clip(lower=0, upper=6)) / 6.0).astype("float32")

    dist_col = _first_existing(x.columns, ["ref_min_dist_km", "ref_nearest_dist_km", "ref_nearest_value_dist_km"])
    if dist_col is not None:
        dist = pd.to_numeric(x[dist_col], errors="coerce")
        x["ref_support_nearest_dist_km"] = dist
        x["ref_support_far_nearest"] = (dist > 2.0).astype("Int8")
        x["ref_support_very_far_nearest"] = (dist > 5.0).astype("Int8")
        x["ref_support_distance_component"] = (dist / 5.0).clip(lower=0, upper=1).astype("float32")

    if "ref_local_spread_c" in x.columns:
        spread = pd.to_numeric(x["ref_local_spread_c"], errors="coerce")
        x["ref_support_local_spread_c"] = spread
        x["ref_support_high_spread"] = (spread > 2.0).astype("Int8")
        x["ref_support_very_high_spread"] = (spread > 4.0).astype("Int8")
        x["ref_support_spread_component"] = (spread / 4.0).clip(lower=0, upper=1).astype("float32")

    lcz_col = _match_fraction_feature(x, "LCZ")
    if lcz_col is not None:
        lcz_mismatch = 1.0 - pd.to_numeric(x[lcz_col], errors="coerce")
        x["ref_support_lcz_mismatch"] = lcz_mismatch.clip(lower=0, upper=1).astype("float32")
        x["ref_support_high_lcz_mismatch"] = (lcz_mismatch > 0.6).astype("Int8")

    lc_col = _match_fraction_feature(x, "LC")
    if lc_col is not None:
        lc_mismatch = 1.0 - pd.to_numeric(x[lc_col], errors="coerce")
        x["ref_support_lc_mismatch"] = lc_mismatch.clip(lower=0, upper=1).astype("float32")
        x["ref_support_high_lc_mismatch"] = (lc_mismatch > 0.6).astype("Int8")

    elev_col = _first_existing(x.columns, ["ref_nearest_abs_elev_diff_m", "ref_idw_abs_elev_diff_m"])
    if elev_col is not None:
        elev = pd.to_numeric(x[elev_col], errors="coerce").abs()
        x["ref_support_abs_elev_diff_m"] = elev
        x["ref_support_large_elev_diff"] = (elev > 100).astype("Int8")
        x["ref_support_elev_component"] = (elev / 150.0).clip(lower=0, upper=1).astype("float32")

    bh_col = _first_existing(x.columns, ["ref_nearest_abs_building_height_diff_m", "ref_idw_abs_building_height_diff_m"])
    if bh_col is not None:
        bh = pd.to_numeric(x[bh_col], errors="coerce").abs()
        x["ref_support_abs_building_height_diff_m"] = bh
        x["ref_support_large_building_height_diff"] = (bh > 10).astype("Int8")
        x["ref_support_building_height_component"] = (bh / 20.0).clip(lower=0, upper=1).astype("float32")



    if {"ssrd_wm2", "NDVI"}.issubset(x.columns):
        ssrd = pd.to_numeric(x["ssrd_wm2"], errors="coerce")
        ndvi = pd.to_numeric(x["NDVI"], errors="coerce")
        x["radiation_x_ndvi"] = ssrd * ndvi
        x["radiation_x_one_minus_ndvi"] = ssrd * (1.0 - ndvi)

    lst_candidates = ["LST_C_gapfill_clim", "LST_C_gapfill_3h", "LST_C_asof", "LST_C"]
    lst_col = _first_existing(x.columns, lst_candidates)
    if lst_col is not None and "ssrd_wm2" in x.columns:
        x["radiation_x_lst"] = pd.to_numeric(x["ssrd_wm2"], errors="coerce") * pd.to_numeric(x[lst_col], errors="coerce")
    if lst_col is not None and "NDVI" in x.columns:
        x["lst_x_ndvi"] = pd.to_numeric(x[lst_col], errors="coerce") * pd.to_numeric(x["NDVI"], errors="coerce")

    if "LCZ_buffer_fraction" in x.columns and lst_col is not None:
        x["lst_x_lcz_buffer_fraction"] = pd.to_numeric(x[lst_col], errors="coerce") * pd.to_numeric(x["LCZ_buffer_fraction"], errors="coerce")
    if "LC_buffer_fraction" in x.columns and "ssrd_wm2" in x.columns:
        x["radiation_x_lc_buffer_fraction"] = pd.to_numeric(x["ssrd_wm2"], errors="coerce") * pd.to_numeric(x["LC_buffer_fraction"], errors="coerce")


    if "NDVI" in x.columns and "ows_idw_ndvi" in x.columns and "ndvi_minus_ows_idw_ndvi" not in x.columns:
        x["ndvi_minus_ows_idw_ndvi"] = pd.to_numeric(x["NDVI"], errors="coerce") - pd.to_numeric(x["ows_idw_ndvi"], errors="coerce")
        x["ndvi_minus_ows_idw_ndvi_abs"] = x["ndvi_minus_ows_idw_ndvi"].abs()

    if lst_col is not None and "ows_idw_lst" in x.columns:
        out_name = f"{lst_col.lower()}_minus_ows_idw_lst"
        if out_name not in x.columns:
            x[out_name] = pd.to_numeric(x[lst_col], errors="coerce") - pd.to_numeric(x["ows_idw_lst"], errors="coerce")
            x[f"{out_name}_abs"] = x[out_name].abs()



    component_cols = [
        "ref_support_density_component",
        "ref_support_distance_component",
        "ref_support_spread_component",
        "ref_support_lcz_mismatch",
        "ref_support_lc_mismatch",
        "ref_support_elev_component",
        "ref_support_building_height_component",
    ]
    available = [c for c in component_cols if c in x.columns]
    if available:
        comp = x[available].apply(pd.to_numeric, errors="coerce")
        x["ref_support_component_n"] = comp.notna().sum(axis=1).astype("int16")
        x["ref_support_score"] = comp.mean(axis=1).astype("float32")
        x["ref_support_class"] = pd.cut(
            x["ref_support_score"],
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=["easy", "moderate", "hard", "very_hard"],
        ).astype("string").fillna("Missing")

    return downcast_dataframe(x)


def fit_station_bias_summary(
    train_df: pd.DataFrame,
    residual_col: str = "cws_ref_resid",
    z_col: str = "cws_ref_z",
    group_cols: Sequence[str] = ("network", "station_id"),
    min_rows: int = 24,
    high_solar_quantile: float = 0.75,
    low_wind_quantile: float = 0.25,
) -> pd.DataFrame:
    """Fit station residual-bias summaries from training rows only.

    These features capture systemic station bias and persistent siting/radiation
    patterns without letting validation/test residuals leak into the model.
    """
    required = set(group_cols) | {residual_col}
    missing = required - set(train_df.columns)
    if missing:
        raise KeyError(f"Training frame missing required columns for station summary: {sorted(missing)}")

    x = train_df.copy()
    x["date"] = ensure_utc_datetime(x["date"])
    x[residual_col] = pd.to_numeric(x[residual_col], errors="coerce")
    if z_col in x.columns:
        x[z_col] = pd.to_numeric(x[z_col], errors="coerce")
    x["abs_resid"] = x[residual_col].abs()
    x["resid_sign"] = np.sign(x[residual_col])
    x["month"] = x["date"].dt.month

    if "is_daylight" not in x.columns:
        if "ssrd_wm2" in x.columns:
            x["is_daylight"] = (pd.to_numeric(x["ssrd_wm2"], errors="coerce") > 5).astype("int8")
        else:
            x["is_daylight"] = x["date"].dt.hour.between(8, 18).astype("int8")

    if "ssrd_wm2" in x.columns:
        ssrd = pd.to_numeric(x["ssrd_wm2"], errors="coerce")
        q = ssrd.quantile(high_solar_quantile) if ssrd.notna().any() else np.nan
        x["_is_high_solar"] = (ssrd >= q).astype("int8") if pd.notna(q) else 0
    else:
        x["_is_high_solar"] = 0

    if "wind_speed" in x.columns:
        wind = pd.to_numeric(x["wind_speed"], errors="coerce")
    elif {"u10_ms", "v10_ms"}.issubset(x.columns):
        wind = np.sqrt(pd.to_numeric(x["u10_ms"], errors="coerce") ** 2 + pd.to_numeric(x["v10_ms"], errors="coerce") ** 2)
        x["wind_speed"] = wind
    else:
        wind = pd.Series(np.nan, index=x.index)
    if wind.notna().any():
        wq = wind.quantile(low_wind_quantile)
        x["_is_low_wind"] = (wind <= wq).astype("int8")
    else:
        x["_is_low_wind"] = 0

    rows = []
    for keys, g in x.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        resid = pd.to_numeric(g[residual_col], errors="coerce").dropna()
        abs_resid = resid.abs()
        row = dict(zip(group_cols, keys))

        row["station_ref_n_train"] = int(resid.size)
        if resid.size < min_rows:


            rows.append(row)
            continue

        med = float(resid.median())
        row["station_ref_mean_resid_train"] = float(resid.mean())
        row["station_ref_median_resid_train"] = med
        row["station_ref_abs_median_resid_train"] = abs(med)
        row["station_ref_mad_resid_train"] = float(np.median(np.abs(resid - med)))
        row["station_ref_sigma_mad_train"] = robust_mad_sigma(resid)
        row["station_ref_p90_abs_resid_train"] = float(abs_resid.quantile(0.90))
        row["station_ref_p95_abs_resid_train"] = float(abs_resid.quantile(0.95))
        row["station_ref_p99_abs_resid_train"] = float(abs_resid.quantile(0.99))

        sign = np.sign(resid)
        row["station_ref_frac_positive_resid_train"] = float((resid > 0).mean())
        row["station_ref_frac_negative_resid_train"] = float((resid < 0).mean())
        row["station_ref_sign_consistency_train"] = float(abs(np.nanmean(sign)))
        row["station_ref_bias_direction_train"] = float(np.sign(med))

        if z_col in g.columns:
            z = pd.to_numeric(g[z_col], errors="coerce")
            row["station_ref_frac_abs_z_gt_1p5_train"] = float((z.abs() > 1.5).mean())
            row["station_ref_frac_abs_z_gt_2_train"] = float((z.abs() > 2.0).mean())
            row["station_ref_frac_abs_z_gt_2p5_train"] = float((z.abs() > 2.5).mean())
            row["station_ref_frac_abs_z_gt_3_train"] = float((z.abs() > 3.0).mean())
            row["station_ref_median_abs_z_train"] = float(z.abs().median())

        day = g["is_daylight"].astype(bool)
        row["station_ref_day_median_resid_train"] = float(g.loc[day, residual_col].median()) if day.any() else np.nan
        row["station_ref_night_median_resid_train"] = float(g.loc[~day, residual_col].median()) if (~day).any() else np.nan
        if pd.notna(row["station_ref_day_median_resid_train"]) and pd.notna(row["station_ref_night_median_resid_train"]):
            row["station_ref_day_minus_night_bias_train"] = row["station_ref_day_median_resid_train"] - row["station_ref_night_median_resid_train"]
            row["station_ref_abs_day_minus_night_bias_train"] = abs(row["station_ref_day_minus_night_bias_train"])

        hs = g["_is_high_solar"].astype(bool)
        row["station_ref_high_solar_median_resid_train"] = float(g.loc[hs, residual_col].median()) if hs.any() else np.nan
        row["station_ref_low_solar_median_resid_train"] = float(g.loc[~hs, residual_col].median()) if (~hs).any() else np.nan
        if pd.notna(row["station_ref_high_solar_median_resid_train"]) and pd.notna(row["station_ref_low_solar_median_resid_train"]):
            row["station_ref_high_minus_low_solar_bias_train"] = row["station_ref_high_solar_median_resid_train"] - row["station_ref_low_solar_median_resid_train"]
            row["station_ref_abs_high_minus_low_solar_bias_train"] = abs(row["station_ref_high_minus_low_solar_bias_train"])

        lw_hs = g["_is_low_wind"].astype(bool) & hs
        row["station_ref_low_wind_high_solar_median_resid_train"] = float(g.loc[lw_hs, residual_col].median()) if lw_hs.any() else np.nan

        if "ssrd_wm2" in g.columns:
            row["station_ref_solar_resid_slope_train"] = _safe_slope(g["ssrd_wm2"], g[residual_col], min_n=min_rows)

        if "wind_speed" in g.columns:
            row["station_ref_wind_resid_slope_train"] = _safe_slope(g["wind_speed"], g[residual_col], min_n=min_rows)



        month_med = g.groupby("month")[residual_col].median().dropna()
        row["station_ref_active_months_train"] = int(month_med.size)
        if month_med.size:
            row["station_ref_month_median_abs_train"] = float(month_med.abs().median())
            row["station_ref_month_median_iqr_train"] = float(month_med.quantile(0.75) - month_med.quantile(0.25)) if month_med.size >= 2 else 0.0
            row["station_ref_month_sign_consistency_train"] = float(abs(np.sign(month_med).mean()))
            row["station_ref_month_frac_same_sign_as_station_train"] = float((np.sign(month_med) == np.sign(med)).mean()) if med != 0 else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    return downcast_dataframe(out)


def apply_station_bias_summary(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    group_cols: Sequence[str] = ("network", "station_id"),
) -> pd.DataFrame:
    """Merge train-only station residual summaries into any split."""
    x = df.merge(summary, on=list(group_cols), how="left", validate="many_to_one")
    x["station_bias_summary_missing"] = x["station_ref_n_train"].isna().astype("int8") if "station_ref_n_train" in x.columns else 1
    if "station_ref_n_train" in x.columns:
        x["station_ref_n_train"] = pd.to_numeric(x["station_ref_n_train"], errors="coerce")
        x["station_ref_n_train_log1p"] = np.log1p(x["station_ref_n_train"]).astype("float32")
    return downcast_dataframe(x)


def add_split_safe_station_bias_features(
    df: pd.DataFrame,
    split_col: str,
    cfg: ResidualRiskConfig,
    residual_col: str = "cws_ref_resid",
    z_col: str = "cws_ref_z",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit station bias on training rows only, then merge into all rows."""
    train = df[df[split_col] == "train"].copy()
    summary = fit_station_bias_summary(
        train,
        residual_col=residual_col,
        z_col=z_col,
        min_rows=cfg.station_min_train_rows,
        high_solar_quantile=cfg.high_solar_quantile,
        low_wind_quantile=cfg.low_wind_quantile,
    )
    out = apply_station_bias_summary(df, summary)
    return out, summary


def is_coordinate_like_feature(col: str) -> bool:
    c = str(col).strip().lower()
    exact = {
        "lat", "lon", "long", "latitude", "longitude",
        "station_lat", "station_lon", "station_long",
        "cws_lat", "cws_lon", "cws_long",
        "ows_lat", "ows_lon", "ows_long",
    }
    suffixes = ("_lat", "_lon", "_long", "_latitude", "_longitude")
    return c in exact or c.endswith(suffixes)


def numeric_landcover_code_columns(df: pd.DataFrame) -> set[str]:
    """Find redundant LC/LCZ numeric code columns when *_lg labels exist."""
    pairs = [
        ("LC_point", "LC_point_lg"),
        ("LC_buffer", "LC_buffer_lg"),
        ("LCZ_point", "LCZ_point_lg"),
        ("LCZ_buffer", "LCZ_buffer_lg"),
    ]
    return {code for code, label in pairs if code in df.columns and label in df.columns}


def _is_current_reference_temperature_feature(col: str) -> bool:
    """Return True for columns that expose the current CWS/reference comparison.

    The residual-risk target is built from the current CWS observation and the
    current OWS-reference estimate.  Therefore, context-style models must remove
    not only cws_ref_resid, but also the ingredients that can reconstruct it:
    temp_raw, ref_mu*, ref_sigma, local OWS temperature summaries, and engineered
    differences involving those reference temperatures.
    """
    c = str(col)
    low = c.lower()

    exact = {
        "temp_raw", "temperature", "temp_c", "ta_c",
        "ref_mu", "ref_sigma", "ref_z",
        "ref_idw_mu", "ref_nearest_mu",
        "ref_local_mean_c", "ref_local_median_c", "ref_local_min_c", "ref_local_max_c",
        "ows_idw_temp", "ows_nearest_temp", "ows_local_mean_temp", "ows_local_median_temp",
    }
    if low in exact:
        return True


    if re.match(r"^ref_mu($|_)", low):
        return True
    if re.match(r"^ref_(idw|nearest).*_?mu$", low):
        return True



    if re.match(r"^ref_local_(mean|median|min|max)_c$", low):
        return True
    if re.match(r"^ows_(idw|nearest|local_mean|local_median).*temp", low):
        return True


    if low.startswith("temp_minus_"):
        return True
    if "minus_ows" in low or "minus_lst" in low:
        return True
    if low.startswith("ref_") and "minus_" in low:
        return True

    return False


def _is_station_residual_history_feature(col: str) -> bool:
    """Train-only station residual-history features.

    These are leak-safe for a time-split if computed only on the train period,
    but they use past residual labels.  Keep them in context_history and remove
    them in context_static_met.
    """
    low = str(col).lower()
    return low.startswith("station_ref_")


def feature_leakage_audit_table(feature_cols: Sequence[str], feature_mode: str) -> pd.DataFrame:
    """Audit selected features for paper-facing leakage categories."""
    rows = []
    mode = str(feature_mode)
    direct_reference_allowed = mode in {"operational_no_direct_residual", "reference_available_diagnostic"}
    station_history_allowed = mode in {"context_history", "context_only", "operational_no_direct_residual", "reference_available_diagnostic"}

    for col in feature_cols:
        low = str(col).lower()
        categories = []
        severity = "ok"
        if low.startswith("target_") or "risk" in low:
            categories.append("target_or_risk_column")
            severity = "fatal"
        if "ref_resid" in low or low in {"cws_ref_resid", "abs_cws_ref_resid", "cws_ref_z", "abs_cws_ref_z"}:
            categories.append("direct_residual_column")
            severity = "fatal"
        if _is_current_reference_temperature_feature(col):
            categories.append("current_temp_or_reference_temperature")
            if not direct_reference_allowed:
                severity = "fatal"
            else:
                severity = max([severity, "diagnostic"], key=["ok", "warning", "diagnostic", "fatal"].index)
        if _is_station_residual_history_feature(col):
            categories.append("train_only_station_residual_history")
            if not station_history_allowed:
                severity = "fatal"
            else:
                severity = max([severity, "warning"], key=["ok", "warning", "diagnostic", "fatal"].index)
        rows.append({
            "feature_mode": mode,
            "feature": str(col),
            "audit_category": ";".join(categories) if categories else "ok",
            "severity": severity,
        })
    return pd.DataFrame(rows)


def forbidden_feature_columns(
    df: pd.DataFrame,
    feature_mode: str,
    include_qc_flags: bool = False,
    include_coordinates: bool = False,
    drop_numeric_lc_codes: bool = True,
) -> set[str]:
    """Columns that should not be used as classifier inputs.

    Important interpretation: the residual-risk label is derived from
    abs((temp_raw - ref_mu) / ref_sigma).  Removing only cws_ref_resid is not
    enough; a model that sees temp_raw, ref_mu and ref_sigma can reconstruct the
    label.  Paper-facing context modes therefore remove all current CWS/reference
    temperature ingredients and keep only metadata, time, meteorology, satellite,
    reference-support geometry, and optionally train-only station history.
    """
    mode = str(feature_mode)


    if mode == "context_only":
        mode = "context_history"

    forbidden = {
        "date", "station_id", "split", "city",
        "target_ref_risk", "target_ref_risk_abs", "target_ref_risk_z",
        "target_ref_risk_band_abs", "target_ref_risk_band_z",
        "target_ref_risk_hiconf", "is_model_labeled",
        "pred_ref_risk_prob", "risk_band_fixed", "risk_band_quantile",
        "cws_ref_resid", "abs_cws_ref_resid", "cws_ref_z", "abs_cws_ref_z",
        "ows_ref_risk_abs", "ows_ref_risk_z",
        "ows_ref_risk_band_abs", "ows_ref_risk_band_z",
    }


    for c in df.columns:
        low = str(c).lower()
        if low.startswith("target_"):
            forbidden.add(c)
        if low.endswith("_station_id") or low in {"ref_nearest_value_source_station_id", "ref_nearest_source_station_id"}:
            forbidden.add(c)
        if "ref_resid" in low or "abs_ref_resid" in low:
            forbidden.add(c)
        if low.startswith("pred_ref_risk") or low.startswith("risk_band"):
            forbidden.add(c)

    if not include_qc_flags:
        for c in df.columns:
            low = str(c).lower()
            if low.startswith("qc_flag") or "qc_flag" in low or low.startswith("ultra_strict") or low.startswith("strict_qc") or low.startswith("lenient_qc"):
                forbidden.add(c)

    if not include_coordinates:
        for c in df.columns:
            if is_coordinate_like_feature(str(c)):
                forbidden.add(c)

    if drop_numeric_lc_codes:
        forbidden.update(numeric_landcover_code_columns(df))

    if mode in {"context_history", "context_static_met"}:
        for c in df.columns:
            if _is_current_reference_temperature_feature(c):
                forbidden.add(c)
        if mode == "context_static_met":
            for c in df.columns:
                if _is_station_residual_history_feature(c):
                    forbidden.add(c)
    elif mode in {"operational_no_direct_residual", "reference_available_diagnostic"}:


        pass
    else:
        raise ValueError(
            "feature_mode must be one of: context_history, context_static_met, "
            "context_only, operational_no_direct_residual, reference_available_diagnostic"
        )

    return set(map(str, forbidden))


def get_residual_risk_feature_columns(
    df: pd.DataFrame,
    feature_mode: str = "operational_no_direct_residual",
    include_qc_flags: bool = False,
    include_coordinates: bool = False,
    drop_numeric_lc_codes: bool = True,
) -> tuple[list[str], list[str]]:
    forbidden = forbidden_feature_columns(
        df,
        feature_mode=feature_mode,
        include_qc_flags=include_qc_flags,
        include_coordinates=include_coordinates,
        drop_numeric_lc_codes=drop_numeric_lc_codes,
    )

    feature_cols: list[str] = []
    for c in df.columns:
        c_str = str(c)
        if c_str in forbidden:
            continue
        if c_str.endswith("_id") or c_str.endswith("_station_id"):
            continue
        dtype = df[c].dtype
        if (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or dtype == object
        ):
            feature_cols.append(c_str)

    categorical_cols = []
    for c in feature_cols:
        if (
            str(df[c].dtype) == "category"
            or pd.api.types.is_string_dtype(df[c])
            or df[c].dtype == object
            or c in {"network", "ref_support_class"}
            or c.endswith("_lg")
            or c.endswith("_bin")
        ):
            categorical_cols.append(c)

    return feature_cols, categorical_cols


def _prepare_catboost_matrix(df: pd.DataFrame, feature_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    X = df[list(feature_cols)].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = _as_string_category(X[c])
    for c in X.columns:
        if c not in categorical_cols and pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("int8")
    return X


def maybe_subsample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def train_catboost_residual_risk(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    cfg: ResidualRiskConfig,
    model_path: str | Path | None = None,
):
    """Train a CatBoost residual-risk classifier."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as e:
        raise ImportError("catboost is not installed. Install catboost or set run_catboost=False.") from e

    tr = train_df[train_df[target_col].isin([0, 1])].copy()
    va = valid_df[valid_df[target_col].isin([0, 1])].copy()
    tr = maybe_subsample_rows(tr, cfg.max_train_rows, random_state=cfg.random_state)
    va = maybe_subsample_rows(va, cfg.max_valid_rows, random_state=cfg.random_state)

    if len(tr) == 0 or len(va) == 0:
        raise ValueError("Need non-empty labeled train and validation frames.")
    if tr[target_col].nunique() < 2:
        raise ValueError("Training target has only one class after filtering.")

    X_train = _prepare_catboost_matrix(tr, feature_cols, categorical_cols)
    X_valid = _prepare_catboost_matrix(va, feature_cols, categorical_cols)
    y_train = tr[target_col].astype(int)
    y_valid = va[target_col].astype(int)

    cat_features = [c for c in categorical_cols if c in feature_cols]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_features)

    thread_count = cfg.thread_count
    if thread_count is None:
        cpu_count = os.cpu_count() or 8
        thread_count = max(1, cpu_count - 2)

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(cfg.iterations),
        "learning_rate": float(cfg.learning_rate),
        "depth": int(cfg.depth),
        "l2_leaf_reg": float(cfg.l2_leaf_reg),
        "random_seed": int(cfg.random_state),
        "early_stopping_rounds": int(cfg.early_stopping_rounds),
        "allow_writing_files": False,
        "verbose": cfg.verbose,
        "thread_count": int(thread_count),
    }
    if cfg.auto_class_weights:
        params["auto_class_weights"] = cfg.auto_class_weights
    if cfg.used_ram_limit:
        params["used_ram_limit"] = cfg.used_ram_limit
    if cfg.use_gpu:
        params["task_type"] = "GPU"

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

    return model


def predict_catboost_proba(model, df: pd.DataFrame, feature_cols: Sequence[str], categorical_cols: Sequence[str]) -> np.ndarray:
    from catboost import Pool

    cat_features = [c for c in categorical_cols if c in feature_cols]
    X = _prepare_catboost_matrix(df, feature_cols, categorical_cols)
    pool = Pool(X, cat_features=cat_features)
    return np.asarray(model.predict_proba(pool))[:, 1]


def choose_f1_threshold(y_true, pred_prob) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def expected_calibration_error(y_true, pred_prob, n_bins: int = 15) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype=float)
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += abs(float(y[mask].mean()) - float(p[mask].mean())) * float(mask.mean())
    return float(ece)


def reliability_curve_table(y_true, pred_prob, n_bins: int = 15, split_name: str = "test", model_name: str = "model") -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        mask = (p >= lo) & (p <= hi) if hi == 1 else (p >= lo) & (p < hi)
        rows.append({
            "model": model_name,
            "split": split_name,
            "bin": i,
            "prob_lo": float(lo),
            "prob_hi": float(hi),
            "n": int(mask.sum()),
            "mean_pred_prob": float(p[mask].mean()) if mask.any() else np.nan,
            "event_rate": float(y[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_probability_predictions(y_true, pred_prob, split_name: str, model_name: str, threshold: float | None = None) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype=float)
    if threshold is None:
        threshold = choose_f1_threshold(y, p)
    pred = (p >= threshold).astype(int)

    out = {
        "model": model_name,
        "split": split_name,
        "n": int(len(y)),
        "event_rate": float(y.mean()) if len(y) else np.nan,
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "ece_15": expected_calibration_error(y, p, n_bins=15) if len(np.unique(y)) > 1 else np.nan,
        "f1": float(f1_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
    }
    try:
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        out["log_loss"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def assign_probability_risk_bands(
    prob: pd.Series | np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> pd.Series:
    p = pd.to_numeric(pd.Series(prob), errors="coerce")
    out = pd.Series("ambiguous", index=p.index, dtype="object")
    out[p <= low_cutoff] = "high_confidence_low_risk"
    out[p >= high_cutoff] = "high_confidence_high_risk"
    out[p.isna()] = pd.NA
    return out.astype("string")


def add_probability_bands(
    df: pd.DataFrame,
    prob_col: str,
    cfg: ResidualRiskConfig,
    valid_df: pd.DataFrame | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    x = df.copy()
    p = pd.to_numeric(x[prob_col], errors="coerce")
    x[f"pred_ref_risk_band_fixed{suffix}"] = assign_probability_risk_bands(
        p,
        low_cutoff=cfg.fixed_low_risk_prob,
        high_cutoff=cfg.fixed_high_risk_prob,
    )

    if valid_df is not None and prob_col in valid_df.columns:
        pv = pd.to_numeric(valid_df[prob_col], errors="coerce").dropna()
        if len(pv):
            low_q = float(pv.quantile(cfg.quantile_low))
            high_q = float(pv.quantile(cfg.quantile_high))
        else:
            low_q, high_q = cfg.fixed_low_risk_prob, cfg.fixed_high_risk_prob
    else:
        low_q, high_q = cfg.fixed_low_risk_prob, cfg.fixed_high_risk_prob

    x[f"pred_ref_risk_band_quantile{suffix}"] = assign_probability_risk_bands(p, low_q, high_q)
    x[f"pred_ref_risk_band_quantile_low_cutoff{suffix}"] = low_q
    x[f"pred_ref_risk_band_quantile_high_cutoff{suffix}"] = high_q
    return downcast_dataframe(x)


def residual_error_summary(values: pd.Series) -> dict:
    r = pd.to_numeric(values, errors="coerce").dropna()
    if len(r) == 0:
        return {
            "residual_n": 0,
            "residual_bias": np.nan,
            "residual_mae": np.nan,
            "residual_rmse": np.nan,
            "residual_p95_abs": np.nan,
            "residual_p99_abs": np.nan,
        }
    return {
        "residual_n": int(len(r)),
        "residual_bias": float(r.mean()),
        "residual_mae": float(r.abs().mean()),
        "residual_rmse": float(np.sqrt(np.mean(r ** 2))),
        "residual_p95_abs": float(r.abs().quantile(0.95)),
        "residual_p99_abs": float(r.abs().quantile(0.99)),
    }


def make_retention_curve(
    df: pd.DataFrame,
    risk_col: str,
    residual_col: str = "cws_ref_resid",
    target_col: str | None = "target_ref_risk",
    fractions: Sequence[float] | None = None,
    method_name: str = "model",
    split_name: str = "test",
) -> pd.DataFrame:
    if fractions is None:
        fractions = np.round(np.linspace(0.50, 0.99, 20), 3)

    x = df[df[risk_col].notna()].copy()
    if len(x) == 0:
        return pd.DataFrame()

    rows = []
    for frac in fractions:
        cutoff = float(x[risk_col].quantile(float(frac)))
        kept = x[x[risk_col] <= cutoff].copy()
        row = {
            "method": method_name,
            "split": split_name,
            "retention_target": float(frac),
            "risk_cutoff": cutoff,
            "n_total": int(len(x)),
            "n_kept": int(len(kept)),
            "retention_actual": float(len(kept) / len(x)) if len(x) else np.nan,
            "n_stations_kept": int(kept[["network", "station_id"]].drop_duplicates().shape[0]) if {"network", "station_id"}.issubset(kept.columns) else np.nan,
        }
        if target_col is not None and target_col in kept.columns:
            lab = kept[kept[target_col].isin([0, 1])]
            row["event_rate_kept"] = float(lab[target_col].mean()) if len(lab) else np.nan
        if residual_col in kept.columns:
            row.update(residual_error_summary(kept[residual_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_band_summary(
    df: pd.DataFrame,
    band_col: str,
    residual_col: str = "cws_ref_resid",
    target_col: str = "target_ref_risk",
    method_name: str = "band",
    split_name: str = "test",
) -> pd.DataFrame:
    if band_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for band, g in df.groupby(band_col, dropna=False):
        row = {
            "method": method_name,
            "split": split_name,
            "band": str(band),
            "n": int(len(g)),
            "fraction": float(len(g) / len(df)) if len(df) else np.nan,
        }
        if target_col in g.columns:
            lab = g[g[target_col].isin([0, 1])]
            row["event_rate"] = float(lab[target_col].mean()) if len(lab) else np.nan
        if residual_col in g.columns:
            row.update(residual_error_summary(g[residual_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_feature_modes(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def _split_counts(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("split", dropna=False):
        lab = g[g[target_col].isin([0, 1])]
        rows.append({
            "split": split,
            "n_rows": int(len(g)),
            "n_labeled": int(len(lab)),
            "event_rate_labeled": float(lab[target_col].mean()) if len(lab) else np.nan,
            "n_stations": int(g[["network", "station_id"]].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(rows)


def _slugify_for_path(value: object) -> str:
    """Return a compact path-safe label."""
    s = str(value).strip()
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {".", " ", "/", "\\", ":"}:
            out.append("p" if ch == "." else "-")
    slug = "".join(out).strip("-_")
    return slug or "run"


def _infer_reference_method_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "catboost" in stem:
        return "catboost"
    if "idw" in stem:
        return "idw"
    return "reference"


def _infer_calibration_mode_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "all_ows_reference" in stem:
        return "all_ows_reference"
    if "time_train" in stem:
        return "time_train"
    return "calibration"


def make_residual_risk_run_label(cfg: ResidualRiskConfig, input_path: str | Path) -> str:
    """Create a readable default run label when the notebook did not provide one."""
    if cfg.run_label:
        return _slugify_for_path(cfg.run_label)

    input_path = Path(input_path)
    method = cfg.reference_method if cfg.reference_method != "auto" else _infer_reference_method_from_path(input_path)
    cal = cfg.calibration_mode or _infer_calibration_mode_from_path(input_path)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pieces = [
        _slugify_for_path(cfg.city),
        "residual-risk",
        _slugify_for_path(method),
        _slugify_for_path(cal),
        _slugify_for_path(cfg.split_strategy),
        timestamp,
    ]
    return "__".join(pieces)


def make_unique_run_dir(
    base_output_dir: str | Path,
    run_label: str,
    overwrite_existing_run: bool = False,
) -> tuple[Path, str]:
    """Return a non-overwriting output directory and the final resolved label."""
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _slugify_for_path(run_label)
    candidate = base_output_dir / safe_label

    if overwrite_existing_run or (not candidate.exists()) or (candidate.exists() and not any(candidate.iterdir())):
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate, safe_label

    for idx in range(2, 1000):
        label_i = f"{safe_label}__repeat{idx:03d}"
        candidate_i = base_output_dir / label_i
        if (not candidate_i.exists()) or (candidate_i.exists() and not any(candidate_i.iterdir())):
            candidate_i.mkdir(parents=True, exist_ok=True)
            return candidate_i, label_i

    raise RuntimeError(f"Could not create a unique run directory under {base_output_dir}")

def run_residual_risk_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: ResidualRiskConfig | None = None,
) -> dict:
    """Run the full downstream residual-risk workflow.

    ``output_dir`` is treated as a base folder. The function creates a unique
    run-labeled subfolder inside it, matching the v4 OWS-reference runner's
    non-overwriting behavior.
    """
    cfg = cfg or ResidualRiskConfig()

    if input_path is None:
        if cfg.input_path is None:
            raise ValueError("input_path must be provided either directly or in cfg.input_path")
        input_path = cfg.input_path
    if output_dir is None:
        output_dir = cfg.output_dir

    input_path = Path(input_path)
    base_output_dir = Path(output_dir)
    requested_run_label = make_residual_risk_run_label(cfg, input_path)
    output_dir, final_run_label = make_unique_run_dir(
        base_output_dir,
        requested_run_label,
        overwrite_existing_run=cfg.overwrite_existing_run,
    )



    cfg.input_path = str(input_path)
    cfg.output_dir = str(base_output_dir)

    print("Input CWS reference table:", input_path)
    print("Residual-risk run label:", final_run_label)
    print("Residual-risk output dir:", output_dir)

    df_raw = load_dataframe_auto(input_path)
    df = prepare_base_frame(df_raw)
    df = add_environmental_support_features(df)
    df = assign_splits(df, cfg)
    df, target_col = resolve_training_target(df, cfg)
    df, station_summary = add_split_safe_station_bias_features(df, "split", cfg)


    augmented_path = save_dataframe_auto(df, output_dir / f"{cfg.city}_cws_reference_augmented_features.parquet")
    station_summary_path = output_dir / f"{cfg.city}_station_bias_summary_train_only.csv"
    station_summary.to_csv(station_summary_path, index=False)

    counts = _split_counts(df, target_col)
    counts_path = output_dir / f"{cfg.city}_split_counts.csv"
    counts.to_csv(counts_path, index=False)

    results: dict[str, object] = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "input_path": str(input_path),
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "requested_run_label": requested_run_label,
        "run_label": final_run_label,
        "reference_method": cfg.reference_method,
        "calibration_mode": cfg.calibration_mode,
        "source_reference_run_label": cfg.source_reference_run_label,
        "source_reference_manifest_path": cfg.source_reference_manifest_path,
        "augmented_features_path": str(augmented_path),
        "station_summary_path": str(station_summary_path),
        "split_counts_path": str(counts_path),
        "target_col": target_col,
        "models": {},
    }

    if not cfg.run_catboost:
        manifest_path = output_dir / f"{cfg.city}_residual_risk_manifest.json"
        manifest_path.write_text(json.dumps(results, indent=2, default=str))
        results["manifest_path"] = str(manifest_path)
        return results

    all_metrics = []
    all_reliability = []
    all_retention = []
    all_band_summary = []
    scored = df.copy()
    feature_audit = {}
    leakage_audit_tables = []

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    for mode in _parse_feature_modes(cfg.feature_modes):
        feature_cols, categorical_cols = get_residual_risk_feature_columns(
            df,
            feature_mode=mode,
            include_qc_flags=cfg.include_qc_flags,
            include_coordinates=cfg.include_coordinates,
            drop_numeric_lc_codes=cfg.drop_numeric_landcover_codes,
        )
        leakage_audit_df = feature_leakage_audit_table(feature_cols, mode)
        leakage_audit_tables.append(leakage_audit_df)
        leakage_counts = leakage_audit_df["severity"].value_counts(dropna=False).to_dict()
        suspicious = leakage_audit_df[leakage_audit_df["severity"].isin(["fatal", "diagnostic"])]
        if len(suspicious):
            warnings.warn(
                f"Feature audit for mode={mode!r} found {len(suspicious)} diagnostic/fatal current-reference or target-like features. "
                f"Check the saved leakage audit before using this model in paper results."
            )
        feature_audit[mode] = {
            "n_features": len(feature_cols),
            "n_categorical": len(categorical_cols),
            "feature_cols": feature_cols,
            "categorical_cols": categorical_cols,
            "leakage_counts": leakage_counts,
            "fatal_or_diagnostic_features": suspicious[["feature", "audit_category", "severity"]].to_dict(orient="records"),
        }

        model_name = f"catboost_{mode}"
        model_path = output_dir / f"{cfg.city}_{model_name}.cbm"
        model = train_catboost_residual_risk(
            train_df,
            valid_df,
            feature_cols,
            categorical_cols,
            target_col=target_col,
            cfg=cfg,
            model_path=model_path,
        )

        prob_col = f"pred_ref_risk_prob_{mode}"
        scored[prob_col] = predict_catboost_proba(model, scored, feature_cols, categorical_cols)

        valid_labeled = scored[(scored["split"] == "valid") & (scored[target_col].isin([0, 1]))].copy()
        threshold = choose_f1_threshold(valid_labeled[target_col].astype(int), valid_labeled[prob_col])

        for split in ["train", "valid", "test"]:
            g = scored[(scored["split"] == split) & (scored[target_col].isin([0, 1]))].copy()
            if len(g) == 0:
                continue
            all_metrics.append(
                evaluate_probability_predictions(
                    g[target_col].astype(int),
                    g[prob_col],
                    split_name=split,
                    model_name=model_name,
                    threshold=threshold,
                )
            )
            all_reliability.append(
                reliability_curve_table(
                    g[target_col].astype(int),
                    g[prob_col],
                    split_name=split,
                    model_name=model_name,
                )
            )


        scored = add_probability_bands(scored, prob_col, cfg, valid_df=scored[scored["split"] == "valid"], suffix=f"_{mode}")


        test_scored = scored[scored["split"] == "test"].copy()
        all_retention.append(
            make_retention_curve(
                test_scored,
                risk_col=prob_col,
                residual_col="cws_ref_resid",
                target_col=target_col,
                fractions=cfg.retention_fractions,
                method_name=model_name,
                split_name="test",
            )
        )

        for band_col in [f"pred_ref_risk_band_fixed_{mode}", f"pred_ref_risk_band_quantile_{mode}"]:
            all_band_summary.append(
                fixed_band_summary(
                    test_scored,
                    band_col=band_col,
                    residual_col="cws_ref_resid",
                    target_col=target_col,
                    method_name=f"{model_name}:{band_col}",
                    split_name="test",
                )
            )


        try:
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.get_feature_importance(),
                "model": model_name,
            }).sort_values("importance", ascending=False)
            fi_path = output_dir / f"{cfg.city}_{model_name}_feature_importance.csv"
            fi.to_csv(fi_path, index=False)
        except Exception as e:
            warnings.warn(f"Could not save feature importance for {model_name}: {e}")
            fi_path = None

        results["models"][mode] = {
            "model_name": model_name,
            "model_path": str(model_path),
            "prob_col": prob_col,
            "valid_f1_threshold": float(threshold),
            "feature_importance_path": str(fi_path) if fi_path is not None else None,
        }



    test_scored = scored[scored["split"] == "test"].copy()
    if "abs_cws_ref_resid" in test_scored.columns:
        all_retention.append(
            make_retention_curve(
                test_scored,
                risk_col="abs_cws_ref_resid",
                residual_col="cws_ref_resid",
                target_col=target_col,
                fractions=cfg.retention_fractions,
                method_name="baseline_abs_cws_ref_resid",
                split_name="test",
            )
        )
    if "abs_cws_ref_z" in test_scored.columns:
        all_retention.append(
            make_retention_curve(
                test_scored,
                risk_col="abs_cws_ref_z",
                residual_col="cws_ref_resid",
                target_col=target_col,
                fractions=cfg.retention_fractions,
                method_name="baseline_abs_cws_ref_z",
                split_name="test",
            )
        )


    scored_path = save_dataframe_auto(scored, output_dir / f"{cfg.city}_cws_residual_risk_scored.parquet")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / f"{cfg.city}_residual_risk_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    reliability_df = pd.concat(all_reliability, ignore_index=True) if all_reliability else pd.DataFrame()
    reliability_path = output_dir / f"{cfg.city}_residual_risk_reliability.csv"
    reliability_df.to_csv(reliability_path, index=False)

    retention_df = pd.concat([x for x in all_retention if x is not None and len(x)], ignore_index=True) if all_retention else pd.DataFrame()
    retention_path = output_dir / f"{cfg.city}_residual_risk_retention_curves.csv"
    retention_df.to_csv(retention_path, index=False)

    band_summary_df = pd.concat([x for x in all_band_summary if x is not None and len(x)], ignore_index=True) if all_band_summary else pd.DataFrame()
    band_summary_path = output_dir / f"{cfg.city}_risk_band_summary.csv"
    band_summary_df.to_csv(band_summary_path, index=False)

    feature_audit_path = output_dir / f"{cfg.city}_feature_audit.json"
    feature_audit_path.write_text(json.dumps(feature_audit, indent=2, default=str))
    leakage_audit_path = output_dir / f"{cfg.city}_feature_leakage_audit.csv"
    if leakage_audit_tables:
        pd.concat(leakage_audit_tables, ignore_index=True).to_csv(leakage_audit_path, index=False)
    else:
        pd.DataFrame(columns=["feature_mode", "feature", "audit_category", "severity"]).to_csv(leakage_audit_path, index=False)

    results.update({
        "scored_path": str(scored_path),
        "metrics_path": str(metrics_path),
        "reliability_path": str(reliability_path),
        "retention_path": str(retention_path),
        "band_summary_path": str(band_summary_path),
        "feature_audit_path": str(feature_audit_path),
        "feature_leakage_audit_path": str(leakage_audit_path),
    })

    manifest_path = output_dir / f"{cfg.city}_residual_risk_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, default=str))
    results["manifest_path"] = str(manifest_path)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run split-safe CWS residual-risk modeling.")
    p.add_argument("--input-path", required=True, help="CWS reference residual features parquet/pkl/csv from the OWS reference step.")
    p.add_argument("--output-dir", required=True, help="Base directory for residual-risk outputs. A unique run subfolder is created inside it.")
    p.add_argument("--city", default="project_id")
    p.add_argument("--split-strategy", default="time", choices=["time", "station_holdout"])
    p.add_argument("--valid-start", default="2021-10-01")
    p.add_argument("--test-start", default="2021-11-01")
    p.add_argument("--target-mode", default="auto", choices=["auto", "hard", "high_confidence_bands"])
    p.add_argument("--feature-modes", default="context_history,context_static_met")
    p.add_argument("--include-qc-flags", action="store_true")
    p.add_argument("--include-coordinates", action="store_true", help="Include raw latitude/longitude columns as model inputs. Distance/support features are still used when this is off.")
    p.add_argument("--exclude-coordinates", action="store_true", help="Backward-compatible flag; coordinates are excluded by default unless --include-coordinates is set.")
    p.add_argument("--reference-method", default="auto", help="Reference method used upstream, for manifest labeling only: auto/idw/catboost.")
    p.add_argument("--calibration-mode", default="time_train", help="Reference calibration mode used upstream, for manifest labeling only.")
    p.add_argument("--run-label", default=None, help="Optional run label. Existing non-empty labels get repeat suffixes unless overwrite is enabled.")
    p.add_argument("--overwrite-existing-run", action="store_true")
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--thread-count", type=int, default=None)
    p.add_argument("--used-ram-limit", default="42gb")
    p.add_argument("--verbose", type=int, default=100)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-valid-rows", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    cfg = ResidualRiskConfig(
        city=args.city,
        input_path=args.input_path,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        valid_start=args.valid_start,
        test_start=args.test_start,
        target_mode=args.target_mode,
        feature_modes=args.feature_modes,
        include_qc_flags=args.include_qc_flags,
        include_coordinates=bool(args.include_coordinates and not args.exclude_coordinates),
        reference_method=args.reference_method,
        calibration_mode=args.calibration_mode,
        run_label=args.run_label,
        overwrite_existing_run=args.overwrite_existing_run,
        run_catboost=not args.no_catboost,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        thread_count=args.thread_count,
        used_ram_limit=args.used_ram_limit,
        verbose=args.verbose,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
    )
    results = run_residual_risk_pipeline(cfg=cfg)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
