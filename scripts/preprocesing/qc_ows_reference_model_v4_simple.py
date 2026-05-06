"""OWS reference-model utilities for CWS temperature quality-control benchmarking."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import qc_benchmark_helpers_ows_patch as qcb
except Exception:
    qcb = None


KEY_COLS = ["date", "station_id", "network"]


@dataclass
class ReferenceModelConfig:
    city: str = "project_id"


    k_nearest_ows: int = 8
    ows_radius_km: float | None = 8.0
    fallback_to_k_nearest: bool = True
    idw_power: float = 2.0
    eps_km: float = 0.05



    valid_start: str = "2021-10-01"
    test_start: str = "2021-11-01"
    calibration_mode: str = "time_train"


    ndvi_asof_tolerance: str = "21D"
    ndvi_asof_direction: str = "backward"
    lst_asof_tolerance: str = "3h"
    lst_asof_direction: str = "nearest"
    use_dynamic_env_columns: bool = True
    use_ows_satellite_context: bool = True


    ambiguous_lower_quantile: float = 0.95


    run_catboost_regressor: bool = False
    catboost_cv_mode: str = "group_kfold"
    catboost_n_splits: int = 5
    catboost_iterations: int = 1200
    catboost_learning_rate: float = 0.04
    catboost_depth: int = 8
    catboost_l2_leaf_reg: float = 8.0
    catboost_include_coordinates: bool = False
    catboost_thread_count: int = 15
    catboost_used_ram_limit: str | None = "42gb"
    catboost_use_gpu: bool = False
    catboost_verbose: int = 100


    sigma_floor_c: float = 0.20
    sigma_min_group_n: int = 200
    abnormal_quantile: float = 0.995


    output_subdir: str = "ows_reference"
    rebuild: bool = False
    random_state: int = 42
    chunk_freq: str | None = "M"


    run_label: str | None = None
    save_both_calibration_modes: bool = True
    overwrite_existing_run: bool = False



    reference_method: str = "auto"


def _require_qcb():
    global qcb
    if qcb is not None:
        return qcb

    try:
        import qc_benchmark_helpers_ows_patch as _qcb
    except Exception as e:
        raise ImportError(
            "Could not import qc_benchmark_helpers_ows_patch. Place this script "
            "in cfm.cwd_scripts_preprocesing or add that folder to PYTHONPATH."
        ) from e
    qcb = _qcb
    return qcb


def normalize_network_name(x: object) -> str:
    helper = _require_qcb()
    return helper.normalize_network_name(x)


def ensure_utc_datetime(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def clean_station_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().astype(str)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    return _require_qcb().downcast_dataframe(df)


def save_dataframe_auto(df: pd.DataFrame, path: str | Path) -> Path:
    return _require_qcb().save_dataframe_auto(df, path)


def load_dataframe_auto(path: str | Path) -> pd.DataFrame:
    return _require_qcb().load_dataframe_auto(path)


def maybe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        print("Missing optional file:", path)
        return None
    print("Reading:", path)
    return pd.read_csv(path, **kwargs)


def infer_hourly_value_cols(
    df: pd.DataFrame | None,
    preferred_cols: Sequence[str] | None = None,
    allow_numericish_objects: bool = True,
) -> list[str]:
    """Infer usable environmental value columns from a long hourly/as-of table.

    This keeps the standard known variables first, then adds any other
    numeric-like columns that are not obvious IDs/coordinates/join keys.
    """
    if df is None:
        return []
    x = df.copy()
    preferred_cols = list(preferred_cols or [])

    nonvalue_exact = {
        "date",
        "network",
        "station_id",
        "station",
        "id",
        "_id",
        "module_final",
        "src_id",
        "station_id_alias_used",
        "station_id_raw_for_join",
        "station_id_meta_original",
        "station_lat",
        "station_long",
        "station_lon",
        "lat",
        "lon",
        "long",
        "latitude",
        "longitude",
    }

    cols: list[str] = []
    seen: set[str] = set()

    def _maybe_add(col: str) -> None:
        if col in seen:
            return
        low = str(col).strip().lower()
        if low in nonvalue_exact or low.endswith("_id") or is_coordinate_like_feature(str(col)):
            return
        cols.append(col)
        seen.add(col)

    for c in preferred_cols:
        if c in x.columns:
            _maybe_add(c)

    for c in x.columns:
        if c in seen:
            continue
        low = str(c).strip().lower()
        if low in nonvalue_exact or low.endswith("_id") or is_coordinate_like_feature(str(c)):
            continue
        s = x[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            _maybe_add(c)
            continue
        if allow_numericish_objects:
            sample = s.dropna().head(250)
            if len(sample) == 0:
                continue
            coerced = pd.to_numeric(sample, errors="coerce")
            if float(coerced.notna().mean()) >= 0.8:
                _maybe_add(c)

    return cols


def _rename_pair_table_for_patch(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Adapt this script's target/source pair schema to the helper-patch schema."""
    rename_map = {
        "target_station_id": "cws_station_id",
        "target_network": "cws_network",
        "source_station_id": "ows_station_id",
        "source_network": "ows_network",
        "ref_rank": "ows_rank",
        "ref_dist_km": "ows_dist_km",
        "ref_weight_idw": "ows_weight_idw",
    }
    cols = [c for c in rename_map if c in pair_table.columns]
    out = pair_table.rename(columns={c: rename_map[c] for c in cols}).copy()
    return out


def merge_local_ows_satellite_context(
    frame: pd.DataFrame,
    pair_table: pd.DataFrame,
    ows_env: pd.DataFrame | None,
    ows_station_ids: Iterable[str],
    value_col: str,
    feature_prefix: str,
    tolerance: str,
    direction: str,
    age_unit: str,
    chunk_freq: str | None = "M",
    cache_path: str | Path | None = None,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    """Attach local OWS-side LST/NDVI context features to a target frame.

    The target frame may be OWS rows or CWS rows. The pair table determines
    whether the local OWS context is exact leave-one-out or full CWS-vs-OWS.
    """
    helper = _require_qcb()
    if ows_env is None or value_col not in ows_env.columns:
        return frame, None

    actual_path = None
    if cache_path is not None:
        cache_path = Path(cache_path)

    if cache_path is not None and cache_path.exists() and not rebuild:
        local_ctx = load_dataframe_auto(cache_path)
        actual_path = str(cache_path)
    else:
        pkl_fallback = cache_path.with_suffix(".pkl") if cache_path is not None else None
        if pkl_fallback is not None and pkl_fallback.exists() and not rebuild:
            local_ctx = load_dataframe_auto(pkl_fallback)
            actual_path = str(pkl_fallback)
        else:
            env_asof = helper.prepare_ows_env_asof_table(
                ows_env=ows_env,
                ows_station_ids=ows_station_ids,
                dates=frame["date"],
                value_col=value_col,
                tolerance=tolerance,
                direction=direction,
                age_unit=age_unit,
            )
            local_ctx = helper.build_local_ows_env_features(
                cws_keys=frame[KEY_COLS],
                ows_env_hourly=env_asof,
                pair_table=_rename_pair_table_for_patch(pair_table),
                value_col=value_col,
                feature_prefix=feature_prefix,
                age_col=f"{value_col}_age_{age_unit}",
                chunk_freq=chunk_freq,
            )
            if cache_path is not None:
                actual_written = save_dataframe_auto(local_ctx, cache_path)
                actual_path = str(actual_written)

    merged = frame.merge(local_ctx, on=KEY_COLS, how="left", validate="one_to_one")
    return downcast(merged), actual_path

def _safe_json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (Path,)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def robust_sigma(values: Sequence[float], sigma_floor: float = 0.20) -> float:
    """Robust residual standard deviation estimate using 1.4826*MAD."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(arr)
    if not np.isfinite(sig) or sig <= 0:
        sig = sigma_floor
    return float(max(sig, sigma_floor))


def residual_metrics(values: Sequence[float]) -> dict:
    r = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    if r.size == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs": np.nan,
            "p90_abs": np.nan,
            "p95_abs": np.nan,
            "p99_abs": np.nan,
        }
    abs_r = np.abs(r)
    return {
        "n": int(r.size),
        "bias": float(np.mean(r)),
        "mae": float(np.mean(abs_r)),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "median_abs": float(np.quantile(abs_r, 0.50)),
        "p90_abs": float(np.quantile(abs_r, 0.90)),
        "p95_abs": float(np.quantile(abs_r, 0.95)),
        "p99_abs": float(np.quantile(abs_r, 0.99)),
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").astype("float64")
    w = pd.to_numeric(weights, errors="coerce").astype("float64")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def _first_existing(cols: Iterable[str], candidates: Sequence[str]) -> str | None:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def build_reference_pair_table(
    target_meta: pd.DataFrame,
    source_ows_meta: pd.DataFrame,
    k_nearest: int = 8,
    max_radius_km: float | None = 8.0,
    fallback_to_k_nearest: bool = True,
    idw_power: float = 2.0,
    eps_km: float = 0.05,
    exclude_self: bool = False,
) -> pd.DataFrame:
    """Build target-station -> source-OWS station pairs.

    For OWS reference validation, call this with ``target_meta=meta_ows`` and
    ``exclude_self=True``.  That gives exact leave-one-station-out IDW/local
    features: the hidden OWS station is never used as one of its own anchors.

    For CWS application, call with ``target_meta=meta_cws`` and
    ``exclude_self=False``.
    """
    helper = _require_qcb()
    required = {"station_id", "network", "station_lat", "station_long"}
    missing_target = required - set(target_meta.columns)
    missing_source = required - set(source_ows_meta.columns)
    if missing_target:
        raise KeyError(f"target_meta missing required columns: {sorted(missing_target)}")
    if missing_source:
        raise KeyError(f"source_ows_meta missing required columns: {sorted(missing_source)}")

    target = target_meta.copy()
    source = source_ows_meta.copy()
    target["station_id"] = clean_station_id(target["station_id"])
    source["station_id"] = clean_station_id(source["station_id"])
    target["network"] = target["network"].map(normalize_network_name)
    source["network"] = source["network"].map(normalize_network_name)
    target = target.dropna(subset=["station_lat", "station_long"]).reset_index(drop=True)
    source = source.dropna(subset=["station_lat", "station_long"]).reset_index(drop=True)

    if len(target) == 0 or len(source) == 0:
        raise ValueError("Need at least one target station and one source OWS station with coordinates.")

    num_meta_cols = [
        "elev_meters",
        "building_height_m",
        "LC_buffer_fraction",
        "LCZ_buffer_fraction",
    ]
    cat_meta_cols = [
        "LC_point",
        "LC_point_lg",
        "LC_buffer",
        "LC_buffer_lg",
        "LCZ_point",
        "LCZ_point_lg",
        "LCZ_buffer",
        "LCZ_buffer_lg",
    ]

    src_lat = source["station_lat"].to_numpy(dtype="float64")
    src_lon = source["station_long"].to_numpy(dtype="float64")

    records: list[dict] = []
    for _, trow in target.iterrows():
        dist = helper.haversine_km(float(trow["station_lat"]), float(trow["station_long"]), src_lat, src_lon)
        order = np.argsort(dist)

        if exclude_self:
            same_station = (
                (source["station_id"].astype(str).to_numpy() == str(trow["station_id"]))
                & (source["network"].astype(str).to_numpy() == str(trow["network"]))
            )
            order = np.array([idx for idx in order if not same_station[idx]], dtype=int)

        if max_radius_km is not None:
            in_radius = np.array([idx for idx in order if dist[idx] <= max_radius_km], dtype=int)
        else:
            in_radius = order

        if len(in_radius) == 0 and fallback_to_k_nearest:
            selected = order[:k_nearest]
        else:
            selected = in_radius[:k_nearest]

        if len(selected) == 0:
            continue

        raw_w = 1.0 / np.maximum(dist[selected], eps_km) ** idw_power
        weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones_like(raw_w) / len(raw_w)

        for rank, (src_idx, weight) in enumerate(zip(selected, weights), start=1):
            srow = source.iloc[int(src_idx)]
            rec: dict = {
                "target_station_id": str(trow["station_id"]),
                "target_network": normalize_network_name(trow["network"]),
                "source_station_id": str(srow["station_id"]),
                "source_network": normalize_network_name(srow["network"]),
                "ref_rank": int(rank),
                "ref_dist_km": float(dist[int(src_idx)]),
                "ref_weight_idw": float(weight),
            }



            for c in num_meta_cols:
                if c in target.columns:
                    rec[f"target_{c}"] = trow.get(c, np.nan)
                if c in source.columns:
                    rec[f"source_{c}"] = srow.get(c, np.nan)
                if c in target.columns and c in source.columns:
                    tv = pd.to_numeric(pd.Series([trow.get(c, np.nan)]), errors="coerce").iloc[0]
                    sv = pd.to_numeric(pd.Series([srow.get(c, np.nan)]), errors="coerce").iloc[0]
                    rec[f"target_minus_source_{c}"] = tv - sv if pd.notna(tv) and pd.notna(sv) else np.nan
                    rec[f"abs_target_minus_source_{c}"] = abs(tv - sv) if pd.notna(tv) and pd.notna(sv) else np.nan

            for c in cat_meta_cols:
                if c in target.columns:
                    rec[f"target_{c}"] = str(trow.get(c, "Missing"))
                if c in source.columns:
                    rec[f"source_{c}"] = str(srow.get(c, "Missing"))
                if c in target.columns and c in source.columns:
                    rec[f"same_{c}"] = int(str(trow.get(c, "")) == str(srow.get(c, "")))

            records.append(rec)

    out = pd.DataFrame(records)
    if len(out) == 0:
        raise ValueError("Reference pair table is empty. Check coordinates/radius settings.")
    return downcast(out)


def build_pair_static_summary(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse target-source pair metadata to one row per target station."""
    p = pair_table.copy()
    group_cols = ["target_station_id", "target_network"]
    p["ref_weight_idw"] = pd.to_numeric(p["ref_weight_idw"], errors="coerce")
    p["ref_dist_km"] = pd.to_numeric(p["ref_dist_km"], errors="coerce")

    base = (
        p.groupby(group_cols, as_index=False, observed=True)
        .agg(
            ref_pair_n=("source_station_id", "count"),
            ref_min_dist_km=("ref_dist_km", "min"),
            ref_mean_dist_km=("ref_dist_km", "mean"),
            ref_max_dist_km=("ref_dist_km", "max"),
            ref_sum_idw_weight=("ref_weight_idw", "sum"),
        )
    )


    nearest_cols = [
        "source_station_id",
        "ref_dist_km",
        "target_minus_source_elev_meters",
        "abs_target_minus_source_elev_meters",
        "target_minus_source_building_height_m",
        "abs_target_minus_source_building_height_m",
    ]
    nearest_cols = [c for c in nearest_cols if c in p.columns]
    nearest = (
        p.sort_values(group_cols + ["ref_rank", "ref_dist_km"])
        .groupby(group_cols, as_index=False, observed=True)
        .first()[group_cols + nearest_cols]
    )
    nearest = nearest.rename(
        columns={
            "source_station_id": "ref_nearest_source_station_id",
            "ref_dist_km": "ref_nearest_dist_km",
            "target_minus_source_elev_meters": "ref_nearest_elev_diff_m",
            "abs_target_minus_source_elev_meters": "ref_nearest_abs_elev_diff_m",
            "target_minus_source_building_height_m": "ref_nearest_building_height_diff_m",
            "abs_target_minus_source_building_height_m": "ref_nearest_abs_building_height_diff_m",
        }
    )
    out = base.merge(nearest, on=group_cols, how="left", validate="one_to_one")


    weighted_specs = [
        ("source_elev_meters", "ref_idw_source_elev_m"),
        ("source_building_height_m", "ref_idw_source_building_height_m"),
        ("source_LC_buffer_fraction", "ref_idw_source_LC_buffer_fraction"),
        ("source_LCZ_buffer_fraction", "ref_idw_source_LCZ_buffer_fraction"),
        ("abs_target_minus_source_elev_meters", "ref_idw_abs_elev_diff_m"),
        ("abs_target_minus_source_building_height_m", "ref_idw_abs_building_height_diff_m"),
    ]

    for src_col, out_col in weighted_specs:
        if src_col not in p.columns:
            continue
        tmp_src = p[group_cols + [src_col, "ref_weight_idw"]].dropna(subset=[src_col]).copy()
        tmp_src["_wv"] = pd.to_numeric(tmp_src[src_col], errors="coerce") * pd.to_numeric(
            tmp_src["ref_weight_idw"], errors="coerce"
        )
        tmp_src["_w_nonmissing"] = pd.to_numeric(tmp_src["ref_weight_idw"], errors="coerce").where(
            pd.to_numeric(tmp_src[src_col], errors="coerce").notna(), 0.0
        )
        tmp = (
            tmp_src.groupby(group_cols, as_index=False, observed=True)
            .agg(_wv_sum=("_wv", "sum"), _w_sum=("_w_nonmissing", "sum"))
        )
        tmp[out_col] = tmp["_wv_sum"] / tmp["_w_sum"].replace(0, np.nan)
        tmp = tmp[group_cols + [out_col]]
        out = out.merge(tmp, on=group_cols, how="left", validate="one_to_one")


    for c in [col for col in p.columns if col.startswith("same_")]:
        tmp = p.groupby(group_cols, as_index=False, observed=True)[c].mean().rename(columns={c: f"ref_frac_{c}"})
        out = out.merge(tmp, on=group_cols, how="left", validate="one_to_one")

    return downcast(out)


def build_reference_local_features(
    target_keys: pd.DataFrame,
    source_ows_long: pd.DataFrame,
    pair_table: pd.DataFrame,
    source_value_col: str = "temp_raw",
    k_nearest: int | None = None,
    chunk_freq: str | None = "M",
) -> pd.DataFrame:
    """Create local OWS reference features for each target station-hour.

    The same function is used for OWS LOO rows and CWS rows.  OWS LOO behavior
    is controlled entirely by the pair table: if the pair table excludes the
    target station, no leakage is possible in IDW/local features.
    """
    required_keys = set(KEY_COLS)
    if required_keys - set(target_keys.columns):
        raise KeyError(f"target_keys must contain {KEY_COLS}")
    required_pairs = {
        "target_station_id",
        "target_network",
        "source_station_id",
        "source_network",
        "ref_rank",
        "ref_dist_km",
        "ref_weight_idw",
    }
    missing_pairs = required_pairs - set(pair_table.columns)
    if missing_pairs:
        raise KeyError(f"pair_table missing required columns: {sorted(missing_pairs)}")

    keys = target_keys[KEY_COLS].drop_duplicates().copy()
    keys["date"] = ensure_utc_datetime(keys["date"])
    keys["station_id"] = clean_station_id(keys["station_id"])
    keys["network"] = keys["network"].map(normalize_network_name)
    keys = keys.dropna(subset=["date", "station_id"])

    src = source_ows_long[["date", "station_id", "network", source_value_col]].copy()
    src["date"] = ensure_utc_datetime(src["date"])
    src["station_id"] = clean_station_id(src["station_id"])
    src["network"] = src["network"].map(normalize_network_name)
    src[source_value_col] = pd.to_numeric(src[source_value_col], errors="coerce", downcast="float")
    src = src.dropna(subset=["date", "station_id", "network"])

    pairs = pair_table.copy()
    pairs["target_station_id"] = clean_station_id(pairs["target_station_id"])
    pairs["source_station_id"] = clean_station_id(pairs["source_station_id"])
    pairs["target_network"] = pairs["target_network"].map(normalize_network_name)
    pairs["source_network"] = pairs["source_network"].map(normalize_network_name)
    for c in ["ref_rank", "ref_dist_km", "ref_weight_idw"]:
        pairs[c] = pd.to_numeric(pairs[c], errors="coerce")
    if k_nearest is not None:
        pairs = pairs[pairs["ref_rank"] <= k_nearest].copy()

    static_summary = build_pair_static_summary(pairs)
    static_summary = static_summary.rename(
        columns={"target_station_id": "station_id", "target_network": "network"}
    )

    if chunk_freq is None:
        chunks = [("__all__", keys)]
    else:
        tmp = keys.copy()
        tmp["_chunk"] = tmp["date"].dt.to_period(chunk_freq).astype(str)
        chunks = list(tmp.groupby("_chunk", sort=True))

    group_cols = KEY_COLS
    parts = []

    src_renamed = src.rename(
        columns={
            "station_id": "source_station_id",
            "network": "source_network",
            source_value_col: "_source_value",
        }
    )

    for chunk_name, key_sub in chunks:
        if "_chunk" in key_sub.columns:
            key_sub = key_sub.drop(columns=["_chunk"])
        if len(key_sub) == 0:
            continue

        chunk_dates = pd.Index(key_sub["date"].dropna().unique())
        src_sub = src_renamed[src_renamed["date"].isin(chunk_dates)].copy()

        joined = key_sub.merge(
            pairs,
            left_on=["station_id", "network"],
            right_on=["target_station_id", "target_network"],
            how="left",
            validate="many_to_many",
        )
        joined = joined.merge(
            src_sub,
            on=["date", "source_station_id", "source_network"],
            how="left",
            validate="many_to_many",
        )

        joined["_source_value"] = pd.to_numeric(joined["_source_value"], errors="coerce")
        joined["_valid_value"] = joined["_source_value"].notna()
        joined["_w"] = pd.to_numeric(joined["ref_weight_idw"], errors="coerce").astype("float64")
        joined["_w_valid"] = joined["_w"].where(joined["_valid_value"], 0.0)
        joined["_wv"] = joined["_w"] * joined["_source_value"].fillna(0.0).astype("float64")

        basic = (
            joined.groupby(group_cols, as_index=False, observed=True)["_source_value"]
            .agg(
                ref_local_n="count",
                ref_local_mean_c="mean",
                ref_local_median_c="median",
                ref_local_std_c="std",
                ref_local_min_c="min",
                ref_local_max_c="max",
            )
        )
        basic["ref_local_spread_c"] = basic["ref_local_max_c"] - basic["ref_local_min_c"]

        weighted = (
            joined.groupby(group_cols, as_index=False, observed=True)
            .agg(
                ref_idw_weight_sum=("_w_valid", "sum"),
                _ref_wv_sum=("_wv", "sum"),
            )
        )
        denom = weighted["ref_idw_weight_sum"].replace(0, np.nan)
        weighted["ref_idw_mu"] = weighted["_ref_wv_sum"] / denom
        weighted = weighted.drop(columns=["_ref_wv_sum"])

        valid_joined = joined[joined["_valid_value"]].copy()
        if len(valid_joined):
            nearest = (
                valid_joined.sort_values(group_cols + ["ref_rank", "ref_dist_km"])
                .groupby(group_cols, as_index=False, observed=True)
                .first()[group_cols + ["_source_value", "ref_rank", "ref_dist_km", "source_station_id"]]
                .rename(
                    columns={
                        "_source_value": "ref_nearest_mu",
                        "ref_rank": "ref_nearest_rank_with_value",
                        "ref_dist_km": "ref_nearest_value_dist_km",
                        "source_station_id": "ref_nearest_value_source_station_id",
                    }
                )
            )
        else:
            nearest = basic[group_cols].copy()
            nearest["ref_nearest_mu"] = np.nan
            nearest["ref_nearest_rank_with_value"] = np.nan
            nearest["ref_nearest_value_dist_km"] = np.nan
            nearest["ref_nearest_value_source_station_id"] = np.nan

        out = key_sub[group_cols].drop_duplicates().copy()
        out = out.merge(basic, on=group_cols, how="left", validate="one_to_one")
        out = out.merge(weighted, on=group_cols, how="left", validate="one_to_one")
        out = out.merge(nearest, on=group_cols, how="left", validate="one_to_one")
        parts.append(out)

    if parts:
        local = pd.concat(parts, ignore_index=True)
    else:
        local = keys.copy()

    local = keys.merge(local, on=group_cols, how="left", validate="one_to_one")
    local = local.merge(static_summary, on=["station_id", "network"], how="left", validate="many_to_one")

    if "ref_local_n" in local.columns:
        local["ref_local_n"] = local["ref_local_n"].fillna(0).astype("int16")
    else:
        local["ref_local_n"] = 0

    return downcast(local)


def attach_hourly_environment(
    base: pd.DataFrame,
    exact_env_tables: Sequence[pd.DataFrame | None] | None = None,
    asof_env_specs: Sequence[Mapping] | None = None,
) -> pd.DataFrame:
    """Attach station-hour environmental features to a target frame.

    ``exact_env_tables`` are merged exactly on date/station/network (good for
    ERA5-Land hourly data). ``asof_env_specs`` use qcb.merge_stationwise_asof_features
    and are appropriate for sparse LST/NDVI tables.
    """
    helper = _require_qcb()
    out = base.copy()
    out["date"] = ensure_utc_datetime(out["date"])
    out["station_id"] = clean_station_id(out["station_id"])
    out["network"] = out["network"].map(normalize_network_name)

    if exact_env_tables:
        for env in exact_env_tables:
            if env is None or len(env) == 0:
                continue
            env_x = env.copy()
            env_x["date"] = ensure_utc_datetime(env_x["date"])
            env_x["station_id"] = clean_station_id(env_x["station_id"])
            env_x["network"] = env_x["network"].map(normalize_network_name)
            value_cols = [c for c in env_x.columns if c not in KEY_COLS]

            value_cols = [c for c in value_cols if c not in out.columns]
            if not value_cols:
                continue
            out = out.merge(env_x[KEY_COLS + value_cols], on=KEY_COLS, how="left", validate="many_to_one")

    if asof_env_specs:
        for spec in asof_env_specs:
            features = spec.get("features")
            if features is None or len(features) == 0:
                continue
            value_cols = list(spec.get("value_cols", []))
            value_cols = [c for c in value_cols if c in features.columns and c not in out.columns]
            if not value_cols:
                continue
            out = helper.merge_stationwise_asof_features(
                base=out,
                features=features,
                value_cols=value_cols,
                rename_map=spec.get("rename_map"),
                tolerance=spec.get("tolerance", "14D"),
                direction=spec.get("direction", "backward"),
                add_missing_indicators=spec.get("add_missing_indicators", True),
                add_age=spec.get("add_age", True),
                age_unit=spec.get("age_unit", "hours"),
            )

    return downcast(out)


def add_reference_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features without using target temperature."""
    x = df.copy()
    x["date"] = ensure_utc_datetime(x["date"])
    x["hour"] = x["date"].dt.hour.astype("int8")
    x["dayofyear"] = x["date"].dt.dayofyear.astype("int16")
    x["month"] = x["date"].dt.month.astype("int8")
    x["dow"] = x["date"].dt.dayofweek.astype("int8")
    x["is_weekend"] = (x["dow"] >= 5).astype("int8")
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24).astype("float32")
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24).astype("float32")
    x["doy_sin"] = np.sin(2 * np.pi * x["dayofyear"] / 365.25).astype("float32")
    x["doy_cos"] = np.cos(2 * np.pi * x["dayofyear"] / 365.25).astype("float32")
    return x


def add_reference_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Interactions that do not use target observed temperature."""
    x = df.copy()

    if {"u10_ms", "v10_ms"}.issubset(x.columns):
        x["wind_speed"] = np.sqrt(x["u10_ms"] ** 2 + x["v10_ms"] ** 2).astype("float32")

    if {"t2m_c", "d2m_c"}.issubset(x.columns):
        x["era5_t_td_spread"] = (x["t2m_c"] - x["d2m_c"]).astype("float32")

    if {"ref_idw_mu", "t2m_c"}.issubset(x.columns):
        x["ref_idw_minus_era5"] = (x["ref_idw_mu"] - x["t2m_c"]).astype("float32")
        x["ref_idw_minus_era5_abs"] = (x["ref_idw_mu"] - x["t2m_c"]).abs().astype("float32")

    for lst_col in ["LST_C", "LST_C_asof", "LST_C_gapfill_3h", "LST_C_gapfill_clim"]:
        if {"ref_idw_mu", lst_col}.issubset(x.columns):
            safe = lst_col.lower()
            x[f"ref_idw_minus_{safe}"] = (x["ref_idw_mu"] - x[lst_col]).astype("float32")
            x[f"ref_idw_minus_{safe}_abs"] = (x["ref_idw_mu"] - x[lst_col]).abs().astype("float32")
        if {"t2m_c", lst_col}.issubset(x.columns):
            safe = lst_col.lower()
            x[f"era5_minus_{safe}"] = (x["t2m_c"] - x[lst_col]).astype("float32")

    if {"ssrd_wm2", "NDVI"}.issubset(x.columns):
        x["solar_x_ndvi"] = (x["ssrd_wm2"] * x["NDVI"]).astype("float32")

    if {"wind_speed", "NDVI"}.issubset(x.columns):
        x["wind_x_ndvi"] = (x["wind_speed"] * x["NDVI"]).astype("float32")

    for lst_col in ["LST_C", "LST_C_asof", "LST_C_gapfill_3h", "LST_C_gapfill_clim", "ows_idw_lst", "ows_nearest_lst"]:
        if {"ssrd_wm2", lst_col}.issubset(x.columns):
            safe = lst_col.lower()
            x[f"solar_x_{safe}"] = (x["ssrd_wm2"] * x[lst_col]).astype("float32")

    if {"NDVI", "LST_C_gapfill_clim"}.issubset(x.columns):
        x["ndvi_x_lst_gapfill_clim"] = (x["NDVI"] * x["LST_C_gapfill_clim"]).astype("float32")

    if {"building_height_m", "ssrd_wm2"}.issubset(x.columns):
        x["building_x_solar"] = (x["building_height_m"] * x["ssrd_wm2"]).astype("float32")

    if "ssrd_wm2" in x.columns:
        x["is_daylight"] = (pd.to_numeric(x["ssrd_wm2"], errors="coerce") > 5).astype("int8")

    for cat_col in ["LCZ_point_lg", "LCZ_buffer_lg"]:
        if cat_col in x.columns and "is_daylight" in x.columns:
            x[f"{cat_col}_x_daylight"] = (
                x[cat_col].astype("string").fillna("Missing") + "__" + x["is_daylight"].astype(str)
            ).astype("string")

    return downcast(x)


def build_target_reference_frame(
    target_long: pd.DataFrame,
    target_meta: pd.DataFrame,
    source_ows_long: pd.DataFrame,
    source_ows_meta: pd.DataFrame,
    exact_env_tables: Sequence[pd.DataFrame | None] | None = None,
    asof_env_specs: Sequence[Mapping] | None = None,
    k_nearest: int = 8,
    max_radius_km: float | None = 8.0,
    fallback_to_k_nearest: bool = True,
    idw_power: float = 2.0,
    eps_km: float = 0.05,
    exclude_self: bool = False,
    chunk_freq: str | None = "M",
    pair_table_path: str | Path | None = None,
    local_features_path: str | Path | None = None,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one reference-model design matrix for OWS or CWS target rows."""
    target = target_long.copy()
    target["date"] = ensure_utc_datetime(target["date"])
    target["station_id"] = clean_station_id(target["station_id"])
    target["network"] = target["network"].map(normalize_network_name)
    target["temp_raw"] = pd.to_numeric(target["temp_raw"], errors="coerce", downcast="float")
    target = target.dropna(subset=["date", "station_id", "network", "temp_raw"]).copy()

    meta = target_meta.copy()
    meta["station_id"] = clean_station_id(meta["station_id"])
    meta["network"] = meta["network"].map(normalize_network_name)
    meta = meta.drop_duplicates(["station_id", "network"]).copy()


    if pair_table_path is not None:
        pair_table_path = Path(pair_table_path)
    if pair_table_path is not None and pair_table_path.exists() and not rebuild:
        print("Loading pair table:", pair_table_path)
        pair_table = pd.read_csv(pair_table_path)
    else:
        pair_table = build_reference_pair_table(
            target_meta=meta,
            source_ows_meta=source_ows_meta,
            k_nearest=k_nearest,
            max_radius_km=max_radius_km,
            fallback_to_k_nearest=fallback_to_k_nearest,
            idw_power=idw_power,
            eps_km=eps_km,
            exclude_self=exclude_self,
        )
        if pair_table_path is not None:
            pair_table_path.parent.mkdir(parents=True, exist_ok=True)
            pair_table.to_csv(pair_table_path, index=False)


    actual_local_path = None
    if local_features_path is not None:
        local_features_path = Path(local_features_path)
    if local_features_path is not None and local_features_path.exists() and not rebuild:
        print("Loading local reference features:", local_features_path)
        local = load_dataframe_auto(local_features_path)
        actual_local_path = local_features_path
    else:
        pkl_fallback = local_features_path.with_suffix(".pkl") if local_features_path is not None else None
        if pkl_fallback is not None and pkl_fallback.exists() and not rebuild:
            print("Loading local reference features:", pkl_fallback)
            local = load_dataframe_auto(pkl_fallback)
            actual_local_path = pkl_fallback
        else:
            local = build_reference_local_features(
                target_keys=target[KEY_COLS],
                source_ows_long=source_ows_long,
                pair_table=pair_table,
                source_value_col="temp_raw",
                k_nearest=k_nearest,
                chunk_freq=chunk_freq,
            )
            if local_features_path is not None:
                actual_local_path = save_dataframe_auto(local, local_features_path)

    frame = target.merge(meta, on=["station_id", "network"], how="left", validate="many_to_one")
    frame = frame.merge(local, on=KEY_COLS, how="left", validate="one_to_one")
    frame = attach_hourly_environment(frame, exact_env_tables=exact_env_tables, asof_env_specs=asof_env_specs)
    frame = add_reference_temporal_features(frame)
    frame = add_reference_interactions(frame)

    return downcast(frame), pair_table, local


def add_reference_residual_columns(
    df: pd.DataFrame,
    mu_col: str,
    method: str,
    temp_col: str = "temp_raw",
) -> pd.DataFrame:
    """Add method-specific residual columns."""
    x = df.copy()
    x[f"ref_mu_{method}"] = pd.to_numeric(x[mu_col], errors="coerce")
    x[f"ref_resid_{method}"] = pd.to_numeric(x[temp_col], errors="coerce") - x[f"ref_mu_{method}"]
    x[f"abs_ref_resid_{method}"] = x[f"ref_resid_{method}"].abs()
    return downcast(x)


def _calibration_subset(
    df: pd.DataFrame,
    residual_col: str,
    valid_start: str | None,
    calibration_mode: str,
) -> pd.DataFrame:
    x = df.copy()
    x["date"] = ensure_utc_datetime(x["date"])
    x[residual_col] = pd.to_numeric(x[residual_col], errors="coerce")
    x = x.dropna(subset=[residual_col, "date"]).copy()

    if calibration_mode == "time_train":
        if valid_start is None:
            raise ValueError("valid_start is required when calibration_mode='time_train'")
        cutoff = pd.Timestamp(valid_start, tz="UTC")
        x = x[x["date"] < cutoff].copy()
    elif calibration_mode == "all_ows_reference":
        pass
    else:
        raise ValueError("calibration_mode must be 'time_train' or 'all_ows_reference'")

    if len(x) == 0:
        raise ValueError("Reference calibration subset is empty.")
    return x


def fit_reference_uncertainty(
    ows_reference_df: pd.DataFrame,
    residual_col: str,
    valid_start: str | None = "2021-10-01",
    calibration_mode: str = "time_train",
    sigma_floor_c: float = 0.20,
    min_group_n: int = 200,
    abnormal_quantile: float = 0.995,
    ambiguous_lower_quantile: float = 0.95,
) -> dict:
    """Fit robust reference uncertainty and abnormal-residual thresholds.

    Sigma is calibrated from both time context and local reference difficulty.
    The time component gives a stable baseline, while difficulty tables capture
    cases where the local OWS problem is intrinsically harder, such as sparse
    nearby OWS coverage, large nearest-station distance, strong local spread,
    or LCZ mismatch. During application we use the maximum available sigma from
    these sources, which is conservative and avoids overconfident z-scores in
    heterogeneous urban settings.
    """
    cal = _calibration_subset(ows_reference_df, residual_col, valid_start, calibration_mode)
    cal = add_reference_temporal_features(cal)
    cal = add_reference_summary_bins(cal)

    global_sigma = robust_sigma(cal[residual_col], sigma_floor=sigma_floor_c)

    def _group_sigma(group_cols: list[str], name: str) -> pd.DataFrame:
        rows = []
        for keys, g in cal.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row["n"] = int(g[residual_col].notna().sum())
            row[name] = robust_sigma(g[residual_col], sigma_floor=sigma_floor_c) if row["n"] >= min_group_n else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    hour_sigma = _group_sigma(["hour"], "sigma_hour")
    month_sigma = _group_sigma(["month"], "sigma_month")
    month_hour_sigma = _group_sigma(["month", "hour"], "sigma_month_hour")

    difficulty_specs = [
        (["ref_local_n_bin"], "sigma_ref_local_n_bin"),
        (["ref_nearest_dist_bin"], "sigma_ref_nearest_dist_bin"),
        (["ref_local_spread_bin"], "sigma_ref_local_spread_bin"),
        (["ref_lcz_mismatch_bin"], "sigma_ref_lcz_mismatch_bin"),
        (["ref_lc_mismatch_bin"], "sigma_ref_lc_mismatch_bin"),
        (["ref_abs_elev_diff_bin"], "sigma_ref_abs_elev_diff_bin"),
        (["ref_abs_building_height_diff_bin"], "sigma_ref_abs_building_height_diff_bin"),
        (["ref_support_class"], "sigma_ref_support_class"),
    ]
    difficulty_tables = {}
    for cols, name in difficulty_specs:
        if all(c in cal.columns for c in cols):
            difficulty_tables[name] = _group_sigma(cols, name)

    cal_with_sigma = apply_reference_uncertainty(
        cal,
        calibration={
            "global_sigma": global_sigma,
            "hour_sigma": hour_sigma,
            "month_sigma": month_sigma,
            "month_hour_sigma": month_hour_sigma,
            "difficulty_sigma_tables": difficulty_tables,
            "sigma_floor_c": sigma_floor_c,
        },
        residual_col=residual_col,
        sigma_col="_ref_sigma_tmp",
        z_col="_ref_z_tmp",
    )
    abs_resid = cal_with_sigma[residual_col].abs()
    abs_z = cal_with_sigma["_ref_z_tmp"].abs()
    abs_resid_threshold = float(abs_resid.quantile(abnormal_quantile))
    abs_z_threshold = float(abs_z.quantile(abnormal_quantile))
    abs_resid_lower = float(abs_resid.quantile(ambiguous_lower_quantile))
    abs_z_lower = float(abs_z.quantile(ambiguous_lower_quantile))

    return {
        "calibration_mode": calibration_mode,
        "valid_start": valid_start,
        "n_calibration_rows": int(len(cal)),
        "abnormal_quantile": float(abnormal_quantile),
        "ambiguous_lower_quantile": float(ambiguous_lower_quantile),
        "sigma_floor_c": float(sigma_floor_c),
        "global_sigma": float(global_sigma),
        "hour_sigma": hour_sigma,
        "month_sigma": month_sigma,
        "month_hour_sigma": month_hour_sigma,
        "difficulty_sigma_tables": difficulty_tables,
        "thresholds": {
            "abs_resid_c": abs_resid_threshold,
            "abs_z": abs_z_threshold,
            "abs_resid_low_c": abs_resid_lower,
            "abs_z_low": abs_z_lower,
        },
    }


def apply_reference_uncertainty(
    df: pd.DataFrame,
    calibration: Mapping,
    residual_col: str,
    sigma_col: str = "ref_sigma",
    z_col: str = "ref_z",
) -> pd.DataFrame:
    """Apply fitted robust sigma lookup to a reference dataframe.

    Base sigma comes from temporal context. Difficulty-conditioned sigmas are
    then merged and used to conservatively inflate sigma via row-wise maxima.
    """
    x = add_reference_temporal_features(df)
    x = add_reference_summary_bins(x)
    global_sigma = float(calibration.get("global_sigma", np.nan))
    sigma_floor = float(calibration.get("sigma_floor_c", 0.20))

    x[sigma_col] = global_sigma

    month_sigma = calibration.get("month_sigma")
    if isinstance(month_sigma, pd.DataFrame) and len(month_sigma):
        x = x.merge(month_sigma[["month", "sigma_month"]], on="month", how="left", validate="many_to_one")
        x[sigma_col] = x["sigma_month"].combine_first(x[sigma_col])
        x = x.drop(columns=["sigma_month"])

    hour_sigma = calibration.get("hour_sigma")
    if isinstance(hour_sigma, pd.DataFrame) and len(hour_sigma):
        x = x.merge(hour_sigma[["hour", "sigma_hour"]], on="hour", how="left", validate="many_to_one")
        x[sigma_col] = x["sigma_hour"].combine_first(x[sigma_col])
        x = x.drop(columns=["sigma_hour"])

    month_hour_sigma = calibration.get("month_hour_sigma")
    if isinstance(month_hour_sigma, pd.DataFrame) and len(month_hour_sigma):
        x = x.merge(month_hour_sigma[["month", "hour", "sigma_month_hour"]], on=["month", "hour"], how="left", validate="many_to_one")
        x[sigma_col] = x["sigma_month_hour"].combine_first(x[sigma_col])
        x = x.drop(columns=["sigma_month_hour"])

    difficulty_sigma_tables = calibration.get("difficulty_sigma_tables", {})
    if isinstance(difficulty_sigma_tables, Mapping):
        sigma_components = [pd.to_numeric(x[sigma_col], errors="coerce")]
        for sigma_name, table in difficulty_sigma_tables.items():
            if not isinstance(table, pd.DataFrame) or len(table) == 0:
                continue
            join_cols = [c for c in table.columns if c not in {"n", sigma_name}]
            if not join_cols or sigma_name not in table.columns:
                continue
            x = x.merge(table[join_cols + [sigma_name]], on=join_cols, how="left", validate="many_to_one")
            sigma_components.append(pd.to_numeric(x[sigma_name], errors="coerce"))
        if sigma_components:
            x[sigma_col] = pd.concat(sigma_components, axis=1).max(axis=1, skipna=True)
        for sigma_name in difficulty_sigma_tables:
            if sigma_name in x.columns:
                x = x.drop(columns=[sigma_name])

    x[sigma_col] = pd.to_numeric(x[sigma_col], errors="coerce").clip(lower=sigma_floor)
    x[z_col] = pd.to_numeric(x[residual_col], errors="coerce") / x[sigma_col]
    return downcast(x)


def _assign_ambiguous_band(
    abs_value: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    band = pd.Series(index=abs_value.index, dtype="string")
    band.loc[abs_value <= low_threshold] = "high_confidence_low_risk"
    band.loc[abs_value > high_threshold] = "high_confidence_high_risk"
    mid = abs_value.gt(low_threshold) & abs_value.le(high_threshold)
    band.loc[mid] = "ambiguous"
    return band


def standardize_ows_reference_columns(
    df: pd.DataFrame,
    method: str,
    calibration: Mapping,
) -> pd.DataFrame:
    """Create canonical OWS columns ref_mu/ref_sigma/ref_resid/ref_z."""
    mu_col = f"ref_mu_{method}"
    resid_col = f"ref_resid_{method}"
    if mu_col not in df.columns or resid_col not in df.columns:
        raise KeyError(f"Missing method-specific columns for method={method}: {mu_col}, {resid_col}")

    x = df.copy()
    x["ref_mu"] = x[mu_col]
    x["ref_resid"] = x[resid_col]
    x["abs_ref_resid"] = x["ref_resid"].abs()
    x = apply_reference_uncertainty(x, calibration=calibration, residual_col="ref_resid", sigma_col="ref_sigma", z_col="ref_z")

    thresholds = calibration.get("thresholds", {})
    if thresholds:
        high_abs = float(thresholds.get("abs_resid_c", np.inf))
        high_z = float(thresholds.get("abs_z", np.inf))
        low_abs = float(thresholds.get("abs_resid_low_c", high_abs))
        low_z = float(thresholds.get("abs_z_low", high_z))
        x["ows_ref_risk_abs"] = (x["abs_ref_resid"] > high_abs).astype("int8")
        x["ows_ref_risk_z"] = (x["ref_z"].abs() > high_z).astype("int8")
        x["ows_ref_risk_band_abs"] = _assign_ambiguous_band(x["abs_ref_resid"], low_abs, high_abs)
        x["ows_ref_risk_band_z"] = _assign_ambiguous_band(x["ref_z"].abs(), low_z, high_z)
    return downcast(x)


def apply_reference_to_cws(
    cws_frame: pd.DataFrame,
    mu_col: str,
    calibration: Mapping,
) -> pd.DataFrame:
    """Generate requested CWS residual features and residual-risk weak targets."""
    if mu_col not in cws_frame.columns:
        raise KeyError(f"{mu_col!r} is not present in cws_frame")

    x = cws_frame.copy()
    x["ref_mu"] = pd.to_numeric(x[mu_col], errors="coerce")
    x["cws_ref_resid"] = pd.to_numeric(x["temp_raw"], errors="coerce") - x["ref_mu"]
    x["abs_cws_ref_resid"] = x["cws_ref_resid"].abs()
    x = apply_reference_uncertainty(
        x,
        calibration=calibration,
        residual_col="cws_ref_resid",
        sigma_col="ref_sigma",
        z_col="cws_ref_z",
    )

    thresholds = calibration.get("thresholds", {})
    abs_resid_thr = float(thresholds.get("abs_resid_c", np.inf))
    abs_z_thr = float(thresholds.get("abs_z", np.inf))
    x["target_ref_risk_abs"] = (x["abs_cws_ref_resid"] > abs_resid_thr).astype("int8")
    x["target_ref_risk_z"] = (x["cws_ref_z"].abs() > abs_z_thr).astype("int8")
    low_abs_resid_thr = float(thresholds.get("abs_resid_low_c", abs_resid_thr))
    low_abs_z_thr = float(thresholds.get("abs_z_low", abs_z_thr))
    x["target_ref_risk_band_abs"] = _assign_ambiguous_band(x["abs_cws_ref_resid"], low_abs_resid_thr, abs_resid_thr)
    x["target_ref_risk_band_z"] = _assign_ambiguous_band(x["cws_ref_z"].abs(), low_abs_z_thr, abs_z_thr)

    x["target_ref_risk"] = x["target_ref_risk_z"].astype("int8")
    return downcast(x)


def is_identifier_like_feature(col: str) -> bool:
    c = col.lower()
    if c in {"station_id", "network", "date"}:
        return True
    if "source_station_id" in c or c.endswith("_station_id"):
        return True
    if c in {"ref_nearest_value_source_station_id", "ref_nearest_source_station_id"}:
        return True
    return False


def is_forbidden_reference_feature(col: str) -> bool:
    c = col.lower()
    forbidden_exact = {
        "temp_raw",
        "ref_mu",
        "ref_sigma",
        "ref_resid",
        "ref_z",
        "abs_ref_resid",
        "cws_ref_resid",
        "abs_cws_ref_resid",
        "cws_ref_z",
        "target_ref_risk",
        "target_ref_risk_abs",
        "target_ref_risk_z",
    }
    if c in forbidden_exact:
        return True
    forbidden_prefixes = (
        "ref_mu_",
        "ref_resid_",
        "abs_ref_resid_",
        "ows_ref_risk",
    )
    return c.startswith(forbidden_prefixes)


def is_coordinate_like_feature(col: str) -> bool:
    c = col.lower()
    exact = {
        "lat",
        "lon",
        "long",
        "latitude",
        "longitude",
        "station_lat",
        "station_long",
        "station_lon",
        "station_longitude",
    }
    suffixes = ("_lat", "_lon", "_long", "_latitude", "_longitude")
    return c in exact or c.endswith(suffixes)


def numeric_landcover_code_columns(df: pd.DataFrame) -> set[str]:
    """Find redundant LC/LCZ numeric code columns when *_lg labels exist.

    Examples:
      LC_point=50 and LC_point_lg="Built-up" encode the same class identity.
    CatBoost should receive the label/categorical representation by default,
    not the arbitrary numeric class code.
    """
    pairs = [
        ("LC_point", "LC_point_lg"),
        ("LC_buffer", "LC_buffer_lg"),
        ("LCZ_point", "LCZ_point_lg"),
        ("LCZ_buffer", "LCZ_buffer_lg"),
    ]
    return {code for code, label in pairs if code in df.columns and label in df.columns}


def get_reference_regression_feature_columns(
    df: pd.DataFrame,
    include_coordinates: bool = True,
    extra_exclude: Sequence[str] | None = None,
    drop_numeric_landcover_codes: bool = True,
) -> tuple[list[str], list[str]]:
    """Select features for the OWS reference CatBoostRegressor.

    The selection deliberately excludes the target temperature and any residual
    columns.  It also excludes station identifiers, because station-held-out
    validation should test spatial/context generalization rather than ID lookup.
    """
    extra_exclude = set(extra_exclude or [])
    numeric_lc_codes = numeric_landcover_code_columns(df) if drop_numeric_landcover_codes else set()
    feature_cols: list[str] = []

    for c in df.columns:
        if c in extra_exclude:
            continue
        if c in numeric_lc_codes:
            continue
        if is_identifier_like_feature(c) or is_forbidden_reference_feature(c):
            continue
        if not include_coordinates and is_coordinate_like_feature(c):
            continue
        if c == "date":
            continue

        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            feature_cols.append(c)
        elif pd.api.types.is_categorical_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object:


            nunique = s.nunique(dropna=True)
            if nunique <= 100:
                feature_cols.append(c)

    categorical_cols = []
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object:
            categorical_cols.append(c)

    return feature_cols, categorical_cols


def _prepare_catboost_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    X = df[list(feature_cols)].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("Missing").astype(str)
    return X


def train_catboost_reference_cv(
    ows_frame: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    cv_mode: str = "group_kfold",
    n_splits: int = 5,
    iterations: int = 1200,
    learning_rate: float = 0.04,
    depth: int = 8,
    l2_leaf_reg: float = 8.0,
    random_state: int = 42,
    thread_count: int = 15,
    used_ram_limit: str | None = "42gb",
    use_gpu: bool = False,
    verbose: int = 100,
    model_dir: str | Path | None = None,
    model_prefix: str = "catboost_reference",
) -> tuple[pd.DataFrame, list]:
    """Station-held-out CatBoostRegressor predictions for OWS rows.

    ``group_kfold`` is the practical default for 270+ OWS stations.  Use
    ``leave_one_station_out`` for exact station-LOO if runtime permits.
    """
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    x = ows_frame.copy()
    x = x.dropna(subset=["temp_raw"]).copy()
    groups = x["station_id"].astype(str).values
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two OWS stations for station-held-out CatBoost.")

    if cv_mode == "group_kfold":
        n_splits_eff = min(int(n_splits), len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits_eff).split(x, x["temp_raw"], groups=groups)
    elif cv_mode == "leave_one_station_out":
        splitter = LeaveOneGroupOut().split(x, x["temp_raw"], groups=groups)
    else:
        raise ValueError("cv_mode must be 'group_kfold' or 'leave_one_station_out'")

    preds = pd.Series(index=x.index, dtype="float64")
    fold_models = []
    model_dir_path = Path(model_dir) if model_dir is not None else None
    if model_dir_path is not None:
        model_dir_path.mkdir(parents=True, exist_ok=True)

    cat_features = [c for c in categorical_cols if c in feature_cols]

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter, start=1):
        train_part = x.iloc[train_idx]
        valid_part = x.iloc[valid_idx]
        held_groups = sorted(pd.Series(valid_part["station_id"].astype(str)).unique().tolist())
        print(
            f"CatBoost reference fold {fold_idx}: "
            f"train_rows={len(train_part):,}, heldout_rows={len(valid_part):,}, "
            f"heldout_stations={len(held_groups):,}"
        )

        X_train = _prepare_catboost_matrix(train_part, feature_cols, cat_features)
        X_valid = _prepare_catboost_matrix(valid_part, feature_cols, cat_features)

        train_pool = Pool(X_train, label=train_part["temp_raw"].astype(float), cat_features=cat_features)
        valid_pool = Pool(X_valid, label=valid_part["temp_raw"].astype(float), cat_features=cat_features)

        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "l2_leaf_reg": float(l2_leaf_reg),
            "random_seed": int(random_state + fold_idx),
            "od_type": "Iter",
            "od_wait": 100,
            "allow_writing_files": False,
            "verbose": verbose,
            "thread_count": int(thread_count),
        }
        if used_ram_limit:
            params["used_ram_limit"] = used_ram_limit
        if use_gpu:
            params["task_type"] = "GPU"

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        preds.iloc[valid_idx] = model.predict(valid_pool)
        fold_models.append(model)

        if model_dir_path is not None:
            model.save_model(str(model_dir_path / f"{model_prefix}_fold{fold_idx:03d}.cbm"))

    out = ows_frame.copy()
    out["ref_mu_catboost"] = np.nan
    out.loc[preds.index, "ref_mu_catboost"] = preds
    out = add_reference_residual_columns(out, mu_col="ref_mu_catboost", method="catboost")
    return downcast(out), fold_models


def fit_final_catboost_reference_model(
    ows_frame: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    iterations: int = 1200,
    learning_rate: float = 0.04,
    depth: int = 8,
    l2_leaf_reg: float = 8.0,
    random_state: int = 42,
    thread_count: int = 15,
    used_ram_limit: str | None = "42gb",
    use_gpu: bool = False,
    verbose: int = 100,
    model_path: str | Path | None = None,
):
    """Train final CatBoostRegressor on all OWS rows for CWS application."""
    from catboost import CatBoostRegressor, Pool

    x = ows_frame.dropna(subset=["temp_raw"]).copy()
    cat_features = [c for c in categorical_cols if c in feature_cols]
    X = _prepare_catboost_matrix(x, feature_cols, cat_features)
    pool = Pool(X, label=x["temp_raw"].astype(float), cat_features=cat_features)

    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_state),
        "allow_writing_files": False,
        "verbose": verbose,
        "thread_count": int(thread_count),
    }
    if used_ram_limit:
        params["used_ram_limit"] = used_ram_limit
    if use_gpu:
        params["task_type"] = "GPU"

    model = CatBoostRegressor(**params)
    model.fit(pool)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

    return model


def predict_catboost_reference(
    model,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> np.ndarray:
    from catboost import Pool

    cat_features = [c for c in categorical_cols if c in feature_cols]
    X = _prepare_catboost_matrix(df, feature_cols, cat_features)
    pool = Pool(X, cat_features=cat_features)
    return model.predict(pool)


def add_reference_summary_bins(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    if "ref_nearest_abs_elev_diff_m" in x.columns:
        elev = pd.to_numeric(x["ref_nearest_abs_elev_diff_m"], errors="coerce")
    elif "ref_idw_abs_elev_diff_m" in x.columns:
        elev = pd.to_numeric(x["ref_idw_abs_elev_diff_m"], errors="coerce")
    else:
        elev = pd.Series(np.nan, index=x.index)
    x["ref_abs_elev_diff_bin"] = pd.cut(
        elev,
        bins=[-np.inf, 25, 50, 100, 200, np.inf],
        labels=["<=25m", "25-50m", "50-100m", "100-200m", ">200m"],
    ).astype("string")

    if "ref_nearest_abs_building_height_diff_m" in x.columns:
        bh = pd.to_numeric(x["ref_nearest_abs_building_height_diff_m"], errors="coerce")
    elif "ref_idw_abs_building_height_diff_m" in x.columns:
        bh = pd.to_numeric(x["ref_idw_abs_building_height_diff_m"], errors="coerce")
    else:
        bh = pd.Series(np.nan, index=x.index)
    x["ref_abs_building_height_diff_bin"] = pd.cut(
        bh,
        bins=[-np.inf, 2, 5, 10, 20, np.inf],
        labels=["<=2m", "2-5m", "5-10m", "10-20m", ">20m"],
    ).astype("string")

    if "ref_local_n" in x.columns:
        x["ref_local_n_bin"] = pd.cut(
            pd.to_numeric(x["ref_local_n"], errors="coerce"),
            bins=[-np.inf, 0, 2, 4, 6, np.inf],
            labels=["0", "1-2", "3-4", "5-6", ">6"],
        ).astype("string")

    dist_col = _first_existing(x.columns, ["ref_min_dist_km", "ref_nearest_dist_km", "ref_nearest_value_dist_km"])
    if dist_col is not None:
        x["ref_nearest_dist_bin"] = pd.cut(
            pd.to_numeric(x[dist_col], errors="coerce"),
            bins=[-np.inf, 0.5, 1, 2, 5, np.inf],
            labels=["<=0.5km", "0.5-1km", "1-2km", "2-5km", ">5km"],
        ).astype("string")

    if "ref_local_spread_c" in x.columns:
        x["ref_local_spread_bin"] = pd.cut(
            pd.to_numeric(x["ref_local_spread_c"], errors="coerce"),
            bins=[-np.inf, 0.5, 1, 2, 4, np.inf],
            labels=["<=0.5C", "0.5-1C", "1-2C", "2-4C", ">4C"],
        ).astype("string")

    lcz_match_col = _first_existing(
        x.columns,
        [
            "ref_frac_same_LCZ_point_lg",
            "ref_frac_same_LCZ_buffer_lg",
            "ref_frac_same_LCZ_point",
            "ref_frac_same_LCZ_buffer",
            "ref_frac_same_same_LCZ_point_lg",
            "ref_frac_same_same_LCZ_buffer_lg",
            "ref_frac_same_same_LCZ_point",
            "ref_frac_same_same_LCZ_buffer",
        ],
    )
    if lcz_match_col is not None:
        mismatch = 1.0 - pd.to_numeric(x[lcz_match_col], errors="coerce")
        x["ref_lcz_mismatch_bin"] = pd.cut(
            mismatch,
            bins=[-np.inf, 0.1, 0.4, 0.7, np.inf],
            labels=["low", "moderate", "high", "very_high"],
        ).astype("string")

    lc_match_col = _first_existing(
        x.columns,
        [
            "ref_frac_same_LC_point_lg",
            "ref_frac_same_LC_buffer_lg",
            "ref_frac_same_LC_point",
            "ref_frac_same_LC_buffer",
            "ref_frac_same_same_LC_point_lg",
            "ref_frac_same_same_LC_buffer_lg",
            "ref_frac_same_same_LC_point",
            "ref_frac_same_same_LC_buffer",
        ],
    )
    if lc_match_col is not None:
        mismatch = 1.0 - pd.to_numeric(x[lc_match_col], errors="coerce")
        x["ref_lc_mismatch_bin"] = pd.cut(
            mismatch,
            bins=[-np.inf, 0.1, 0.4, 0.7, np.inf],
            labels=["low", "moderate", "high", "very_high"],
        ).astype("string")


    if "ref_nearest_dist_bin" in x.columns or "ref_local_n" in x.columns or "ref_local_spread_c" in x.columns:
        dist = pd.to_numeric(
            x[_first_existing(x.columns, ["ref_min_dist_km", "ref_nearest_dist_km", "ref_nearest_value_dist_km"])]
            if _first_existing(x.columns, ["ref_min_dist_km", "ref_nearest_dist_km", "ref_nearest_value_dist_km"]) is not None
            else pd.Series(np.nan, index=x.index),
            errors="coerce",
        )
        n_local = pd.to_numeric(x.get("ref_local_n", pd.Series(np.nan, index=x.index)), errors="coerce")
        spread = pd.to_numeric(x.get("ref_local_spread_c", pd.Series(np.nan, index=x.index)), errors="coerce")
        support = pd.Series("moderate_support", index=x.index, dtype="string")
        support.loc[(dist <= 2.0) & (n_local >= 4) & ((spread <= 2.0) | spread.isna())] = "high_support"
        support.loc[(dist > 5.0) | (n_local <= 1) | (spread > 4.0)] = "low_support"
        support.loc[(dist > 10.0) | (n_local <= 0)] = "outside_support"
        support.loc[dist.isna() & n_local.isna() & spread.isna()] = "unknown_support"
        x["ref_support_class"] = support

    return x


def summarize_reference_errors(
    df: pd.DataFrame,
    residual_col: str = "ref_resid",
) -> dict[str, pd.DataFrame]:
    """Reference error summary overall and by hour/month/site-context bins."""
    x = add_reference_summary_bins(df)
    x["date"] = ensure_utc_datetime(x["date"])
    x = add_reference_temporal_features(x)
    x[residual_col] = pd.to_numeric(x[residual_col], errors="coerce")

    def _by(cols: Sequence[str]) -> pd.DataFrame:
        rows = []
        for keys, g in x.groupby(list(cols), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(cols, keys))
            row.update(residual_metrics(g[residual_col]))
            rows.append(row)
        return pd.DataFrame(rows)

    summaries = {
        "overall": pd.DataFrame([residual_metrics(x[residual_col])]),
        "by_hour": _by(["hour"]),
        "by_month": _by(["month"]),
    }

    for col, name in [
        ("LCZ_point_lg", "by_lcz_point_lg"),
        ("LCZ_buffer_lg", "by_lcz_buffer_lg"),
        ("ref_abs_elev_diff_bin", "by_abs_elev_diff_bin"),
        ("ref_abs_building_height_diff_bin", "by_abs_building_height_diff_bin"),
        ("ref_local_n_bin", "by_ows_density_bin"),
        ("ref_nearest_dist_bin", "by_nearest_ows_distance_bin"),
        ("ref_local_spread_bin", "by_local_ows_spread_bin"),
        ("ref_lcz_mismatch_bin", "by_lcz_mismatch_bin"),
        ("ref_lc_mismatch_bin", "by_lc_mismatch_bin"),
        ("ref_support_class", "by_reference_support_class"),
    ]:
        if col in x.columns:
            summaries[name] = _by([col])

    return summaries


def write_reference_summaries(
    summaries: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in summaries.items():
        path = output_dir / f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def make_reference_retention_curves(
    cws_reference_df: pd.DataFrame,
    residual_col: str = "cws_ref_resid",
) -> pd.DataFrame:
    """Simple retention-error curves using residual magnitude/z as risk scores."""
    helper = _require_qcb()
    parts = []
    if "abs_cws_ref_resid" in cws_reference_df.columns:
        parts.append(
            helper.make_retention_curve(
                cws_reference_df,
                risk_col="abs_cws_ref_resid",
                residual_col=residual_col,
                target_col="target_ref_risk",
                method_name="abs_cws_ref_resid",
            )
        )
    if "cws_ref_z" in cws_reference_df.columns:
        tmp = cws_reference_df.copy()
        tmp["abs_cws_ref_z"] = tmp["cws_ref_z"].abs()
        parts.append(
            helper.make_retention_curve(
                tmp,
                risk_col="abs_cws_ref_z",
                residual_col=residual_col,
                target_col="target_ref_risk",
                method_name="abs_cws_ref_z",
            )
        )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if "weak_bad_rate_kept" in out.columns:
        out = out.rename(columns={"weak_bad_rate_kept": "ref_risk_rate_kept"})
    return out


def fit_station_residual_summary(
    train_df: pd.DataFrame,
    residual_col: str = "cws_ref_resid",
    group_cols: Sequence[str] = ("network", "station_id"),
) -> pd.DataFrame:
    """Fit station-level residual summaries using training rows only.

    Use this in later residual-risk models to avoid computing station-bias
    features from validation/test rows.
    """
    x = train_df.copy()
    x[residual_col] = pd.to_numeric(x[residual_col], errors="coerce")
    x["abs_resid"] = x[residual_col].abs()
    x["resid_sign"] = np.sign(x[residual_col])
    x["is_daytime"] = x["date"].dt.hour.between(8, 18).astype("int8") if "date" in x.columns else 0

    rows = []
    for keys, g in x.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        resid = g[residual_col].dropna()
        row = dict(zip(group_cols, keys))
        row["station_ref_n_train"] = int(resid.size)
        row["station_ref_median_resid_train"] = float(resid.median()) if resid.size else np.nan
        row["station_ref_mad_resid_train"] = float(np.median(np.abs(resid - resid.median()))) if resid.size else np.nan
        row["station_ref_frac_large_abs_z_train"] = (
            float((g.get("cws_ref_z", pd.Series(index=g.index, dtype=float)).abs() > 2.5).mean())
            if "cws_ref_z" in g.columns else np.nan
        )
        row["station_ref_sign_consistency_train"] = float(abs(np.nanmean(np.sign(resid)))) if resid.size else np.nan
        if "is_daytime" in g.columns:
            row["station_ref_day_median_resid_train"] = float(g.loc[g["is_daytime"] == 1, residual_col].median())
            row["station_ref_night_median_resid_train"] = float(g.loc[g["is_daytime"] == 0, residual_col].median())
            row["station_ref_day_minus_night_bias_train"] = (
                row["station_ref_day_median_resid_train"] - row["station_ref_night_median_resid_train"]
            )
        rows.append(row)

    return downcast(pd.DataFrame(rows))


def apply_station_residual_summary(
    df: pd.DataFrame,
    station_summary: pd.DataFrame,
    group_cols: Sequence[str] = ("network", "station_id"),
) -> pd.DataFrame:
    """Merge train-only station residual summaries into any split."""
    return downcast(df.merge(station_summary, on=list(group_cols), how="left", validate="many_to_one"))


def import_project_configs(city: str):
    """Import config_tool/config_project using the same conventions as notebooks."""
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]

    config_tool_module = None
    for cand in candidates:
        if (cand / "config_tool.py").exists():
            sys.path.insert(0, str(cand))
            config_tool_module = importlib.import_module("config_tool")
            break

    if config_tool_module is None:

        config_tool_module = importlib.import_module("config_tool")

    cfm = config_tool_module
    project_dir = Path(cfm.cwd_data) / city
    sys.path.insert(0, str(project_dir))
    sys.modules.pop("config_project", None)
    cfp = importlib.import_module("config_project")
    return cfm, cfp, project_dir


def load_project_station_data(cfp):
    cwd_data_str = Path(cfp.cwd_data_str)
    cwd_data_meta = Path(cfp.cwd_data_meta)

    start_date = pd.to_datetime(cfp.first_date, format="%d-%m-%Y %H:%M")
    end_date = pd.to_datetime(cfp.last_date, format="%d-%m-%Y %H:%M")
    year_span = f"{start_date.year}-{end_date.year}"

    CWS_coordinates_wunder = pd.read_csv(cwd_data_meta / f"Coordinates_{cfp.city}_CWS_Wunderground_str_all.csv")
    CWS_ta_wunder = pd.read_csv(
        cwd_data_str / f"ta_{cfp.city}_{year_span}_h_CWS_Wunderground_str.csv",
        index_col="date",
        parse_dates=True,
    )

    CWS_coordinates_net = pd.read_csv(cwd_data_meta / f"Coordinates_{cfp.city}_CWS_Netatmo_str_all.csv")
    CWS_ta_net = pd.read_csv(
        cwd_data_str / f"ta_{cfp.city}_{year_span}_h_CWS_Netatmo_str.csv",
        index_col="date",
        parse_dates=True,
    )

    OWS_coordinates = pd.read_csv(cwd_data_meta / f"Coordinates_{cfp.city}_OWS_str_all.csv")
    OWS_ta = pd.read_csv(
        cwd_data_str / f"ta_{cfp.city}_{year_span}_h_OWS_str.csv",
        index_col="date",
        parse_dates=True,
    )

    return {
        "year_span": year_span,
        "CWS_coordinates_wunder": CWS_coordinates_wunder,
        "CWS_ta_wunder": CWS_ta_wunder,
        "CWS_coordinates_net": CWS_coordinates_net,
        "CWS_ta_net": CWS_ta_net,
        "OWS_coordinates": OWS_coordinates,
        "OWS_ta": OWS_ta,
    }


def load_optional_environment(cfp, year_span: str) -> dict:
    cwd_data_str = Path(cfp.cwd_data_str)
    folder_era5 = cwd_data_str / "ERA5Land_hourly"
    folder_lst = cwd_data_str / "LST_hourly"
    folder_ndvi = cwd_data_str / "NDVI_S2"

    return {
        "era5_net": maybe_read_csv(folder_era5 / "CWS_Netatmo" / f"ERA5Land_hourly_{cfp.city}_{year_span}_CWS_Netatmo_str_long.csv"),
        "era5_wunder": maybe_read_csv(folder_era5 / "CWS_Wunderground" / f"ERA5Land_hourly_{cfp.city}_{year_span}_CWS_Wunderground_str_long.csv"),
        "era5_ows": maybe_read_csv(folder_era5 / "OWS" / f"ERA5Land_hourly_{cfp.city}_{year_span}_OWS_str_long.csv"),
        "lst_net": maybe_read_csv(folder_lst / f"LST_hourly_{cfp.city}_CWS_Netatmo_long.csv"),
        "lst_wunder": maybe_read_csv(folder_lst / f"LST_hourly_{cfp.city}_CWS_Wunderground_long.csv"),
        "lst_ows": maybe_read_csv(folder_lst / f"LST_hourly_{cfp.city}_OWS_long.csv"),
        "ndvi_net": maybe_read_csv(folder_ndvi / f"NDVI_S2_{cfp.city}_CWS_Netatmo_long.csv"),
        "ndvi_wunder": maybe_read_csv(folder_ndvi / f"NDVI_S2_{cfp.city}_CWS_Wunderground_long.csv"),
        "ndvi_ows": maybe_read_csv(folder_ndvi / f"NDVI_S2_{cfp.city}_OWS_long.csv"),
    }


def prep_optional_hourly(df, value_cols, network_name, obs_ids):
    if df is None:
        return None
    helper = _require_qcb()
    return helper.prep_hourly_env(
        df,
        value_cols=value_cols,
        network_name=network_name,
        observed_station_ids=obs_ids,
    )


def _slugify_for_path(value: object) -> str:
    """Small path-safe slug used for run labels and filenames."""
    s = str(value).strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {".", "/", " ", ":"}:
            out.append("p" if ch == "." else "-")
    slug = "".join(out).strip("-_")
    return slug or "run"


def reference_method_from_config(config: ReferenceModelConfig) -> str:
    if config.reference_method == "auto":
        return "catboost" if config.run_catboost_regressor else "idw"
    return str(config.reference_method)


def calibration_modes_to_fit(config: ReferenceModelConfig) -> list[str]:
    modes = [str(config.calibration_mode)]
    if config.save_both_calibration_modes:
        for mode in ["time_train", "all_ows_reference"]:
            if mode not in modes:
                modes.append(mode)
    return modes


def make_reference_run_label(config: ReferenceModelConfig, cfp_city: str) -> str:
    """Create a deterministic-ish, human-readable label plus timestamp.

    The timestamp prevents accidental overwrites while still making it obvious
    which settings produced the run.
    """
    if config.run_label:
        return _slugify_for_path(config.run_label)

    method = reference_method_from_config(config)
    radius = "all" if config.ows_radius_km is None else f"{config.ows_radius_km:g}km"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pieces = [
        _slugify_for_path(cfp_city),
        f"ref-{method}",
        f"cal-{config.calibration_mode}",
        f"k{config.k_nearest_ows}",
        f"r{_slugify_for_path(radius)}",
        timestamp,
    ]
    return "__".join(pieces)


def make_unique_run_dir(base_output_dir: Path, run_label: str, overwrite_existing_run: bool = False) -> tuple[Path, str]:
    """Return a non-overwriting run directory and final label."""
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


def _station_id_set_from_frame(df: pd.DataFrame | None) -> set[str]:
    if df is None or len(df) == 0 or "station_id" not in df.columns:
        return set()
    return set(pd.Series(df["station_id"]).dropna().astype(str).str.strip().unique())


def build_station_coverage_audit(
    base_station_ids_by_network: Mapping[str, Iterable[str]],
    prepared_tables: Mapping[str, tuple[str, pd.DataFrame | None]],
) -> pd.DataFrame:
    """Compare prepared metadata/env tables to observed station universes.

    Parameters
    ----------
    base_station_ids_by_network:
        Mapping from normalized network name to observed station IDs from the
        temperature table.
    prepared_tables:
        Mapping from source name to (network_name, prepared_df). The prepared
        dataframes should already have canonical station_id columns.
    """
    base_sets = {
        normalize_network_name(network): set(pd.Series(list(ids)).dropna().astype(str).str.strip().unique())
        for network, ids in base_station_ids_by_network.items()
    }
    rows = []
    for source_name, (network_name, df) in prepared_tables.items():
        network = normalize_network_name(network_name)
        base = base_sets.get(network, set())
        ids = _station_id_set_from_frame(df)
        overlap = ids & base
        rows.append({
            "network": network,
            "source": source_name,
            "n_base_observed_stations": int(len(base)),
            "n_source_stations": int(len(ids)),
            "n_overlap_with_observed": int(len(overlap)),
            "overlap_frac_observed": float(len(overlap) / len(base)) if base else np.nan,
            "n_observed_missing_from_source": int(len(base - ids)),
            "n_source_extra_not_observed": int(len(ids - base)),
            "example_missing_observed_ids": ", ".join(sorted(base - ids)[:15]),
            "example_extra_source_ids": ", ".join(sorted(ids - base)[:15]),
        })
    return pd.DataFrame(rows).sort_values(["network", "source"]).reset_index(drop=True)


def write_reference_calibration_artifacts(
    calibration: Mapping,
    output_dir: str | Path,
    city: str,
    method: str,
    calibration_mode: str,
) -> dict[str, str]:
    """Save one calibration object without trying to JSON-serialize DataFrames."""
    output_dir = Path(output_dir)
    cal_dir = output_dir / f"calibration__{method}__{calibration_mode}"
    cal_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for table_name in ["hour_sigma", "month_sigma", "month_hour_sigma"]:
        table = calibration.get(table_name)
        if isinstance(table, pd.DataFrame):
            path = cal_dir / f"{city}_{method}_{calibration_mode}_{table_name}.csv"
            table.to_csv(path, index=False)
            paths[table_name] = str(path)

    difficulty_tables = calibration.get("difficulty_sigma_tables", {})
    if isinstance(difficulty_tables, Mapping):
        for sigma_name, table in difficulty_tables.items():
            if isinstance(table, pd.DataFrame):
                path = cal_dir / f"{city}_{method}_{calibration_mode}_{sigma_name}.csv"
                table.to_csv(path, index=False)
                paths[sigma_name] = str(path)

    serializable = {}
    for k, v in calibration.items():
        if isinstance(v, pd.DataFrame):
            continue
        if k == "difficulty_sigma_tables":
            serializable[k] = {
                name: paths.get(name)
                for name in (v.keys() if isinstance(v, Mapping) else [])
            }
        else:
            serializable[k] = v

    path_json = cal_dir / f"{city}_{method}_{calibration_mode}_reference_calibration.json"
    with open(path_json, "w") as f:
        json.dump(serializable, f, indent=2, default=_safe_json_default)
    paths["json"] = str(path_json)
    return paths


def fit_reference_calibrations_for_modes(
    ows_reference_frame: pd.DataFrame,
    residual_col: str,
    config: ReferenceModelConfig,
) -> dict[str, dict]:
    """Fit all requested uncertainty calibrations for one reference method."""
    out = {}
    for mode in calibration_modes_to_fit(config):
        out[mode] = fit_reference_uncertainty(
            ows_reference_frame,
            residual_col=residual_col,
            valid_start=config.valid_start,
            calibration_mode=mode,
            sigma_floor_c=config.sigma_floor_c,
            min_group_n=config.sigma_min_group_n,
            abnormal_quantile=config.abnormal_quantile,
            ambiguous_lower_quantile=config.ambiguous_lower_quantile,
        )
    return out

def run_reference_pipeline(
    config: ReferenceModelConfig,
    *,
    cfm=None,
    cfp=None,
    project_dir: str | Path | None = None,
    raw: Mapping | None = None,
    env_raw: Mapping | None = None,
) -> dict:
    """End-to-end pipeline runnable from the companion notebook or CLI.

    In notebook mode, pass ``cfp``, ``project_dir``, ``raw``, and ``env_raw`` so
    the notebook controls the city/files while this script does the processing.

    v4 output behavior:
      - each execution writes into a unique run directory under
        ``qc_benchmark/<output_subdir>/<run_label>``;
      - file names include reference method and calibration mode;
      - by default, both ``time_train`` and ``all_ows_reference`` calibrations
        are fitted and saved side by side, while ``config.calibration_mode``
        remains the primary/canonical mode in the manifest.
    """
    helper = _require_qcb()

    if cfp is None:

        cfm, cfp, project_dir = import_project_configs(config.city)
    else:


        if project_dir is None:
            project_dir = Path(cfp.cwd_data_str).parent
        else:
            project_dir = Path(project_dir)
    os.chdir(project_dir)
    print("Project dir:", project_dir)

    benchmark_dir = Path(cfp.cwd_results_qc) / "qc_benchmark"
    base_output_dir = benchmark_dir / config.output_subdir
    requested_run_label = make_reference_run_label(config, cfp.city)
    output_dir, final_run_label = make_unique_run_dir(
        base_output_dir,
        requested_run_label,
        overwrite_existing_run=config.overwrite_existing_run,
    )
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("Run label:", final_run_label)
    print("Output dir:", output_dir)

    primary_calibration_mode = str(config.calibration_mode)
    calibration_modes = calibration_modes_to_fit(config)
    selected_method_hint = reference_method_from_config(config)

    if raw is None:
        raw = load_project_station_data(cfp)
    else:
        raw = dict(raw)
        if "year_span" not in raw:
            start_date = pd.to_datetime(cfp.first_date, format="%d-%m-%Y %H:%M")
            end_date = pd.to_datetime(cfp.last_date, format="%d-%m-%Y %H:%M")
            raw["year_span"] = f"{start_date.year}-{end_date.year}"

    if env_raw is None:
        env_raw = load_optional_environment(cfp, raw["year_span"])
    else:
        env_raw = dict(env_raw)


    CWS_ta_net = helper.dedupe_temp_wide(raw["CWS_ta_net"], how="mean")
    CWS_ta_wunder = helper.dedupe_temp_wide(raw["CWS_ta_wunder"], how="mean")
    OWS_ta = helper.dedupe_temp_wide(raw["OWS_ta"], how="mean")

    cws_net_long = helper.melt_ta_data(CWS_ta_net, "Netatmo")
    cws_wunder_long = helper.melt_ta_data(CWS_ta_wunder, "Wunderground")
    cws_all = pd.concat([cws_net_long, cws_wunder_long], ignore_index=True)
    ows_long = helper.melt_ta_data(OWS_ta, "OWS")

    net_obs_ids = cws_net_long["station_id"].dropna().astype(str).str.strip().unique()
    wunder_obs_ids = cws_wunder_long["station_id"].dropna().astype(str).str.strip().unique()
    ows_obs_ids = ows_long["station_id"].dropna().astype(str).str.strip().unique()

    meta_net = helper.prep_station_meta(raw["CWS_coordinates_net"], "Netatmo", observed_station_ids=net_obs_ids)
    meta_wunder = helper.prep_station_meta(raw["CWS_coordinates_wunder"], "Wunderground", observed_station_ids=wunder_obs_ids)
    meta_ows = helper.prep_station_meta(raw["OWS_coordinates"], "OWS", observed_station_ids=ows_obs_ids)


    meta_net = meta_net[meta_net["station_id"].astype(str).isin(set(CWS_ta_net.columns.astype(str)))].copy()
    meta_wunder = meta_wunder[meta_wunder["station_id"].astype(str).isin(set(CWS_ta_wunder.columns.astype(str)))].copy()
    meta_ows = meta_ows[meta_ows["station_id"].astype(str).isin(set(OWS_ta.columns.astype(str)))].copy()
    meta_cws = pd.concat([meta_net, meta_wunder], ignore_index=True).drop_duplicates(["station_id", "network"])


    era5_pref = ["t2m_c", "d2m_c", "u10_ms", "v10_ms", "ssrd_wm2", "tp_mm"]
    lst_pref = ["LST_C"]
    ndvi_pref = ["NDVI"]
    era5_cols_net = infer_hourly_value_cols(env_raw["era5_net"], era5_pref) if config.use_dynamic_env_columns else era5_pref
    era5_cols_wunder = infer_hourly_value_cols(env_raw["era5_wunder"], era5_pref) if config.use_dynamic_env_columns else era5_pref
    era5_cols_ows = infer_hourly_value_cols(env_raw["era5_ows"], era5_pref) if config.use_dynamic_env_columns else era5_pref
    lst_cols_net = infer_hourly_value_cols(env_raw["lst_net"], lst_pref) if config.use_dynamic_env_columns else lst_pref
    lst_cols_wunder = infer_hourly_value_cols(env_raw["lst_wunder"], lst_pref) if config.use_dynamic_env_columns else lst_pref
    lst_cols_ows = infer_hourly_value_cols(env_raw["lst_ows"], lst_pref) if config.use_dynamic_env_columns else lst_pref
    ndvi_cols_net = infer_hourly_value_cols(env_raw["ndvi_net"], ndvi_pref) if config.use_dynamic_env_columns else ndvi_pref
    ndvi_cols_wunder = infer_hourly_value_cols(env_raw["ndvi_wunder"], ndvi_pref) if config.use_dynamic_env_columns else ndvi_pref
    ndvi_cols_ows = infer_hourly_value_cols(env_raw["ndvi_ows"], ndvi_pref) if config.use_dynamic_env_columns else ndvi_pref
    print("ERA5 value cols (Netatmo/Wunderground/OWS):", era5_cols_net, era5_cols_wunder, era5_cols_ows)
    print("LST value cols (Netatmo/Wunderground/OWS):", lst_cols_net, lst_cols_wunder, lst_cols_ows)
    print("NDVI value cols (Netatmo/Wunderground/OWS):", ndvi_cols_net, ndvi_cols_wunder, ndvi_cols_ows)

    era5_net_p = prep_optional_hourly(env_raw["era5_net"], era5_cols_net, "Netatmo", net_obs_ids)
    era5_wunder_p = prep_optional_hourly(env_raw["era5_wunder"], era5_cols_wunder, "Wunderground", wunder_obs_ids)
    era5_ows_p = prep_optional_hourly(env_raw["era5_ows"], era5_cols_ows, "OWS", ows_obs_ids)

    lst_net_p = prep_optional_hourly(env_raw["lst_net"], lst_cols_net, "Netatmo", net_obs_ids)
    lst_wunder_p = prep_optional_hourly(env_raw["lst_wunder"], lst_cols_wunder, "Wunderground", wunder_obs_ids)
    lst_ows_p = prep_optional_hourly(env_raw["lst_ows"], lst_cols_ows, "OWS", ows_obs_ids)

    ndvi_net_p = prep_optional_hourly(env_raw["ndvi_net"], ndvi_cols_net, "Netatmo", net_obs_ids)
    ndvi_wunder_p = prep_optional_hourly(env_raw["ndvi_wunder"], ndvi_cols_wunder, "Wunderground", wunder_obs_ids)
    ndvi_ows_p = prep_optional_hourly(env_raw["ndvi_ows"], ndvi_cols_ows, "OWS", ows_obs_ids)

    coverage_audit = build_station_coverage_audit(
        base_station_ids_by_network={
            "OWS": ows_obs_ids,
            "Netatmo": net_obs_ids,
            "Wunderground": wunder_obs_ids,
        },
        prepared_tables={
            "metadata": ("OWS", meta_ows),
            "era5": ("OWS", era5_ows_p),
            "lst": ("OWS", lst_ows_p),
            "ndvi": ("OWS", ndvi_ows_p),
            "metadata_netatmo": ("Netatmo", meta_net),
            "era5_netatmo": ("Netatmo", era5_net_p),
            "lst_netatmo": ("Netatmo", lst_net_p),
            "ndvi_netatmo": ("Netatmo", ndvi_net_p),
            "metadata_wunderground": ("Wunderground", meta_wunder),
            "era5_wunderground": ("Wunderground", era5_wunder_p),
            "lst_wunderground": ("Wunderground", lst_wunder_p),
            "ndvi_wunderground": ("Wunderground", ndvi_wunder_p),
        },
    )
    coverage_audit_path = output_dir / f"{cfp.city}_station_covariate_coverage_audit.csv"
    coverage_audit.to_csv(coverage_audit_path, index=False)
    print("Saved station/covariate coverage audit:", coverage_audit_path)

    era5_cws = (
        pd.concat([x for x in [era5_net_p, era5_wunder_p] if x is not None], ignore_index=True)
        if any(x is not None for x in [era5_net_p, era5_wunder_p])
        else None
    )
    lst_cws = (
        pd.concat([x for x in [lst_net_p, lst_wunder_p] if x is not None], ignore_index=True)
        if any(x is not None for x in [lst_net_p, lst_wunder_p])
        else None
    )
    ndvi_cws = (
        pd.concat([x for x in [ndvi_net_p, ndvi_wunder_p] if x is not None], ignore_index=True)
        if any(x is not None for x in [ndvi_net_p, ndvi_wunder_p])
        else None
    )

    ows_asof_specs = [
        {
            "features": lst_ows_p,
            "value_cols": ["LST_C"],
            "tolerance": config.lst_asof_tolerance,
            "direction": config.lst_asof_direction,
            "age_unit": "hours",
        },
        {
            "features": ndvi_ows_p,
            "value_cols": ["NDVI"],
            "tolerance": config.ndvi_asof_tolerance,
            "direction": config.ndvi_asof_direction,
            "age_unit": "days",
        },
    ]
    cws_asof_specs = [
        {
            "features": lst_cws,
            "value_cols": ["LST_C"],
            "tolerance": config.lst_asof_tolerance,
            "direction": config.lst_asof_direction,
            "age_unit": "hours",
        },
        {
            "features": ndvi_cws,
            "value_cols": ["NDVI"],
            "tolerance": config.ndvi_asof_tolerance,
            "direction": config.ndvi_asof_direction,
            "age_unit": "days",
        },
    ]


    ows_frame, ows_pair_table, ows_local = build_target_reference_frame(
        target_long=ows_long,
        target_meta=meta_ows,
        source_ows_long=ows_long,
        source_ows_meta=meta_ows,
        exact_env_tables=[era5_ows_p],
        asof_env_specs=ows_asof_specs,
        k_nearest=config.k_nearest_ows,
        max_radius_km=config.ows_radius_km,
        fallback_to_k_nearest=config.fallback_to_k_nearest,
        idw_power=config.idw_power,
        eps_km=config.eps_km,
        exclude_self=True,
        chunk_freq=config.chunk_freq,
        pair_table_path=output_dir / f"{cfp.city}_ows_loo_pair_table.csv",
        local_features_path=output_dir / f"{cfp.city}_ows_loo_local_features.parquet",
        rebuild=config.rebuild,
    )
    ows_frame = add_reference_residual_columns(ows_frame, mu_col="ref_idw_mu", method="idw")

    ows_local_context_paths: dict[str, str | None] = {}
    if config.use_ows_satellite_context:
        if lst_ows_p is not None and "LST_C" in lst_ows_p.columns:
            ows_frame, ows_local_context_paths["ows_local_lst_features_path"] = merge_local_ows_satellite_context(
                ows_frame,
                pair_table=ows_pair_table,
                ows_env=lst_ows_p,
                ows_station_ids=ows_obs_ids,
                value_col="LST_C",
                feature_prefix="lst",
                tolerance=config.lst_asof_tolerance,
                direction=config.lst_asof_direction,
                age_unit="hours",
                chunk_freq=config.chunk_freq,
                cache_path=output_dir / f"{cfp.city}_ows_reference_local_lst_features.parquet",
                rebuild=config.rebuild,
            )
        if ndvi_ows_p is not None and "NDVI" in ndvi_ows_p.columns:
            ows_frame, ows_local_context_paths["ows_local_ndvi_features_path"] = merge_local_ows_satellite_context(
                ows_frame,
                pair_table=ows_pair_table,
                ows_env=ndvi_ows_p,
                ows_station_ids=ows_obs_ids,
                value_col="NDVI",
                feature_prefix="ndvi",
                tolerance=config.ndvi_asof_tolerance,
                direction=config.ndvi_asof_direction,
                age_unit="days",
                chunk_freq=config.chunk_freq,
                cache_path=output_dir / f"{cfp.city}_ows_reference_local_ndvi_features.parquet",
                rebuild=config.rebuild,
            )
        ows_frame = add_reference_interactions(ows_frame)
    ows_frame_path = save_dataframe_auto(ows_frame, output_dir / f"{cfp.city}_ows_reference_frame_idw_design.parquet")

    idw_calibrations = fit_reference_calibrations_for_modes(
        ows_frame,
        residual_col="ref_resid_idw",
        config=config,
    )


    catboost_model_path = None
    catboost_feature_cols: list[str] = []
    catboost_categorical_cols: list[str] = []
    catboost_calibrations: dict[str, dict] = {}
    ows_cat = None

    if config.run_catboost_regressor:
        catboost_feature_cols, catboost_categorical_cols = get_reference_regression_feature_columns(
            ows_frame,
            include_coordinates=config.catboost_include_coordinates,
        )
        print("CatBoost reference features:", len(catboost_feature_cols))
        print("CatBoost reference categorical features:", catboost_categorical_cols)

        ows_cat, _fold_models = train_catboost_reference_cv(
            ows_frame,
            feature_cols=catboost_feature_cols,
            categorical_cols=catboost_categorical_cols,
            cv_mode=config.catboost_cv_mode,
            n_splits=config.catboost_n_splits,
            iterations=config.catboost_iterations,
            learning_rate=config.catboost_learning_rate,
            depth=config.catboost_depth,
            l2_leaf_reg=config.catboost_l2_leaf_reg,
            random_state=config.random_state,
            thread_count=config.catboost_thread_count,
            used_ram_limit=config.catboost_used_ram_limit,
            use_gpu=config.catboost_use_gpu,
            verbose=config.catboost_verbose,
            model_dir=model_dir,
            model_prefix=f"{cfp.city}_catboost_reference_{config.catboost_cv_mode}_{final_run_label}",
        )
        ows_cat_path = save_dataframe_auto(ows_cat, output_dir / f"{cfp.city}_ows_reference_frame_catboost_cv.parquet")
        catboost_calibrations = fit_reference_calibrations_for_modes(
            ows_cat,
            residual_col="ref_resid_catboost",
            config=config,
        )

        catboost_model_path = model_dir / f"{cfp.city}_catboost_reference_final_{final_run_label}.cbm"
        final_model = fit_final_catboost_reference_model(
            ows_frame,
            feature_cols=catboost_feature_cols,
            categorical_cols=catboost_categorical_cols,
            iterations=config.catboost_iterations,
            learning_rate=config.catboost_learning_rate,
            depth=config.catboost_depth,
            l2_leaf_reg=config.catboost_l2_leaf_reg,
            random_state=config.random_state,
            thread_count=config.catboost_thread_count,
            used_ram_limit=config.catboost_used_ram_limit,
            use_gpu=config.catboost_use_gpu,
            verbose=config.catboost_verbose,
            model_path=catboost_model_path,
        )
    else:
        final_model = None
        ows_cat_path = None


    selected_method = reference_method_from_config(config)
    if selected_method == "catboost" and not config.run_catboost_regressor:
        raise ValueError("reference_method='catboost' requires run_catboost_regressor=True")
    if primary_calibration_mode not in calibration_modes:
        raise ValueError(f"Primary calibration mode {primary_calibration_mode!r} was not fitted.")

    calibrations_by_method: dict[str, dict[str, dict]] = {"idw": idw_calibrations}
    if config.run_catboost_regressor:
        calibrations_by_method["catboost"] = catboost_calibrations

    reference_frame_by_method = {"idw": ows_frame}
    if ows_cat is not None:
        reference_frame_by_method["catboost"] = ows_cat


    ows_reference_paths: dict[str, str] = {}
    ows_summary_paths_by_mode: dict[str, dict[str, str]] = {}
    calibration_paths_by_method_mode: dict[str, dict[str, str]] = {}

    for method, method_calibrations in calibrations_by_method.items():
        frame_for_method = reference_frame_by_method[method]
        for mode, calibration in method_calibrations.items():
            mode_key = f"{method}__{mode}"
            ows_reference = standardize_ows_reference_columns(frame_for_method, method=method, calibration=calibration)
            ows_reference_path = save_dataframe_auto(
                ows_reference,
                output_dir / f"{cfp.city}_ows_reference_predictions_{method}_{mode}.parquet",
            )
            ows_reference_paths[mode_key] = str(ows_reference_path)
            ows_summary_paths_by_mode[mode_key] = write_reference_summaries(
                summarize_reference_errors(ows_reference, residual_col="ref_resid"),
                output_dir,
                prefix=f"{cfp.city}_ows_reference_{method}_{mode}",
            )
            calibration_paths_by_method_mode[mode_key] = write_reference_calibration_artifacts(
                calibration,
                output_dir=output_dir,
                city=cfp.city,
                method=method,
                calibration_mode=mode,
            )

    selected_calibrations = calibrations_by_method[selected_method]
    selected_calibration = selected_calibrations[primary_calibration_mode]


    cws_frame, cws_pair_table, cws_local = build_target_reference_frame(
        target_long=cws_all,
        target_meta=meta_cws,
        source_ows_long=ows_long,
        source_ows_meta=meta_ows,
        exact_env_tables=[era5_cws],
        asof_env_specs=cws_asof_specs,
        k_nearest=config.k_nearest_ows,
        max_radius_km=config.ows_radius_km,
        fallback_to_k_nearest=config.fallback_to_k_nearest,
        idw_power=config.idw_power,
        eps_km=config.eps_km,
        exclude_self=False,
        chunk_freq=config.chunk_freq,
        pair_table_path=output_dir / f"{cfp.city}_cws_to_ows_pair_table.csv",
        local_features_path=output_dir / f"{cfp.city}_cws_local_reference_features.parquet",
        rebuild=config.rebuild,
    )
    cws_frame = add_reference_residual_columns(cws_frame, mu_col="ref_idw_mu", method="idw")

    cws_local_context_paths: dict[str, str | None] = {}
    if config.use_ows_satellite_context:
        if lst_ows_p is not None and "LST_C" in lst_ows_p.columns:
            cws_frame, cws_local_context_paths["cws_local_ows_lst_features_path"] = merge_local_ows_satellite_context(
                cws_frame,
                pair_table=cws_pair_table,
                ows_env=lst_ows_p,
                ows_station_ids=ows_obs_ids,
                value_col="LST_C",
                feature_prefix="lst",
                tolerance=config.lst_asof_tolerance,
                direction=config.lst_asof_direction,
                age_unit="hours",
                chunk_freq=config.chunk_freq,
                cache_path=output_dir / f"{cfp.city}_cws_local_ows_lst_features.parquet",
                rebuild=config.rebuild,
            )
        if ndvi_ows_p is not None and "NDVI" in ndvi_ows_p.columns:
            cws_frame, cws_local_context_paths["cws_local_ows_ndvi_features_path"] = merge_local_ows_satellite_context(
                cws_frame,
                pair_table=cws_pair_table,
                ows_env=ndvi_ows_p,
                ows_station_ids=ows_obs_ids,
                value_col="NDVI",
                feature_prefix="ndvi",
                tolerance=config.ndvi_asof_tolerance,
                direction=config.ndvi_asof_direction,
                age_unit="days",
                chunk_freq=config.chunk_freq,
                cache_path=output_dir / f"{cfp.city}_cws_local_ows_ndvi_features.parquet",
                rebuild=config.rebuild,
            )
        cws_frame = add_reference_interactions(cws_frame)

    if selected_method == "catboost":
        if final_model is None:
            raise RuntimeError("CatBoost selected but final_model is missing.")
        cws_frame["ref_mu_catboost"] = predict_catboost_reference(
            final_model,
            cws_frame,
            feature_cols=catboost_feature_cols,
            categorical_cols=catboost_categorical_cols,
        )
        cws_frame = add_reference_residual_columns(cws_frame, mu_col="ref_mu_catboost", method="catboost")
        cws_mu_col = "ref_mu_catboost"
    else:
        cws_mu_col = "ref_idw_mu"

    cws_reference_paths: dict[str, str] = {}
    retention_curve_paths: dict[str, str] = {}
    for mode, calibration in selected_calibrations.items():
        cws_reference = apply_reference_to_cws(cws_frame, mu_col=cws_mu_col, calibration=calibration)
        cws_reference_path = save_dataframe_auto(
            cws_reference,
            output_dir / f"{cfp.city}_cws_reference_residual_features_{selected_method}_{mode}.parquet",
        )
        cws_reference_paths[mode] = str(cws_reference_path)

        retention = make_reference_retention_curves(cws_reference, residual_col="cws_ref_resid")
        retention_path = output_dir / f"{cfp.city}_cws_reference_retention_curves_{selected_method}_{mode}.csv"
        retention.to_csv(retention_path, index=False)
        retention_curve_paths[mode] = str(retention_path)

    selected_mode_key = f"{selected_method}__{primary_calibration_mode}"
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "run_label": final_run_label,
        "requested_run_label": requested_run_label,
        "config": asdict(config),
        "city_project_folder": config.city,
        "cfp_city": cfp.city,
        "project_dir": str(project_dir),
        "benchmark_dir": str(benchmark_dir),
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "selected_reference_method": selected_method,
        "primary_calibration_mode": primary_calibration_mode,
        "calibration_modes_fitted": calibration_modes,
        "station_coverage_audit_path": str(coverage_audit_path),
        "ows_frame_path": str(ows_frame_path),
        "ows_catboost_cv_frame_path": str(ows_cat_path) if ows_cat_path is not None else None,
        "ows_reference_paths": ows_reference_paths,
        "ows_primary_reference_path": ows_reference_paths.get(selected_mode_key),
        "ows_summary_paths_by_method_mode": ows_summary_paths_by_mode,
        "cws_reference_paths_by_calibration_mode": cws_reference_paths,
        "cws_primary_reference_path": cws_reference_paths.get(primary_calibration_mode),
        "ows_local_context_paths": ows_local_context_paths,
        "cws_local_context_paths": cws_local_context_paths,
        "retention_curve_paths_by_calibration_mode": retention_curve_paths,
        "primary_retention_curve_path": retention_curve_paths.get(primary_calibration_mode),
        "calibration_paths_by_method_mode": calibration_paths_by_method_mode,
        "catboost_model_path": str(catboost_model_path) if catboost_model_path is not None else None,
        "catboost_feature_cols": catboost_feature_cols,
        "catboost_categorical_cols": catboost_categorical_cols,
        "thresholds_by_method_mode": {
            f"{method}__{mode}": cal.get("thresholds")
            for method, cals in calibrations_by_method.items()
            for mode, cal in cals.items()
        },
        "idw_thresholds_primary_mode": idw_calibrations.get(primary_calibration_mode, {}).get("thresholds"),
        "selected_thresholds_primary_mode": selected_calibration.get("thresholds"),
        "valid_start": config.valid_start,
        "test_start": config.test_start,
        "year_span": raw["year_span"],
        "n_ows_rows": int(len(ows_long)),
        "n_cws_rows": int(len(cws_all)),
        "n_ows_stations": int(len(ows_obs_ids)),
        "n_cws_stations": int(cws_all[["network", "station_id"]].drop_duplicates().shape[0]),
    }
    manifest_path = output_dir / f"{cfp.city}_ows_reference_manifest_{selected_method}_{primary_calibration_mode}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_safe_json_default)

    print("Saved OWS primary reference predictions:", manifest["ows_primary_reference_path"])
    print("Saved CWS primary reference residual features:", manifest["cws_primary_reference_path"])
    print("Saved manifest:", manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OWS reference model and CWS residual features.")
    parser.add_argument("--city", default="project_id")
    parser.add_argument("--k-nearest-ows", type=int, default=8)
    parser.add_argument("--ows-radius-km", type=float, default=8.0)
    parser.add_argument("--no-radius", action="store_true", help="Ignore radius and always use nearest K OWS stations.")
    parser.add_argument("--valid-start", default="2021-10-01")
    parser.add_argument("--test-start", default="2021-11-01")
    parser.add_argument("--calibration-mode", choices=["time_train", "all_ows_reference"], default="time_train")
    parser.add_argument("--run-catboost-regressor", action="store_true")
    parser.add_argument("--catboost-cv-mode", choices=["group_kfold", "leave_one_station_out"], default="group_kfold")
    parser.add_argument("--catboost-n-splits", type=int, default=5)
    parser.add_argument("--catboost-iterations", type=int, default=1200)
    parser.add_argument("--catboost-thread-count", type=int, default=15)
    parser.add_argument("--catboost-use-gpu", action="store_true")
    parser.add_argument("--reference-method", choices=["auto", "idw", "catboost"], default="auto")
    parser.add_argument("--abnormal-quantile", type=float, default=0.995)
    parser.add_argument("--sigma-floor-c", type=float, default=0.20)
    parser.add_argument("--output-subdir", default="ows_reference")
    parser.add_argument("--run-label", default=None, help="Optional human-readable run label. Existing labels get repeat suffixes unless overwrite is enabled.")
    parser.add_argument("--single-calibration-mode", action="store_true", help="Only fit --calibration-mode instead of fitting both time_train and all_ows_reference.")
    parser.add_argument("--overwrite-existing-run", action="store_true", help="Allow writing into an existing non-empty run-label directory.")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ReferenceModelConfig:
    return ReferenceModelConfig(
        city=args.city,
        k_nearest_ows=args.k_nearest_ows,
        ows_radius_km=None if args.no_radius else args.ows_radius_km,
        valid_start=args.valid_start,
        test_start=args.test_start,
        calibration_mode=args.calibration_mode,
        run_catboost_regressor=args.run_catboost_regressor,
        catboost_cv_mode=args.catboost_cv_mode,
        catboost_n_splits=args.catboost_n_splits,
        catboost_iterations=args.catboost_iterations,
        catboost_thread_count=args.catboost_thread_count,
        catboost_use_gpu=args.catboost_use_gpu,
        reference_method=args.reference_method,
        abnormal_quantile=args.abnormal_quantile,
        sigma_floor_c=args.sigma_floor_c,
        output_subdir=args.output_subdir,
        run_label=args.run_label,
        save_both_calibration_modes=not args.single_calibration_mode,
        overwrite_existing_run=args.overwrite_existing_run,
        rebuild=args.rebuild,
    )


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    config = config_from_args(args)
    return run_reference_pipeline(config)


if __name__ == "__main__":
    main()
