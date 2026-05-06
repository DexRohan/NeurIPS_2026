"""Supplemental helper functions for CWS/OWS feature construction."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import qc_benchmark_helpers as _base
from qc_benchmark_helpers import *



_ensure_utc_datetime = _base._ensure_utc_datetime
_clean_station_series = _base._clean_station_series
_maybe_repair_single_column_csv_df = _base._maybe_repair_single_column_csv_df
infer_station_column = _base.infer_station_column
normalize_network_name = _base.normalize_network_name
downcast_dataframe = _base.downcast_dataframe
merge_stationwise_asof_features = _base.merge_stationwise_asof_features


def _station_id_variants_one(value: object) -> list[str]:
    """Return plausible join-key variants for one station id.

    Netatmo satellite covariate files often store ids as
    base_station_id_module_id, while the hourly CWS/ERA5/QC tables often use
    only the module_id. The observed station ids decide the safest key.
    """
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"nan", "none", "nat"}:
        return []
    variants = [raw]
    if "_" in raw:
        parts = [part.strip() for part in raw.split("_") if str(part).strip()]
        if parts:
            variants.append(parts[-1])
            variants.append(parts[0])
    out = []
    seen = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def canonicalize_station_series(
    s: pd.Series,
    observed_station_ids: Iterable[str] | None = None,
    network_name: object | None = None,
) -> pd.Series:
    """Canonicalize station ids for joins.

    Example fixed by this function:
      <base_station_id>_<module_id>  ->  <module_id>
    when the observed Netatmo temperature table uses only the module id.
    """
    raw = s.astype("string").str.strip()

    obs_set: set[str] | None = None
    if observed_station_ids is not None:
        obs_set = set(
            pd.Series(list(observed_station_ids))
            .astype("string")
            .str.strip()
            .dropna()
            .astype(str)
            .unique()
        )

    if isinstance(network_name, pd.Series):
        net = network_name.reindex(s.index).astype("string").fillna("").map(normalize_network_name)
    else:
        net_value = normalize_network_name(network_name) if network_name is not None else ""
        net = pd.Series([net_value for _ in range(len(s))], index=s.index)

    def _choose(value: object, network_value: object) -> str:
        variants = _station_id_variants_one(value)
        if not variants:
            return ""
        if obs_set:
            for v in variants:
                if v in obs_set:
                    return v
        if str(network_value).lower() == "netatmo" and len(variants) > 1:
            return variants[1]
        return variants[0]

    return pd.Series(
        [_choose(v, n) for v, n in zip(raw.tolist(), net.tolist())],
        index=s.index,
        dtype="string",
    )


def station_id_variant_overlap_report(
    df: pd.DataFrame,
    observed_station_ids: Iterable[str],
    network_name: str | None = None,
    station_col: str | None = None,
    value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Report exact/prefix/suffix/canonical station-id overlap before filtering."""
    x = _maybe_repair_single_column_csv_df(df.copy())
    chosen = infer_station_column(
        x,
        observed_station_ids=observed_station_ids,
        network_name=network_name,
        explicit_station_col=station_col,
    )
    obs = set(
        pd.Series(list(observed_station_ids))
        .astype("string")
        .str.strip()
        .dropna()
        .astype(str)
        .unique()
    )
    raw = x[chosen].astype("string").str.strip()

    def _candidate(series: pd.Series, pos: int) -> pd.Series:
        def _one(v):
            variants = _station_id_variants_one(v)
            if len(variants) > pos:
                return variants[pos]
            return str(v).strip() if pd.notna(v) else ""
        return series.map(_one)

    candidates = {
        "exact": raw,
        "underscore_suffix": _candidate(raw, 1),
        "underscore_prefix": _candidate(raw, 2),
        "canonical": canonicalize_station_series(raw, observed_station_ids=obs, network_name=network_name),
    }

    rows = []
    for name, series in candidates.items():
        ids = set(series.dropna().astype(str).unique())
        rows.append({
            "network": normalize_network_name(network_name) if network_name is not None else None,
            "station_col": chosen,
            "candidate": name,
            "n_candidate_ids": int(len(ids)),
            "n_observed_ids": int(len(obs)),
            "n_overlap": int(len(ids & obs)),
            "overlap_frac_observed": float(len(ids & obs) / len(obs)) if obs else np.nan,
            "overlap_frac_candidate": float(len(ids & obs) / len(ids)) if ids else np.nan,
        })
    out = pd.DataFrame(rows)
    if value_cols:
        for c in value_cols:
            if c in x.columns:
                out[f"{c}_nonmissing_rows"] = int(pd.to_numeric(x[c], errors="coerce").notna().sum())
    return out


def prep_hourly_env(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    network_name: str,
    observed_station_ids: Iterable[str] | None = None,
    station_col: str | None = None,
    date_col: str = "date",
    filter_to_observed: bool = True,
) -> pd.DataFrame:
    """Prepare an hourly environmental table for merge with robust station-id aliases."""
    x = _maybe_repair_single_column_csv_df(df.copy())

    if date_col not in x.columns:
        if getattr(x.index, "name", None) == date_col or pd.api.types.is_datetime64_any_dtype(x.index):
            x = x.reset_index()
            if "index" in x.columns and date_col not in x.columns:
                x = x.rename(columns={"index": date_col})
        else:
            raise ValueError(f"Hourly table for {network_name} must contain a '{date_col}' column")

    chosen_station_col = infer_station_column(
        x,
        observed_station_ids=observed_station_ids,
        network_name=network_name,
        explicit_station_col=station_col,
    )

    x["date"] = _ensure_utc_datetime(x[date_col])
    x["station_id"] = canonicalize_station_series(
        x[chosen_station_col],
        observed_station_ids=observed_station_ids,
        network_name=network_name,
    )

    if observed_station_ids is not None and filter_to_observed:
        obs = set(pd.Series(list(observed_station_ids)).astype(str).str.strip().dropna().unique())
        x = x[x["station_id"].astype(str).isin(obs)].copy()
    x["network"] = normalize_network_name(network_name)

    keep = ["date", "station_id", "network"] + [c for c in value_cols if c in x.columns]
    x = x[keep].copy()

    for c in value_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce", downcast="float")

    group_cols = ["date", "station_id", "network"]
    if x.duplicated(group_cols).any():
        agg = {c: "mean" for c in x.columns if c not in group_cols}
        x = x.groupby(group_cols, as_index=False).agg(agg)

    return downcast_dataframe(x.sort_values(group_cols))


def prepare_ows_env_asof_table(
    ows_env: pd.DataFrame | None,
    ows_station_ids: Iterable[str],
    dates: Iterable,
    value_col: str,
    tolerance: str | pd.Timedelta = "3h",
    direction: str = "nearest",
    age_unit: str = "hours",
) -> pd.DataFrame | None:
    """Create an OWS station-date environmental table aligned to CWS timestamps."""
    if ows_env is None or value_col not in ows_env.columns:
        return None

    station_ids = (
        pd.Series(list(ows_station_ids))
        .astype("string")
        .str.strip()
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    date_index = pd.Index(_ensure_utc_datetime(pd.Series(list(dates))).dropna().unique()).sort_values()
    if len(station_ids) == 0 or len(date_index) == 0:
        return None

    grid = pd.MultiIndex.from_product(
        [date_index, station_ids], names=["date", "station_id"]
    ).to_frame(index=False)
    grid["network"] = "OWS"

    env = ows_env[["date", "station_id", "network", value_col]].copy()
    env["date"] = _ensure_utc_datetime(env["date"])
    env["station_id"] = _clean_station_series(env["station_id"])
    env["network"] = env["network"].map(normalize_network_name)
    env[value_col] = pd.to_numeric(env[value_col], errors="coerce", downcast="float")
    env = env.dropna(subset=["date", "station_id"])
    env = env[env[value_col].notna()].copy()

    if len(env) == 0:
        out = grid.copy()
        out[value_col] = np.nan
        out[f"{value_col}_missing"] = 1
        out[f"{value_col}_age_{age_unit}"] = np.nan
        return downcast_dataframe(out)

    out = merge_stationwise_asof_features(
        grid,
        env,
        value_cols=[value_col],
        rename_map={value_col: value_col},
        tolerance=tolerance,
        direction=direction,
        add_missing_indicators=True,
        add_age=True,
        age_unit=age_unit,
    )
    return downcast_dataframe(out)


def build_local_ows_env_features(
    cws_keys: pd.DataFrame,
    ows_env_hourly: pd.DataFrame | None,
    pair_table: pd.DataFrame,
    value_col: str,
    feature_prefix: str,
    age_col: str | None = None,
    k_nearest: int | None = None,
    chunk_freq: str | None = "M",
) -> pd.DataFrame | None:
    """Aggregate OWS-side LST/NDVI/etc. over nearby OWS anchors."""
    if ows_env_hourly is None or value_col not in ows_env_hourly.columns:
        return None

    required_pairs = {"cws_station_id", "cws_network", "ows_station_id", "ows_rank", "ows_dist_km", "ows_weight_idw"}
    missing_pairs = required_pairs - set(pair_table.columns)
    if missing_pairs:
        raise KeyError(f"pair_table missing required columns: {sorted(missing_pairs)}")

    keys = cws_keys[["date", "station_id", "network"]].drop_duplicates().copy()
    keys["date"] = _ensure_utc_datetime(keys["date"])
    keys["station_id"] = _clean_station_series(keys["station_id"])
    keys["network"] = keys["network"].map(normalize_network_name)

    pairs = pair_table.copy()
    pairs["cws_station_id"] = _clean_station_series(pairs["cws_station_id"])
    pairs["cws_network"] = pairs["cws_network"].map(normalize_network_name)
    pairs["ows_station_id"] = _clean_station_series(pairs["ows_station_id"])
    pairs["ows_rank"] = pd.to_numeric(pairs["ows_rank"], errors="coerce")
    pairs["ows_dist_km"] = pd.to_numeric(pairs["ows_dist_km"], errors="coerce")
    pairs["ows_weight_idw"] = pd.to_numeric(pairs["ows_weight_idw"], errors="coerce")
    if k_nearest is not None:
        pairs = pairs[pairs["ows_rank"] <= k_nearest].copy()

    env_cols = ["date", "station_id", "network", value_col]
    if age_col is not None and age_col in ows_env_hourly.columns:
        env_cols.append(age_col)
    else:
        age_col = None

    env = ows_env_hourly[env_cols].copy()
    env["date"] = _ensure_utc_datetime(env["date"])
    env["station_id"] = _clean_station_series(env["station_id"])
    env["network"] = env["network"].map(normalize_network_name)
    env[value_col] = pd.to_numeric(env[value_col], errors="coerce", downcast="float")
    if age_col is not None:
        env[age_col] = pd.to_numeric(env[age_col], errors="coerce", downcast="float")
    env = env.dropna(subset=["date", "station_id"])

    if chunk_freq is None:
        chunks = [("__all__", env)]
    else:
        tmp = env.copy()
        tmp["_chunk"] = tmp["date"].dt.to_period(chunk_freq).astype(str)
        chunks = list(tmp.groupby("_chunk", sort=True))

    group_cols = ["date", "station_id", "network"]
    results = []

    for _, env_sub in chunks:
        if "_chunk" in env_sub.columns:
            env_sub = env_sub.drop(columns=["_chunk"])
        if len(env_sub) == 0:
            continue

        chunk_dates = pd.Index(env_sub["date"].dropna().unique())
        base_sub = keys[keys["date"].isin(chunk_dates)].copy()
        if len(base_sub) == 0:
            continue

        joined = base_sub.merge(
            pairs,
            left_on=["station_id", "network"],
            right_on=["cws_station_id", "cws_network"],
            how="left",
            validate="many_to_many",
        )
        joined = joined.merge(
            env_sub,
            left_on=["date", "ows_station_id"],
            right_on=["date", "station_id"],
            how="left",
            suffixes=("", "_ows_env"),
            validate="many_to_many",
        )
        for c in ["station_id_ows_env", "network_ows_env"]:
            if c in joined.columns:
                joined = joined.drop(columns=[c])

        joined[value_col] = pd.to_numeric(joined[value_col], errors="coerce")
        joined["_valid_value"] = joined[value_col].notna()
        joined["_w"] = pd.to_numeric(joined["ows_weight_idw"], errors="coerce").astype("float64")
        joined["_w_valid"] = joined["_w"].where(joined["_valid_value"], 0.0)
        joined["_wv"] = joined["_w"] * joined[value_col].fillna(0.0).astype("float64")

        basic = (
            joined.groupby(group_cols, as_index=False)[value_col]
            .agg(
                **{
                    f"ows_local_n_{feature_prefix}": "count",
                    f"ows_local_mean_{feature_prefix}": "mean",
                    f"ows_local_median_{feature_prefix}": "median",
                    f"ows_local_std_{feature_prefix}": "std",
                    f"ows_local_min_{feature_prefix}": "min",
                    f"ows_local_max_{feature_prefix}": "max",
                }
            )
        )
        basic[f"ows_local_range_{feature_prefix}"] = basic[f"ows_local_max_{feature_prefix}"] - basic[f"ows_local_min_{feature_prefix}"]

        weighted = (
            joined.groupby(group_cols, as_index=False)
            .agg(
                **{
                    f"ows_idw_{feature_prefix}_weight_sum": ("_w_valid", "sum"),
                    f"_ows_w_{feature_prefix}_sum": ("_wv", "sum"),
                    f"ows_local_mean_dist_{feature_prefix}_km": ("ows_dist_km", "mean"),
                    f"ows_local_min_dist_{feature_prefix}_km": ("ows_dist_km", "min"),
                }
            )
        )
        denom = weighted[f"ows_idw_{feature_prefix}_weight_sum"].replace(0, np.nan)
        weighted[f"ows_idw_{feature_prefix}"] = weighted[f"_ows_w_{feature_prefix}_sum"] / denom
        weighted = weighted.drop(columns=[f"_ows_w_{feature_prefix}_sum"])

        valid_joined = joined[joined["_valid_value"]].copy()
        if len(valid_joined):
            nearest = (
                valid_joined.sort_values(group_cols + ["ows_rank", "ows_dist_km"])
                .groupby(group_cols, as_index=False)
                .first()[group_cols + [value_col, "ows_rank", "ows_dist_km"]]
                .rename(columns={
                    value_col: f"ows_nearest_{feature_prefix}",
                    "ows_rank": f"ows_nearest_{feature_prefix}_rank",
                    "ows_dist_km": f"ows_nearest_{feature_prefix}_dist_km",
                })
            )
        else:
            nearest = basic[group_cols].copy()
            nearest[f"ows_nearest_{feature_prefix}"] = np.nan
            nearest[f"ows_nearest_{feature_prefix}_rank"] = np.nan
            nearest[f"ows_nearest_{feature_prefix}_dist_km"] = np.nan

        out = basic.merge(weighted, on=group_cols, how="left", validate="one_to_one")
        out = out.merge(nearest, on=group_cols, how="left", validate="one_to_one")

        if age_col is not None and age_col in joined.columns:
            joined[age_col] = pd.to_numeric(joined[age_col], errors="coerce")
            joined["_w_age"] = joined["_w"].where(joined["_valid_value"] & joined[age_col].notna(), 0.0)
            joined["_w_age_value"] = joined["_w_age"] * joined[age_col].fillna(0.0).astype("float64")
            age_basic = (
                joined[joined["_valid_value"]]
                .groupby(group_cols, as_index=False)[age_col]
                .agg(**{
                    f"ows_local_mean_{feature_prefix}_age": "mean",
                    f"ows_local_median_{feature_prefix}_age": "median",
                })
            )
            age_weighted = (
                joined.groupby(group_cols, as_index=False)
                .agg(**{
                    f"_ows_{feature_prefix}_age_w_sum": ("_w_age", "sum"),
                    f"_ows_{feature_prefix}_age_wv_sum": ("_w_age_value", "sum"),
                })
            )
            denom_age = age_weighted[f"_ows_{feature_prefix}_age_w_sum"].replace(0, np.nan)
            age_weighted[f"ows_idw_{feature_prefix}_age"] = age_weighted[f"_ows_{feature_prefix}_age_wv_sum"] / denom_age
            age_weighted = age_weighted.drop(columns=[f"_ows_{feature_prefix}_age_w_sum", f"_ows_{feature_prefix}_age_wv_sum"])

            if len(valid_joined):
                nearest_age = (
                    valid_joined.sort_values(group_cols + ["ows_rank", "ows_dist_km"])
                    .groupby(group_cols, as_index=False)
                    .first()[group_cols + [age_col]]
                    .rename(columns={age_col: f"ows_nearest_{feature_prefix}_age"})
                )
            else:
                nearest_age = basic[group_cols].copy()
                nearest_age[f"ows_nearest_{feature_prefix}_age"] = np.nan

            out = out.merge(age_basic, on=group_cols, how="left", validate="one_to_one")
            out = out.merge(age_weighted, on=group_cols, how="left", validate="one_to_one")
            out = out.merge(nearest_age, on=group_cols, how="left", validate="one_to_one")

        results.append(out)

    if results:
        result = pd.concat(results, ignore_index=True)
    else:
        result = keys.copy()
        result[f"ows_local_n_{feature_prefix}"] = 0

    result = keys.merge(result, on=group_cols, how="left", validate="one_to_one")
    n_col = f"ows_local_n_{feature_prefix}"
    if n_col in result.columns:
        result[n_col] = result[n_col].fillna(0).astype("int16")
    else:
        result[n_col] = 0
    return downcast_dataframe(result)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extend base interactions with CWS-vs-local-OWS LST/NDVI deltas."""
    x = _base.add_interaction_features(df)

    for c in ["LST_C", "LST_C_asof", "LST_C_gapfill_3h", "LST_C_gapfill_clim"]:
        if c in x.columns and "ows_idw_lst" in x.columns:
            safe = c.replace("LST_C", "lst").lower()
            x[f"{safe}_minus_ows_idw_lst"] = x[c] - x["ows_idw_lst"]
            x[f"{safe}_minus_ows_idw_lst_abs"] = (x[c] - x["ows_idw_lst"]).abs()
        if c in x.columns and "ows_nearest_lst" in x.columns:
            safe = c.replace("LST_C", "lst").lower()
            x[f"{safe}_minus_ows_nearest_lst"] = x[c] - x["ows_nearest_lst"]
            x[f"{safe}_minus_ows_nearest_lst_abs"] = (x[c] - x["ows_nearest_lst"]).abs()

    if "NDVI" in x.columns and "ows_idw_ndvi" in x.columns:
        x["ndvi_minus_ows_idw_ndvi"] = x["NDVI"] - x["ows_idw_ndvi"]
        x["ndvi_minus_ows_idw_ndvi_abs"] = (x["NDVI"] - x["ows_idw_ndvi"]).abs()
    if "NDVI" in x.columns and "ows_nearest_ndvi" in x.columns:
        x["ndvi_minus_ows_nearest_ndvi"] = x["NDVI"] - x["ows_nearest_ndvi"]
        x["ndvi_minus_ows_nearest_ndvi_abs"] = (x["NDVI"] - x["ows_nearest_ndvi"]).abs()

    return downcast_dataframe(x)


_base_build_master_feature_table_benchmark = _base.build_master_feature_table_benchmark


def build_master_feature_table_benchmark(*args, **kwargs):
    """Wrapper around the base builder that also adds CWS-vs-OWS LST/NDVI deltas."""
    df, labels_summary = _base_build_master_feature_table_benchmark(*args, **kwargs)
    df = add_interaction_features(df)
    return df, labels_summary
