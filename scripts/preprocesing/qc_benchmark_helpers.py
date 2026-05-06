
"""Helper functions for CWS/OWS quality-control benchmarking."""
from __future__ import annotations

import csv
import io
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _ensure_utc_datetime(obj) -> pd.Series:
    return pd.to_datetime(obj, utc=True, errors="coerce")


def _clean_station_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _station_id_alias_variants(s: pd.Series) -> dict[str, pd.Series]:
    """Return common station-ID aliases used by the project files.

    The most important case is Netatmo:
      LST/NDVI sometimes store ``parent_module`` IDs such as
      ``<base_station_id>_<module_id>`` while TA/ERA5/QC tables often use only the
      outdoor module ID after the underscore.
    """
    raw = _clean_station_series(s)
    variants: dict[str, pd.Series] = {"raw": raw}


    if raw.astype(str).str.contains("_", regex=False, na=False).any():
        variants["after_last_underscore"] = raw.astype(str).str.rsplit("_", n=1).str[-1].str.strip()
        variants["before_first_underscore"] = raw.astype(str).str.split("_", n=1).str[0].str.strip()



    if raw.astype(str).str.contains(r"\|\|", regex=True, na=False).any():
        variants["after_last_doublepipe"] = raw.astype(str).str.rsplit("||", n=1).str[-1].str.strip()
    if raw.astype(str).str.contains("/", regex=False, na=False).any():
        variants["after_last_slash"] = raw.astype(str).str.rsplit("/", n=1).str[-1].str.strip()

    return variants


def _observed_id_set(observed_station_ids: Iterable[str] | None) -> set[str]:
    if observed_station_ids is None:
        return set()
    return set(
        pd.Series(list(observed_station_ids))
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda z: ~z.isin(["", "nan", "None", "NaT"])]
        .unique()
    )


def resolve_station_id_alias(
    values: pd.Series,
    observed_station_ids: Iterable[str] | None = None,
    network_name: str | None = None,
    prefer_raw_without_observed: bool = True,
) -> tuple[pd.Series, str, dict[str, int]]:
    """Choose the station-ID representation with the best overlap to observed IDs.

    This prevents silent loss of Netatmo LST/NDVI rows when one file uses a
    composite parent+module ID and the main CWS table uses only the module ID.
    """
    variants = _station_id_alias_variants(values)
    obs = _observed_id_set(observed_station_ids)

    if not obs:


        raw = variants["raw"]
        return raw, "raw", {k: 0 for k in variants}

    scores: dict[str, int] = {}
    for name, ser in variants.items():
        vals = set(ser.dropna().astype(str).str.strip().unique())
        scores[name] = len(vals & obs)

    best = max(scores, key=scores.get)
    if scores.get(best, 0) > 0:
        return variants[best], best, scores

    return variants["raw"], "raw", scores


def station_id_alias_report(
    observed_station_ids: Iterable[str],
    df: pd.DataFrame,
    station_col: str | None = None,
    network_name: str | None = None,
    max_examples: int = 5,
) -> pd.DataFrame:
    """Report overlap for raw and derived station-ID aliases.

    Use this before preparing ERA5/LST/NDVI files. For Netatmo LST/NDVI, a high
    overlap for ``after_last_underscore`` means the file stores
    ``parent_module`` but your TA/QC table uses only the module ID.
    """
    x = _maybe_repair_single_column_csv_df(df.copy())
    if station_col is None:
        station_col = infer_station_column(
            x,
            observed_station_ids=observed_station_ids,
            network_name=network_name,
            explicit_station_col=None,
        )
    if station_col not in x.columns:
        raise KeyError(f"station_col={station_col!r} not in dataframe columns")

    obs = _observed_id_set(observed_station_ids)
    rows = []
    for alias_name, ser in _station_id_alias_variants(x[station_col]).items():
        vals = pd.Series(ser).dropna().astype(str).str.strip()
        vals = vals[~vals.isin(["", "nan", "None", "NaT"])]
        unique_vals = set(vals.unique())
        overlap = sorted(unique_vals & obs)
        rows.append({
            "network": normalize_network_name(network_name) if "normalize_network_name" in globals() else str(network_name),
            "station_col": station_col,
            "alias": alias_name,
            "n_ids_after_alias": int(len(unique_vals)),
            "n_observed_ids": int(len(obs)),
            "n_overlap": int(len(overlap)),
            "overlap_fraction_observed": float(len(overlap) / len(obs)) if obs else np.nan,
            "example_overlap_ids": ", ".join(overlap[:max_examples]),
            "example_alias_ids": ", ".join(sorted(unique_vals)[:max_examples]),
        })
    return pd.DataFrame(rows).sort_values("n_overlap", ascending=False, ignore_index=True)


def memory_report(df: pd.DataFrame, name: str = "df") -> None:
    mem_gb = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"{name} memory: {mem_gb:.2f} GB")


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="float")
        elif pd.api.types.is_integer_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="integer")
        elif x[col].dtype == "object":

            nunique = x[col].nunique(dropna=True)
            if nunique > 0 and nunique <= min(500, max(20, len(x) // 50)):
                try:
                    x[col] = x[col].astype("category")
                except Exception:
                    pass
    return x


def _maybe_repair_single_column_csv_df(df: pd.DataFrame, sep: str = ",") -> pd.DataFrame:
    """Repair a DataFrame that was read with the wrong separator and ended up as one column."""
    if df.shape[1] != 1:
        return df

    col = df.columns[0]

    if sep not in str(col):
        return df

    buffer = io.StringIO()
    buffer.write(str(col) + "\n")
    series = df.iloc[:, 0].astype(str)
    for val in series:
        buffer.write(val + "\n")
    buffer.seek(0)

    repaired = pd.read_csv(buffer, sep=sep)
    return repaired


def read_csv_auto(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV with delimiter auto-detection."""

    kwargs = dict(kwargs)
    if "sep" not in kwargs:
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    return pd.read_csv(path, **kwargs)


def _candidate_station_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "module_final",
        "src_id",
        "station_id",
        "_id",
        "station",
        "id",
    ]
    return [c for c in preferred if c in df.columns]


def infer_station_column(
    df: pd.DataFrame,
    observed_station_ids: Iterable[str] | None = None,
    network_name: str | None = None,
    explicit_station_col: str | None = None,
) -> str:
    """Infer which column should be used as the station join key.

    Uses overlap with observed station IDs when available. This is crucial for
    Netatmo metadata where `module_final` matches the TA table but `station_id`
    is a composite metadata ID that should NOT be used for joins.
    """
    if explicit_station_col is not None:
        if explicit_station_col not in df.columns:
            raise KeyError(f"Explicit station column '{explicit_station_col}' not in columns {list(df.columns)}")
        return explicit_station_col

    candidates = _candidate_station_columns(df)
    if not candidates:
        raise ValueError(f"Could not infer station column from columns: {list(df.columns)}")

    obs_set = None
    if observed_station_ids is not None:
        obs_set = set(pd.Series(list(observed_station_ids)).astype(str).str.strip().dropna().unique())

    if obs_set:
        scores = {}
        for c in candidates:
            vals = set(_clean_station_series(df[c]).dropna().unique())
            scores[c] = len(vals & obs_set)
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best


    network_lower = (network_name or "").lower()
    if "ows" in network_lower and "src_id" in candidates:
        return "src_id"
    if ("netatmo" in network_lower or "wunderground" in network_lower) and "module_final" in candidates:
        return "module_final"
    if "station_id" in candidates:
        return "station_id"
    return candidates[0]


def _safe_stack_frame(x: pd.DataFrame) -> pd.Series:
    """Version-compatible stack that works on older and newer pandas."""
    try:
        return x.stack(future_stack=True)
    except TypeError:

        return x.stack(dropna=False)
    except ValueError as e:

        if "dropna must be unspecified" in str(e):
            return x.stack(future_stack=True)
        raise


def maybe_subsample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def get_station_ids_from_wide(df: pd.DataFrame, date_col: str = "date") -> pd.Index:
    x = ensure_wide_hourly_datetime_index(df, date_col=date_col)
    return pd.Index([str(c).strip() for c in x.columns])


def report_station_id_alignment(
    observed_station_ids: Iterable[str],
    df_meta: pd.DataFrame | None = None,
    df_hourly: pd.DataFrame | None = None,
    network_name: str = "",
    explicit_station_col: str | None = None,
) -> dict:
    """Return a small overlap report to confirm the chosen station key."""
    obs = pd.Index(pd.Series(list(observed_station_ids)).astype(str).str.strip().unique())
    report = {
        "network": network_name,
        "n_observed_station_ids": int(len(obs)),
    }

    if df_meta is not None:
        meta = _maybe_repair_single_column_csv_df(df_meta.copy())
        chosen = infer_station_column(meta, observed_station_ids=obs, network_name=network_name, explicit_station_col=explicit_station_col)
        resolved_ids, alias_used, alias_scores = resolve_station_id_alias(
            meta[chosen],
            observed_station_ids=obs,
            network_name=network_name,
        )
        meta_ids = pd.Index(pd.Series(resolved_ids).dropna().astype(str).str.strip().unique())
        report.update({
            "meta_station_col": chosen,
            "meta_station_alias_used": alias_used,
            "meta_station_alias_scores": alias_scores,
            "n_meta_station_ids": int(len(meta_ids)),
            "n_overlap_obs_meta": int(len(meta_ids.intersection(obs))),
            "n_obs_missing_in_meta": int(len(obs.difference(meta_ids))),
            "n_meta_extra_not_in_obs": int(len(meta_ids.difference(obs))),
        })

    if df_hourly is not None:
        hourly = _maybe_repair_single_column_csv_df(df_hourly.copy())
        chosen = infer_station_column(hourly, observed_station_ids=obs, network_name=network_name, explicit_station_col=explicit_station_col)
        resolved_ids, alias_used, alias_scores = resolve_station_id_alias(
            hourly[chosen],
            observed_station_ids=obs,
            network_name=network_name,
        )
        hourly_ids = pd.Index(pd.Series(resolved_ids).dropna().astype(str).str.strip().unique())
        report.update({
            "hourly_station_col": chosen,
            "hourly_station_alias_used": alias_used,
            "hourly_station_alias_scores": alias_scores,
            "n_hourly_station_ids": int(len(hourly_ids)),
            "n_overlap_obs_hourly": int(len(hourly_ids.intersection(obs))),
            "n_obs_missing_in_hourly": int(len(obs.difference(hourly_ids))),
            "n_hourly_extra_not_in_obs": int(len(hourly_ids.difference(obs))),
        })

    return report


def ensure_wide_hourly_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    x = df.copy()


    x = _maybe_repair_single_column_csv_df(x)

    if date_col in x.columns:
        x[date_col] = _ensure_utc_datetime(x[date_col])
        x = x.set_index(date_col)
    else:
        x.index = _ensure_utc_datetime(x.index)


    x.columns = [str(c).strip() for c in x.columns]
    return x.sort_index()


def dedupe_temp_wide(df: pd.DataFrame, how: str = "mean", date_col: str = "date") -> pd.DataFrame:
    x = ensure_wide_hourly_datetime_index(df, date_col=date_col)

    if not x.index.duplicated().any():
        return x

    if how == "first":
        x = x[~x.index.duplicated(keep="first")]
    elif how == "last":
        x = x[~x.index.duplicated(keep="last")]
    elif how == "mean":
        x = x.groupby(level=0).mean(numeric_only=True)
    else:
        raise ValueError("how must be one of: 'mean', 'first', 'last'")

    return x.sort_index()


def dedupe_flag_wide(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    x = _maybe_repair_single_column_csv_df(df.copy())

    if date_col is None:
        date_col = "date" if "date" in x.columns else x.columns[0]

    x = x.rename(columns={date_col: "date"})
    x["date"] = _ensure_utc_datetime(x["date"])


    for c in x.columns:
        if c != "date":
            x[c] = pd.to_numeric(x[c])

    if x["date"].duplicated().any():
        value_cols = [c for c in x.columns if c != "date"]

        x = x.groupby("date", as_index=False)[value_cols].max()

    return x.sort_values("date").reset_index(drop=True)


def melt_ta_data(
    df: pd.DataFrame,
    network: str,
    value_name: str = "temp_raw",
    drop_missing_values: bool = True,
    date_col: str = "date",
) -> pd.DataFrame:
    """Convert a wide hourly TA table into long format.

    Handles either:
    - date as index
    - date as a named column
    """
    x = dedupe_temp_wide(df, how="mean", date_col=date_col)

    if x.index.duplicated().any():
        raise ValueError(f"{network}: duplicate timestamps remain before melt")

    out = (
        _safe_stack_frame(x.rename_axis("date"))
        .rename(value_name)
        .reset_index()
    )
    out.columns = ["date", "station_id", value_name]
    out["date"] = _ensure_utc_datetime(out["date"])
    out["station_id"] = _clean_station_series(out["station_id"])
    out["network"] = network

    if drop_missing_values:
        out = out[out[value_name].notna()].reset_index(drop=True)

    out[value_name] = pd.to_numeric(out[value_name], errors="coerce", downcast="float")
    out["network"] = out["network"].astype("category")
    return downcast_dataframe(out)


def build_labels(strict: pd.DataFrame, lenient: pd.DataFrame) -> pd.DataFrame:
    """Build weak labels from strict/lenient QC agreement.

    IMPORTANT:
    QC files produced by the existing pipeline may contain both raw temperature
    columns and corresponding *_is_outlier flag columns, for example:

        station_id, station_id_is_outlier, ...

    When *_is_outlier columns are present, ONLY those columns are valid labels.
    Melting raw temperature columns as labels creates the failure mode where the
    model learns to classify temp_raw == 0 or temp_raw == 1.
    """
    strict = dedupe_flag_wide(strict)
    lenient = dedupe_flag_wide(lenient)

    def _reshape_flags(x: pd.DataFrame, value_name: str) -> pd.DataFrame:
        all_value_cols = [c for c in x.columns if c != "date"]
        if not all_value_cols:
            raise ValueError("QC table has no station/flag columns")

        flag_cols = [c for c in all_value_cols if str(c).endswith("_is_outlier")]



        value_cols = flag_cols if flag_cols else all_value_cols

        long = x.melt(
            id_vars=["date"],
            value_vars=value_cols,
            var_name="raw_col",
            value_name=value_name,
        )
        long["station_id"] = (
            long["raw_col"]
            .astype(str)
            .str.replace("_is_outlier", "", regex=False)
            .str.strip()
        )
        long = long.drop(columns=["raw_col"])
        long[value_name] = pd.to_numeric(long[value_name], errors="coerce")

        non_na = long[value_name].dropna()
        unexpected = sorted(
            pd.unique(non_na[~non_na.isin([0, 1, 0.0, 1.0])]).tolist()
        )
        if unexpected:
            raise ValueError(
                f"{value_name} contains non-binary values such as {unexpected[:10]}. "
                "This usually means raw temperature columns were used as labels."
            )

        dup_count = int(long.duplicated(["date", "station_id"]).sum())
        if dup_count:
            raise ValueError(
                f"{value_name} has {dup_count} duplicate date/station rows after reshaping. "
                "Check for duplicate station columns in the QC file."
            )

        return long

    strict_long = _reshape_flags(strict, "qc_flag_strict")
    lenient_long = _reshape_flags(lenient, "qc_flag_lenient")

    labels = strict_long.merge(
        lenient_long,
        on=["date", "station_id"],
        how="inner",
        validate="one_to_one",
    )

    labels["target_bad"] = np.where(
        (labels["qc_flag_strict"] == 1) & (labels["qc_flag_lenient"] == 1),
        1,
        np.where(
            (labels["qc_flag_strict"] == 0) & (labels["qc_flag_lenient"] == 0),
            0,
            np.nan,
        ),
    )

    labels = labels[["date", "station_id", "target_bad"]].copy()
    labels["station_id"] = _clean_station_series(labels["station_id"])
    labels["target_bad"] = pd.to_numeric(labels["target_bad"], downcast="float")
    return labels


def prep_station_meta(
    df_meta: pd.DataFrame,
    network_name: str,
    observed_station_ids: Iterable[str] | None = None,
    station_col: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "long",
    filter_to_observed: bool = True,
) -> pd.DataFrame:
    """Prepare station metadata and infer the correct join key.

    Important:
    - For Netatmo metadata, this will typically choose `module_final`, not the
      composite `station_id`.
    - For Wunderground metadata, this will choose `module_final`.
    - For OWS metadata, this will choose `src_id`.
    """
    x = _maybe_repair_single_column_csv_df(df_meta.copy())

    chosen_station_col = infer_station_column(
        x,
        observed_station_ids=observed_station_ids,
        network_name=network_name,
        explicit_station_col=station_col,
    )


    x["station_id_raw_for_join"] = _clean_station_series(x[chosen_station_col])
    x["station_id"], station_alias_used, station_alias_scores = resolve_station_id_alias(
        x["station_id_raw_for_join"],
        observed_station_ids=observed_station_ids,
        network_name=network_name,
    )

    if observed_station_ids is not None and filter_to_observed:
        obs = _observed_id_set(observed_station_ids)
        x = x[x["station_id"].isin(obs)].copy()

    if lat_col in x.columns:
        x = x.rename(columns={lat_col: "station_lat"})
    if lon_col in x.columns:
        x = x.rename(columns={lon_col: "station_long"})

    x["network"] = network_name


    for extra in ["module_final", "src_id", "_id", "station_id_raw", "station_uid"]:
        if extra in x.columns:

            pass
    if "station_id" in df_meta.columns and chosen_station_col != "station_id":
        x["station_id_meta_original"] = _clean_station_series(df_meta["station_id"])

    cat_cols = ["LC_point_lg", "LC_buffer_lg", "LCZ_point_lg", "LCZ_buffer_lg"]
    for c in cat_cols:
        if c in x.columns:
            x[c] = x[c].astype("string").fillna("Missing")

    keep_cols = [
        "station_id",
        "network",
        "station_long",
        "station_lat",
        "building_height_m",
        "elev_meters",

        "LC_point_lg",

        "LC_buffer_lg",
        "LC_buffer_fraction",

        "LCZ_point_lg",

        "LCZ_buffer_lg",
        "LCZ_buffer_fraction",
    ]

    keep_cols = [c for c in keep_cols if c in x.columns]

    x = x[keep_cols].drop_duplicates(["station_id", "network"])
    return downcast_dataframe(x)


def prep_hourly_env(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    network_name: str,
    observed_station_ids: Iterable[str] | None = None,
    station_col: str | None = None,
    date_col: str = "date",
    filter_to_observed: bool = True,
) -> pd.DataFrame:
    """Prepare an hourly environmental table for merge.

    Auto-detects the correct station key column among station_id/module_final/src_id.
    """
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
    x["station_id_raw_for_join"] = _clean_station_series(x[chosen_station_col])
    x["station_id"], station_alias_used, station_alias_scores = resolve_station_id_alias(
        x["station_id_raw_for_join"],
        observed_station_ids=observed_station_ids,
        network_name=network_name,
    )

    x["station_id_alias_used"] = station_alias_used

    if observed_station_ids is not None and filter_to_observed:
        obs = _observed_id_set(observed_station_ids)
        x = x[x["station_id"].isin(obs)].copy()
    x["network"] = network_name

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


def build_ows_hourly_summary(ows_long: pd.DataFrame) -> pd.DataFrame:
    x = ows_long.copy()
    x["date"] = _ensure_utc_datetime(x["date"])
    x["temp_raw"] = pd.to_numeric(x["temp_raw"], errors="coerce", downcast="float")

    out = (
        x.groupby("date", as_index=False)["temp_raw"]
        .agg(
            ows_n_available="count",
            ows_mean_temp="mean",
            ows_median_temp="median",
            ows_std_temp="std",
            ows_min_temp="min",
            ows_max_temp="max",
        )
    )
    out["ows_temp_range"] = out["ows_max_temp"] - out["ows_min_temp"]
    return downcast_dataframe(out.sort_values("date"))


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
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


def add_station_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    grp = x.groupby(["network", "station_id"], sort=False)["temp_raw"]

    x["temp_lag1"] = grp.shift(1)
    x["temp_diff1"] = x["temp_raw"] - x["temp_lag1"]
    x["temp_abs_diff1"] = x["temp_diff1"].abs()
    return downcast_dataframe(x)


def fill_stationwise_slow_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    ffill: bool = True,
    bfill: bool = False,
) -> pd.DataFrame:
    """Fill slow-varying station features without silently losing missingness.

    The missing indicator is computed BEFORE filling. By default this uses only
    forward fill, because backward fill can leak future satellite observations
    into earlier timestamps when the model is evaluated with a time split.
    """
    x = df.copy()
    group_keys = ["network", "station_id"]

    for c in cols:
        if c not in x.columns:
            continue

        original_missing = x[c].isna()
        x[f"{c}_missing"] = original_missing.astype("int8")

        def _fill_one(s: pd.Series) -> pd.Series:
            out = s
            if ffill:
                out = out.ffill()
            if bfill:
                out = out.bfill()
            return out

        x[c] = x.groupby(group_keys, sort=False)[c].transform(_fill_one)

    return x


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    if {"u10_ms", "v10_ms"}.issubset(x.columns):
        x["wind_speed"] = np.sqrt(x["u10_ms"] ** 2 + x["v10_ms"] ** 2)

    if {"t2m_c", "d2m_c"}.issubset(x.columns):
        x["era5_t_td_spread"] = x["t2m_c"] - x["d2m_c"]

    if {"temp_raw", "t2m_c"}.issubset(x.columns):
        x["temp_minus_era5"] = x["temp_raw"] - x["t2m_c"]
        x["temp_minus_era5_abs"] = (x["temp_raw"] - x["t2m_c"]).abs()

    if {"temp_raw", "d2m_c"}.issubset(x.columns):
        x["temp_minus_dew"] = x["temp_raw"] - x["d2m_c"]

    if {"temp_raw", "LST_C"}.issubset(x.columns):
        x["temp_minus_lst"] = x["temp_raw"] - x["LST_C"]

    if {"temp_raw", "ows_mean_temp"}.issubset(x.columns):
        x["temp_minus_ows_mean"] = x["temp_raw"] - x["ows_mean_temp"]

    if {"temp_raw", "ows_median_temp"}.issubset(x.columns):
        x["temp_minus_ows_median"] = x["temp_raw"] - x["ows_median_temp"]

    if {"temp_raw", "ows_mean_temp", "ows_std_temp"}.issubset(x.columns):
        denom = x["ows_std_temp"].replace(0, np.nan)
        x["temp_zscore_ows"] = (x["temp_raw"] - x["ows_mean_temp"]) / denom

    if {"ssrd_wm2", "wind_speed"}.issubset(x.columns):
        x["solar_x_wind"] = x["ssrd_wm2"] * x["wind_speed"]
        x["solar_over_wind"] = x["ssrd_wm2"] / (x["wind_speed"] + 0.5)

    if "ssrd_wm2" in x.columns:
        x["is_daylight"] = (x["ssrd_wm2"] > 5).astype("int8")

    if "tp_mm" in x.columns:
        x["is_rain"] = (x["tp_mm"] > 0).astype("int8")

    if {"building_height_m", "ssrd_wm2"}.issubset(x.columns):
        x["building_x_solar"] = x["building_height_m"] * x["ssrd_wm2"]

    return downcast_dataframe(x)


def build_master_feature_table_full(
    cws_long: pd.DataFrame,
    strict: pd.DataFrame,
    lenient: pd.DataFrame,
    station_meta_all: pd.DataFrame,
    era5_all: pd.DataFrame | None = None,
    lst_all: pd.DataFrame | None = None,
    ndvi_all: pd.DataFrame | None = None,
    ows_long: pd.DataFrame | None = None,
    add_station_dynamics: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = build_labels(strict, lenient)

    df = cws_long.copy()
    df["date"] = _ensure_utc_datetime(df["date"])
    df["station_id"] = _clean_station_series(df["station_id"])
    df["network"] = df["network"].astype(str).str.strip()


    df = df.merge(labels, on=["date", "station_id"], how="left", validate="many_to_one")
    df = downcast_dataframe(df)

    df = df.merge(
        station_meta_all,
        on=["station_id", "network"],
        how="left",
        validate="many_to_one",
    )
    df = downcast_dataframe(df)

    if era5_all is not None:
        df = df.merge(
            era5_all,
            on=["date", "station_id", "network"],
            how="left",
            validate="many_to_one",
        )
        df = downcast_dataframe(df)

    if lst_all is not None:
        df = df.merge(
            lst_all,
            on=["date", "station_id", "network"],
            how="left",
            validate="many_to_one",
        )
        df = downcast_dataframe(df)

    if ndvi_all is not None:
        df = df.merge(
            ndvi_all,
            on=["date", "station_id", "network"],
            how="left",
            validate="many_to_one",
        )
        df = downcast_dataframe(df)

    if ows_long is not None:
        ows_summary = build_ows_hourly_summary(ows_long)
        df = df.merge(ows_summary, on="date", how="left", validate="many_to_one")
        df = downcast_dataframe(df)

    df = df.sort_values(["network", "station_id", "date"]).reset_index(drop=True)
    df = add_temporal_features(df)
    if add_station_dynamics:
        df = add_station_dynamics_features(df)
    df = fill_stationwise_slow_features(df, cols=[c for c in ["NDVI"] if c in df.columns])
    df = add_interaction_features(df)

    for c in ["network", "LC_point_lg", "LC_buffer_lg", "LCZ_point_lg", "LCZ_buffer_lg"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = downcast_dataframe(df)

    dup_count = df.duplicated(["date", "station_id", "network"]).sum()
    if dup_count:
        raise ValueError(f"Master table still has {dup_count} duplicate station-hour rows")

    return df, df["target_bad"].value_counts(dropna=False)


def get_full_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidate_features = [
        "temp_raw",
        "network",
        "hour",
        "dayofyear",
        "month",
        "dow",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "temp_lag1",
        "temp_diff1",
        "temp_abs_diff1",
        "station_lat",
        "station_long",
        "building_height_m",
        "elev_meters",

        "LC_point_lg",

        "LC_buffer_lg",
        "LC_buffer_fraction",

        "LCZ_point_lg",

        "LCZ_buffer_lg",
        "LCZ_buffer_fraction",
        "t2m_c",
        "d2m_c",
        "u10_ms",
        "v10_ms",
        "wind_speed",
        "ssrd_wm2",
        "tp_mm",
        "LST_C",
        "NDVI",
        "NDVI_missing",
        "ows_n_available",
        "ows_mean_temp",
        "ows_median_temp",
        "ows_std_temp",
        "ows_min_temp",
        "ows_max_temp",
        "ows_temp_range",
        "era5_t_td_spread",
        "temp_minus_era5",
        "temp_minus_era5_abs",
        "temp_minus_dew",
        "temp_minus_lst",
        "temp_minus_ows_mean",
        "temp_minus_ows_median",
        "temp_zscore_ows",
        "solar_x_wind",
        "solar_over_wind",
        "is_daylight",
        "is_rain",
        "building_x_solar",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    categorical_cols = [c for c in ["network", "LC_point_lg", "LC_buffer_lg", "LCZ_point_lg", "LCZ_buffer_lg"] if c in feature_cols]
    return feature_cols, categorical_cols


def make_time_split(df: pd.DataFrame, valid_start, test_start):
    valid_start = pd.to_datetime(valid_start, utc=True)
    test_start = pd.to_datetime(test_start, utc=True)

    x = df.copy()
    x["date"] = _ensure_utc_datetime(x["date"])

    train = x[x["date"] < valid_start].copy()
    valid = x[(x["date"] >= valid_start) & (x["date"] < test_start)].copy()
    test = x[x["date"] >= test_start].copy()
    return train, valid, test


def train_catboost_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    random_state: int = 42,
    verbose: int = 100,
    use_gpu: bool = False,
    thread_count: int | None = None,
    used_ram_limit: str = "36gb",
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
    fast_mode: bool = True,
):
    train_df = maybe_subsample_rows(train_df, max_rows=max_train_rows, random_state=random_state)
    valid_df = maybe_subsample_rows(valid_df, max_rows=max_valid_rows, random_state=random_state)

    X_train = train_df[feature_cols]
    y_train = train_df["target_bad"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target_bad"].astype(int)

    cat_idx = [feature_cols.index(c) for c in categorical_cols if c in feature_cols]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)

    if thread_count is None:
        cpu_count = os.cpu_count() or 8
        thread_count = max(1, cpu_count - 2)

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=random_state,
        early_stopping_rounds=100,
        thread_count=thread_count,
        verbose=verbose,
    )

    if use_gpu:
        params["task_type"] = "GPU"
    else:
        params["used_ram_limit"] = used_ram_limit

    if fast_mode:
        params.update(
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.8,
            border_count=128,
        )
    else:
        params.update(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            border_count=254,
        )

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model, train_pool, valid_pool


def _choose_threshold(y_true: np.ndarray, pred_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, pred_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_catboost_model(
    model,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    split_name: str = "test",
    threshold: float | None = None,
):
    X = df[feature_cols]
    y = df["target_bad"].astype(int).values
    pred_prob = model.predict_proba(X)[:, 1]

    if threshold is None:
        threshold = _choose_threshold(y, pred_prob)

    pred = (pred_prob >= threshold).astype(int)

    return {
        "split": split_name,
        "n": len(df),
        "bad_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "auroc": roc_auc_score(y, pred_prob) if len(np.unique(y)) > 1 else None,
        "auprc": average_precision_score(y, pred_prob) if len(np.unique(y)) > 1 else None,
        "f1": f1_score(y, pred),
        "classification_report": classification_report(y, pred),
    }


def score_ambiguous_rows(model, df_all: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    amb = df_all[df_all["target_bad"].isna()].copy()
    if len(amb) == 0:
        return amb
    amb["pred_bad_prob"] = model.predict_proba(amb[feature_cols])[:, 1]
    return downcast_dataframe(amb)


def get_feature_importance_df(model, feature_cols: Sequence[str]) -> pd.DataFrame:
    fi = pd.DataFrame({
        "feature": list(feature_cols),
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False, ignore_index=True)
    return fi


import hashlib
import warnings
from pathlib import Path
from typing import Mapping

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    confusion_matrix,
)


def normalize_network_name(value: object) -> str:
    """Normalize common network labels used across CWS/OWS covariate files."""
    s = str(value).strip()
    low = s.lower()
    if "wunderground" in low:
        return "Wunderground"
    if "netatmo" in low:
        return "Netatmo"
    if low in {"ows", "official", "official_weather_station"} or "ows" in low:
        return "OWS"
    return s


def build_labels_with_components(strict: pd.DataFrame, lenient: pd.DataFrame) -> pd.DataFrame:
    """Build weak labels and keep strict/lenient component flags.

    target_bad:
      1   if strict == 1 and lenient == 1
      0   if strict == 0 and lenient == 0
      NaN if strict and lenient disagree
    """
    strict = dedupe_flag_wide(strict)
    lenient = dedupe_flag_wide(lenient)

    def _reshape_flags(x: pd.DataFrame, value_name: str) -> pd.DataFrame:
        all_value_cols = [c for c in x.columns if c != "date"]
        flag_cols = [c for c in all_value_cols if str(c).endswith("_is_outlier")]
        value_cols = flag_cols if flag_cols else all_value_cols

        long = x.melt(
            id_vars=["date"],
            value_vars=value_cols,
            var_name="raw_col",
            value_name=value_name,
        )
        long["station_id"] = (
            long["raw_col"].astype(str)
            .str.replace("_is_outlier", "", regex=False)
            .str.strip()
        )
        long = long.drop(columns=["raw_col"])
        long[value_name] = pd.to_numeric(long[value_name], errors="coerce")

        non_na = long[value_name].dropna()
        unexpected = sorted(pd.unique(non_na[~non_na.isin([0, 1, 0.0, 1.0])]).tolist())
        if unexpected:
            raise ValueError(
                f"{value_name} contains non-binary values such as {unexpected[:10]}. "
                "This usually means raw temperature columns were used as labels."
            )

        dup_count = int(long.duplicated(["date", "station_id"]).sum())
        if dup_count:
            raise ValueError(
                f"{value_name} has {dup_count} duplicate date/station rows after reshaping."
            )
        return long

    strict_long = _reshape_flags(strict, "qc_flag_strict")
    lenient_long = _reshape_flags(lenient, "qc_flag_lenient")

    labels = strict_long.merge(
        lenient_long,
        on=["date", "station_id"],
        how="inner",
        validate="one_to_one",
    )
    labels["target_bad"] = np.where(
        (labels["qc_flag_strict"] == 1) & (labels["qc_flag_lenient"] == 1),
        1,
        np.where(
            (labels["qc_flag_strict"] == 0) & (labels["qc_flag_lenient"] == 0),
            0,
            np.nan,
        ),
    )

    labels["station_id"] = _clean_station_series(labels["station_id"])
    for c in ["qc_flag_strict", "qc_flag_lenient", "target_bad"]:
        labels[c] = pd.to_numeric(labels[c], errors="coerce", downcast="float")

    return downcast_dataframe(
        labels[["date", "station_id", "qc_flag_strict", "qc_flag_lenient", "target_bad"]]
    )


def qc_station_ids_with_raw_data(qc_df: pd.DataFrame, date_col: str = "date") -> set[str]:
    """Return QC station ids that have at least one non-missing raw value.

    Flagged QC files may contain *_is_outlier columns filled with zeros even
    when the corresponding raw temperature column is entirely missing. This
    helper counts only stations with actual raw observations.
    """
    x = _maybe_repair_single_column_csv_df(qc_df.copy())
    flag_cols = [c for c in x.columns if str(c).endswith("_is_outlier")]
    if flag_cols:
        station_ids = [str(c).replace("_is_outlier", "").strip() for c in flag_cols]
    else:
        station_ids = [str(c).strip() for c in x.columns if str(c) != date_col]

    active: set[str] = set()
    for station_id in station_ids:
        if station_id in x.columns:
            n_obs = pd.to_numeric(x[station_id], errors="coerce").notna().sum()
            if int(n_obs) > 0:
                active.add(station_id)
    return active


def mask_qc_flags_where_raw_missing(qc_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Set *_is_outlier flags to NaN where the paired raw value is missing.

    This prevents all-missing stations from looking like fully clean stations
    in label diagnostics. It does not create model rows; the feature table still
    starts from observed CWS temperature rows.
    """
    x = _maybe_repair_single_column_csv_df(qc_df.copy())
    flag_cols = [c for c in x.columns if str(c).endswith("_is_outlier")]
    for flag_col in flag_cols:
        raw_col = str(flag_col).replace("_is_outlier", "")
        if raw_col in x.columns:
            raw_missing = pd.to_numeric(x[raw_col], errors="coerce").isna()
            x.loc[raw_missing, flag_col] = np.nan
    return x


def qc_flagged_station_audit(qc_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Summarize raw-value and flag coverage in a flagged QC wide table."""
    x = _maybe_repair_single_column_csv_df(qc_df.copy())
    flag_cols = [c for c in x.columns if str(c).endswith("_is_outlier")]
    if flag_cols:
        station_ids = [str(c).replace("_is_outlier", "").strip() for c in flag_cols]
    else:
        station_ids = [str(c).strip() for c in x.columns if str(c) != date_col]

    rows = []
    for station_id in station_ids:
        flag_col = f"{station_id}_is_outlier"
        raw_nonmissing = int(pd.to_numeric(x[station_id], errors="coerce").notna().sum()) if station_id in x.columns else 0
        if flag_col in x.columns:
            flag = pd.to_numeric(x[flag_col], errors="coerce")
            flag_nonmissing = int(flag.notna().sum())
            flag_sum = float(flag.sum(skipna=True)) if flag_nonmissing else np.nan
        else:
            flag_nonmissing = 0
            flag_sum = np.nan
        rows.append({
            "station_id": station_id,
            "raw_temp_nonmissing_n": raw_nonmissing,
            "flag_nonmissing_n": flag_nonmissing,
            "flag_sum_outliers": flag_sum,
            "raw_all_missing": bool(raw_nonmissing == 0),
        })
    return pd.DataFrame(rows).sort_values(["raw_all_missing", "station_id"], ascending=[False, True]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in kilometers."""
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    r = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _station_key(df: pd.DataFrame, station_col: str = "station_id", network_col: str = "network") -> pd.Series:
    return df[network_col].astype(str).str.strip() + "||" + df[station_col].astype(str).str.strip()


def build_station_pair_table(
    cws_meta_all: pd.DataFrame,
    ows_meta: pd.DataFrame,
    k_nearest: int = 5,
    max_radius_km: float | None = 5.0,
    fallback_to_k_nearest: bool = True,
    idw_power: float = 2.0,
    eps_km: float = 0.05,
) -> pd.DataFrame:
    """Precompute CWS -> nearby OWS station pairs and site-context deltas.

    The output is small: roughly n_cws * k_nearest rows. It is later joined
    with hourly OWS temperatures to create local anchor features.

    Parameters
    ----------
    cws_meta_all:
        Prepared CWS metadata with station_id, network, station_lat, station_long.
    ows_meta:
        Prepared OWS metadata with station_id, network, station_lat, station_long.
    k_nearest:
        Maximum number of OWS anchors per CWS station.
    max_radius_km:
        Keep anchors within this radius. If no anchor exists and
        fallback_to_k_nearest=True, the nearest k anchors are used anyway.
    fallback_to_k_nearest:
        Avoids losing isolated CWS stations.
    idw_power:
        Inverse-distance weighting power.
    eps_km:
        Distance floor to avoid infinite weights.
    """
    required = {"station_id", "network", "station_lat", "station_long"}
    missing_cws = required - set(cws_meta_all.columns)
    missing_ows = required - set(ows_meta.columns)
    if missing_cws:
        raise KeyError(f"cws_meta_all missing required columns: {sorted(missing_cws)}")
    if missing_ows:
        raise KeyError(f"ows_meta missing required columns: {sorted(missing_ows)}")

    cws = cws_meta_all.copy()
    ows = ows_meta.copy()
    cws["network"] = cws["network"].map(normalize_network_name)
    ows["network"] = ows["network"].map(normalize_network_name)

    cws = cws.dropna(subset=["station_lat", "station_long"]).reset_index(drop=True)
    ows = ows.dropna(subset=["station_lat", "station_long"]).reset_index(drop=True)

    if len(cws) == 0 or len(ows) == 0:
        raise ValueError("Need at least one CWS station and one OWS station with coordinates.")

    num_meta_cols = [
        "elev_meters",
        "building_height_m",
        "LC_buffer_fraction",
        "LCZ_buffer_fraction",
    ]
    cat_meta_cols = [

        "LC_point_lg",

        "LC_buffer_lg",

        "LCZ_point_lg",

        "LCZ_buffer_lg",
    ]

    records: list[dict] = []
    ows_lat = ows["station_lat"].to_numpy(dtype="float64")
    ows_lon = ows["station_long"].to_numpy(dtype="float64")

    for _, crow in cws.iterrows():
        dist = haversine_km(float(crow["station_lat"]), float(crow["station_long"]), ows_lat, ows_lon)
        order = np.argsort(dist)

        if max_radius_km is not None:
            in_radius = order[dist[order] <= max_radius_km]
        else:
            in_radius = order

        if len(in_radius) == 0 and fallback_to_k_nearest:
            chosen = order[:k_nearest]
            used_fallback = True
        else:
            chosen = in_radius[:k_nearest]
            used_fallback = False

        for rank, oi in enumerate(chosen, start=1):
            orow = ows.iloc[int(oi)]
            d = float(dist[int(oi)])
            rec = {
                "cws_station_id": str(crow["station_id"]).strip(),
                "cws_network": normalize_network_name(crow["network"]),
                "ows_station_id": str(orow["station_id"]).strip(),
                "ows_network": normalize_network_name(orow["network"]),
                "ows_rank": int(rank),
                "ows_dist_km": d,
                "ows_weight_idw": float(1.0 / ((max(d, eps_km)) ** idw_power)),
                "ows_pair_used_radius_fallback": int(used_fallback),
            }

            for col in num_meta_cols:
                if col in crow.index and col in orow.index:
                    cval = pd.to_numeric(pd.Series([crow[col]]), errors="coerce").iloc[0]
                    oval = pd.to_numeric(pd.Series([orow[col]]), errors="coerce").iloc[0]
                    rec[f"cws_{col}"] = cval
                    rec[f"ows_{col}"] = oval
                    if pd.notna(cval) and pd.notna(oval):
                        rec[f"pair_delta_{col}"] = float(cval - oval)
                        rec[f"pair_abs_delta_{col}"] = float(abs(cval - oval))
                    else:
                        rec[f"pair_delta_{col}"] = np.nan
                        rec[f"pair_abs_delta_{col}"] = np.nan

            for col in cat_meta_cols:
                if col in crow.index and col in orow.index:
                    cval = crow[col]
                    oval = orow[col]
                    rec[f"cws_{col}"] = cval
                    rec[f"ows_{col}"] = oval
                    if pd.isna(cval) or pd.isna(oval):
                        rec[f"pair_same_{col}"] = np.nan
                    else:
                        rec[f"pair_same_{col}"] = int(str(cval) == str(oval))

            records.append(rec)

    out = pd.DataFrame.from_records(records)
    if len(out) == 0:
        raise ValueError("No CWS-OWS station pairs were created. Check radius and coordinates.")
    return downcast_dataframe(out)


def build_local_ows_features(
    cws_keys: pd.DataFrame,
    ows_long: pd.DataFrame,
    pair_table: pd.DataFrame,
    value_col: str = "temp_raw",
    k_nearest: int | None = None,
    chunk_freq: str | None = "M",
) -> pd.DataFrame:
    """Create local OWS anchor features for each CWS station-hour.

    This uses the pair table from build_station_pair_table() and same-hour OWS
    temperatures. It aggregates nearest, mean, median, std, min/max, and IDW
    temperatures, plus weighted OWS metadata and weighted CWS/OWS site deltas.

    Parameters
    ----------
    cws_keys:
        DataFrame containing at least date, station_id, network for desired CWS rows.
    ows_long:
        Long OWS table with date, station_id, temp_raw.
    pair_table:
        Output of build_station_pair_table().
    k_nearest:
        Optional additional cap on the number of anchors used per CWS station.
    chunk_freq:
        Group OWS rows by month by default to reduce memory. Set None to process
        in one pass.
    """
    required_pairs = {"cws_station_id", "cws_network", "ows_station_id", "ows_rank", "ows_dist_km", "ows_weight_idw"}
    missing = required_pairs - set(pair_table.columns)
    if missing:
        raise KeyError(f"pair_table missing columns: {sorted(missing)}")

    keys = cws_keys[["date", "station_id", "network"]].copy()
    keys["date"] = _ensure_utc_datetime(keys["date"])
    keys["station_id"] = _clean_station_series(keys["station_id"])
    keys["network"] = keys["network"].map(normalize_network_name)
    keys = keys.drop_duplicates().rename(
        columns={"station_id": "cws_station_id", "network": "cws_network"}
    )

    pairs = pair_table.copy()
    pairs["cws_station_id"] = _clean_station_series(pairs["cws_station_id"])
    pairs["cws_network"] = pairs["cws_network"].map(normalize_network_name)
    pairs["ows_station_id"] = _clean_station_series(pairs["ows_station_id"])
    if k_nearest is not None:
        pairs = pairs[pairs["ows_rank"] <= k_nearest].copy()

    ows = ows_long[["date", "station_id", value_col]].copy()
    ows["date"] = _ensure_utc_datetime(ows["date"])
    ows["station_id"] = _clean_station_series(ows["station_id"])
    ows[value_col] = pd.to_numeric(ows[value_col], errors="coerce", downcast="float")
    ows = ows.dropna(subset=["date", "station_id"])

    if chunk_freq is None:
        chunks = [("__all__", ows)]
    else:
        tmp = ows.copy()

        if str(chunk_freq).upper() in {"M", "MS", "MONTH", "MONTHLY"}:
            tmp["_chunk"] = tmp["date"].dt.strftime("%Y-%m")
        elif str(chunk_freq).upper() in {"D", "DAY", "DAILY"}:
            tmp["_chunk"] = tmp["date"].dt.strftime("%Y-%m-%d")
        else:
            tmp["_chunk"] = tmp["date"].dt.to_period(chunk_freq).astype(str)
        chunks = list(tmp.groupby("_chunk", sort=True))

    out_parts = []
    group_cols = ["date", "cws_station_id", "cws_network"]

    for _, ows_sub in chunks:
        if "_chunk" in ows_sub.columns:
            ows_sub = ows_sub.drop(columns=["_chunk"])


        chunk_dates = pd.Index(ows_sub["date"].dropna().unique())
        keys_sub = keys[keys["date"].isin(chunk_dates)]
        if len(keys_sub) == 0:
            continue

        joined = pairs.merge(
            ows_sub,
            left_on="ows_station_id",
            right_on="station_id",
            how="left",
            validate="many_to_many",
        ).drop(columns=["station_id"])

        joined = joined[joined["date"].isin(chunk_dates)]
        joined = joined[joined[value_col].notna()].copy()
        if len(joined) == 0:
            continue


        joined = joined.merge(keys_sub, on=group_cols, how="inner")
        if len(joined) == 0:
            continue

        joined["ows_w"] = pd.to_numeric(joined["ows_weight_idw"], errors="coerce").astype("float64")
        joined["ows_w_temp"] = joined["ows_w"] * joined[value_col].astype("float64")

        basic = (
            joined.groupby(group_cols, as_index=False)[value_col]
            .agg(
                ows_local_n="count",
                ows_local_mean_temp="mean",
                ows_local_median_temp="median",
                ows_local_std_temp="std",
                ows_local_min_temp="min",
                ows_local_max_temp="max",
            )
        )
        basic["ows_local_temp_range"] = basic["ows_local_max_temp"] - basic["ows_local_min_temp"]

        weighted = (
            joined.groupby(group_cols, as_index=False)
            .agg(
                ows_idw_weight_sum=("ows_w", "sum"),
                _ows_w_temp_sum=("ows_w_temp", "sum"),
                ows_local_mean_dist_km=("ows_dist_km", "mean"),
                ows_local_min_dist_km=("ows_dist_km", "min"),
            )
        )
        weighted["ows_idw_temp"] = weighted["_ows_w_temp_sum"] / weighted["ows_idw_weight_sum"].replace(0, np.nan)
        weighted = weighted.drop(columns=["_ows_w_temp_sum"])


        nearest = (
            joined.sort_values(group_cols + ["ows_rank", "ows_dist_km"])
            .groupby(group_cols, as_index=False)
            .first()
        )
        nearest_keep = [
            "date",
            "cws_station_id",
            "cws_network",
            "ows_station_id",
            "ows_rank",
            "ows_dist_km",
            value_col,
        ]
        nearest = nearest[nearest_keep].rename(
            columns={
                "ows_station_id": "ows_nearest_station_id",
                "ows_rank": "ows_nearest_rank",
                "ows_dist_km": "ows_nearest_dist_km",
                value_col: "ows_nearest_temp",
            }
        )


        weighted_meta_frames = []
        meta_candidate_cols = []
        excluded_meta_cols = {
            "ows_rank",
            "ows_dist_km",
            "ows_weight_idw",
            "ows_pair_used_radius_fallback",
        }
        for c in joined.columns:
            if c in group_cols or c in {value_col, "ows_w", "ows_w_temp"} or c in excluded_meta_cols:
                continue
            if (
                c.startswith("ows_")
                or c.startswith("pair_abs_delta_")
                or c.startswith("pair_delta_")
                or c.startswith("pair_same_")
            ):
                if pd.api.types.is_numeric_dtype(joined[c]):
                    meta_candidate_cols.append(c)

        for c in sorted(set(meta_candidate_cols)):
            numeric_c = pd.to_numeric(joined[c], errors="coerce")
            valid_c = numeric_c.notna()
            tmp_col = f"_w_{c}"
            tmp_w_col = f"_w_nonmissing_{c}"
            joined[tmp_col] = joined["ows_w"] * numeric_c.fillna(0.0)
            joined[tmp_w_col] = joined["ows_w"].where(valid_c, 0.0)
            agg = joined.groupby(group_cols, as_index=False).agg(
                **{
                    f"{c}_idw": (tmp_col, "sum"),
                    f"_{c}_w_nonmissing": (tmp_w_col, "sum"),
                }
            )
            denom = agg[f"_{c}_w_nonmissing"].replace(0, np.nan)
            agg[f"{c}_idw"] = agg[f"{c}_idw"] / denom
            agg = agg.drop(columns=[f"_{c}_w_nonmissing"])
            weighted_meta_frames.append(agg)

        out = basic.merge(weighted, on=group_cols, how="outer")
        out = out.merge(nearest, on=group_cols, how="left")
        for wm in weighted_meta_frames:
            out = out.merge(wm, on=group_cols, how="left")

        out_parts.append(out)

    if not out_parts:
        result = keys.copy()
        result["ows_local_n"] = 0
    else:
        result = pd.concat(out_parts, ignore_index=True)

    result = result.rename(
        columns={"cws_station_id": "station_id", "cws_network": "network"}
    )
    result["station_id"] = _clean_station_series(result["station_id"])
    result["network"] = result["network"].map(normalize_network_name)


    return downcast_dataframe(result.sort_values(["network", "station_id", "date"]).reset_index(drop=True))


def diagnose_hourly_env_coverage(
    base_long: pd.DataFrame,
    env_long: pd.DataFrame,
    value_cols: Sequence[str],
    network_name: str | None = None,
    station_col: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Diagnose exact-join coverage for ERA5/LST/NDVI-like long tables.

    This is especially useful for LST: the file can contain no NaN values but
    still have sparse timestamp coverage relative to hourly CWS observations.
    """
    env = _maybe_repair_single_column_csv_df(env_long.copy())

    if network_name is not None:
        env["network"] = normalize_network_name(network_name)
    elif "network" in env.columns:
        env["network"] = env["network"].map(normalize_network_name)
    else:
        env["network"] = "unknown"

    observed_for_alias = base_long["station_id"].dropna().astype(str).unique() if "station_id" in base_long else None
    chosen_station_col = infer_station_column(
        env,
        observed_station_ids=observed_for_alias,
        network_name=network_name,
        explicit_station_col=station_col,
    )
    env["station_id"], station_alias_used, station_alias_scores = resolve_station_id_alias(
        env[chosen_station_col],
        observed_station_ids=observed_for_alias,
        network_name=network_name,
    )
    env["date"] = _ensure_utc_datetime(env[date_col])

    base = base_long[["date", "station_id", "network"]].copy()
    base["date"] = _ensure_utc_datetime(base["date"])
    base["station_id"] = _clean_station_series(base["station_id"])
    base["network"] = base["network"].map(normalize_network_name)
    if network_name is not None:
        base = base[base["network"] == normalize_network_name(network_name)]

    reports = []
    base_keys = base.drop_duplicates(["date", "station_id", "network"])
    env_keys = env[["date", "station_id", "network"] + [c for c in value_cols if c in env.columns]].copy()

    joined = base_keys.merge(
        env_keys,
        on=["date", "station_id", "network"],
        how="left",
        validate="many_to_one",
    )

    for c in value_cols:
        if c not in env.columns:
            reports.append({
                "feature": c,
                "present_in_env": False,
                "env_rows": len(env),
                "env_nonmissing_rows": 0,
                "env_n_stations": env["station_id"].nunique(),
                "env_n_timestamps": env["date"].nunique(),
                "base_rows": len(base_keys),
                "exact_join_nonmissing_rows": 0,
                "exact_join_coverage": 0.0,
            })
            continue

        reports.append({
            "feature": c,
            "present_in_env": True,
            "station_col": chosen_station_col,
            "station_alias_used": station_alias_used,
            "station_alias_scores": station_alias_scores,
            "env_rows": int(len(env)),
            "env_nonmissing_rows": int(env[c].notna().sum()),
            "env_n_stations": int(env.loc[env[c].notna(), "station_id"].nunique()),
            "env_n_timestamps": int(env.loc[env[c].notna(), "date"].nunique()),
            "base_rows": int(len(base_keys)),
            "base_n_stations": int(base_keys["station_id"].nunique()),
            "base_n_timestamps": int(base_keys["date"].nunique()),
            "station_overlap": int(len(set(base_keys["station_id"]) & set(env.loc[env[c].notna(), "station_id"]))),
            "exact_join_nonmissing_rows": int(joined[c].notna().sum()),
            "exact_join_coverage": float(joined[c].notna().mean()) if len(joined) else np.nan,
        })
    return pd.DataFrame(reports)


def merge_stationwise_asof_features(
    base: pd.DataFrame,
    features: pd.DataFrame,
    value_cols: Sequence[str],
    rename_map: Mapping[str, str] | None = None,
    tolerance: str | pd.Timedelta = "14D",
    direction: str = "backward",
    add_missing_indicators: bool = True,
    add_age: bool = True,
    age_unit: str = "hours",
) -> pd.DataFrame:
    """Attach latest/nearest station-wise features with a tolerance.

    Use this for slow features like NDVI. It can also be used for sparse LST
    with a short tolerance, but LST should not usually be forward-filled over
    long gaps because it has a strong diurnal cycle.

    direction:
      - 'backward': previous available value only; safest for time splits.
      - 'nearest': nearest available value within tolerance; useful for offline QC.
      - 'forward': future value only; usually avoid for predictive settings.
    """
    if rename_map is None:
        rename_map = {c: c for c in value_cols}
    else:
        rename_map = {**{c: c for c in value_cols}, **dict(rename_map)}

    missing_value_cols = [c for c in value_cols if c not in features.columns]
    if missing_value_cols:
        raise KeyError(f"features missing value columns: {missing_value_cols}")

    left = base[["date", "station_id", "network"]].copy()
    left["_row_id"] = np.arange(len(left), dtype="int64")
    left["date"] = _ensure_utc_datetime(left["date"])
    left["station_id"] = _clean_station_series(left["station_id"])
    left["network"] = left["network"].map(normalize_network_name)

    right = features[["date", "station_id", "network"] + list(value_cols)].copy()
    right["date"] = _ensure_utc_datetime(right["date"])
    right["_feature_date"] = right["date"]
    right["station_id"] = _clean_station_series(right["station_id"])
    right["network"] = right["network"].map(normalize_network_name)
    for c in value_cols:
        right[c] = pd.to_numeric(right[c], errors="coerce", downcast="float")
    right = right.dropna(subset=["date", "station_id", "network"])
    right = right.dropna(subset=list(value_cols), how="all")

    tol = pd.Timedelta(tolerance)
    parts = []



    for (network, station_id), g_left in left.groupby(["network", "station_id"], sort=False):
        g_right = right[(right["network"] == network) & (right["station_id"] == station_id)]
        g_left = g_left.sort_values("date")

        if len(g_right) == 0:
            empty = g_left[["_row_id"]].copy()
            for c in value_cols:
                out_c = rename_map[c]
                empty[out_c] = np.nan
                if add_missing_indicators:
                    empty[f"{out_c}_missing"] = 1
                if add_age:
                    empty[f"{out_c}_age_{age_unit}"] = np.nan
            parts.append(empty)
            continue

        g_right = g_right.sort_values("date")
        merged = pd.merge_asof(
            g_left[["date", "_row_id"]],
            g_right[["date", "_feature_date"] + list(value_cols)],
            on="date",
            direction=direction,
            tolerance=tol,
        )

        out = merged[["_row_id"]].copy()
        for c in value_cols:
            out_c = rename_map[c]
            out[out_c] = merged[c].values
            if add_missing_indicators:
                out[f"{out_c}_missing"] = pd.isna(merged[c]).astype("int8").values
            if add_age:
                delta = merged["date"] - merged["_feature_date"]
                if direction == "nearest":
                    delta = delta.abs()
                if age_unit == "days":
                    age = delta.dt.total_seconds() / 86400.0
                elif age_unit == "hours":
                    age = delta.dt.total_seconds() / 3600.0
                else:
                    age = delta.dt.total_seconds()
                out[f"{out_c}_age_{age_unit}"] = age.astype("float32")
        parts.append(out)

    attached = pd.concat(parts, ignore_index=True).sort_values("_row_id")
    attached = attached.drop(columns=["_row_id"]).reset_index(drop=True)

    out = base.reset_index(drop=True).copy()
    for c in attached.columns:
        out[c] = attached[c].values
    return downcast_dataframe(out)


def add_lst_month_hour_climatology(
    df: pd.DataFrame,
    lst_all: pd.DataFrame,
    value_col: str = "LST_C",
) -> pd.DataFrame:
    """Add station-month-hour and network-month-hour LST climatology features.

    This does not pretend to be contemporaneous LST. It gives the model a
    low-missing seasonal/diurnal surface-temperature context feature.
    """
    if value_col not in lst_all.columns:
        return df

    x = df.copy()
    x["date"] = _ensure_utc_datetime(x["date"])
    x["month"] = x["date"].dt.month.astype("int8") if "month" not in x.columns else x["month"]
    x["hour"] = x["date"].dt.hour.astype("int8") if "hour" not in x.columns else x["hour"]
    x["station_id"] = _clean_station_series(x["station_id"])
    x["network"] = x["network"].map(normalize_network_name)

    lst = lst_all[["date", "station_id", "network", value_col]].copy()
    lst["date"] = _ensure_utc_datetime(lst["date"])
    lst["station_id"] = _clean_station_series(lst["station_id"])
    lst["network"] = lst["network"].map(normalize_network_name)
    lst[value_col] = pd.to_numeric(lst[value_col], errors="coerce", downcast="float")
    lst = lst[lst[value_col].notna()].copy()
    if len(lst) == 0:
        return x

    lst["month"] = lst["date"].dt.month.astype("int8")
    lst["hour"] = lst["date"].dt.hour.astype("int8")

    station_clim = (
        lst.groupby(["network", "station_id", "month", "hour"], as_index=False)[value_col]
        .median()
        .rename(columns={value_col: "LST_C_station_month_hour_median"})
    )
    network_clim = (
        lst.groupby(["network", "month", "hour"], as_index=False)[value_col]
        .median()
        .rename(columns={value_col: "LST_C_network_month_hour_median"})
    )

    x = x.merge(station_clim, on=["network", "station_id", "month", "hour"], how="left", validate="many_to_one")
    x = x.merge(network_clim, on=["network", "month", "hour"], how="left", validate="many_to_one")

    if "LST_C" in x.columns:
        base_lst = x["LST_C"].copy()
    else:
        base_lst = pd.Series(np.nan, index=x.index, dtype="float32")

    if "LST_C_asof" in x.columns:
        base_lst = base_lst.combine_first(x["LST_C_asof"])

    x["LST_C_gapfill_clim"] = base_lst
    x["LST_C_gapfill_clim"] = x["LST_C_gapfill_clim"].combine_first(x["LST_C_station_month_hour_median"])
    x["LST_C_gapfill_clim"] = x["LST_C_gapfill_clim"].combine_first(x["LST_C_network_month_hour_median"])
    x["LST_C_gapfill_clim_missing"] = x["LST_C_gapfill_clim"].isna().astype("int8")
    return downcast_dataframe(x)



_add_interaction_features_original = add_interaction_features


def add_station_dynamics_features(
    df: pd.DataFrame,
    lags: Sequence[int] = (1, 2, 3, 6, 12, 24),
    rolling_windows: Sequence[int] = (3, 6, 24),
) -> pd.DataFrame:
    """Add richer station-wise temporal dynamics.

    Rolling features are lagged by one row within station so the current
    temperature is not included in its own rolling baseline.
    """
    x = df.copy()
    x = x.sort_values(["network", "station_id", "date"]).reset_index(drop=True)
    grp = x.groupby(["network", "station_id"], sort=False)["temp_raw"]

    for lag in lags:
        x[f"temp_lag{lag}"] = grp.shift(lag)
        x[f"temp_diff{lag}"] = x["temp_raw"] - x[f"temp_lag{lag}"]
        x[f"temp_abs_diff{lag}"] = x[f"temp_diff{lag}"].abs()

    for window in rolling_windows:
        min_periods = max(2, min(window, int(np.ceil(window / 2))))
        x[f"temp_roll_mean_{window}h"] = grp.transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(w, min_periods=mp).mean()
        )
        x[f"temp_roll_std_{window}h"] = grp.transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(w, min_periods=mp).std()
        )
        x[f"temp_roll_median_{window}h"] = grp.transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(w, min_periods=mp).median()
        )
        denom = x[f"temp_roll_std_{window}h"].replace(0, np.nan)
        x[f"temp_roll_zscore_{window}h"] = (x["temp_raw"] - x[f"temp_roll_mean_{window}h"]) / denom

    return downcast_dataframe(x)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add original interactions plus local-OWS, satellite, and metadata deltas."""
    x = _add_interaction_features_original(df)


    ows_temp_cols = [
        "ows_idw_temp",
        "ows_nearest_temp",
        "ows_local_mean_temp",
        "ows_local_median_temp",
        "ows_city_mean_temp",
        "ows_city_median_temp",
    ]
    for c in ows_temp_cols:
        if {"temp_raw", c}.issubset(x.columns):
            stem = c.replace("_temp", "")
            x[f"temp_minus_{stem}"] = x["temp_raw"] - x[c]
            x[f"temp_minus_{stem}_abs"] = (x["temp_raw"] - x[c]).abs()

    if {"temp_raw", "ows_idw_temp", "ows_local_std_temp"}.issubset(x.columns):
        denom = x["ows_local_std_temp"].replace(0, np.nan)
        x["temp_zscore_ows_local"] = (x["temp_raw"] - x["ows_idw_temp"]) / denom

    if {"temp_raw", "LST_C_asof"}.issubset(x.columns):
        x["temp_minus_lst_asof"] = x["temp_raw"] - x["LST_C_asof"]
        x["temp_minus_lst_asof_abs"] = x["temp_minus_lst_asof"].abs()

    if {"temp_raw", "LST_C_gapfill_3h"}.issubset(x.columns):
        x["temp_minus_lst_gapfill_3h"] = x["temp_raw"] - x["LST_C_gapfill_3h"]
        x["temp_minus_lst_gapfill_3h_abs"] = x["temp_minus_lst_gapfill_3h"].abs()

    if {"temp_raw", "LST_C_gapfill_clim"}.issubset(x.columns):
        x["temp_minus_lst_gapfill_clim"] = x["temp_raw"] - x["LST_C_gapfill_clim"]
        x["temp_minus_lst_gapfill_clim_abs"] = x["temp_minus_lst_gapfill_clim"].abs()

    if {"temp_raw", "NDVI"}.issubset(x.columns):

        x["temp_x_ndvi"] = x["temp_raw"] * x["NDVI"]



    for base_col in [
        "elev_meters",
        "building_height_m",
        "LC_buffer_fraction",
        "LCZ_buffer_fraction",
    ]:
        weighted_col = f"ows_{base_col}_idw"
        if base_col in x.columns and weighted_col in x.columns:
            x[f"cws_minus_ows_idw_{base_col}"] = x[base_col] - x[weighted_col]
            x[f"cws_minus_ows_idw_{base_col}_abs"] = (x[base_col] - x[weighted_col]).abs()

    return downcast_dataframe(x)


def _dedupe_env_for_merge(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    x = df.copy()
    x["date"] = _ensure_utc_datetime(x["date"])
    x["station_id"] = _clean_station_series(x["station_id"])
    x["network"] = x["network"].map(normalize_network_name)
    keep = ["date", "station_id", "network"] + [c for c in value_cols if c in x.columns]
    x = x[keep].copy()
    for c in value_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce", downcast="float")
    if x.duplicated(["date", "station_id", "network"]).any():
        agg = {c: "mean" for c in x.columns if c not in ["date", "station_id", "network"]}
        x = x.groupby(["date", "station_id", "network"], as_index=False).agg(agg)
    return downcast_dataframe(x)


def build_master_feature_table_benchmark(
    cws_long: pd.DataFrame,
    strict: pd.DataFrame,
    lenient: pd.DataFrame,
    station_meta_all: pd.DataFrame,
    era5_all: pd.DataFrame | None = None,
    lst_all: pd.DataFrame | None = None,
    ndvi_all: pd.DataFrame | None = None,
    ows_long: pd.DataFrame | None = None,
    ows_local_features: pd.DataFrame | None = None,
    add_city_ows_summary: bool = True,
    add_station_dynamics: bool = True,
    ndvi_asof_tolerance: str = "21D",
    ndvi_asof_direction: str = "backward",
    lst_asof_tolerance: str = "3h",
    lst_asof_direction: str = "nearest",
    add_lst_climatology: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the extended benchmark feature table.

    Differences from build_master_feature_table_full:
      - keeps qc_flag_strict and qc_flag_lenient for strict/lenient baselines
      - supports local OWS anchor features
      - attaches NDVI using station-wise as-of/forward-fill with age indicators
      - attaches sparse LST exactly plus optional short-tolerance as-of and climatology
    """
    labels = build_labels_with_components(strict, lenient)

    df = cws_long.copy()
    df["date"] = _ensure_utc_datetime(df["date"])
    df["station_id"] = _clean_station_series(df["station_id"])
    df["network"] = df["network"].map(normalize_network_name)

    df = df.merge(labels, on=["date", "station_id"], how="left", validate="many_to_one")
    df = downcast_dataframe(df)

    meta = station_meta_all.copy()
    meta["station_id"] = _clean_station_series(meta["station_id"])
    meta["network"] = meta["network"].map(normalize_network_name)

    df = df.merge(meta, on=["station_id", "network"], how="left", validate="many_to_one")
    df = downcast_dataframe(df)

    if era5_all is not None:
        era5_cols = [c for c in ["t2m_c", "d2m_c", "u10_ms", "v10_ms", "ssrd_wm2", "tp_mm"] if c in era5_all.columns]
        era5 = _dedupe_env_for_merge(era5_all, era5_cols)
        df = df.merge(era5, on=["date", "station_id", "network"], how="left", validate="many_to_one")
        df = downcast_dataframe(df)

    if lst_all is not None and "LST_C" in lst_all.columns:
        lst = _dedupe_env_for_merge(lst_all, ["LST_C"])

        df = df.merge(lst, on=["date", "station_id", "network"], how="left", validate="many_to_one")
        df["LST_C_exact_missing"] = df["LST_C"].isna().astype("int8")
        df = downcast_dataframe(df)

        if lst_asof_tolerance is not None:
            df = merge_stationwise_asof_features(
                df,
                lst,
                value_cols=["LST_C"],
                rename_map={"LST_C": "LST_C_asof"},
                tolerance=lst_asof_tolerance,
                direction=lst_asof_direction,
                add_missing_indicators=True,
                add_age=True,
                age_unit="hours",
            )
            df["LST_C_gapfill_3h"] = df["LST_C"].combine_first(df["LST_C_asof"])
            df["LST_C_gapfill_3h_missing"] = df["LST_C_gapfill_3h"].isna().astype("int8")
            df = downcast_dataframe(df)

        if add_lst_climatology:
            df = add_lst_month_hour_climatology(df, lst, value_col="LST_C")

    if ndvi_all is not None and "NDVI" in ndvi_all.columns:
        ndvi = _dedupe_env_for_merge(ndvi_all, ["NDVI"])
        df = merge_stationwise_asof_features(
            df,
            ndvi,
            value_cols=["NDVI"],
            rename_map={"NDVI": "NDVI"},
            tolerance=ndvi_asof_tolerance,
            direction=ndvi_asof_direction,
            add_missing_indicators=True,
            add_age=True,
            age_unit="days",
        )
        df = downcast_dataframe(df)

    if ows_long is not None and add_city_ows_summary:
        ows_summary = build_ows_hourly_summary(ows_long)
        rename = {
            "ows_n_available": "ows_city_n_available",
            "ows_mean_temp": "ows_city_mean_temp",
            "ows_median_temp": "ows_city_median_temp",
            "ows_std_temp": "ows_city_std_temp",
            "ows_min_temp": "ows_city_min_temp",
            "ows_max_temp": "ows_city_max_temp",
            "ows_temp_range": "ows_city_temp_range",
        }
        ows_summary = ows_summary.rename(columns=rename)
        df = df.merge(ows_summary, on="date", how="left", validate="many_to_one")
        df = downcast_dataframe(df)

    if ows_local_features is not None:
        loc = ows_local_features.copy()
        loc["date"] = _ensure_utc_datetime(loc["date"])
        loc["station_id"] = _clean_station_series(loc["station_id"])
        loc["network"] = loc["network"].map(normalize_network_name)
        if loc.duplicated(["date", "station_id", "network"]).any():
            raise ValueError("ows_local_features contains duplicate date/station/network rows.")
        df = df.merge(loc, on=["date", "station_id", "network"], how="left", validate="many_to_one")
        df = downcast_dataframe(df)

    df = df.sort_values(["network", "station_id", "date"]).reset_index(drop=True)
    df = add_temporal_features(df)
    if add_station_dynamics:
        df = add_station_dynamics_features(df)
    df = add_interaction_features(df)

    for c in [
        "network",
        "LC_point_lg",
        "LC_buffer_lg",
        "LCZ_point_lg",
        "LCZ_buffer_lg",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = downcast_dataframe(df)

    dup_count = int(df.duplicated(["date", "station_id", "network"]).sum())
    if dup_count:
        raise ValueError(f"Master table still has {dup_count} duplicate station-hour rows")

    return df, df["target_bad"].value_counts(dropna=False)


def is_coordinate_like_feature(col: object) -> bool:
    """Return True for raw coordinate columns and common coordinate aliases."""
    c = str(col).strip().lower()
    exact = {
        "lat",
        "lon",
        "long",
        "latitude",
        "longitude",
        "station_lat",
        "station_lon",
        "station_long",
        "station_latitude",
        "station_longitude",
        "cws_lat",
        "cws_lon",
        "cws_long",
        "cws_latitude",
        "cws_longitude",
        "ows_lat",
        "ows_lon",
        "ows_long",
        "ows_latitude",
        "ows_longitude",
    }
    suffixes = (
        "_lat",
        "_lon",
        "_long",
        "_latitude",
        "_longitude",
    )
    return c in exact or c.endswith(suffixes)


def is_city_ows_like_feature(col: object) -> bool:
    """Return True for city-level OWS features and residuals derived from them.

    Covers both benchmark names (ows_city_*) and legacy names from
    build_master_feature_table_full() such as ows_mean_temp and
    temp_minus_ows_mean. Local OWS features like temp_minus_ows_local_mean are
    intentionally not matched.
    """
    c = str(col).strip().lower()
    legacy_city_ows = {
        "ows_n_available",
        "ows_mean_temp",
        "ows_median_temp",
        "ows_std_temp",
        "ows_min_temp",
        "ows_max_temp",
        "ows_temp_range",
        "temp_minus_ows_mean",
        "temp_minus_ows_median",
        "temp_zscore_ows",
    }
    return c.startswith("ows_city_") or c.startswith("temp_minus_ows_city") or c in legacy_city_ows


def is_station_static_like_feature(col: object) -> bool:
    """Return True for static site/context features.

    This includes base CWS metadata, coordinate-like fields, CWS-vs-OWS static
    deltas, OWS static metadata aggregates, and static proximity/rank/weight
    descriptors. Dynamic OWS temperatures such as ows_idw_temp and
    ows_local_mean_temp are intentionally not removed here.
    """
    c = str(col).strip()
    low = c.lower()

    base_static = {
        "building_height_m",
        "elev_meters",

        "lc_point_lg",

        "lc_buffer_lg",
        "lc_buffer_fraction",

        "lcz_point_lg",

        "lcz_buffer_lg",
        "lcz_buffer_fraction",
    }
    if low in base_static or is_coordinate_like_feature(c):
        return True

    if low.startswith(("pair_delta_", "pair_abs_delta_", "pair_same_", "cws_minus_ows_idw_")):
        return True


    static_tokens = (
        "elev_meters",
        "building_height_m",
        "lc_point_lg",
        "lc_buffer_lg",
        "lc_buffer_fraction",
        "lcz_point_lg",
        "lcz_buffer_lg",
        "lcz_buffer_fraction",
    )
    if low.startswith("ows_") and any(token in low for token in static_tokens):
        return True



    if low.startswith("ows_") and ("dist" in low or "rank" in low or "weight_sum" in low):
        return True

    return False


def get_benchmark_feature_columns(
    df: pd.DataFrame,
    include_coordinates: bool = True,
    include_station_static: bool = True,
    include_satellite: bool = True,
    include_ows: bool = True,
    include_city_ows: bool = True,
    include_temporal_dynamics: bool = True,
    extra_exclude: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer a safe feature list for benchmark models.

    Excludes labels, timestamps, station IDs, city labels, and diagnostic text IDs.

    Notes
    -----
    - include_coordinates=False removes station_lat/station_long plus common
      coordinate aliases such as lat/lon/latitude/longitude and cws_/ows_ variants.
    - include_city_ows=False removes both raw city-level OWS features
      (ows_city_*) and derived residuals (temp_minus_ows_city_*).
    - include_station_static=False removes base station metadata plus derived
      static CWS-vs-OWS site-context features.
    """
    exclude = {
        "date",
        "station_id",
        "target_bad",
        "qc_flag_strict",
        "qc_flag_lenient",
        "pred_bad_prob",
        "split",
        "city",
        "ows_nearest_station_id",
    }
    if extra_exclude:
        exclude.update(extra_exclude)

    feature_cols = []
    for c in df.columns:
        c_str = str(c)

        if c in exclude or c_str in exclude:
            continue
        if c_str.endswith("_id") or c_str.endswith("_station_id"):
            continue
        if c_str.startswith("cws_") and not c_str.startswith("cws_minus_"):

            continue
        if not include_coordinates and is_coordinate_like_feature(c_str):
            continue
        if not include_station_static and is_station_static_like_feature(c_str):
            continue
        if not include_satellite and (
            c_str.startswith("LST")
            or c_str.startswith("NDVI")
            or "lst" in c_str.lower()
            or "ndvi" in c_str.lower()
        ):
            continue
        if not include_ows and (
            c_str.startswith("ows_")
            or "ows_" in c_str
            or c_str.startswith("pair_")
            or c_str.startswith("temp_minus_ows")
        ):
            continue
        if not include_city_ows and is_city_ows_like_feature(c_str):
            continue
        if not include_temporal_dynamics and (
            c_str.startswith("temp_lag")
            or c_str.startswith("temp_diff")
            or c_str.startswith("temp_abs_diff")
            or c_str.startswith("temp_roll")
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype) == "category" or pd.api.types.is_bool_dtype(df[c]):
            feature_cols.append(c)

    categorical_cols = [
        c for c in feature_cols
        if str(df[c].dtype) == "category" or pd.api.types.is_string_dtype(df[c])
    ]
    return feature_cols, categorical_cols


def missingness_report(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "target_bad",
) -> pd.DataFrame:
    rows = []
    labeled = df[df[target_col].isin([0, 1])].copy() if target_col in df.columns else df.copy()

    for c in feature_cols:
        miss = labeled[c].isna()
        rows.append({
            "feature": c,
            "missing_pct": float(miss.mean()) if len(labeled) else np.nan,
            "bad_rate_when_missing": float(labeled.loc[miss, target_col].mean()) if target_col in labeled and miss.any() else np.nan,
            "bad_rate_when_present": float(labeled.loc[~miss, target_col].mean()) if target_col in labeled and (~miss).any() else np.nan,
            "n_missing": int(miss.sum()),
            "n_present": int((~miss).sum()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["missing_pct", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def make_group_holdout_split(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("network", "station_id"),
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by station/network groups so stations do not appear in multiple splits."""
    if valid_frac < 0 or test_frac < 0 or valid_frac + test_frac >= 1:
        raise ValueError("valid_frac and test_frac must be non-negative and sum to < 1.")

    groups = df[list(group_cols)].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(groups))

    n_test = int(round(len(groups) * test_frac))
    n_valid = int(round(len(groups) * valid_frac))

    split = np.array(["train"] * len(groups), dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_valid]] = "valid"
    groups["_split"] = split

    tagged = df.merge(groups, on=list(group_cols), how="left", validate="many_to_one")
    return (
        tagged[tagged["_split"] == "train"].drop(columns=["_split"]).copy(),
        tagged[tagged["_split"] == "valid"].drop(columns=["_split"]).copy(),
        tagged[tagged["_split"] == "test"].drop(columns=["_split"]).copy(),
    )


def assign_spatial_blocks(
    df: pd.DataFrame,
    n_lat_bins: int = 5,
    n_lon_bins: int = 5,
    lat_col: str = "station_lat",
    lon_col: str = "station_long",
) -> pd.Series:
    """Assign each row to a quantile-based lat/lon block."""
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Need {lat_col} and {lon_col} for spatial blocks.")

    lat_bins = pd.qcut(df[lat_col], q=n_lat_bins, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df[lon_col], q=n_lon_bins, labels=False, duplicates="drop")
    return lat_bins.astype("Int64").astype(str) + "_" + lon_bins.astype("Int64").astype(str)


def make_spatial_block_split(
    df: pd.DataFrame,
    n_lat_bins: int = 5,
    n_lon_bins: int = 5,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out entire spatial blocks.

    This original version assigns blocks row-wise. Use
    make_station_level_spatial_block_split() for the recommended station-level
    variant where each station contributes once when quantile bins are created.
    """
    x = df.copy()
    x["_spatial_block"] = assign_spatial_blocks(x, n_lat_bins=n_lat_bins, n_lon_bins=n_lon_bins)
    train, valid, test = make_group_holdout_split(
        x,
        group_cols=("_spatial_block",),
        valid_frac=valid_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    return (
        train.drop(columns=["_spatial_block"]).copy(),
        valid.drop(columns=["_spatial_block"]).copy(),
        test.drop(columns=["_spatial_block"]).copy(),
    )


def make_station_level_spatial_block_split(
    df: pd.DataFrame,
    n_lat_bins: int = 5,
    n_lon_bins: int = 5,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    lat_col: str = "station_lat",
    lon_col: str = "station_long",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out spatial blocks using unique stations to define lat/lon bins.

    This avoids row-count-weighted qcut boundaries when some stations have more
    observations than others. All rows from a station remain in the same split.
    """
    required = {"network", "station_id", lat_col, lon_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Need columns for station-level spatial split: {sorted(missing)}")

    station_xy = (
        df[["network", "station_id", lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .groupby(["network", "station_id"], as_index=False)
        .agg(**{
            lat_col: (lat_col, "median"),
            lon_col: (lon_col, "median"),
        })
    )
    if len(station_xy) == 0:
        raise ValueError("No stations with non-missing coordinates for spatial split.")

    station_xy["_lat_bin"] = pd.qcut(
        station_xy[lat_col],
        q=n_lat_bins,
        labels=False,
        duplicates="drop",
    )
    station_xy["_lon_bin"] = pd.qcut(
        station_xy[lon_col],
        q=n_lon_bins,
        labels=False,
        duplicates="drop",
    )
    station_xy["_spatial_block"] = (
        station_xy["_lat_bin"].astype("Int64").astype(str)
        + "_"
        + station_xy["_lon_bin"].astype("Int64").astype(str)
    )

    x = df.merge(
        station_xy[["network", "station_id", "_spatial_block"]],
        on=["network", "station_id"],
        how="left",
        validate="many_to_one",
    )
    if x["_spatial_block"].isna().any():
        n_missing = int(x["_spatial_block"].isna().sum())
        raise ValueError(f"{n_missing} rows could not be assigned to a spatial block due to missing coordinates.")

    train, valid, test = make_group_holdout_split(
        x,
        group_cols=("_spatial_block",),
        valid_frac=valid_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    return (
        train.drop(columns=["_spatial_block"]).copy(),
        valid.drop(columns=["_spatial_block"]).copy(),
        test.drop(columns=["_spatial_block"]).copy(),
    )


def _model_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities for CatBoost/sklearn/lightgbm-like models."""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if isinstance(prob, list):
            prob = np.asarray(prob)
        return np.asarray(prob)[:, 1]

    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-z))

    raise TypeError("Model does not expose predict_proba or decision_function.")


def expected_calibration_error(y_true, pred_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob, dtype="float64")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return np.nan
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (pred_prob >= lo) & (pred_prob <= hi)
        else:
            mask = (pred_prob >= lo) & (pred_prob < hi)
        if not np.any(mask):
            continue
        conf = pred_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def evaluate_probability_predictions(
    y_true,
    pred_prob,
    split_name: str = "test",
    threshold: float | None = None,
) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype="float64")
    if threshold is None:
        threshold = _choose_threshold(y, p)
    pred = (p >= threshold).astype(int)

    out = {
        "split": split_name,
        "n": int(len(y)),
        "bad_rate": float(np.mean(y)) if len(y) else np.nan,
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "ece_15": expected_calibration_error(y, p, n_bins=15) if len(np.unique(y)) > 1 else np.nan,
    }
    try:
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        out["log_loss"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def evaluate_model(
    model,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    split_name: str = "test",
    threshold: float | None = None,
) -> dict:
    y = df["target_bad"].astype(int).values
    p = _model_predict_proba(model, df[list(feature_cols)])
    return evaluate_probability_predictions(y, p, split_name=split_name, threshold=threshold)


def add_model_score_column(
    model,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    score_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[score_col] = _model_predict_proba(model, out[list(feature_cols)])
    return downcast_dataframe(out)


def train_lightgbm_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.04,
    num_leaves: int = 96,
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
):
    """Train a LightGBM baseline if lightgbm is installed."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is not installed. Install lightgbm or set RUN_LIGHTGBM=False.") from e

    tr = maybe_subsample_rows(train_df, max_rows=max_train_rows, random_state=random_state)
    va = maybe_subsample_rows(valid_df, max_rows=max_valid_rows, random_state=random_state)

    X_train = tr[list(feature_cols)].copy()
    X_valid = va[list(feature_cols)].copy()
    for c in categorical_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_valid[c] = X_valid[c].astype("category")

    y_train = tr["target_bad"].astype(int)
    y_valid = va["target_bad"].astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=[c for c in categorical_cols if c in X_train.columns],
        callbacks=callbacks,
    )
    return clf


def _make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def train_sgd_logistic_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    random_state: int = 42,
    max_train_rows: int | None = 1_500_000,
):
    """Fast linear baseline using SGD logistic regression.

    This is intended as a sanity baseline, not a model-selection winner.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    tr = maybe_subsample_rows(train_df, max_rows=max_train_rows, random_state=random_state)
    X_train = tr[list(feature_cols)].copy()
    y_train = tr["target_bad"].astype(int)

    cat = [c for c in categorical_cols if c in feature_cols]
    num = [c for c in feature_cols if c not in cat]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num),
            ("cat", categorical_pipe, cat),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=25,
        tol=1e-3,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def make_retention_curve(
    df: pd.DataFrame,
    risk_col: str,
    residual_col: str | None = None,
    target_col: str | None = "target_bad",
    fractions: Sequence[float] | None = None,
    method_name: str = "model",
) -> pd.DataFrame:
    """Compute metrics when keeping the lowest-risk fraction of rows."""
    if fractions is None:
        fractions = np.round(np.linspace(0.50, 0.99, 20), 3)

    x = df.copy()
    x = x[x[risk_col].notna()].copy()
    if len(x) == 0:
        return pd.DataFrame()

    rows = []
    for frac in fractions:
        frac = float(frac)
        cutoff = x[risk_col].quantile(frac)
        kept = x[x[risk_col] <= cutoff]
        row = {
            "method": method_name,
            "retention_target": frac,
            "risk_cutoff": float(cutoff),
            "n_kept": int(len(kept)),
            "retention_actual": float(len(kept) / len(x)),
        }

        if target_col is not None and target_col in kept.columns:
            labeled = kept[kept[target_col].isin([0, 1])]
            row["weak_bad_rate_kept"] = float(labeled[target_col].mean()) if len(labeled) else np.nan

        if residual_col is not None and residual_col in kept.columns:
            resid = pd.to_numeric(kept[residual_col], errors="coerce").dropna()
            row["residual_n"] = int(len(resid))
            row["residual_bias"] = float(resid.mean()) if len(resid) else np.nan
            row["residual_mae"] = float(resid.abs().mean()) if len(resid) else np.nan
            row["residual_rmse"] = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan
            row["residual_p95_abs"] = float(resid.abs().quantile(0.95)) if len(resid) else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def fixed_filter_metrics(
    df: pd.DataFrame,
    keep_mask: pd.Series,
    method_name: str,
    residual_col: str | None = None,
    target_col: str = "target_bad",
) -> dict:
    """Metrics for fixed filters such as raw, strict QC, and lenient QC."""
    mask = keep_mask.fillna(False).astype(bool)
    kept = df[mask]
    row = {
        "method": method_name,
        "n_total": int(len(df)),
        "n_kept": int(len(kept)),
        "retention_actual": float(len(kept) / len(df)) if len(df) else np.nan,
    }
    if target_col in kept.columns:
        labeled = kept[kept[target_col].isin([0, 1])]
        row["weak_bad_rate_kept"] = float(labeled[target_col].mean()) if len(labeled) else np.nan

    if residual_col is not None and residual_col in kept.columns:
        resid = pd.to_numeric(kept[residual_col], errors="coerce").dropna()
        row["residual_n"] = int(len(resid))
        row["residual_bias"] = float(resid.mean()) if len(resid) else np.nan
        row["residual_mae"] = float(resid.abs().mean()) if len(resid) else np.nan
        row["residual_rmse"] = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan
        row["residual_p95_abs"] = float(resid.abs().quantile(0.95)) if len(resid) else np.nan
    return row


def baseline_filter_table(
    df: pd.DataFrame,
    residual_col: str | None = "temp_minus_ows_idw",
) -> pd.DataFrame:
    """Raw/lenient/strict fixed-filter retention metrics."""
    rows = [fixed_filter_metrics(df, pd.Series(True, index=df.index), "raw", residual_col=residual_col)]
    if "qc_flag_lenient" in df.columns:
        rows.append(fixed_filter_metrics(df, df["qc_flag_lenient"] == 0, "lenient_qc", residual_col=residual_col))
    if "qc_flag_strict" in df.columns:
        rows.append(fixed_filter_metrics(df, df["qc_flag_strict"] == 0, "strict_qc", residual_col=residual_col))
    return pd.DataFrame(rows)


def save_dataframe_auto(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame to parquet if possible, otherwise pickle.

    Returns the actual written path.
    """
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
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def print_metric_block(metrics: dict) -> None:
    """Compact notebook-friendly metric printer."""
    print(f"\n--- {metrics.get('split', 'split').upper()} ---")
    for k in ["n", "bad_rate", "auroc", "auprc", "brier", "ece_15", "threshold", "f1"]:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
