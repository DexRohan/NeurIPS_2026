
"""QC and residual-risk fusion utilities for CWS temperature quality control."""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


KEY_COLS = ["date", "station_id", "network"]


@dataclass
class FusionConfig:
    city: str = "project_id"

    scored_input: str | None = None
    residual_risk_manifest: str | None = None
    residual_risk_dir: str | None = None
    output_dir: str = "outputs/qc_risk_fusion"
    run_label: str | None = None
    overwrite: bool = False

    risk_model_key: str = "context_history"
    risk_model_name: str | None = None
    prob_col: str | None = None
    reliability_path: str | None = None
    reliability_calibration_split: str = "valid"
    use_reliability_calibration: bool = True


    qc_lenient: str | None = None
    qc_strict: str | None = None
    qc_ultra_strict: str | None = None
    qc_dir: str | None = None
    auto_discover_qc: bool = True
    force_rebuild_qc_cache: bool = False


    evaluation_split: str = "test"



    qc_weight_lenient: float = 0.95
    qc_weight_strict: float = 0.70
    qc_weight_ultra_strict: float = 0.25


    resid_z_center: float = 3.8
    resid_z_slope: float = 1.25
    low_abs_z_for_valid: float = 2.05


    model_low_prob: float = 0.05
    model_high_prob: float = 0.50
    final_low_error_prob: float = 0.15
    final_moderate_error_prob: float = 0.30
    final_high_error_prob: float = 0.70
    reference_uncertainty_high: float = 0.80
    reference_uncertainty_moderate: float = 0.55
    bias_explained_high: float = 0.55
    bias_explained_moderate: float = 0.35
    microclimate_support_high: float = 0.55
    microclimate_support_moderate: float = 0.35
    environmental_directional_support_high: float = 0.50
    environmental_directional_support_moderate: float = 0.30
    radiation_bias_high: float = 0.60
    radiation_bias_moderate: float = 0.40
    radiation_low_wind_quantile: float = 0.25
    radiation_min_positive_resid_c: float = 0.75
    min_abs_expected_bias_c: float = 0.50


    risk_keep_cutoffs: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.80)
    final_keep_error_cutoffs: tuple[float, ...] = (0.15, 0.30, 0.50, 0.70)
    sample_rows_per_category: int = 250
    random_state: int = 42


    add_station_bias_correction_preview: bool = True
    save_policy_output_tables: bool = True


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


def ensure_key_types(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "date" in x.columns:
        x["date"] = ensure_utc_datetime(x["date"])
    if "station_id" in x.columns:
        x["station_id"] = clean_station_id(x["station_id"])
    if "network" in x.columns:
        x["network"] = x["network"].map(normalize_network_name)
    return x


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
    raise ValueError(f"Unsupported file type: {path}")


def save_dataframe_auto(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as e:
            warnings.warn(f"Parquet save failed for {path} ({e}); falling back to pickle.")
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


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="float")
        elif pd.api.types.is_integer_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], downcast="integer")
        elif x[col].dtype == "object":
            nunique = x[col].nunique(dropna=True)
            if 0 < nunique <= min(1000, max(50, len(x) // 50)):
                try:
                    x[col] = x[col].astype("category")
                except Exception:
                    pass
    return x


def sigmoid(z) -> np.ndarray:
    arr = np.asarray(z, dtype="float64")
    arr = np.clip(arr, -50, 50)
    return 1.0 / (1.0 + np.exp(-arr))


def _first_existing(cols: Iterable[str], candidates: Sequence[str]) -> str | None:
    colset = set(map(str, cols))
    for c in candidates:
        if c in colset:
            return c
    return None


def _safe_numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _wind_speed_from_available_columns(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return wind speed in m/s when available.

    The residual-risk tables have used several wind column conventions across
    runs.  This helper keeps the fusion logic robust without requiring one exact
    column name.
    """
    direct = _first_existing(
        df.columns,
        [
            "wind_speed_ms",
            "wind_speed",
            "wind_speed_10m_ms",
            "wind_10m_ms",
            "ws10_ms",
            "era5_wind_speed_ms",
        ],
    )
    if direct:
        return pd.to_numeric(df[direct], errors="coerce"), direct

    if {"u10_ms", "v10_ms"}.issubset(df.columns):
        u = pd.to_numeric(df["u10_ms"], errors="coerce")
        v = pd.to_numeric(df["v10_ms"], errors="coerce")
        return np.sqrt(u * u + v * v), "sqrt(u10_ms^2+v10_ms^2)"

    if {"u10", "v10"}.issubset(df.columns):
        u = pd.to_numeric(df["u10"], errors="coerce")
        v = pd.to_numeric(df["v10"], errors="coerce")
        return np.sqrt(u * u + v * v), "sqrt(u10^2+v10^2)"

    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def resolve_inputs(cfg: FusionConfig) -> tuple[Path, Path | None, dict]:
    manifest = {}
    if cfg.residual_risk_manifest:
        manifest_path = Path(cfg.residual_risk_manifest)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
    elif cfg.residual_risk_dir:
        candidate = Path(cfg.residual_risk_dir) / f"{cfg.city}_residual_risk_manifest.json"
        if candidate.exists():
            manifest = json.loads(candidate.read_text())
            cfg.residual_risk_manifest = str(candidate)

    scored_path = Path(cfg.scored_input) if cfg.scored_input else None
    if scored_path is None:
        value = manifest.get("scored_path") or manifest.get("outputs", {}).get("scored_path")
        if value:
            scored_path = Path(value)
    if scored_path is None:
        raise FileNotFoundError("No scored input path supplied and none found in manifest.")

    reliability_path = Path(cfg.reliability_path) if cfg.reliability_path else None
    if reliability_path is None:
        value = manifest.get("reliability_path") or manifest.get("outputs", {}).get("reliability_path")
        if value:
            reliability_path = Path(value)
        elif cfg.residual_risk_dir:
            candidate = Path(cfg.residual_risk_dir) / f"{cfg.city}_residual_risk_reliability.csv"
            if candidate.exists():
                reliability_path = candidate

    if cfg.prob_col is None:
        cfg.prob_col = f"pred_ref_risk_prob_{cfg.risk_model_key}"
    if cfg.risk_model_name is None:
        cfg.risk_model_name = f"catboost_{cfg.risk_model_key}"

    return scored_path, reliability_path, manifest


FEATURE_GROUP_PATTERNS: dict[str, list[str]] = {
    "core_keys": ["date", "station_id", "network", "split"],
    "temperature_reference": [
        "temp_raw", "ref_mu", "ref_sigma", "cws_ref_resid",
        "abs_cws_ref_resid", "cws_ref_z", "abs_cws_ref_z",
    ],
    "risk_model": ["pred_ref_risk_prob_context_history", "pred_ref_risk_prob_context_static_met"],
    "targets": ["target_ref_risk", "target_ref_risk_hiconf", "target_ref_risk_band_z"],
    "station_history": ["station_ref_"],
    "reference_support": ["ref_support_", "ref_local_", "ref_nearest_", "ref_idw_", "ref_frac_"],
    "static_metadata": [
        "building_height_m", "elev_meters", "LC_point_lg", "LC_buffer_lg",
        "LCZ_point_lg", "LCZ_buffer_lg", "LC_buffer_fraction", "LCZ_buffer_fraction",
    ],
    "era5_meteorology": ["t2m_c", "d2m_c", "u10_ms", "v10_ms", "wind_speed", "ssrd_wm2", "tp_mm", "is_daylight", "is_rain"],
    "satellite_context": ["LST_C", "LST_C_asof", "LST_C_gapfill_3h", "LST_C_gapfill_clim", "NDVI", "lst_", "ndvi_", "ows_idw_lst", "ows_idw_ndvi", "ows_nearest_lst", "ows_nearest_ndvi"],
    "qc_flags": ["qc_"],
}


def _columns_matching(df: pd.DataFrame, patterns: Sequence[str]) -> list[str]:
    cols = []
    for c in df.columns:
        for p in patterns:
            if p.endswith("_"):
                if str(c).startswith(p):
                    cols.append(c)
                    break
            elif str(c) == p or str(c).startswith(p):
                cols.append(c)
                break
    return sorted(set(cols))


def feature_coverage_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for group, patterns in FEATURE_GROUP_PATTERNS.items():
        cols = _columns_matching(df, patterns)
        for c in cols:
            s = df[c]
            rows.append({
                "feature_group": group,
                "feature": c,
                "dtype": str(s.dtype),
                "nonmissing_n": int(s.notna().sum()),
                "nonmissing_fraction": float(s.notna().mean()) if n else np.nan,
                "n_unique": int(s.nunique(dropna=True)) if n else 0,
                "example_values": ", ".join(map(str, s.dropna().astype(str).head(3).tolist())),
            })
    return pd.DataFrame(rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)


def compact_feature_group_audit(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby("feature_group", as_index=False)
        .agg(
            n_features=("feature", "count"),
            median_nonmissing_fraction=("nonmissing_fraction", "median"),
            min_nonmissing_fraction=("nonmissing_fraction", "min"),
            max_nonmissing_fraction=("nonmissing_fraction", "max"),
        )
        .sort_values("feature_group")
    )


def _infer_qc_level_from_filename(path: str | Path) -> str | None:
    """Infer lenient/strict/ultra_strict from the QC flagged filename."""
    name = Path(path).name.lower()
    if "ultra" in name:
        return "ultra_strict"
    if "strict" in name:
        return "strict"
    if "lenient" in name or "loose" in name:
        return "lenient"
    return None


def _qc_candidate_rank(path: Path, level: str) -> tuple:
    """Prefer the expected CWS TA QC files when several matches exist."""
    name = path.name.lower()
    expected_level_token = level.replace("_", "")
    name_no_underscore = name.replace("_", "")
    return (
        0 if "qc_flagged" in name else 1,
        0 if "cws" in name else 1,
        0 if name.startswith("ta_") else 1,
        0 if expected_level_token in name_no_underscore else 1,
        len(path.parts),
        len(name),
        str(path),
    )


def discover_qc_flagged_files(qc_dir: str | Path | None) -> dict[str, Path]:
    """Discover lenient/strict/ultra-strict CrowdQC+ flagged CSV files.

    Expected project-specific names are for example:
      ta_project_2021-2021_h_CWS_lenient_qc_flagged.csv
      ta_project_2021-2021_h_CWS_strict_qc_flagged.csv
      ta_project_2021-2021_h_CWS_ultra_strict_qc_flagged.csv

    The ultra-strict test is evaluated before the strict test so that
    ``ultra_strict`` is never accidentally categorized as plain ``strict``.
    """
    if qc_dir is None:
        return {}
    qc_dir = Path(qc_dir)
    if not qc_dir.exists():
        return {}

    candidates = sorted(
        set(qc_dir.rglob("*qc_flagged*.csv"))
        | set(qc_dir.rglob("*QC*flagged*.csv"))
        | set(qc_dir.rglob("*flagged*.csv"))
    )
    by_level: dict[str, list[Path]] = {"lenient": [], "strict": [], "ultra_strict": []}
    for p in candidates:
        level = _infer_qc_level_from_filename(p)
        if level in by_level:
            by_level[level].append(p)

    found: dict[str, Path] = {}
    for level, paths in by_level.items():
        if not paths:
            continue
        found[level] = sorted(paths, key=lambda p: _qc_candidate_rank(p, level))[0]
    return found


def _read_qc_wide_csv(path: str | Path) -> pd.DataFrame:
    """Read a wide QC CSV robustly and normalize the date column name.

    The QC files are comma-separated, but this also handles accidental
    tab/semicolon exports. Column names are stripped but station IDs containing
    colons are left unchanged.
    """
    path = Path(path)
    try:
        wide = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        wide = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")



    if wide.shape[1] == 1:
        wide = pd.read_csv(path, sep=None, engine="python", low_memory=False)



    wide = wide.rename(columns={c: str(c).strip().lstrip("\ufeff") for c in wide.columns})

    if "date" not in wide.columns:
        date_like = [c for c in wide.columns if str(c).strip().lower() == "date"]
        if date_like:
            wide = wide.rename(columns={date_like[0]: "date"})
        elif len(wide.columns) and str(wide.columns[0]).strip().lower().startswith("date"):
            wide = wide.rename(columns={wide.columns[0]: "date"})

    return wide


def _qc_parse_audit_dict(
    *,
    path: Path,
    level: str,
    source_format: str,
    wide: pd.DataFrame,
    paired_station_cols: Sequence[str],
    paired_flag_cols: Sequence[str],
    orphan_raw_cols: Sequence[str],
    orphan_flag_cols: Sequence[str],
    out: pd.DataFrame,
    flag_col: str,
    duplicate_rows_collapsed: int,
) -> dict:
    available = pd.to_numeric(out[flag_col], errors="coerce").notna() if flag_col in out.columns else pd.Series(False, index=out.index)
    flagged = pd.to_numeric(out[flag_col], errors="coerce").eq(1) if flag_col in out.columns else pd.Series(False, index=out.index)
    return {
        "qc_level": level,
        "source_file": str(path),
        "source_format": source_format,
        "wide_rows": int(len(wide)),
        "wide_columns": int(len(wide.columns)),
        "paired_station_count": int(len(paired_station_cols)),
        "paired_flag_count": int(len(paired_flag_cols)),
        "orphan_raw_column_count": int(len(orphan_raw_cols)),
        "orphan_flag_column_count": int(len(orphan_flag_cols)),
        "orphan_raw_columns_example": ", ".join(map(str, list(orphan_raw_cols)[:10])),
        "orphan_flag_columns_example": ", ".join(map(str, list(orphan_flag_cols)[:10])),
        "long_rows_before_merge": int(len(out)),
        "long_available_flag_rows": int(available.sum()),
        "long_flagged_rows": int(flagged.sum()),
        "long_flagged_fraction": float(flagged[available].mean()) if bool(available.any()) else np.nan,
        "duplicate_date_station_rows_collapsed": int(duplicate_rows_collapsed),
    }


def qc_wide_to_long(path: str | Path, level: str, cache_dir: str | Path, force: bool = False) -> pd.DataFrame:
    """Convert a wide QC flagged file to long date/station_id flag rows.

    This handles the paired raw/flag format exactly:

        date,
        <station_id>,
        <station_id>_is_outlier,
        <station_id_2>,
        <station_id_2>_is_outlier,
        ...

    Only columns ending in ``_is_outlier`` become QC flags.  The paired raw
    temperature columns are used only to mask missing observations; they are
    never interpreted as labels and are not carried into the fused table.
    """
    path = Path(path)
    level = str(level)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    cache_base = cache_dir / f"qc_flags_long_v2__{level}__{safe_stem}"
    cache_path = cache_base.with_suffix(".parquet")
    cache_fallback_path = cache_base.with_suffix(".pkl")
    audit_path = cache_base.with_suffix(".audit.json")
    if not force:
        if cache_path.exists():
            out = load_dataframe_auto(cache_path)
            out = ensure_key_types(out)
            if audit_path.exists():
                try:
                    out.attrs["qc_parse_audit"] = json.loads(audit_path.read_text())
                except Exception:
                    pass
            return out
        if cache_fallback_path.exists():
            out = load_dataframe_auto(cache_fallback_path)
            out = ensure_key_types(out)
            if audit_path.exists():
                try:
                    out.attrs["qc_parse_audit"] = json.loads(audit_path.read_text())
                except Exception:
                    pass
            return out

    wide = _read_qc_wide_csv(path)
    if "date" not in wide.columns:
        raise KeyError(
            f"{path} has no 'date' column after CSV parsing. "
            f"Parsed columns begin with: {list(wide.columns[:8])}"
        )


    flag_cols_all = [c for c in wide.columns if c != "date" and str(c).endswith("_is_outlier")]
    flag_to_station = {c: str(c).replace("_is_outlier", "", 1).strip() for c in flag_cols_all}
    raw_cols_all = [c for c in wide.columns if c != "date" and not str(c).endswith("_is_outlier")]
    raw_col_set = set(map(str, raw_cols_all))

    paired_flag_cols = [c for c in flag_cols_all if flag_to_station[c] in raw_col_set]
    paired_station_cols = [flag_to_station[c] for c in paired_flag_cols]
    orphan_flag_cols = [c for c in flag_cols_all if flag_to_station[c] not in raw_col_set]
    paired_station_set = set(paired_station_cols)
    orphan_raw_cols = [c for c in raw_cols_all if str(c) not in paired_station_set]

    flag_col = f"qc_{level}_is_outlier"
    duplicate_rows_collapsed = 0

    if paired_station_cols:


        values_long = wide[["date"] + paired_station_cols].melt(
            id_vars="date", var_name="station_id", value_name=f"temp_in_qc_file_{level}"
        )
        flags_wide = wide[["date"] + paired_flag_cols].rename(
            columns={flag: station for flag, station in zip(paired_flag_cols, paired_station_cols)}
        )
        flags_long = flags_wide.melt(id_vars="date", var_name="station_id", value_name=flag_col)
        out = values_long.merge(flags_long, on=["date", "station_id"], how="inner", validate="one_to_one")
        out = out[pd.notna(out[f"temp_in_qc_file_{level}"])].copy()
        out = out.drop(columns=[f"temp_in_qc_file_{level}"])
        source_format = "paired_raw_temperature_plus_is_outlier_flags"
    elif flag_cols_all:

        flags_long = wide[["date"] + flag_cols_all].melt(
            id_vars="date", var_name="raw_col", value_name=flag_col
        )
        flags_long["station_id"] = flags_long["raw_col"].astype(str).str.replace("_is_outlier", "", regex=False).str.strip()
        out = flags_long.drop(columns=["raw_col"])
        source_format = "flag_columns_only_no_raw_temperature_mask"
    else:
        raise ValueError(
            f"No *_is_outlier QC flag columns found in {path}. "
            f"Parsed columns begin with: {list(wide.columns[:10])}"
        )

    out = ensure_key_types(out)
    flag_numeric = pd.to_numeric(out[flag_col], errors="coerce")
    nonmissing_flags = flag_numeric.dropna()
    unexpected = sorted(pd.unique(nonmissing_flags[~nonmissing_flags.isin([0, 1, 0.0, 1.0])]).tolist())
    if unexpected:
        raise ValueError(
            f"{path} produced non-binary QC flags for {level}: {unexpected[:10]}. "
            "This usually means a raw temperature column was accidentally read as a flag."
        )
    out[flag_col] = flag_numeric.astype("Float32")

    duplicate_rows_collapsed = int(out.duplicated(["date", "station_id"]).sum())
    if duplicate_rows_collapsed:

        out = (
            out.groupby(["date", "station_id"], as_index=False)[flag_col]
            .max()
        )
        out = ensure_key_types(out)

    audit = _qc_parse_audit_dict(
        path=path,
        level=level,
        source_format=source_format,
        wide=wide,
        paired_station_cols=paired_station_cols,
        paired_flag_cols=paired_flag_cols,
        orphan_raw_cols=orphan_raw_cols,
        orphan_flag_cols=orphan_flag_cols,
        out=out,
        flag_col=flag_col,
        duplicate_rows_collapsed=duplicate_rows_collapsed,
    )
    out.attrs["qc_parse_audit"] = audit

    save_dataframe_auto(out, cache_path)
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    return out


def merge_qc_flags(scored: pd.DataFrame, cfg: FusionConfig, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Path], pd.DataFrame]:
    qc_files: dict[str, Path] = {}
    explicit = {
        "lenient": cfg.qc_lenient,
        "strict": cfg.qc_strict,
        "ultra_strict": cfg.qc_ultra_strict,
    }
    for k, v in explicit.items():
        if v and Path(v).exists():
            qc_files[k] = Path(v)
    if cfg.auto_discover_qc:
        discovered = discover_qc_flagged_files(cfg.qc_dir)
        for k, v in discovered.items():
            qc_files.setdefault(k, v)

    cache_dir = output_dir / "cache"
    qc_long = None
    parse_audits: list[dict] = []
    for level, path in qc_files.items():
        q = qc_wide_to_long(path, level, cache_dir=cache_dir, force=cfg.force_rebuild_qc_cache)
        if isinstance(q.attrs.get("qc_parse_audit"), dict):
            parse_audits.append(q.attrs["qc_parse_audit"])
        if qc_long is None:
            qc_long = q
        else:
            qc_long = qc_long.merge(q, on=["date", "station_id"], how="outer", validate="one_to_one")

    if qc_long is None:
        fused = scored.copy()
        qc_long = pd.DataFrame(columns=["date", "station_id"])
    else:
        qc_long = ensure_key_types(qc_long)


        fused = scored.merge(qc_long, on=["date", "station_id"], how="left", validate="many_to_one")

    qc_cols = [c for c in fused.columns if c.startswith("qc_") and c.endswith("_is_outlier")]
    for c in qc_cols:
        fused[c] = pd.to_numeric(fused[c], errors="coerce").astype("Float32")

    parse_by_level = {str(a.get("qc_level")): a for a in parse_audits}
    audit_rows = []
    for c in qc_cols:
        level = c.replace("qc_", "").replace("_is_outlier", "")
        available = fused[c].notna()
        row = {
            "qc_col": c,
            "qc_level": level,
            "available_rows": int(available.sum()),
            "available_fraction": float(available.mean()) if len(fused) else np.nan,
            "flagged_rows": int((fused[c] == 1).sum()),
            "flagged_fraction_available": float((fused.loc[available, c] == 1).mean()) if available.any() else np.nan,
        }
        if level in parse_by_level:
            row.update({k: v for k, v in parse_by_level[level].items() if k not in row})
        audit_rows.append(row)


    for level, audit in parse_by_level.items():
        expected_col = f"qc_{level}_is_outlier"
        if expected_col not in qc_cols:
            row = {"qc_col": expected_col, "qc_level": level, "available_rows": 0, "available_fraction": 0.0, "flagged_rows": 0, "flagged_fraction_available": np.nan}
            row.update(audit)
            audit_rows.append(row)

    qc_audit = pd.DataFrame(audit_rows)
    return fused, qc_files, qc_audit


def add_qc_evidence(df: pd.DataFrame, cfg: FusionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()
    qc_cols = [c for c in x.columns if c.startswith("qc_") and c.endswith("_is_outlier")]
    weights = {
        "lenient": cfg.qc_weight_lenient,
        "strict": cfg.qc_weight_strict,
        "ultra_strict": cfg.qc_weight_ultra_strict,
    }

    if not qc_cols:
        x["qc_available_count"] = 0
        x["qc_flag_count"] = 0.0
        x["p_qc_crowdqc_available_normalized"] = np.nan
        x["p_qc_crowdqc_full_normalized"] = 0.0
        x["p_qc_crowdqc"] = 0.0
        x["qc_any_flag"] = False
        x["qc_strong_flag"] = False
        x["qc_flag_pattern"] = "no_qc_available"
        return x, pd.DataFrame()

    weighted_sum = pd.Series(0.0, index=x.index, dtype="float64")
    available_weight = pd.Series(0.0, index=x.index, dtype="float64")
    full_weight = 0.0

    pattern_parts = []
    for c in qc_cols:
        level = c.replace("qc_", "").replace("_is_outlier", "")
        w = float(weights.get(level, 0.5))
        full_weight += w
        val = pd.to_numeric(x[c], errors="coerce")
        weighted_sum += val.fillna(0) * w
        available_weight += val.notna().astype(float) * w
        pattern_parts.append(level + "=" + val.map(lambda z: "NA" if pd.isna(z) else str(int(z))).astype(str))

    x["qc_available_count"] = x[qc_cols].notna().sum(axis=1).astype("int16")
    x["qc_flag_count"] = x[qc_cols].fillna(0).sum(axis=1).astype("float32")
    x["p_qc_crowdqc_available_normalized"] = (weighted_sum / available_weight.replace(0, np.nan)).clip(0, 1).astype("float32")
    x["p_qc_crowdqc_full_normalized"] = (weighted_sum / full_weight).clip(0, 1).astype("float32") if full_weight > 0 else 0.0



    x["p_qc_crowdqc"] = x["p_qc_crowdqc_full_normalized"].fillna(0).astype("float32")
    x["qc_any_flag"] = x[qc_cols].fillna(0).sum(axis=1).gt(0)
    x["qc_strong_flag"] = False
    for level in ["lenient", "strict"]:
        col = f"qc_{level}_is_outlier"
        if col in x.columns:
            x["qc_strong_flag"] = x["qc_strong_flag"] | (x[col] == 1)


    pattern_df = pd.DataFrame(pattern_parts).T if pattern_parts else pd.DataFrame(index=x.index)
    x["qc_flag_pattern"] = pattern_df.apply(lambda row: "|".join(row.values.astype(str)), axis=1).astype("string")

    rows = []
    for pattern, g in x.groupby("qc_flag_pattern", dropna=False):
        rows.append({
            "qc_flag_pattern": str(pattern),
            "n": int(len(g)),
            "fraction": float(len(g) / max(len(x), 1)),
            "mean_p_qc_crowdqc": float(pd.to_numeric(g["p_qc_crowdqc"], errors="coerce").mean()),
            "qc_any_flag_rate": float(g["qc_any_flag"].mean()),
            "qc_strong_flag_rate": float(g["qc_strong_flag"].mean()),
        })
    pattern_summary = pd.DataFrame(rows).sort_values("n", ascending=False)
    return x, pattern_summary


def apply_reliability_bin_calibration(
    df: pd.DataFrame,
    prob_col: str,
    reliability_path: str | Path | None,
    model_name: str,
    calibration_split: str = "valid",
    enabled: bool = True,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    x = df.copy()
    raw = pd.to_numeric(x[prob_col], errors="coerce").clip(0, 1).astype("float64")
    out_col = f"{prob_col}_reliability_calibrated"
    x[out_col] = raw.astype("float32")

    if not enabled or reliability_path is None or not Path(reliability_path).exists():
        return x, out_col, pd.DataFrame()

    rel = pd.read_csv(reliability_path)
    rel_use = rel[(rel["model"].astype(str) == str(model_name)) & (rel["split"].astype(str) == str(calibration_split))].copy()
    if rel_use.empty:
        return x, out_col, rel

    calibrated = pd.Series(np.nan, index=x.index, dtype="float64")
    for _, r in rel_use.iterrows():
        lo, hi = float(r["prob_lo"]), float(r["prob_hi"])
        event = r.get("event_rate", np.nan)
        if pd.isna(event):
            continue
        mask = raw.ge(lo) & (raw.le(hi) if hi >= 1.0 else raw.lt(hi))
        calibrated.loc[mask] = float(event)

    x[out_col] = calibrated.fillna(raw).clip(0, 1).astype("float32")
    return x, out_col, rel_use


def choose_expected_station_bias(df: pd.DataFrame, cfg: FusionConfig) -> pd.DataFrame:
    x = df.copy()
    base = _safe_numeric_series(x, "station_ref_median_resid_train")
    expected = base.copy()

    if {"is_daylight", "station_ref_day_median_resid_train", "station_ref_night_median_resid_train"}.issubset(x.columns):
        is_day = pd.to_numeric(x["is_daylight"], errors="coerce").fillna(0).astype(bool)
        day_bias = pd.to_numeric(x["station_ref_day_median_resid_train"], errors="coerce")
        night_bias = pd.to_numeric(x["station_ref_night_median_resid_train"], errors="coerce")
        expected = expected.where(~(is_day & day_bias.notna()), day_bias)
        expected = expected.where(~((~is_day) & night_bias.notna()), night_bias)

    if {"ssrd_wm2", "station_ref_high_solar_median_resid_train", "station_ref_low_solar_median_resid_train"}.issubset(x.columns):
        train_mask = x.get("split", pd.Series("train", index=x.index)).astype(str).eq("train")
        solar_train = pd.to_numeric(x.loc[train_mask, "ssrd_wm2"], errors="coerce").dropna()
        high_solar_cutoff = float(solar_train.quantile(0.75)) if len(solar_train) else 300.0
        high_solar = pd.to_numeric(x["ssrd_wm2"], errors="coerce") >= high_solar_cutoff
        high_bias = pd.to_numeric(x["station_ref_high_solar_median_resid_train"], errors="coerce")
        low_bias = pd.to_numeric(x["station_ref_low_solar_median_resid_train"], errors="coerce")
        expected = expected.where(~(high_solar & high_bias.notna()), high_bias)
        expected = expected.where(~((~high_solar) & low_bias.notna()), low_bias)
        x["fusion_high_solar_cutoff_wm2"] = high_solar_cutoff
        x["fusion_is_high_solar"] = high_solar
    else:
        x["fusion_high_solar_cutoff_wm2"] = np.nan
        x["fusion_is_high_solar"] = False

    x["expected_train_station_context_bias_c"] = expected.astype("float32")

    resid = _safe_numeric_series(x, "cws_ref_resid")
    expected = pd.to_numeric(x["expected_train_station_context_bias_c"], errors="coerce")
    after = resid - expected
    x["resid_after_train_station_context_bias_c"] = after.astype("float32")
    sign_aligned = np.sign(resid) == np.sign(expected)
    enough_bias = expected.abs() >= cfg.min_abs_expected_bias_c
    explained = 1.0 - (after.abs() / resid.abs().replace(0, np.nan))
    explained = explained.clip(lower=0, upper=1)
    explained = explained.where(sign_aligned & enough_bias, 0.0)
    x["p_residual_explained_by_persistent_station_context"] = explained.fillna(0).clip(0, 1).astype("float32")
    return x


def add_radiation_exposure_bias_score(df: pd.DataFrame, cfg: FusionConfig) -> pd.DataFrame:
    """Add an explicit sunny-day / low-wind radiation-or-siting bias signal.

    High values mean the current residual is positive and occurs under
    high-solar/daytime conditions, especially with low wind and a train-only
    station pattern showing larger high-solar residuals.  This is a bias/siting
    candidate score, not proof of bad data.
    """
    x = df.copy()
    resid = _safe_numeric_series(x, "cws_ref_resid")
    ssrd = _safe_numeric_series(x, "ssrd_wm2")

    if "is_daylight" in x.columns:
        is_day = pd.to_numeric(x["is_daylight"], errors="coerce").fillna(0).astype(bool)
    else:
        is_day = ssrd.fillna(0) > 20

    if "fusion_is_high_solar" in x.columns:
        high_solar = x["fusion_is_high_solar"].fillna(False).astype(bool)
        solar_cutoff = pd.to_numeric(x.get("fusion_high_solar_cutoff_wm2", np.nan), errors="coerce")
        solar_cutoff_value = float(solar_cutoff.dropna().iloc[0]) if solar_cutoff.notna().any() else np.nan
    else:
        train_mask = x.get("split", pd.Series("train", index=x.index)).astype(str).eq("train")
        solar_train = ssrd.loc[train_mask].dropna()
        solar_cutoff_value = float(solar_train.quantile(0.75)) if len(solar_train) else 300.0
        high_solar = ssrd >= solar_cutoff_value
        x["fusion_high_solar_cutoff_wm2"] = solar_cutoff_value
        x["fusion_is_high_solar"] = high_solar

    wind, wind_source = _wind_speed_from_available_columns(x)
    train_mask = x.get("split", pd.Series("train", index=x.index)).astype(str).eq("train")
    wind_train = wind.loc[train_mask].dropna()
    if len(wind_train):
        low_wind_cutoff = float(wind_train.quantile(cfg.radiation_low_wind_quantile))
    else:
        low_wind_cutoff = float(wind.dropna().quantile(cfg.radiation_low_wind_quantile)) if wind.notna().any() else np.nan

    if np.isfinite(low_wind_cutoff):


        p_low_wind = sigmoid((low_wind_cutoff - wind) / max(0.35, abs(low_wind_cutoff) * 0.35))
        low_wind_bool = wind <= low_wind_cutoff
    else:
        p_low_wind = pd.Series(0.5, index=x.index, dtype="float64")
        low_wind_bool = pd.Series(False, index=x.index)

    high_bias = _safe_numeric_series(x, "station_ref_high_solar_median_resid_train")
    low_bias = _safe_numeric_series(x, "station_ref_low_solar_median_resid_train")
    base_bias = _safe_numeric_series(x, "station_ref_median_resid_train")
    solar_excess = high_bias - low_bias
    solar_excess = solar_excess.where(solar_excess.notna(), high_bias - base_bias)

    station_solar_pattern = sigmoid((solar_excess - 0.50) * 2.0)
    station_high_solar_positive = sigmoid((high_bias - 0.50) * 2.0)
    station_training_score = pd.concat(
        [
            pd.Series(station_solar_pattern, index=x.index),
            pd.Series(station_high_solar_positive, index=x.index),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    positive_resid_score = sigmoid((resid - cfg.radiation_min_positive_resid_c) * 1.25)
    positive_resid_score = pd.Series(positive_resid_score, index=x.index).where(resid > 0, 0.0)

    current_context_score = (
        0.45 * high_solar.astype(float)
        + 0.25 * is_day.astype(float)
        + 0.30 * pd.Series(p_low_wind, index=x.index).fillna(0.5)
    ).clip(0, 1)


    station_training_score = station_training_score.fillna(0.25).clip(0, 1)

    p_rad = (
        pd.Series(current_context_score, index=x.index).fillna(0)
        * pd.Series(positive_resid_score, index=x.index).fillna(0)
        * (0.45 + 0.55 * station_training_score)
    ).clip(0, 1)

    x["wind_speed_for_radiation_ms"] = pd.to_numeric(wind, errors="coerce").astype("float32")
    x["wind_speed_for_radiation_source"] = wind_source
    x["radiation_low_wind_cutoff_ms"] = low_wind_cutoff
    x["fusion_is_low_wind"] = pd.Series(low_wind_bool, index=x.index).fillna(False).astype(bool)
    x["station_solar_excess_bias_train_c"] = solar_excess.astype("float32")
    x["radiation_current_context_score"] = pd.Series(current_context_score, index=x.index).astype("float32")
    x["radiation_positive_residual_score"] = pd.Series(positive_resid_score, index=x.index).astype("float32")
    x["radiation_station_training_score"] = station_training_score.astype("float32")
    x["p_radiation_or_siting_bias_signal"] = p_rad.astype("float32")
    return x


def add_reference_uncertainty_score(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "ref_support_score" in x.columns:
        s = pd.to_numeric(x["ref_support_score"], errors="coerce")
        train_mask = x.get("split", pd.Series("train", index=x.index)).astype(str).eq("train")
        q95 = float(s[train_mask].quantile(0.95)) if s[train_mask].notna().any() else float(s.quantile(0.95))
        if not np.isfinite(q95) or q95 <= 0:
            q95 = float(s.max()) if s.notna().any() else 1.0
        x["p_reference_uncertainty"] = (s / q95).clip(0, 1).astype("float32")
        x["reference_uncertainty_source"] = "ref_support_score"
        return x

    components = []
    dist_col = _first_existing(x.columns, ["ref_support_nearest_dist_km", "ref_min_dist_km", "ref_nearest_dist_km", "ref_nearest_value_dist_km"])
    if dist_col:
        components.append((pd.to_numeric(x[dist_col], errors="coerce") / 8.0).clip(0, 1))
    spread_col = _first_existing(x.columns, ["ref_support_local_spread_c", "ref_local_spread_c"])
    if spread_col:
        components.append((pd.to_numeric(x[spread_col], errors="coerce") / 3.0).clip(0, 1))
    if "ref_local_n" in x.columns:
        n = pd.to_numeric(x["ref_local_n"], errors="coerce")
        components.append((1.0 - (n / 8.0).clip(0, 1)))
    if components:
        score = pd.concat(components, axis=1).mean(axis=1)
        x["p_reference_uncertainty"] = score.clip(0, 1).astype("float32")
        x["reference_uncertainty_source"] = "derived_distance_spread_density"
    else:
        x["p_reference_uncertainty"] = np.nan
        x["reference_uncertainty_source"] = "none"
    return x


def _scaled_abs(series: pd.Series, q95: float | None = None, fallback: float = 1.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").abs()
    if q95 is None or not np.isfinite(q95) or q95 <= 0:
        q95 = float(s.quantile(0.95)) if s.notna().any() else fallback
    if not np.isfinite(q95) or q95 <= 0:
        q95 = fallback
    return (s / q95).clip(0, 1)


def add_microclimate_context_support_score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute conservative and sign-aware local-context support.

    Two related scores are produced:

    * ``p_context_mismatch_support``: the CWS and OWS-reference contexts differ
      in LCZ/LC/elevation/building/LST/NDVI or the local OWS field is spatially
      variable.  This is an abstention signal.
    * ``p_context_directional_microclimate_support``: the *direction* of the
      observed residual is consistent with signed LST/elevation/NDVI differences
      where such signed features are available.

    ``p_context_microclimate_support`` / ``p_environmental_difference_signal``
    combine the two, giving priority to sign-aware evidence when available.
    """
    x = df.copy()
    mismatch_components: dict[str, pd.Series] = {}
    directional_components: dict[str, pd.Series] = {}
    derived_sources: list[dict] = []

    cols = list(map(str, x.columns))
    lower_lookup = {str(c).lower(): str(c) for c in cols}

    def first_col_by_tokens(required: Sequence[str], forbidden: Sequence[str] = ()) -> str | None:
        required_l = [r.lower() for r in required]
        forbidden_l = [f.lower() for f in forbidden]
        for c in cols:
            low = c.lower()
            if all(r in low for r in required_l) and not any(f in low for f in forbidden_l):
                return c
        return None

    def first_existing_case_insensitive(candidates: Sequence[str]) -> str | None:
        for c in candidates:
            if c in x.columns:
                return c
            low = c.lower()
            if low in lower_lookup:
                return lower_lookup[low]
        return None

    def scaled_directional_component(
        diff: pd.Series,
        expected_temperature_signal: pd.Series,
        scale: float,
        min_abs_resid: float = 0.50,
    ) -> pd.Series:
        resid = _safe_numeric_series(x, "cws_ref_resid")
        diff = pd.to_numeric(diff, errors="coerce")
        expected_temperature_signal = pd.to_numeric(expected_temperature_signal, errors="coerce")
        mag = (diff.abs() / scale).clip(0, 1)
        valid = diff.notna() & expected_temperature_signal.notna() & resid.notna()
        aligns = (
            valid
            & (resid.abs() >= min_abs_resid)
            & (np.sign(expected_temperature_signal) == np.sign(resid))
        )
        out = mag.where(aligns, 0.0)
        return out.where(valid, np.nan)


    for c in ["ref_support_lcz_mismatch", "ref_support_lc_mismatch"]:
        if c in x.columns:
            mismatch_components[c] = pd.to_numeric(x[c], errors="coerce").clip(0, 1)

    if "ref_support_lcz_mismatch" not in mismatch_components:
        match_col = _first_existing(x.columns, [
            "ref_frac_same_LCZ_point_lg", "ref_frac_same_LCZ_buffer_lg",
            "ref_frac_same_same_LCZ_point_lg", "ref_frac_same_same_LCZ_buffer_lg",
        ])
        if match_col:
            mismatch_components["lcz_mismatch_from_match_fraction"] = (1 - pd.to_numeric(x[match_col], errors="coerce")).clip(0, 1)

    if "ref_support_lc_mismatch" not in mismatch_components:
        match_col = _first_existing(x.columns, [
            "ref_frac_same_LC_point_lg", "ref_frac_same_LC_buffer_lg",
            "ref_frac_same_same_LC_point_lg", "ref_frac_same_same_LC_buffer_lg",
        ])
        if match_col:
            mismatch_components["lc_mismatch_from_match_fraction"] = (1 - pd.to_numeric(x[match_col], errors="coerce")).clip(0, 1)

    elev_abs_col = _first_existing(x.columns, ["ref_support_abs_elev_diff_m", "ref_nearest_abs_elev_diff_m", "ref_idw_abs_elev_diff_m"])
    if elev_abs_col:
        mismatch_components["elevation_context_difference_abs"] = (pd.to_numeric(x[elev_abs_col], errors="coerce").abs() / 150.0).clip(0, 1)

    bh_col = _first_existing(x.columns, ["ref_support_abs_building_height_diff_m", "ref_nearest_abs_building_height_diff_m", "ref_idw_abs_building_height_diff_m"])
    if bh_col:
        mismatch_components["building_height_context_difference_abs"] = (pd.to_numeric(x[bh_col], errors="coerce").abs() / 20.0).clip(0, 1)

    spread_col = _first_existing(x.columns, ["ref_support_local_spread_c", "ref_local_spread_c"])
    if spread_col:
        mismatch_components["local_ows_temperature_spread"] = (pd.to_numeric(x[spread_col], errors="coerce").abs() / 3.0).clip(0, 1)



    lst_signed_col = first_col_by_tokens(["lst", "minus_ows"], forbidden=["abs"])
    if lst_signed_col is None:
        lst_signed_col = first_col_by_tokens(["lst", "diff"], forbidden=["abs"])
    if lst_signed_col is not None:
        lst_diff = pd.to_numeric(x[lst_signed_col], errors="coerce")
        derived_sources.append({"component": "lst_directional_support", "source": lst_signed_col, "derivation": "signed_diff_column"})
    else:
        local_lst_col = first_existing_case_insensitive(["LST_C_asof", "LST_C", "LST_C_gapfill_3h", "LST_C_gapfill_clim"])
        ows_lst_col = first_col_by_tokens(["ows", "lst"], forbidden=["abs", "minus"])
        if local_lst_col and ows_lst_col:
            lst_diff = pd.to_numeric(x[local_lst_col], errors="coerce") - pd.to_numeric(x[ows_lst_col], errors="coerce")
            derived_sources.append({"component": "lst_directional_support", "source": f"{local_lst_col} - {ows_lst_col}", "derivation": "local_minus_ows"})
        else:
            lst_diff = None
    if lst_diff is not None:
        mismatch_components["lst_context_difference_abs_or_signed"] = (pd.to_numeric(lst_diff, errors="coerce").abs() / 8.0).clip(0, 1)
        directional_components["lst_directional_support"] = scaled_directional_component(lst_diff, lst_diff, scale=8.0)



    elev_signed_col = first_col_by_tokens(["elev", "minus_ows"], forbidden=["abs"])
    if elev_signed_col is None:
        elev_signed_col = first_col_by_tokens(["elev", "diff"], forbidden=["abs"])
    if elev_signed_col is not None:
        elev_diff = pd.to_numeric(x[elev_signed_col], errors="coerce")
        derived_sources.append({"component": "elevation_directional_support", "source": elev_signed_col, "derivation": "signed_diff_column"})
    else:
        local_elev_col = first_existing_case_insensitive(["elev_meters", "elevation_m", "elev_m", "height_m"])
        ows_elev_col = first_col_by_tokens(["ows", "elev"], forbidden=["abs", "minus"])
        if local_elev_col and ows_elev_col:
            elev_diff = pd.to_numeric(x[local_elev_col], errors="coerce") - pd.to_numeric(x[ows_elev_col], errors="coerce")
            derived_sources.append({"component": "elevation_directional_support", "source": f"{local_elev_col} - {ows_elev_col}", "derivation": "local_minus_ows"})
        else:
            elev_diff = None
    if elev_diff is not None:
        mismatch_components.setdefault("elevation_context_difference_abs_or_signed", (pd.to_numeric(elev_diff, errors="coerce").abs() / 150.0).clip(0, 1))
        directional_components["elevation_directional_support"] = scaled_directional_component(elev_diff, -pd.to_numeric(elev_diff, errors="coerce"), scale=150.0)



    ndvi_signed_col = first_col_by_tokens(["ndvi", "minus_ows"], forbidden=["abs"])
    if ndvi_signed_col is None:
        ndvi_signed_col = first_col_by_tokens(["ndvi", "diff"], forbidden=["abs"])
    if ndvi_signed_col is not None:
        ndvi_diff = pd.to_numeric(x[ndvi_signed_col], errors="coerce")
        derived_sources.append({"component": "ndvi_directional_cooling_support", "source": ndvi_signed_col, "derivation": "signed_diff_column"})
    else:
        local_ndvi_col = first_existing_case_insensitive(["NDVI", "ndvi"])
        ows_ndvi_col = first_col_by_tokens(["ows", "ndvi"], forbidden=["abs", "minus"])
        if local_ndvi_col and ows_ndvi_col:
            ndvi_diff = pd.to_numeric(x[local_ndvi_col], errors="coerce") - pd.to_numeric(x[ows_ndvi_col], errors="coerce")
            derived_sources.append({"component": "ndvi_directional_cooling_support", "source": f"{local_ndvi_col} - {ows_ndvi_col}", "derivation": "local_minus_ows"})
        else:
            ndvi_diff = None
    if ndvi_diff is not None:
        mismatch_components["ndvi_context_difference_abs_or_signed"] = (pd.to_numeric(ndvi_diff, errors="coerce").abs() / 0.35).clip(0, 1)
        ndvi_dir = scaled_directional_component(ndvi_diff, -pd.to_numeric(ndvi_diff, errors="coerce"), scale=0.35)
        if "is_daylight" in x.columns:
            day_weight = pd.to_numeric(x["is_daylight"], errors="coerce").fillna(0).clip(0, 1)
            ndvi_dir = ndvi_dir * (0.50 + 0.50 * day_weight)
        else:
            ndvi_dir = ndvi_dir * 0.75
        directional_components["ndvi_directional_cooling_support"] = ndvi_dir.clip(0, 1)


    if not any("ndvi" in k for k in mismatch_components):
        ndvi_abs_col = _first_existing(x.columns, ["ndvi_minus_ows_idw_ndvi_abs", "ndvi_minus_ows_nearest_ndvi_abs"])
        if ndvi_abs_col:
            mismatch_components["ndvi_context_difference_abs"] = (pd.to_numeric(x[ndvi_abs_col], errors="coerce").abs() / 0.35).clip(0, 1)

    if not any("lst" in k for k in mismatch_components):
        lst_abs_candidates = [c for c in x.columns if "minus_ows_idw_lst_abs" in str(c).lower() or "minus_ows_nearest_lst_abs" in str(c).lower()]
        if lst_abs_candidates:
            lst_abs_col = sorted(lst_abs_candidates)[0]
            mismatch_components["lst_context_difference_abs"] = (pd.to_numeric(x[lst_abs_col], errors="coerce").abs() / 8.0).clip(0, 1)

    if mismatch_components:
        mismatch_df = pd.DataFrame(mismatch_components, index=x.index)
        mismatch_score = (0.55 * mismatch_df.max(axis=1, skipna=True) + 0.45 * mismatch_df.mean(axis=1, skipna=True)).clip(0, 1)
        x["microclimate_context_component_n"] = mismatch_df.notna().sum(axis=1).astype("int16")
    else:
        mismatch_df = pd.DataFrame(index=x.index)
        mismatch_score = pd.Series(np.nan, index=x.index, dtype="float64")
        x["microclimate_context_component_n"] = 0

    if directional_components:
        directional_df = pd.DataFrame(directional_components, index=x.index)
        directional_score = (0.65 * directional_df.max(axis=1, skipna=True) + 0.35 * directional_df.mean(axis=1, skipna=True)).clip(0, 1)
        x["microclimate_directional_component_n"] = directional_df.notna().sum(axis=1).astype("int16")
    else:
        directional_df = pd.DataFrame(index=x.index)
        directional_score = pd.Series(np.nan, index=x.index, dtype="float64")
        x["microclimate_directional_component_n"] = 0



    mismatch_for_final = mismatch_score.fillna(0)
    directional_for_final = directional_score.fillna(0)
    environmental_signal = (1 - (1 - directional_for_final) * (1 - 0.45 * mismatch_for_final)).clip(0, 1)
    fallback_only = (directional_score.isna()) & mismatch_score.notna()
    environmental_signal = environmental_signal.where(~fallback_only, (0.75 * mismatch_for_final).clip(0, 1))

    x["p_context_mismatch_support"] = mismatch_score.astype("float32")
    x["p_context_directional_microclimate_support"] = directional_score.astype("float32")
    x["p_environmental_difference_signal"] = environmental_signal.astype("float32")
    x["p_context_microclimate_support"] = environmental_signal.astype("float32")

    audit_rows = []
    for name, values in mismatch_components.items():
        audit_rows.append({
            "component": name,
            "component_type": "context_mismatch",
            "nonmissing_n": int(values.notna().sum()),
            "nonmissing_fraction": float(values.notna().mean()),
            "mean": float(values.mean(skipna=True)),
            "p90": float(values.quantile(0.90)) if values.notna().any() else np.nan,
            "source": "",
            "derivation": "",
        })
    for name, values in directional_components.items():
        match = next((d for d in derived_sources if d["component"] == name), {})
        audit_rows.append({
            "component": name,
            "component_type": "directional_environmental_match",
            "nonmissing_n": int(values.notna().sum()),
            "nonmissing_fraction": float(values.notna().mean()),
            "mean": float(values.mean(skipna=True)),
            "p90": float(values.quantile(0.90)) if values.notna().any() else np.nan,
            "source": match.get("source", ""),
            "derivation": match.get("derivation", ""),
        })

    component_audit = pd.DataFrame(audit_rows)
    return x, component_audit


def add_fusion_scores(df: pd.DataFrame, cfg: FusionConfig, calibrated_prob_col: str) -> pd.DataFrame:
    x = df.copy()

    if "abs_cws_ref_z" not in x.columns and "cws_ref_z" in x.columns:
        x["abs_cws_ref_z"] = pd.to_numeric(x["cws_ref_z"], errors="coerce").abs()
    if "abs_cws_ref_resid" not in x.columns and "cws_ref_resid" in x.columns:
        x["abs_cws_ref_resid"] = pd.to_numeric(x["cws_ref_resid"], errors="coerce").abs()

    if "abs_cws_ref_z" in x.columns:
        x["p_reference_residual_inconsistent"] = sigmoid(
            (pd.to_numeric(x["abs_cws_ref_z"], errors="coerce") - cfg.resid_z_center) * cfg.resid_z_slope
        ).astype("float32")
    else:
        x["p_reference_residual_inconsistent"] = np.nan

    x = choose_expected_station_bias(x, cfg)
    x = add_radiation_exposure_bias_score(x, cfg)
    x = add_reference_uncertainty_score(x)
    x, _component_audit = add_microclimate_context_support_score(x)

    p_model = pd.to_numeric(x[calibrated_prob_col], errors="coerce").clip(0, 1)
    p_qc = pd.to_numeric(x.get("p_qc_crowdqc", 0), errors="coerce").clip(0, 1).fillna(0)
    p_resid = pd.to_numeric(x["p_reference_residual_inconsistent"], errors="coerce").clip(0, 1).fillna(p_model)
    p_explained = pd.to_numeric(x["p_residual_explained_by_persistent_station_context"], errors="coerce").clip(0, 1).fillna(0)
    p_ref_unc = pd.to_numeric(x["p_reference_uncertainty"], errors="coerce").clip(0, 1).fillna(0)
    p_micro = pd.to_numeric(x["p_context_microclimate_support"], errors="coerce").clip(0, 1).fillna(0)
    p_rad = pd.to_numeric(x.get("p_radiation_or_siting_bias_signal", 0), errors="coerce").clip(0, 1).fillna(0)


    x["p_reference_model_issue_no_qc"] = (
        1 - (1 - p_model.fillna(0)) * (1 - p_resid)
    ).clip(0, 1).astype("float32")


    x["p_any_quality_issue"] = (
        1 - (1 - p_model.fillna(0)) * (1 - p_qc) * (1 - p_resid)
    ).clip(0, 1).astype("float32")


    p_unexplained_resid = (p_resid * (1 - p_explained)).clip(0, 1)
    p_model_unexplained = (p_model.fillna(0) * (1 - 0.65 * p_explained)).clip(0, 1)
    x["p_unexplained_reference_model_no_qc"] = (
        1 - (1 - p_model_unexplained) * (1 - p_unexplained_resid)
    ).clip(0, 1).astype("float32")

    x["p_unexplained_residual_or_model_error"] = (
        1 - (1 - p_model_unexplained) * (1 - p_qc) * (1 - p_unexplained_resid)
    ).clip(0, 1).astype("float32")


    preservation_factor = (
        (1 - 0.45 * p_ref_unc)
        * (1 - 0.35 * p_micro)
        * (1 - 0.25 * p_rad)
    )
    x["p_reject_as_unexplained_transient_error"] = (
        p_qc * 0.25 + (1 - 0.25) * x["p_unexplained_residual_or_model_error"] * preservation_factor
    ).clip(0, 1).astype("float32")


    x["p_preserve_or_correct_signal"] = (
        1 - (1 - p_explained) * (1 - p_micro) * (1 - p_ref_unc) * (1 - p_rad)
    ).clip(0, 1).astype("float32")

    return x


def assign_categories(df: pd.DataFrame, cfg: FusionConfig, calibrated_prob_col: str) -> pd.DataFrame:
    x = df.copy()

    p_model = pd.to_numeric(x[calibrated_prob_col], errors="coerce")
    p_ref_no_qc = pd.to_numeric(x.get("p_reference_model_issue_no_qc", np.nan), errors="coerce")
    p_any = pd.to_numeric(x["p_any_quality_issue"], errors="coerce").fillna(0)
    p_reject = pd.to_numeric(x["p_reject_as_unexplained_transient_error"], errors="coerce").fillna(0)
    p_qc = pd.to_numeric(x.get("p_qc_crowdqc", 0), errors="coerce").fillna(0)
    p_ref_unc = pd.to_numeric(x["p_reference_uncertainty"], errors="coerce").fillna(0)
    p_explained = pd.to_numeric(x["p_residual_explained_by_persistent_station_context"], errors="coerce").fillna(0)
    p_micro = pd.to_numeric(x["p_context_microclimate_support"], errors="coerce").fillna(0)
    p_env_dir = pd.to_numeric(x.get("p_context_directional_microclimate_support", np.nan), errors="coerce")
    p_env = pd.to_numeric(x.get("p_environmental_difference_signal", p_micro), errors="coerce").fillna(0)
    p_rad = pd.to_numeric(x.get("p_radiation_or_siting_bias_signal", 0), errors="coerce").fillna(0)
    qc_any = x.get("qc_any_flag", pd.Series(False, index=x.index)).fillna(False).astype(bool)
    qc_strong = x.get("qc_strong_flag", pd.Series(False, index=x.index)).fillna(False).astype(bool)

    if "abs_cws_ref_z" in x.columns:
        abs_z = pd.to_numeric(x["abs_cws_ref_z"], errors="coerce")
    else:
        abs_z = pd.Series(np.nan, index=x.index)

    category = pd.Series("ambiguous_review", index=x.index, dtype="object")
    reason = pd.Series("", index=x.index, dtype="object")

    missing = p_model.isna()
    if "temp_raw" in x.columns:
        missing = missing | pd.to_numeric(x["temp_raw"], errors="coerce").isna()
    category.loc[missing] = "missing_or_unscored"
    reason.loc[missing] = "missing temperature or residual-risk score"



    low_reference_risk = (
        (p_ref_no_qc.fillna(1) <= cfg.final_low_error_prob)
        & ((abs_z <= cfg.low_abs_z_for_valid) | abs_z.isna())
        & ~missing
    )
    rescue = low_reference_risk & qc_any
    category.loc[rescue] = "qc_flagged_low_reference_risk_rescue_candidate"
    reason.loc[rescue] = "CrowdQC flagged, but model/reference residual evidence excluding QC is low-risk"

    valid = low_reference_risk & ~qc_any
    category.loc[valid] = "high_confidence_reference_consistent"
    reason.loc[valid] = "low model/reference risk and small calibrated reference residual"


    ref_limited = (
        (p_any >= cfg.model_high_prob)
        & (p_ref_unc >= cfg.reference_uncertainty_high)
        & ~qc_strong
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[ref_limited] = "reference_limited_or_ows_support_review"
    reason.loc[ref_limited] = "large disagreement but OWS reference support is weak"



    radiation = (
        (p_any >= cfg.final_moderate_error_prob)
        & (p_rad >= cfg.radiation_bias_high)
        & (p_qc < 0.60)
        & (p_ref_unc < cfg.reference_uncertainty_high)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[radiation] = "radiation_or_siting_bias_candidate"
    reason.loc[radiation] = "positive residual under sunny/low-wind conditions with train-only solar-bias support"


    environmental = (
        (p_any >= cfg.final_moderate_error_prob)
        & (
            (p_env_dir.fillna(0) >= cfg.environmental_directional_support_high)
            | (p_env >= cfg.microclimate_support_high)
        )
        & (p_qc < 0.60)
        & (p_reject < cfg.final_high_error_prob)
        & (p_ref_unc < cfg.reference_uncertainty_high)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[environmental] = "environmental_difference_candidate_preserve"
    reason.loc[environmental] = "residual direction/magnitude is plausible from LCZ/LST/elevation/NDVI/local context"



    systemic = (
        (p_any >= cfg.final_moderate_error_prob)
        & (p_explained >= cfg.bias_explained_high)
        & (p_rad < cfg.radiation_bias_high)
        & (p_env < cfg.microclimate_support_high)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[systemic] = "systemic_station_bias_correction_candidate"
    reason.loc[systemic] = "large residual aligns with train-only persistent station residual pattern"


    probable_error = (
        (p_reject >= cfg.final_high_error_prob)
        & (p_explained < cfg.bias_explained_moderate)
        & (p_rad < cfg.radiation_bias_moderate)
        & (p_env < cfg.microclimate_support_high)
        & (p_ref_unc < cfg.reference_uncertainty_high)
        & ~missing
        & category.eq("ambiguous_review")
    )
    confirmed = probable_error & qc_any
    missed = probable_error & ~qc_any
    category.loc[confirmed] = "qc_confirmed_probable_transient_or_sensor_error"
    reason.loc[confirmed] = "high unexplained-error evidence and CrowdQC flag"
    category.loc[missed] = "qc_missed_probable_high_risk_observation"
    reason.loc[missed] = "high unexplained-error evidence but no CrowdQC flag"


    possible_preserve = (
        (p_any >= cfg.final_low_error_prob)
        & (
            (p_explained >= cfg.bias_explained_moderate)
            | (p_micro >= cfg.microclimate_support_moderate)
            | (p_rad >= cfg.radiation_bias_moderate)
        )
        & (p_reject < cfg.final_high_error_prob)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[possible_preserve] = "possible_bias_or_microclimate_preserve_with_caution"
    reason.loc[possible_preserve] = "moderate persistent/context/radiation support; not clean enough for conservative use"

    likely_valid = (
        (p_reject <= cfg.final_moderate_error_prob)
        & (p_ref_no_qc.fillna(1) <= cfg.final_moderate_error_prob)
        & (p_qc < 0.20)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[likely_valid] = "likely_valid_low_unexplained_error"
    reason.loc[likely_valid] = "low unexplained-error evidence"


    review_ref = (
        (p_ref_unc >= cfg.reference_uncertainty_high)
        & (p_any >= cfg.final_moderate_error_prob)
        & ~missing
        & category.eq("ambiguous_review")
    )
    category.loc[review_ref] = "reference_limited_or_ows_support_review"
    reason.loc[review_ref] = "OWS support weak; avoid calling sensor error without review"


    reason.loc[category.eq("ambiguous_review") & ~missing] = "mixed or insufficient evidence"

    x["fusion_category"] = category.astype("string")
    x["fusion_reason"] = reason.astype("string")

    action_map = {
        "missing_or_unscored": "exclude_missing",
        "high_confidence_reference_consistent": "use_as_is",
        "likely_valid_low_unexplained_error": "use_as_is_or_light_downweight",
        "qc_flagged_low_reference_risk_rescue_candidate": "rescue_use_with_caution",
        "systemic_station_bias_correction_candidate": "bias_correct_then_use",
        "radiation_or_siting_bias_candidate": "context_bias_correct_or_downweight",
        "possible_bias_or_microclimate_preserve_with_caution": "preserve_or_downweight",
        "environmental_difference_candidate_preserve": "preserve_for_microclimate_analysis",
        "reference_limited_or_ows_support_review": "review_reference_support_or_downweight",
        "qc_confirmed_probable_transient_or_sensor_error": "exclude_transient_error",
        "qc_missed_probable_high_risk_observation": "exclude_or_manual_review",
        "ambiguous_review": "manual_review_or_downweight",
    }
    x["recommended_action"] = x["fusion_category"].map(action_map).astype("string")

    keep_micro = {
        "high_confidence_reference_consistent",
        "likely_valid_low_unexplained_error",
        "qc_flagged_low_reference_risk_rescue_candidate",
        "systemic_station_bias_correction_candidate",
        "radiation_or_siting_bias_candidate",
        "possible_bias_or_microclimate_preserve_with_caution",
        "environmental_difference_candidate_preserve",
    }
    keep_conservative = {
        "high_confidence_reference_consistent",
        "likely_valid_low_unexplained_error",
    }
    bias_correctable = {
        "systemic_station_bias_correction_candidate",
        "radiation_or_siting_bias_candidate",
    }
    reject = {
        "qc_confirmed_probable_transient_or_sensor_error",
        "qc_missed_probable_high_risk_observation",
        "missing_or_unscored",
    }
    x["recommended_keep_conservative"] = x["fusion_category"].isin(keep_conservative)
    x["recommended_keep_microclimate"] = x["fusion_category"].isin(keep_micro)
    x["recommended_bias_correctable"] = x["fusion_category"].isin(bias_correctable)
    x["recommended_reject_transient_error"] = x["fusion_category"].isin(reject)


    weight = 1 - pd.to_numeric(x["p_reject_as_unexplained_transient_error"], errors="coerce").fillna(1)
    weight = weight.clip(0, 1)
    weight.loc[x["fusion_category"].eq("high_confidence_reference_consistent")] = 1.0
    weight.loc[x["fusion_category"].eq("likely_valid_low_unexplained_error")] = np.maximum(weight.loc[x["fusion_category"].eq("likely_valid_low_unexplained_error")], 0.85)
    weight.loc[x["fusion_category"].eq("systemic_station_bias_correction_candidate")] = np.maximum(weight.loc[x["fusion_category"].eq("systemic_station_bias_correction_candidate")], 0.70)
    weight.loc[x["fusion_category"].eq("radiation_or_siting_bias_candidate")] = np.maximum(weight.loc[x["fusion_category"].eq("radiation_or_siting_bias_candidate")], 0.65)
    weight.loc[x["fusion_category"].eq("environmental_difference_candidate_preserve")] = np.maximum(weight.loc[x["fusion_category"].eq("environmental_difference_candidate_preserve")], 0.65)
    weight.loc[x["fusion_category"].eq("reference_limited_or_ows_support_review")] = np.minimum(weight.loc[x["fusion_category"].eq("reference_limited_or_ows_support_review")], 0.50)
    weight.loc[x["recommended_reject_transient_error"].fillna(False)] = 0.0
    weight.loc[x["fusion_category"].eq("missing_or_unscored")] = 0.0
    x["analysis_weight"] = weight.astype("float32")

    return x


def residual_summary(df: pd.DataFrame, residual_col: str = "cws_ref_resid") -> dict:
    if residual_col not in df.columns:
        return {
            "residual_n": 0,
            "residual_bias": np.nan,
            "residual_mae": np.nan,
            "residual_rmse": np.nan,
            "residual_median_abs": np.nan,
            "residual_p90_abs": np.nan,
            "residual_p95_abs": np.nan,
            "residual_p99_abs": np.nan,
            "frac_abs_resid_gt_1c": np.nan,
            "frac_abs_resid_gt_2c": np.nan,
            "frac_abs_resid_gt_3c": np.nan,
        }
    r = pd.to_numeric(df[residual_col], errors="coerce").dropna()
    if len(r) == 0:
        return {
            "residual_n": 0,
            "residual_bias": np.nan,
            "residual_mae": np.nan,
            "residual_rmse": np.nan,
            "residual_median_abs": np.nan,
            "residual_p90_abs": np.nan,
            "residual_p95_abs": np.nan,
            "residual_p99_abs": np.nan,
            "frac_abs_resid_gt_1c": np.nan,
            "frac_abs_resid_gt_2c": np.nan,
            "frac_abs_resid_gt_3c": np.nan,
        }
    abs_r = r.abs()
    return {
        "residual_n": int(len(r)),
        "residual_bias": float(r.mean()),
        "residual_mae": float(abs_r.mean()),
        "residual_rmse": float(np.sqrt(np.mean(np.square(r)))),
        "residual_median_abs": float(abs_r.median()),
        "residual_p90_abs": float(abs_r.quantile(0.90)),
        "residual_p95_abs": float(abs_r.quantile(0.95)),
        "residual_p99_abs": float(abs_r.quantile(0.99)),
        "frac_abs_resid_gt_1c": float((abs_r > 1).mean()),
        "frac_abs_resid_gt_2c": float((abs_r > 2).mean()),
        "frac_abs_resid_gt_3c": float((abs_r > 3).mean()),
    }


def binary_keep_metrics(df: pd.DataFrame, keep_mask, target_col: str | None = None, residual_col: str = "cws_ref_resid") -> dict:
    keep = pd.Series(keep_mask, index=df.index).fillna(False).astype(bool)
    out = {
        "n_total": int(len(df)),
        "n_kept": int(keep.sum()),
        "retention": float(keep.mean()) if len(keep) else np.nan,
        "n_stations_kept": int(df.loc[keep, "station_id"].nunique()) if "station_id" in df.columns else np.nan,
    }
    out.update(residual_summary(df.loc[keep], residual_col=residual_col))
    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        valid = y.isin([0, 1])
        pred_bad = (~keep).astype(int)
        if valid.any():
            yy = y[valid].astype(int)
            pp = pred_bad[valid].astype(int)
            tp = int(((yy == 1) & (pp == 1)).sum())
            fp = int(((yy == 0) & (pp == 1)).sum())
            tn = int(((yy == 0) & (pp == 0)).sum())
            fn = int(((yy == 1) & (pp == 0)).sum())
            out.update({"label_n": int(valid.sum()), "target_event_rate": float(yy.mean()), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
            out["proxy_accuracy"] = float((tn + tp) / max(tn + fp + fn + tp, 1))
            out["proxy_precision_bad"] = float(tp / max(tp + fp, 1))
            out["proxy_recall_bad"] = float(tp / max(tp + fn, 1))
            out["proxy_f1_bad"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def summarise_by_group(df: pd.DataFrame, group_col: str, target_col: str | None = None, residual_col: str = "cws_ref_resid") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        row = {
            group_col: str(key),
            "n": int(len(g)),
            "fraction": float(len(g) / max(len(df), 1)),
            "n_stations": int(g["station_id"].nunique()) if "station_id" in g.columns else np.nan,
        }
        row.update(residual_summary(g, residual_col=residual_col))
        if target_col and target_col in g.columns:
            y = pd.to_numeric(g[target_col], errors="coerce")
            y = y[y.isin([0, 1])]
            row["target_event_rate"] = float(y.mean()) if len(y) else np.nan
            row["target_label_n"] = int(len(y))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)


def add_station_bias_correction_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw and policy-corrected temperature/residual columns.

    The raw columns are preserved.  The policy correction is applied only to
    categories where the fusion layer recommends correcting a station/radiation
    bias, not to environmental-difference preservation candidates.
    """
    x = df.copy()
    if not {"temp_raw", "ref_mu"}.issubset(x.columns):
        return x

    temp = pd.to_numeric(x["temp_raw"], errors="coerce")
    ref_mu = pd.to_numeric(x["ref_mu"], errors="coerce")

    x["temp_no_bias_correction"] = temp.astype("float32")
    x["cws_ref_resid_no_bias_correction"] = (
        pd.to_numeric(x["cws_ref_resid"], errors="coerce") if "cws_ref_resid" in x.columns else temp - ref_mu
    ).astype("float32")

    if "expected_train_station_context_bias_c" not in x.columns:
        x["station_bias_correction_applied"] = False
        x["station_bias_correction_c"] = 0.0
        x["temp_corrected_station_bias_policy"] = temp.astype("float32")
        x["cws_ref_resid_corrected_station_bias_policy"] = (temp - ref_mu).astype("float32")
        x["temp_corrected_station_bias_all"] = temp.astype("float32")
        x["cws_ref_resid_corrected_station_bias_all"] = (temp - ref_mu).astype("float32")
        return x

    correction_candidate = x["fusion_category"].isin({
        "systemic_station_bias_correction_candidate",
        "radiation_or_siting_bias_candidate",
    })
    expected = pd.to_numeric(x["expected_train_station_context_bias_c"], errors="coerce")

    x["station_bias_correction_applied"] = correction_candidate & expected.notna()
    x["station_bias_correction_c"] = expected.where(x["station_bias_correction_applied"], 0.0).astype("float32")
    x["temp_corrected_station_bias_policy"] = (temp - x["station_bias_correction_c"]).astype("float32")
    x["cws_ref_resid_corrected_station_bias_policy"] = (
        x["temp_corrected_station_bias_policy"] - ref_mu
    ).astype("float32")



    x["temp_corrected_station_bias_all"] = (temp - expected.fillna(0)).astype("float32")
    x["cws_ref_resid_corrected_station_bias_all"] = (
        x["temp_corrected_station_bias_all"] - ref_mu
    ).astype("float32")
    return x


def build_policy_output_table(df: pd.DataFrame, corrected: bool) -> pd.DataFrame:
    """Build a compact product table for raw vs bias-corrected comparison."""
    x = df.copy()
    if corrected and "temp_corrected_station_bias_policy" in x.columns:
        temp_col = "temp_corrected_station_bias_policy"
        resid_col = "cws_ref_resid_corrected_station_bias_policy"
        product = "policy_bias_corrected"
    else:
        temp_col = "temp_raw"
        resid_col = "cws_ref_resid"
        product = "raw_no_bias_correction"

    base_cols = [
        "date", "station_id", "network", "split",
        "temp_raw", "ref_mu", "ref_sigma", "cws_ref_resid", "abs_cws_ref_z",
        "fusion_category", "recommended_action", "fusion_reason",
        "recommended_keep_conservative", "recommended_keep_microclimate",
        "recommended_bias_correctable", "recommended_reject_transient_error",
        "analysis_weight", "p_reference_model_issue_no_qc", "p_any_quality_issue",
        "p_reject_as_unexplained_transient_error", "p_qc_crowdqc",
        "p_residual_explained_by_persistent_station_context",
        "p_radiation_or_siting_bias_signal",
        "p_context_directional_microclimate_support",
        "p_environmental_difference_signal",
        "p_reference_uncertainty", "qc_flag_pattern",
        "expected_train_station_context_bias_c",
        "station_bias_correction_applied", "station_bias_correction_c",
    ]
    base_cols = [c for c in base_cols if c in x.columns]
    out = x[base_cols].copy()
    out["temperature_product"] = pd.to_numeric(x[temp_col], errors="coerce").astype("float32")
    if resid_col in x.columns:
        out["cws_ref_resid_product"] = pd.to_numeric(x[resid_col], errors="coerce").astype("float32")
    elif {"temperature_product", "ref_mu"}.issubset(out.columns):
        out["cws_ref_resid_product"] = (out["temperature_product"] - pd.to_numeric(out["ref_mu"], errors="coerce")).astype("float32")
    out["bias_correction_version"] = product
    return out


def method_comparison_table(eval_df: pd.DataFrame, cfg: FusionConfig, calibrated_prob_col: str) -> pd.DataFrame:
    target_col = None
    for c in ["target_ref_risk_hiconf", "target_ref_risk"]:
        if c in eval_df.columns:
            target_col = c
            break

    masks: dict[str, pd.Series] = {
        "raw_all_observed": pd.Series(True, index=eval_df.index),
    }

    qc_cols = [c for c in eval_df.columns if c.startswith("qc_") and c.endswith("_is_outlier")]
    for c in qc_cols:
        level = c.replace("qc_", "").replace("_is_outlier", "")
        available = eval_df[c].notna()
        masks[f"crowdqc_{level}_clean_available_only"] = available & eval_df[c].eq(0)
        masks[f"crowdqc_{level}_clean_or_unavailable"] = eval_df[c].fillna(0).eq(0)

    for cutoff in cfg.risk_keep_cutoffs:
        masks[f"risk_{cfg.risk_model_key}_p_le_{cutoff:g}"] = pd.to_numeric(eval_df[calibrated_prob_col], errors="coerce") <= cutoff

    for cutoff in cfg.final_keep_error_cutoffs:
        masks[f"fusion_reject_score_p_le_{cutoff:g}"] = pd.to_numeric(eval_df["p_reject_as_unexplained_transient_error"], errors="coerce") <= cutoff

    masks["fusion_keep_conservative"] = eval_df["recommended_keep_conservative"].fillna(False)
    masks["fusion_keep_microclimate"] = eval_df["recommended_keep_microclimate"].fillna(False)
    if "recommended_bias_correctable" in eval_df.columns:
        masks["fusion_bias_correctable_only"] = eval_df["recommended_bias_correctable"].fillna(False)
    masks["fusion_remove_reject_transient_only"] = ~eval_df["recommended_reject_transient_error"].fillna(False)
    masks["fusion_weight_positive"] = pd.to_numeric(eval_df["analysis_weight"], errors="coerce").fillna(0) > 0

    rows = []
    for method, mask in masks.items():
        row = {"method": method}
        row.update(binary_keep_metrics(eval_df, mask, target_col=target_col, residual_col="cws_ref_resid"))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["retention", "residual_mae"], ascending=[False, True]).reset_index(drop=True)


def correction_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keep_micro = eval_df.get("recommended_keep_microclimate", pd.Series(False, index=eval_df.index)).fillna(False)
    keep_cons = eval_df.get("recommended_keep_conservative", pd.Series(False, index=eval_df.index)).fillna(False)
    keep_bias = eval_df.get("recommended_bias_correctable", pd.Series(False, index=eval_df.index)).fillna(False)
    keep_non_reject = ~eval_df.get("recommended_reject_transient_error", pd.Series(False, index=eval_df.index)).fillna(False)

    specs = [
        ("raw_all_observed_no_bias_correction", pd.Series(True, index=eval_df.index), "cws_ref_resid"),
        ("fusion_keep_microclimate_no_bias_correction", keep_micro, "cws_ref_resid"),
        ("fusion_keep_conservative_no_bias_correction", keep_cons, "cws_ref_resid"),
        ("fusion_remove_reject_transient_only_no_bias_correction", keep_non_reject, "cws_ref_resid"),
        ("fusion_bias_correctable_subset_no_bias_correction", keep_bias, "cws_ref_resid"),
    ]
    if "cws_ref_resid_corrected_station_bias_policy" in eval_df.columns:
        specs.extend([
            ("fusion_keep_microclimate_policy_bias_corrected", keep_micro, "cws_ref_resid_corrected_station_bias_policy"),
            ("fusion_bias_correctable_subset_policy_bias_corrected", keep_bias, "cws_ref_resid_corrected_station_bias_policy"),
            ("all_rows_policy_bias_corrected", pd.Series(True, index=eval_df.index), "cws_ref_resid_corrected_station_bias_policy"),
            ("all_rows_station_bias_all_corrected_upper_bound", pd.Series(True, index=eval_df.index), "cws_ref_resid_corrected_station_bias_all"),
        ])
    for name, mask, resid_col in specs:
        kept = pd.Series(mask, index=eval_df.index).fillna(False).astype(bool)
        row = {
            "method": name,
            "residual_col": resid_col,
            "n_total": int(len(eval_df)),
            "n_used": int(kept.sum()),
            "coverage": float(kept.mean()) if len(eval_df) else np.nan,
        }
        row.update(residual_summary(eval_df.loc[kept], residual_col=resid_col))
        rows.append(row)
    return pd.DataFrame(rows)


def category_qc_agreement(eval_df: pd.DataFrame) -> pd.DataFrame:
    qc_cols = [c for c in eval_df.columns if c.startswith("qc_") and c.endswith("_is_outlier")]
    parts = []
    for c in qc_cols:
        available = eval_df[c].notna()
        if not available.any():
            continue
        tab = pd.crosstab(
            eval_df.loc[available, "fusion_category"].astype(str),
            eval_df.loc[available, c].astype(int),
            normalize="index",
        )
        tab.columns = [f"{c}={col}" for col in tab.columns]
        tab = tab.reset_index().rename(columns={"fusion_category": "fusion_category"})
        tab.insert(0, "qc_flag_col", c)
        parts.append(tab)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_candidate_samples(eval_df: pd.DataFrame, output_dir: Path, cfg: FusionConfig) -> dict[str, str]:
    sample_dir = output_dir / "candidate_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_cols = [
        "date", "network", "station_id", "temp_raw", "ref_mu", "ref_sigma",
        "cws_ref_resid", "abs_cws_ref_resid", "cws_ref_z", "abs_cws_ref_z",
        "fusion_category", "recommended_action", "fusion_reason",
        "p_reference_model_issue_no_qc", "p_any_quality_issue", "p_reject_as_unexplained_transient_error",
        "p_residual_explained_by_persistent_station_context",
        "p_radiation_or_siting_bias_signal", "radiation_current_context_score",
        "radiation_station_training_score", "station_solar_excess_bias_train_c",
        "p_reference_uncertainty", "p_context_mismatch_support",
        "p_context_directional_microclimate_support", "p_environmental_difference_signal",
        "p_context_microclimate_support", "p_qc_crowdqc", "qc_flag_pattern",
        "expected_train_station_context_bias_c", "station_bias_correction_c",
        "temp_corrected_station_bias_policy", "analysis_weight",
    ]
    sample_cols = [c for c in sample_cols if c in eval_df.columns]
    paths = {}
    rng = np.random.default_rng(cfg.random_state)
    for category, g in eval_df.groupby("fusion_category", dropna=False):
        if len(g) == 0:
            continue
        n = min(int(cfg.sample_rows_per_category), len(g))
        if n <= 0:
            continue
        sample = g.sample(n=n, random_state=cfg.random_state) if len(g) > n else g
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(category))
        path = sample_dir / f"sample__{safe}.csv"
        sample[sample_cols].to_csv(path, index=False)
        paths[str(category)] = str(path)
    return paths


def station_network_ambiguity_audit(scored: pd.DataFrame) -> pd.DataFrame:
    """Audit whether station_id alone is unique across networks.

    QC wide files usually carry station_id only. If a station_id appears in more
    than one CWS network, QC merging by date+station_id can duplicate/apply the
    same QC flag to multiple network rows. In that case, explicit network-aware
    QC files are preferable.
    """
    if not {"station_id", "network"}.issubset(scored.columns):
        return pd.DataFrame()
    audit = (
        scored[["station_id", "network"]]
        .drop_duplicates()
        .groupby("station_id", as_index=False)
        .agg(n_networks=("network", "nunique"), networks=("network", lambda z: "|".join(sorted(map(str, z.unique())))))
        .sort_values(["n_networks", "station_id"], ascending=[False, True])
    )
    audit["station_id_ambiguous_across_networks"] = audit["n_networks"] > 1
    return audit


def run_fusion(cfg: FusionConfig) -> dict:
    scored_path, reliability_path, upstream_manifest = resolve_inputs(cfg)

    output_dir = Path(cfg.output_dir)
    if cfg.run_label:
        output_dir = output_dir / cfg.run_label
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.overwrite:

        raise FileExistsError(f"Output directory exists and is non-empty: {output_dir}. Use overwrite=True or choose another run_label.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = load_dataframe_auto(scored_path)
    scored = ensure_key_types(scored)
    station_network_audit = station_network_ambiguity_audit(scored)

    if cfg.prob_col is None:
        cfg.prob_col = f"pred_ref_risk_prob_{cfg.risk_model_key}"
    if cfg.risk_model_name is None:
        cfg.risk_model_name = f"catboost_{cfg.risk_model_key}"
    if cfg.prob_col not in scored.columns:
        available = [c for c in scored.columns if str(c).startswith("pred_ref_risk_prob_")]
        raise KeyError(f"Probability column {cfg.prob_col!r} not found. Available: {available}")


    if "target_ref_risk_hiconf" not in scored.columns and "target_ref_risk_band_z" in scored.columns:
        band = scored["target_ref_risk_band_z"].astype("string")
        scored["target_ref_risk_hiconf"] = np.where(
            band.eq("high_confidence_high_risk"),
            1,
            np.where(band.eq("high_confidence_low_risk"), 0, np.nan),
        )

    fused, qc_files, qc_audit = merge_qc_flags(scored, cfg, output_dir)
    fused, qc_pattern_summary = add_qc_evidence(fused, cfg)


    feature_audit = feature_coverage_audit(fused)
    feature_group_audit = compact_feature_group_audit(feature_audit)

    fused, calibrated_prob_col, reliability_used = apply_reliability_bin_calibration(
        fused,
        prob_col=cfg.prob_col,
        reliability_path=reliability_path,
        model_name=cfg.risk_model_name or f"catboost_{cfg.risk_model_key}",
        calibration_split=cfg.reliability_calibration_split,
        enabled=cfg.use_reliability_calibration,
    )

    fused = add_fusion_scores(fused, cfg, calibrated_prob_col)

    _, micro_component_audit = add_microclimate_context_support_score(fused)
    fused = assign_categories(fused, cfg, calibrated_prob_col)

    if cfg.add_station_bias_correction_preview:
        fused = add_station_bias_correction_preview(fused)

    eval_df = fused[fused["split"].astype(str).eq(cfg.evaluation_split)].copy() if "split" in fused.columns else fused.copy()
    target_col = "target_ref_risk_hiconf" if "target_ref_risk_hiconf" in eval_df.columns else ("target_ref_risk" if "target_ref_risk" in eval_df.columns else None)

    category_summary = summarise_by_group(eval_df, "fusion_category", target_col=target_col)
    action_summary = summarise_by_group(eval_df, "recommended_action", target_col=target_col)
    method_comparison = method_comparison_table(eval_df, cfg, calibrated_prob_col)
    correction_summary = correction_metrics(eval_df)
    qc_agreement = category_qc_agreement(eval_df)

    sample_paths = save_candidate_samples(eval_df, output_dir, cfg)


    fused_path = save_dataframe_auto(fused, output_dir / f"{cfg.city}_cws_qc_risk_fused.parquet")

    raw_policy_path = None
    corrected_policy_path = None
    if cfg.save_policy_output_tables:
        raw_policy = build_policy_output_table(fused, corrected=False)
        corrected_policy = build_policy_output_table(fused, corrected=True)
        raw_policy_path = save_dataframe_auto(raw_policy, output_dir / f"{cfg.city}_cws_policy_output__raw_no_bias_correction.parquet")
        corrected_policy_path = save_dataframe_auto(corrected_policy, output_dir / f"{cfg.city}_cws_policy_output__policy_bias_corrected.parquet")

    feature_audit_path = output_dir / f"{cfg.city}_feature_coverage_audit.csv"
    feature_group_audit_path = output_dir / f"{cfg.city}_feature_group_coverage_audit.csv"
    station_network_audit_path = output_dir / f"{cfg.city}_station_network_ambiguity_audit.csv"
    qc_audit_path = output_dir / f"{cfg.city}_qc_flag_merge_audit.csv"
    qc_pattern_path = output_dir / f"{cfg.city}_qc_flag_pattern_summary.csv"
    micro_component_path = output_dir / f"{cfg.city}_microclimate_context_component_audit.csv"
    reliability_used_path = output_dir / f"{cfg.city}_reliability_calibration_used.csv"
    category_path = output_dir / f"{cfg.city}_fusion_category_summary.csv"
    action_path = output_dir / f"{cfg.city}_fusion_action_summary.csv"
    method_path = output_dir / f"{cfg.city}_fusion_method_comparison.csv"
    correction_path = output_dir / f"{cfg.city}_fusion_correction_metrics.csv"
    qc_agreement_path = output_dir / f"{cfg.city}_fusion_qc_category_agreement.csv"

    feature_audit.to_csv(feature_audit_path, index=False)
    feature_group_audit.to_csv(feature_group_audit_path, index=False)
    station_network_audit.to_csv(station_network_audit_path, index=False)
    qc_audit.to_csv(qc_audit_path, index=False)
    qc_pattern_summary.to_csv(qc_pattern_path, index=False)
    micro_component_audit.to_csv(micro_component_path, index=False)
    reliability_used.to_csv(reliability_used_path, index=False)
    category_summary.to_csv(category_path, index=False)
    action_summary.to_csv(action_path, index=False)
    method_comparison.to_csv(method_path, index=False)
    correction_summary.to_csv(correction_path, index=False)
    qc_agreement.to_csv(qc_agreement_path, index=False)

    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "scored_path": str(scored_path),
        "reliability_path": str(reliability_path) if reliability_path is not None else None,
        "upstream_manifest": upstream_manifest,
        "qc_files": {k: str(v) for k, v in qc_files.items()},
        "calibrated_prob_col": calibrated_prob_col,
        "evaluation_split": cfg.evaluation_split,
        "n_rows": int(len(fused)),
        "n_eval_rows": int(len(eval_df)),
        "outputs": {
            "fused_path": str(fused_path),
            "raw_policy_output_path": str(raw_policy_path) if raw_policy_path is not None else None,
            "bias_corrected_policy_output_path": str(corrected_policy_path) if corrected_policy_path is not None else None,
            "feature_audit_path": str(feature_audit_path),
            "feature_group_audit_path": str(feature_group_audit_path),
            "station_network_audit_path": str(station_network_audit_path),
            "qc_audit_path": str(qc_audit_path),
            "qc_pattern_path": str(qc_pattern_path),
            "micro_component_path": str(micro_component_path),
            "reliability_used_path": str(reliability_used_path),
            "category_summary_path": str(category_path),
            "action_summary_path": str(action_path),
            "method_comparison_path": str(method_path),
            "correction_summary_path": str(correction_path),
            "qc_agreement_path": str(qc_agreement_path),
            "candidate_sample_paths": sample_paths,
        },
    }
    manifest_path = output_dir / f"{cfg.city}_qc_risk_fusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audited CWS QC × residual-risk fusion and correction-preview layer.")
    p.add_argument("--city", default="project_id")
    p.add_argument("--scored-input", default=None)
    p.add_argument("--residual-risk-manifest", default=None)
    p.add_argument("--residual-risk-dir", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-label", default=None)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--risk-model-key", default="context_history")
    p.add_argument("--prob-col", default=None)
    p.add_argument("--risk-model-name", default=None)
    p.add_argument("--reliability-path", default=None)
    p.add_argument("--reliability-calibration-split", default="valid")
    p.add_argument("--no-reliability-calibration", action="store_true")

    p.add_argument("--qc-lenient", default=None)
    p.add_argument("--qc-strict", default=None)
    p.add_argument("--qc-ultra-strict", default=None)
    p.add_argument("--qc-dir", default=None)
    p.add_argument("--no-auto-discover-qc", action="store_true")
    p.add_argument("--force-rebuild-qc-cache", action="store_true")

    p.add_argument("--evaluation-split", default="test")
    p.add_argument("--disable-correction-preview", action="store_true")
    p.add_argument("--no-policy-output-tables", action="store_true")

    return p


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    cfg = FusionConfig(
        city=args.city,
        scored_input=args.scored_input,
        residual_risk_manifest=args.residual_risk_manifest,
        residual_risk_dir=args.residual_risk_dir,
        output_dir=args.output_dir,
        run_label=args.run_label,
        overwrite=args.overwrite,
        risk_model_key=args.risk_model_key,
        prob_col=args.prob_col,
        risk_model_name=args.risk_model_name,
        reliability_path=args.reliability_path,
        reliability_calibration_split=args.reliability_calibration_split,
        use_reliability_calibration=not args.no_reliability_calibration,
        qc_lenient=args.qc_lenient,
        qc_strict=args.qc_strict,
        qc_ultra_strict=args.qc_ultra_strict,
        qc_dir=args.qc_dir,
        auto_discover_qc=not args.no_auto_discover_qc,
        force_rebuild_qc_cache=args.force_rebuild_qc_cache,
        evaluation_split=args.evaluation_split,
        add_station_bias_correction_preview=not args.disable_correction_preview,
        save_policy_output_tables=not args.no_policy_output_tables,
    )
    manifest = run_fusion(cfg)
    print(json.dumps(manifest["outputs"], indent=2))
    return manifest


if __name__ == "__main__":
    main()
