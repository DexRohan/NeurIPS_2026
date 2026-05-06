"""Robustness checks for the CWS/OWS quality-control pipeline."""
from __future__ import annotations

import inspect
import json
import inspect
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class ReviewerRobustnessConfig:
    """Configuration for reviewer-facing CWS robustness checks."""

    city: str = "project_id"


    cws_reference_path: str | None = None
    source_reference_manifest_path: str | None = None
    source_reference_run_label: str | None = None
    reference_method: str = "catboost"
    calibration_mode: str = "time_train"



    time_residual_risk_manifest: str | None = None
    time_fusion_manifest: str | None = None
    time_fused_path: str | None = None



    output_dir: str = "outputs/reviewer_tests"
    residual_risk_output_dir: str = "outputs/residual_risk_reviewer_station_holdout"
    fusion_output_dir: str = "outputs/qc_risk_fusion_reviewer_station_holdout"


    run_station_holdout: bool = True
    run_fusion_on_station_holdout: bool = False
    run_bootstrap_existing_fusion: bool = True
    run_station_diagnostics_existing_fusion: bool = True
    run_shift_diagnostics_existing_fusion: bool = True



    station_holdout_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    valid_frac: float = 0.15
    test_frac: float = 0.15
    target_mode: str = "auto"
    feature_modes: str = "context_history,context_static_met"



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



    fusion_risk_model_keys: tuple[str, ...] = ("context_history",)
    qc_dir: str | None = None
    qc_lenient: str | None = None
    qc_strict: str | None = None
    qc_ultra_strict: str | None = None
    fusion_overwrite: bool = True
    fusion_use_reliability_calibration: bool = True



    bootstrap_unit: str = "station"
    n_bootstrap: int = 500
    bootstrap_random_state: int = 2026
    bootstrap_methods_regex: str | None = None


    overwrite_summary_outputs: bool = True


def _now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _as_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(obj: Mapping, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(p)
    elif p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
        except Exception as e:
            warnings.warn(f"Could not save parquet ({e}); saving pickle instead.")
            p = p.with_suffix(".pkl")
            df.to_pickle(p)
    else:
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


def load_dataframe_auto(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    if suf == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported dataframe extension: {p.suffix}")


def import_pipeline_modules(script_dir: str | Path | None = None):
    """Import the existing audited residual-risk and fusion modules."""
    if script_dir is not None:
        script_dir = Path(script_dir)
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
    import importlib
    import qc_cws_residual_risk_model_audited as rrm
    import qc_risk_fusion_audited as qrf

    rrm = importlib.reload(rrm)
    qrf = importlib.reload(qrf)
    return rrm, qrf


def _fusionconfig_supported_fields(qrf) -> tuple[set[str], bool]:
    """Return supported FusionConfig kwargs for older/newer fusion-module versions."""
    fusion_cls = getattr(qrf, "FusionConfig", None)
    if fusion_cls is None:
        raise AttributeError("qc_risk_fusion_audited has no FusionConfig class")

    supported: set[str] = set()
    has_var_kwargs = False
    try:
        sig = inspect.signature(fusion_cls)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                supported.add(name)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_kwargs = True
    except Exception:
        pass

    if not supported and hasattr(fusion_cls, "__dataclass_fields__"):
        supported = set(getattr(fusion_cls, "__dataclass_fields__", {}).keys())

    return supported, has_var_kwargs


def _build_fusion_config_compatible(qrf, **kwargs):
    """Instantiate FusionConfig while gracefully dropping unsupported kwargs."""
    supported, has_var_kwargs = _fusionconfig_supported_fields(qrf)
    if has_var_kwargs:
        return qrf.FusionConfig(**kwargs), []

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(k for k in kwargs if k not in filtered)
    return qrf.FusionConfig(**filtered), dropped


def _fusion_config_field_names(qrf) -> set[str]:
    """Backward-compatible helper kept for notebook diagnostics."""
    supported, _ = _fusionconfig_supported_fields(qrf)
    return supported


def fusion_compatibility_report(script_dir: str | Path | None = None, qrf=None) -> dict:
    """Small notebook-facing report for the installed fusion-module version."""
    if qrf is None:
        _, qrf = import_pipeline_modules(script_dir=script_dir)
    supported, has_var_kwargs = _fusionconfig_supported_fields(qrf)
    return {
        "fusion_module_path": str(getattr(qrf, "__file__", "")),
        "supports_var_kwargs": bool(has_var_kwargs),
        "supported_config_fields": sorted(supported),
        "supports_save_policy_output_tables": bool(has_var_kwargs or "save_policy_output_tables" in supported),
        "supports_add_station_bias_correction_preview": bool(has_var_kwargs or "add_station_bias_correction_preview" in supported),
        "supports_sample_rows_per_category": bool(has_var_kwargs or "sample_rows_per_category" in supported),
        "has_run_fusion": bool(hasattr(qrf, "run_fusion")),
    }


def resolve_fused_path_from_manifest(fusion_manifest_path: str | Path | None, direct_path: str | Path | None = None) -> tuple[Path | None, dict | None]:
    """Resolve the fused parquet path and return the manifest dict if available."""
    if direct_path is not None and str(direct_path).strip():
        manifest = load_json(fusion_manifest_path) if fusion_manifest_path else None
        return Path(direct_path), manifest
    if fusion_manifest_path is None or not str(fusion_manifest_path).strip():
        return None, None
    manifest = load_json(fusion_manifest_path)
    outputs = manifest.get("outputs", {})
    fused = outputs.get("fused_path") or manifest.get("fused_path")
    return (Path(fused) if fused else None), manifest


def _supported_fusion_config_fields(qrf) -> set[str]:
    """Best-effort field discovery for FusionConfig across script versions."""
    fusion_cls = getattr(qrf, "FusionConfig", None)
    if fusion_cls is None:
        return set()


    dataclass_fields = getattr(fusion_cls, "__dataclass_fields__", None)
    if dataclass_fields:
        return set(dataclass_fields.keys())


    try:
        sig = inspect.signature(fusion_cls)
        return {
            name
            for name, param in sig.parameters.items()
            if name != "self" and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
    except Exception:
        return set()


def _instantiate_fusion_config_compat(qrf, **kwargs):
    """Instantiate FusionConfig while dropping unsupported kwargs safely."""
    fusion_cls = getattr(qrf, "FusionConfig")
    supported = _supported_fusion_config_fields(qrf)
    use_kwargs = dict(kwargs)
    dropped: list[str] = []

    if supported:
        dropped = sorted([k for k in use_kwargs if k not in supported])
        use_kwargs = {k: v for k, v in use_kwargs.items() if k in supported}

    while True:
        try:
            if dropped:
                warnings.warn(
                    "FusionConfig compatibility mode: dropping unsupported kwargs "
                    + ", ".join(sorted(set(dropped)))
                )
            return fusion_cls(**use_kwargs), sorted(set(dropped))
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad = m.group(1)
                if bad in use_kwargs:
                    dropped.append(bad)
                    use_kwargs.pop(bad, None)
                    continue
            raise


FUSION_KEEP_CONSERVATIVE_CATEGORIES = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
}
FUSION_KEEP_MICROCLIMATE_CATEGORIES = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
    "qc_flagged_low_reference_risk_rescue_candidate",
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
    "possible_bias_or_microclimate_preserve_with_caution",
    "environmental_difference_candidate_preserve",
}
FUSION_BIAS_CORRECTABLE_CATEGORIES = {
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
}
FUSION_REJECT_CATEGORIES = {
    "missing_or_unscored",
    "qc_confirmed_probable_transient_or_sensor_error",
    "qc_missed_probable_high_risk_observation",
}


def _string_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].astype("string").fillna("")


def _infer_keep_conservative(eval_df: pd.DataFrame) -> pd.Series:
    category = _string_series(eval_df, "fusion_category")
    action = _string_series(eval_df, "recommended_action")
    mask = category.isin(FUSION_KEEP_CONSERVATIVE_CATEGORIES)
    mask = mask | action.isin(["use_as_is", "use_as_is_or_light_downweight"])
    return mask.fillna(False)


def _infer_keep_microclimate(eval_df: pd.DataFrame) -> pd.Series:
    category = _string_series(eval_df, "fusion_category")
    action = _string_series(eval_df, "recommended_action")
    mask = category.isin(FUSION_KEEP_MICROCLIMATE_CATEGORIES)
    mask = mask | action.isin([
        "use_as_is",
        "use_as_is_or_light_downweight",
        "rescue_use_with_caution",
        "bias_correct_then_use",
        "context_bias_correct_or_downweight",
        "preserve_or_downweight",
        "preserve_for_microclimate_analysis",
    ])
    return mask.fillna(False)


def _infer_bias_correctable(eval_df: pd.DataFrame) -> pd.Series:
    category = _string_series(eval_df, "fusion_category")
    action = _string_series(eval_df, "recommended_action")
    mask = category.isin(FUSION_BIAS_CORRECTABLE_CATEGORIES)
    mask = mask | category.str.contains("correction_candidate", case=False, na=False)
    mask = mask | action.str.contains("bias_correct", case=False, na=False)
    if "station_bias_correction_applied" in eval_df.columns:
        mask = mask | _series_bool(eval_df, "station_bias_correction_applied")
    return mask.fillna(False)


def _infer_reject_transient(eval_df: pd.DataFrame) -> pd.Series:
    category = _string_series(eval_df, "fusion_category")
    action = _string_series(eval_df, "recommended_action")
    mask = category.isin(FUSION_REJECT_CATEGORIES)
    mask = mask | action.str.startswith("exclude")
    return mask.fillna(False)


def _ensure_fusion_correction_columns(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = eval_df.copy()
    notes: list[str] = []

    if "recommended_keep_conservative" not in x.columns:
        x["recommended_keep_conservative"] = _infer_keep_conservative(x).astype(bool)
        notes.append("added recommended_keep_conservative via category/action inference")

    if "recommended_keep_microclimate" not in x.columns:
        x["recommended_keep_microclimate"] = _infer_keep_microclimate(x).astype(bool)
        notes.append("added recommended_keep_microclimate via category/action inference")

    if "recommended_bias_correctable" not in x.columns:
        x["recommended_bias_correctable"] = _infer_bias_correctable(x).astype(bool)
        notes.append("added recommended_bias_correctable via category/action inference")

    if "recommended_reject_transient_error" not in x.columns:
        x["recommended_reject_transient_error"] = _infer_reject_transient(x).astype(bool)
        notes.append("added recommended_reject_transient_error via category/action inference")

    temp = pd.to_numeric(x["temp_raw"], errors="coerce") if "temp_raw" in x.columns else None
    ref_mu = pd.to_numeric(x["ref_mu"], errors="coerce") if "ref_mu" in x.columns else None
    expected = pd.to_numeric(x["expected_train_station_context_bias_c"], errors="coerce") if "expected_train_station_context_bias_c" in x.columns else None

    if expected is not None and "station_bias_correction_applied" not in x.columns:
        applied = _series_bool(x, "recommended_bias_correctable") & expected.notna()
        x["station_bias_correction_applied"] = applied.astype(bool)
        notes.append("added station_bias_correction_applied from recommended_bias_correctable and expected_train_station_context_bias_c")

    if expected is not None and "station_bias_correction_c" not in x.columns:
        applied = _series_bool(x, "station_bias_correction_applied")
        x["station_bias_correction_c"] = expected.where(applied, 0.0).astype("float32")
        notes.append("added station_bias_correction_c from expected_train_station_context_bias_c")

    if temp is not None and ref_mu is not None and "cws_ref_resid" not in x.columns:
        x["cws_ref_resid"] = (temp - ref_mu).astype("float32")
        notes.append("reconstructed cws_ref_resid from temp_raw and ref_mu")

    if temp is not None and "temp_no_bias_correction" not in x.columns:
        x["temp_no_bias_correction"] = temp.astype("float32")
    if temp is not None and ref_mu is not None and "cws_ref_resid_no_bias_correction" not in x.columns:
        x["cws_ref_resid_no_bias_correction"] = (temp - ref_mu).astype("float32")

    if temp is not None and ref_mu is not None and "station_bias_correction_c" in x.columns and "temp_corrected_station_bias_policy" not in x.columns:
        corr = pd.to_numeric(x["station_bias_correction_c"], errors="coerce").fillna(0.0)
        x["temp_corrected_station_bias_policy"] = (temp - corr).astype("float32")
        notes.append("added temp_corrected_station_bias_policy from station_bias_correction_c")

    if ref_mu is not None and "temp_corrected_station_bias_policy" in x.columns and "cws_ref_resid_corrected_station_bias_policy" not in x.columns:
        x["cws_ref_resid_corrected_station_bias_policy"] = (
            pd.to_numeric(x["temp_corrected_station_bias_policy"], errors="coerce") - ref_mu
        ).astype("float32")
        notes.append("added cws_ref_resid_corrected_station_bias_policy")

    if expected is not None and temp is not None and "temp_corrected_station_bias_all" not in x.columns:
        x["temp_corrected_station_bias_all"] = (temp - expected.fillna(0.0)).astype("float32")
        notes.append("added temp_corrected_station_bias_all upper-bound correction")

    if ref_mu is not None and "temp_corrected_station_bias_all" in x.columns and "cws_ref_resid_corrected_station_bias_all" not in x.columns:
        x["cws_ref_resid_corrected_station_bias_all"] = (
            pd.to_numeric(x["temp_corrected_station_bias_all"], errors="coerce") - ref_mu
        ).astype("float32")
        notes.append("added cws_ref_resid_corrected_station_bias_all upper-bound correction")

    if "analysis_weight" not in x.columns:
        if "p_reject_as_unexplained_transient_error" in x.columns:
            w = 1.0 - pd.to_numeric(x["p_reject_as_unexplained_transient_error"], errors="coerce").fillna(1.0)
        else:
            w = _series_bool(x, "recommended_keep_microclimate").astype(float)
        w = w.clip(lower=0.0, upper=1.0)
        w.loc[_series_bool(x, "recommended_keep_conservative")] = 1.0
        w.loc[_series_bool(x, "recommended_reject_transient_error")] = 0.0
        x["analysis_weight"] = w.astype("float32")
        notes.append("added analysis_weight from reject probability / inferred recommendation flags")

    return x, notes


def harmonize_fusion_output_schema(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add compatibility columns expected by reviewer summaries."""
    x = eval_df.copy()
    notes: list[str] = []

    x, corr_notes = _ensure_fusion_correction_columns(x)
    notes.extend(corr_notes)

    if "abs_cws_ref_resid" not in x.columns and "cws_ref_resid" in x.columns:
        x["abs_cws_ref_resid"] = pd.to_numeric(x["cws_ref_resid"], errors="coerce").abs().astype("float32")
        notes.append("added abs_cws_ref_resid from cws_ref_resid")

    if "qc_flag_pattern" not in x.columns:
        qc_cols = [c for c in x.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
        if qc_cols:
            pattern_parts = []
            for c in qc_cols:
                level = str(c).replace("qc_", "").replace("_is_outlier", "")
                val = pd.to_numeric(x[c], errors="coerce")
                pattern_parts.append(level + "=" + val.map(lambda z: "NA" if pd.isna(z) else str(int(z))).astype(str))
            x["qc_flag_pattern"] = pd.DataFrame(pattern_parts).T.apply(lambda row: "|".join(row.values.astype(str)), axis=1).astype("string")
            notes.append("added qc_flag_pattern from available qc_*_is_outlier columns")

    return x, notes


def infer_target_col_for_fusion_eval(eval_df: pd.DataFrame) -> str | None:
    for c in ["target_ref_risk_hiconf", "target_ref_risk"]:
        if c in eval_df.columns:
            return c
    return None


def summarize_fusion_group_table(eval_df: pd.DataFrame, group_col: str, target_col: str | None = None, residual_col: str = "cws_ref_resid") -> pd.DataFrame:
    if group_col not in eval_df.columns:
        return pd.DataFrame()
    rows = []
    for key, g in eval_df.groupby(group_col, dropna=False):
        row = {
            group_col: str(key),
            "n": int(len(g)),
            "fraction": float(len(g) / max(len(eval_df), 1)),
            "n_stations": int(g[["network", "station_id"]].drop_duplicates().shape[0]) if {"network", "station_id"}.issubset(g.columns) else np.nan,
        }
        if residual_col in g.columns:
            row.update(residual_metrics_from_series(g[residual_col]))
        if target_col and target_col in g.columns:
            y = pd.to_numeric(g[target_col], errors="coerce")
            y = y[y.isin([0, 1])]
            row["target_event_rate"] = float(y.mean()) if len(y) else np.nan
            row["target_label_n"] = int(len(y))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fraction", "n"], ascending=[False, False]).reset_index(drop=True)


def point_metrics_for_spec(eval_df: pd.DataFrame, spec: PolicySpec, target_col: str | None = None) -> dict:
    resid = pd.to_numeric(eval_df[spec.residual_col], errors="coerce") if spec.residual_col in eval_df.columns else pd.Series(np.nan, index=eval_df.index)
    mask = pd.Series(spec.mask, index=eval_df.index).fillna(False).astype(bool) & resid.notna()
    row = {"method": spec.method}
    if spec.table_family == "correction":
        row["residual_col"] = spec.residual_col
        row["n_total"] = int(len(eval_df))
        row["n_used"] = int(mask.sum())
        row["coverage"] = float(mask.mean()) if len(eval_df) else np.nan
    else:
        row["n_total"] = int(len(eval_df))
        row["n_kept"] = int(mask.sum())
        row["retention"] = float(mask.mean()) if len(eval_df) else np.nan
        row["n_stations_kept"] = int(eval_df.loc[mask, ["network", "station_id"]].drop_duplicates().shape[0]) if {"network", "station_id"}.issubset(eval_df.columns) else np.nan

    row.update(residual_metrics_from_series(resid[mask]))

    if target_col and target_col in eval_df.columns and spec.table_family != "correction":
        y = pd.to_numeric(eval_df[target_col], errors="coerce")
        valid = y.isin([0, 1])
        if valid.any():
            yy = y[valid].astype(int)
            pred_bad = (~mask[valid]).astype(int)
            tp = int(((yy == 1) & (pred_bad == 1)).sum())
            fp = int(((yy == 0) & (pred_bad == 1)).sum())
            tn = int(((yy == 0) & (pred_bad == 0)).sum())
            fn = int(((yy == 1) & (pred_bad == 0)).sum())
            row.update({
                "label_n": int(valid.sum()),
                "target_event_rate": float(yy.mean()),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "proxy_accuracy": float((tn + tp) / max(tn + fp + fn + tp, 1)),
                "proxy_precision_bad": float(tp / max(tp + fp, 1)),
                "proxy_recall_bad": float(tp / max(tp + fn, 1)),
                "proxy_f1_bad": float(2 * tp / max(2 * tp + fp + fn, 1)),
            })
    return row


def make_policy_metrics_table(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> pd.DataFrame:
    target_col = infer_target_col_for_fusion_eval(eval_df)
    rows = [point_metrics_for_spec(eval_df, spec, target_col=target_col) for spec in build_policy_specs(eval_df, manifest=manifest)]
    out = pd.DataFrame(rows)
    if len(out):
        sort_cols = [c for c in ["retention", "residual_mae"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=[False, True][:len(sort_cols)]).reset_index(drop=True)
    return out


def make_correction_metrics_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = [point_metrics_for_spec(eval_df, spec, target_col=None) for spec in build_correction_specs(eval_df)]
    return pd.DataFrame(rows)


def make_category_qc_agreement_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    qc_cols = [c for c in eval_df.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
    parts = []
    if "fusion_category" not in eval_df.columns:
        return pd.DataFrame()
    for c in qc_cols:
        available = eval_df[c].notna()
        if not available.any():
            continue
        tab = pd.crosstab(
            eval_df.loc[available, "fusion_category"].astype(str),
            pd.to_numeric(eval_df.loc[available, c], errors="coerce").astype("Int64"),
            normalize="index",
        )
        tab.columns = [f"{c}={col}" for col in tab.columns]
        tab = tab.reset_index().rename(columns={"fusion_category": "fusion_category"})
        tab.insert(0, "qc_flag_col", c)
        parts.append(tab)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_fusion_compat_tables(eval_df: pd.DataFrame, manifest: Mapping | None, output_dir: str | Path, city: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_col = infer_target_col_for_fusion_eval(eval_df)

    category_summary = summarize_fusion_group_table(eval_df, "fusion_category", target_col=target_col, residual_col="cws_ref_resid")
    action_summary = summarize_fusion_group_table(eval_df, "recommended_action", target_col=target_col, residual_col="cws_ref_resid")
    method_comparison = make_policy_metrics_table(eval_df, manifest=manifest)
    correction_summary = make_correction_metrics_table(eval_df)
    qc_agreement = make_category_qc_agreement_table(eval_df)

    paths = {
        "category_summary_path": str(save_dataframe(category_summary, output_dir / f"{city}_fusion_category_summary.csv")),
        "action_summary_path": str(save_dataframe(action_summary, output_dir / f"{city}_fusion_action_summary.csv")),
        "method_comparison_path": str(save_dataframe(method_comparison, output_dir / f"{city}_fusion_method_comparison.csv")),
        "correction_summary_path": str(save_dataframe(correction_summary, output_dir / f"{city}_fusion_correction_metrics.csv")),
        "qc_agreement_path": str(save_dataframe(qc_agreement, output_dir / f"{city}_fusion_qc_category_agreement.csv")),
    }
    return paths


def _series_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(int(default)).astype(int).astype(bool)
    return s.astype("string").str.lower().isin(["true", "1", "yes", "y"])


FUSION_KEEP_CONSERVATIVE_CATEGORIES = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
}
FUSION_KEEP_MICROCLIMATE_CATEGORIES = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
    "qc_flagged_low_reference_risk_rescue_candidate",
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
    "possible_bias_or_microclimate_preserve_with_caution",
    "environmental_difference_candidate_preserve",
}
FUSION_BIAS_CORRECTABLE_CATEGORIES = {
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
}
FUSION_REJECT_CATEGORIES = {
    "qc_confirmed_probable_transient_or_sensor_error",
    "qc_missed_probable_high_risk_observation",
    "missing_or_unscored",
}
FUSION_KEEP_CONSERVATIVE_ACTIONS = {"use_as_is", "use_as_is_or_light_downweight"}
FUSION_KEEP_MICROCLIMATE_ACTIONS = {
    "use_as_is",
    "use_as_is_or_light_downweight",
    "rescue_use_with_caution",
    "bias_correct_then_use",
    "context_bias_correct_or_downweight",
    "preserve_or_downweight",
    "preserve_for_microclimate_analysis",
}
FUSION_BIAS_CORRECTABLE_ACTIONS = {"bias_correct_then_use", "context_bias_correct_or_downweight"}
FUSION_REJECT_ACTIONS = {"exclude_missing", "exclude_transient_error", "exclude_or_manual_review"}


def _string_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[col].astype("string")


def _legacy_recommended_cols(df: pd.DataFrame, include_tokens: Sequence[str]) -> list[str]:
    cols = []
    include_tokens = [str(tok).lower() for tok in include_tokens]
    for c in df.columns:
        low = str(c).lower()
        if not low.startswith("recommended_"):
            continue
        if all(tok in low for tok in include_tokens):
            cols.append(str(c))
    return cols


def _derive_recommended_mask(df: pd.DataFrame, logical_name: str) -> pd.Series:
    cat = _string_series(df, "fusion_category").fillna("")
    action = _string_series(df, "recommended_action").fillna("")
    cat_low = cat.str.lower()
    action_low = action.str.lower()
    mask = pd.Series(False, index=df.index)

    if logical_name == "recommended_keep_conservative":
        mask |= cat.isin(FUSION_KEEP_CONSERVATIVE_CATEGORIES)
        mask |= action.isin(FUSION_KEEP_CONSERVATIVE_ACTIONS)
        mask |= cat_low.str.contains("high_confidence_reference_consistent", na=False)
        mask |= cat_low.str.contains("likely_valid", na=False)

    elif logical_name == "recommended_keep_microclimate":
        mask |= cat.isin(FUSION_KEEP_MICROCLIMATE_CATEGORIES)
        mask |= action.isin(FUSION_KEEP_MICROCLIMATE_ACTIONS)
        mask |= cat_low.str.contains("microclimate", na=False)
        mask |= cat_low.str.contains("environmental_difference", na=False)
        mask |= cat_low.str.contains("rescue_candidate", na=False)
        mask |= cat_low.str.contains("bias_correction_candidate", na=False)
        mask |= cat_low.str.contains("radiation", na=False)

    elif logical_name == "recommended_bias_correctable":
        mask |= cat.isin(FUSION_BIAS_CORRECTABLE_CATEGORIES)
        mask |= action.isin(FUSION_BIAS_CORRECTABLE_ACTIONS)
        mask |= action_low.str.contains("bias_correct", na=False)
        mask |= cat_low.str.contains("bias_correction_candidate", na=False)
        mask |= cat_low.str.contains("radiation", na=False)
        for c in _legacy_recommended_cols(df, include_tokens=("bias", "correct")):
            mask |= _series_bool(df, c)
        if "station_bias_correction_applied" in df.columns:
            mask |= _series_bool(df, "station_bias_correction_applied")
        if {"cws_ref_resid", "cws_ref_resid_corrected_station_bias_policy"}.issubset(df.columns):
            raw = pd.to_numeric(df["cws_ref_resid"], errors="coerce")
            corr = pd.to_numeric(df["cws_ref_resid_corrected_station_bias_policy"], errors="coerce")
            mask |= raw.notna() & corr.notna() & (raw - corr).abs().gt(1e-9)

    elif logical_name == "recommended_reject_transient_error":
        mask |= cat.isin(FUSION_REJECT_CATEGORIES)
        mask |= action.isin(FUSION_REJECT_ACTIONS)
        mask |= action_low.str.startswith("exclude", na=False)
        mask |= cat_low.str.contains("transient", na=False)
        mask |= cat_low.str.contains("sensor_error", na=False)

    return mask.fillna(False).astype(bool)


def make_eval_df_reviewer_compatible(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Back-fill reviewer-facing fusion columns for older fusion-module outputs."""
    x = eval_df.copy()
    for logical_name in [
        "recommended_keep_conservative",
        "recommended_keep_microclimate",
        "recommended_bias_correctable",
        "recommended_reject_transient_error",
    ]:
        if logical_name in x.columns:
            x[logical_name] = _series_bool(x, logical_name)
        else:
            x[logical_name] = _derive_recommended_mask(x, logical_name)

    if "analysis_weight" not in x.columns and "p_reject_as_unexplained_transient_error" in x.columns:
        p_reject = pd.to_numeric(x["p_reject_as_unexplained_transient_error"], errors="coerce").clip(0, 1)
        x["analysis_weight"] = (1 - p_reject).clip(0, 1)

    return x


def residual_metrics_from_series(resid: pd.Series | np.ndarray) -> dict:
    r = pd.to_numeric(pd.Series(resid), errors="coerce").dropna()
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


_FUSION_CATEGORY_TO_ACTION = {
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

_FUSION_KEEP_CONSERVATIVE_CATS = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
}
_FUSION_KEEP_MICROCLIMATE_CATS = {
    "high_confidence_reference_consistent",
    "likely_valid_low_unexplained_error",
    "qc_flagged_low_reference_risk_rescue_candidate",
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
    "possible_bias_or_microclimate_preserve_with_caution",
    "environmental_difference_candidate_preserve",
}
_FUSION_BIAS_CORRECTABLE_CATS = {
    "systemic_station_bias_correction_candidate",
    "radiation_or_siting_bias_candidate",
}
_FUSION_REJECT_CATS = {
    "qc_confirmed_probable_transient_or_sensor_error",
    "qc_missed_probable_high_risk_observation",
    "missing_or_unscored",
}

_FUSION_KEEP_CONSERVATIVE_ACTIONS = {"use_as_is", "use_as_is_or_light_downweight"}
_FUSION_KEEP_MICROCLIMATE_ACTIONS = _FUSION_KEEP_CONSERVATIVE_ACTIONS | {
    "rescue_use_with_caution",
    "bias_correct_then_use",
    "context_bias_correct_or_downweight",
    "preserve_or_downweight",
    "preserve_for_microclimate_analysis",
}
_FUSION_BIAS_CORRECTABLE_ACTIONS = {"bias_correct_then_use", "context_bias_correct_or_downweight"}
_FUSION_REJECT_ACTIONS = {"exclude_transient_error", "exclude_or_manual_review", "exclude_missing"}


def _string_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[col].astype("string")


def _infer_action_from_category(cat: pd.Series) -> pd.Series:
    out = cat.map(_FUSION_CATEGORY_TO_ACTION)
    return out.astype("string")


def _infer_bool_from_category_or_action(df: pd.DataFrame, categories: set[str], actions: set[str]) -> pd.Series:
    cat = _string_series(df, "fusion_category")
    act = _string_series(df, "recommended_action")
    mask = cat.isin(categories) | act.isin(actions)
    return mask.fillna(False).astype(bool)


def ensure_fusion_compat_columns(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> pd.DataFrame:
    """Backfill columns that older fusion scripts may not emit.

    This keeps the reviewer diagnostics usable even when the local fusion script
    predates fields such as ``save_policy_output_tables`` or
    ``recommended_bias_correctable``. The fallbacks are derived from
    ``fusion_category``/``recommended_action`` and correction-preview columns.
    """
    x = eval_df.copy()

    if "recommended_action" not in x.columns and "fusion_category" in x.columns:
        x["recommended_action"] = _infer_action_from_category(_string_series(x, "fusion_category"))

    if "recommended_keep_conservative" not in x.columns:
        x["recommended_keep_conservative"] = _infer_bool_from_category_or_action(
            x, _FUSION_KEEP_CONSERVATIVE_CATS, _FUSION_KEEP_CONSERVATIVE_ACTIONS
        )
    if "recommended_keep_microclimate" not in x.columns:
        x["recommended_keep_microclimate"] = _infer_bool_from_category_or_action(
            x, _FUSION_KEEP_MICROCLIMATE_CATS, _FUSION_KEEP_MICROCLIMATE_ACTIONS
        )
    if "recommended_bias_correctable" not in x.columns:
        keep_bias = _infer_bool_from_category_or_action(
            x, _FUSION_BIAS_CORRECTABLE_CATS, _FUSION_BIAS_CORRECTABLE_ACTIONS
        )
        if "station_bias_correction_applied" in x.columns:
            keep_bias = keep_bias | _series_bool(x, "station_bias_correction_applied")
        if "station_bias_correction_c" in x.columns:
            keep_bias = keep_bias | (
                pd.to_numeric(x["station_bias_correction_c"], errors="coerce").fillna(0).abs() > 0
            )
        x["recommended_bias_correctable"] = keep_bias.astype(bool)
    if "recommended_reject_transient_error" not in x.columns:
        x["recommended_reject_transient_error"] = _infer_bool_from_category_or_action(
            x, _FUSION_REJECT_CATS, _FUSION_REJECT_ACTIONS
        )

    if "station_bias_correction_applied" not in x.columns and "station_bias_correction_c" in x.columns:
        x["station_bias_correction_applied"] = (
            pd.to_numeric(x["station_bias_correction_c"], errors="coerce").fillna(0).abs() > 0
        )

    if "cws_ref_resid_corrected_station_bias_policy" not in x.columns:
        if {"temp_corrected_station_bias_policy", "ref_mu"}.issubset(x.columns):
            x["cws_ref_resid_corrected_station_bias_policy"] = (
                pd.to_numeric(x["temp_corrected_station_bias_policy"], errors="coerce")
                - pd.to_numeric(x["ref_mu"], errors="coerce")
            )
    if "cws_ref_resid_corrected_station_bias_all" not in x.columns:
        if {"temp_corrected_station_bias_all", "ref_mu"}.issubset(x.columns):
            x["cws_ref_resid_corrected_station_bias_all"] = (
                pd.to_numeric(x["temp_corrected_station_bias_all"], errors="coerce")
                - pd.to_numeric(x["ref_mu"], errors="coerce")
            )

    if "abs_cws_ref_resid" not in x.columns and "cws_ref_resid" in x.columns:
        x["abs_cws_ref_resid"] = pd.to_numeric(x["cws_ref_resid"], errors="coerce").abs()
    if "abs_cws_ref_z" not in x.columns and "cws_ref_z" in x.columns:
        x["abs_cws_ref_z"] = pd.to_numeric(x["cws_ref_z"], errors="coerce").abs()

    if "analysis_weight" not in x.columns and "p_reject_as_unexplained_transient_error" in x.columns:
        weight = 1 - pd.to_numeric(x["p_reject_as_unexplained_transient_error"], errors="coerce").fillna(1.0)
        weight = weight.clip(0, 1)
        weight.loc[_series_bool(x, "recommended_reject_transient_error")] = 0.0
        x["analysis_weight"] = weight.astype("float32")

    return x


def _fusion_config_field_names(qrf) -> set[str]:
    """Best-effort field discovery for legacy/new FusionConfig dataclasses."""
    names: set[str] = set()
    fusion_config_cls = getattr(qrf, "FusionConfig", None)
    if fusion_config_cls is None:
        return names
    dataclass_fields = getattr(fusion_config_cls, "__dataclass_fields__", None)
    if dataclass_fields:
        names.update(map(str, dataclass_fields.keys()))
    annotations = getattr(fusion_config_cls, "__annotations__", None) or {}
    names.update(map(str, annotations.keys()))
    try:
        sig = inspect.signature(fusion_config_cls)
        for name, param in sig.parameters.items():
            if name != "self" and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                names.add(str(name))
    except Exception:
        pass
    return names


def build_compatible_fusion_config(qrf, **kwargs):
    """Construct FusionConfig while dropping unsupported kwargs on older scripts."""
    accepted = _fusion_config_field_names(qrf)
    if not accepted:
        return qrf.FusionConfig(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        warnings.warn(
            "FusionConfig compatibility shim ignored unsupported fields: " + ", ".join(dropped)
        )
    return qrf.FusionConfig(**filtered)


def _infer_target_col(eval_df: pd.DataFrame) -> str | None:
    if "target_ref_risk_hiconf" in eval_df.columns:
        return "target_ref_risk_hiconf"
    if "target_ref_risk" in eval_df.columns:
        return "target_ref_risk"
    return None


def summarize_fusion_group(eval_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    x = ensure_fusion_compat_columns(eval_df)
    if group_col not in x.columns:
        return pd.DataFrame()
    target_col = _infer_target_col(x)
    rows = []
    for key, g in x.groupby(group_col, dropna=False):
        row = {
            group_col: str(key),
            "n": int(len(g)),
            "fraction": float(len(g) / len(x)) if len(x) else np.nan,
            "n_stations": int(g["station_id"].nunique()) if "station_id" in g.columns else np.nan,
        }
        if "cws_ref_resid" in g.columns:
            row.update(residual_metrics_from_series(g["cws_ref_resid"]))
        if target_col is not None:
            y = pd.to_numeric(g[target_col], errors="coerce")
            y = y[y.isin([0, 1])]
            row["target_event_rate"] = float(y.mean()) if len(y) else np.nan
            row["target_label_n"] = int(len(y))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)


def summarize_policy_specs(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> pd.DataFrame:
    x = ensure_fusion_compat_columns(eval_df, manifest=manifest)
    target_col = _infer_target_col(x)
    rows = []
    for spec in build_policy_specs(x, manifest=manifest):
        keep = pd.Series(spec.mask, index=x.index).fillna(False).astype(bool)
        kept = x.loc[keep].copy()
        row = {
            "method": spec.method,
            "n_total": int(len(x)),
            "n_kept": int(keep.sum()),
            "retention": float(keep.mean()) if len(keep) else np.nan,
            "n_stations_kept": int(kept["station_id"].nunique()) if "station_id" in kept.columns else np.nan,
        }
        if spec.residual_col in kept.columns:
            row.update(residual_metrics_from_series(kept[spec.residual_col]))
        else:
            row.update(residual_metrics_from_series(pd.Series(dtype=float)))
        if target_col is not None:
            y = pd.to_numeric(kept[target_col], errors="coerce")
            y = y[y.isin([0, 1])]
            row["target_event_rate"] = float(y.mean()) if len(y) else np.nan
            row["target_label_n"] = int(len(y))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["retention", "residual_mae"], ascending=[False, True]).reset_index(drop=True)


def summarize_correction_specs(eval_df: pd.DataFrame) -> pd.DataFrame:
    x = ensure_fusion_compat_columns(eval_df)
    rows = []
    for spec in build_correction_specs(x):
        keep = pd.Series(spec.mask, index=x.index).fillna(False).astype(bool)
        kept = x.loc[keep].copy()
        row = {
            "method": spec.method,
            "residual_col": spec.residual_col,
            "n_total": int(len(x)),
            "n_used": int(keep.sum()),
            "coverage": float(keep.mean()) if len(keep) else np.nan,
        }
        if spec.residual_col in kept.columns:
            row.update(residual_metrics_from_series(kept[spec.residual_col]))
        else:
            row.update(residual_metrics_from_series(pd.Series(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_qc_category_agreement(eval_df: pd.DataFrame) -> pd.DataFrame:
    x = ensure_fusion_compat_columns(eval_df)
    if "fusion_category" not in x.columns:
        return pd.DataFrame()
    qc_cols = [c for c in x.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
    parts = []
    for c in qc_cols:
        available = x[c].notna()
        if not available.any():
            continue
        tab = pd.crosstab(
            x.loc[available, "fusion_category"].astype(str),
            pd.to_numeric(x.loc[available, c], errors="coerce").fillna(0).astype(int),
            normalize="index",
        )
        tab.columns = [f"{c}={col}" for col in tab.columns]
        tab = tab.reset_index().rename(columns={"fusion_category": "fusion_category"})
        tab.insert(0, "qc_flag_col", c)
        parts.append(tab)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_one_station_holdout_residual_risk(cfg: ReviewerRobustnessConfig, seed: int, rrm=None) -> dict:
    """Run one CWS station-held-out residual-risk experiment."""
    if rrm is None:
        rrm, _ = import_pipeline_modules()
    if cfg.cws_reference_path is None:
        raise ValueError("cws_reference_path is required for station-held-out runs.")

    run_label = (
        f"{cfg.source_reference_run_label or cfg.city}__residual_risk_REVIEWER_CWS_station_holdout__"
        f"{cfg.reference_method}__{cfg.calibration_mode}__seed{seed}"
    )
    risk_cfg = rrm.ResidualRiskConfig(
        city=cfg.city,
        input_path=str(cfg.cws_reference_path),
        output_dir=str(cfg.residual_risk_output_dir),
        run_label=run_label,
        overwrite_existing_run=False,
        reference_method=cfg.reference_method,
        calibration_mode=cfg.calibration_mode,
        source_reference_run_label=cfg.source_reference_run_label,
        source_reference_manifest_path=cfg.source_reference_manifest_path,
        split_strategy="station_holdout",
        valid_frac=float(cfg.valid_frac),
        test_frac=float(cfg.test_frac),
        random_state=int(seed),
        target_mode=cfg.target_mode,
        feature_modes=cfg.feature_modes,
        include_coordinates=False,
        include_qc_flags=False,
        drop_numeric_landcover_codes=True,
        run_catboost=True,
        iterations=int(cfg.iterations),
        learning_rate=float(cfg.learning_rate),
        depth=int(cfg.depth),
        l2_leaf_reg=float(cfg.l2_leaf_reg),
        auto_class_weights=cfg.auto_class_weights,
        early_stopping_rounds=int(cfg.early_stopping_rounds),
        thread_count=cfg.thread_count,
        used_ram_limit=cfg.used_ram_limit,
        use_gpu=bool(cfg.use_gpu),
        verbose=int(cfg.verbose),
        max_train_rows=cfg.max_train_rows,
        max_valid_rows=cfg.max_valid_rows,
    )
    result = rrm.run_residual_risk_pipeline(cfg=risk_cfg)
    result["reviewer_test"] = "cws_station_holdout_residual_risk"
    result["station_holdout_seed"] = int(seed)
    return result


def collect_residual_risk_run_tables(manifests: Sequence[Mapping | str | Path], cfg: ReviewerRobustnessConfig) -> dict[str, pd.DataFrame]:
    """Collect metrics, split counts, retention curves, bands, and feature importance."""
    metrics_parts = []
    counts_parts = []
    retention_parts = []
    band_parts = []
    fi_parts = []
    manifest_rows = []

    for item in manifests:
        m = load_json(item) if isinstance(item, (str, Path)) else dict(item)
        seed = m.get("station_holdout_seed")
        run_label = m.get("run_label")
        outdir = m.get("output_dir")
        row = {
            "run_label": run_label,
            "output_dir": outdir,
            "station_holdout_seed": seed,
            "split_strategy": m.get("config", {}).get("split_strategy"),
            "target_col": m.get("target_col"),
            "manifest_path": m.get("manifest_path"),
        }
        manifest_rows.append(row)

        def _try_read(path_key: str) -> pd.DataFrame | None:
            p = m.get(path_key)
            if not p:
                return None
            p = Path(p)
            if not p.exists():
                warnings.warn(f"Manifest path does not exist: {p}")
                return None
            return pd.read_csv(p)

        metrics = _try_read("metrics_path")
        if metrics is not None:
            metrics.insert(0, "station_holdout_seed", seed)
            metrics.insert(0, "run_label", run_label)
            metrics.insert(0, "robustness_test", "cws_station_holdout")
            metrics_parts.append(metrics)

        counts = _try_read("split_counts_path")
        if counts is not None:
            counts.insert(0, "station_holdout_seed", seed)
            counts.insert(0, "run_label", run_label)
            counts_parts.append(counts)

        retention = _try_read("retention_path")
        if retention is not None:
            retention.insert(0, "station_holdout_seed", seed)
            retention.insert(0, "run_label", run_label)
            retention_parts.append(retention)

        bands = _try_read("band_summary_path")
        if bands is not None:
            bands.insert(0, "station_holdout_seed", seed)
            bands.insert(0, "run_label", run_label)
            band_parts.append(bands)

        models = m.get("models", {})
        for mode, meta in models.items():
            p = meta.get("feature_importance_path")
            if p and Path(p).exists():
                fi = pd.read_csv(p)
                fi.insert(0, "feature_mode", mode)
                fi.insert(0, "station_holdout_seed", seed)
                fi.insert(0, "run_label", run_label)
                fi_parts.append(fi)

    out = {
        "manifest_index": pd.DataFrame(manifest_rows),
        "metrics": pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame(),
        "split_counts": pd.concat(counts_parts, ignore_index=True) if counts_parts else pd.DataFrame(),
        "retention": pd.concat(retention_parts, ignore_index=True) if retention_parts else pd.DataFrame(),
        "band_summary": pd.concat(band_parts, ignore_index=True) if band_parts else pd.DataFrame(),
        "feature_importance": pd.concat(fi_parts, ignore_index=True) if fi_parts else pd.DataFrame(),
    }
    return out


def summarize_station_holdout_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max summary over station-held-out seeds."""
    if metrics is None or len(metrics) == 0:
        return pd.DataFrame()
    test_metrics = metrics[metrics["split"].astype(str).eq("test")].copy()
    numeric = [
        c for c in [
            "n", "event_rate", "auroc", "auprc", "brier", "ece_15", "f1",
            "log_loss", "threshold", "tn", "fp", "fn", "tp",
        ]
        if c in test_metrics.columns
    ]
    rows = []
    for model, g in test_metrics.groupby("model", dropna=False):
        row = {"model": model, "split": "test", "n_runs": int(g["station_holdout_seed"].nunique())}
        for c in numeric:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_feature_importance_stability(fi: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Aggregate feature importance across station-held-out seeds."""
    if fi is None or len(fi) == 0:
        return pd.DataFrame()
    x = fi.copy()
    x["importance"] = pd.to_numeric(x["importance"], errors="coerce")
    x = x.dropna(subset=["importance"])
    rows = []
    for (feature_mode, model, feature), g in x.groupby(["feature_mode", "model", "feature"], dropna=False):
        vals = g["importance"]
        rows.append({
            "feature_mode": feature_mode,
            "model": model,
            "feature": feature,
            "n_runs_present": int(g["station_holdout_seed"].nunique()),
            "importance_mean": float(vals.mean()),
            "importance_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "importance_min": float(vals.min()),
            "importance_max": float(vals.max()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["rank_by_mean_importance"] = out.groupby(["feature_mode", "model"])["importance_mean"].rank(ascending=False, method="dense")
        out = out.sort_values(["feature_mode", "model", "rank_by_mean_importance", "feature"])
        out = out[out["rank_by_mean_importance"] <= int(top_n)].reset_index(drop=True)
    return out


def _target_label_col(eval_df: pd.DataFrame) -> str | None:
    for c in ["target_ref_risk_hiconf", "target_ref_risk"]:
        if c in eval_df.columns:
            return c
    return None


def point_method_comparison_table(eval_df: pd.DataFrame, specs: Sequence[PolicySpec]) -> pd.DataFrame:
    """Point-estimate method table used when older fusion runs lack compatible summaries."""
    x = make_eval_df_reviewer_compatible(eval_df)
    target_col = _target_label_col(x)
    rows = []
    for spec in specs:
        mask = pd.Series(spec.mask, index=x.index).fillna(False).astype(bool)
        row = {
            "method": spec.method,
            "n_total": int(len(x)),
            "n_kept": int(mask.sum()),
            "retention": float(mask.mean()) if len(x) else np.nan,
            "n_stations_kept": int(x.loc[mask, ["network", "station_id"]].drop_duplicates().shape[0]) if {"network", "station_id"}.issubset(x.columns) else np.nan,
        }
        if spec.residual_col in x.columns:
            row.update(residual_metrics_from_series(x.loc[mask, spec.residual_col]))
        if target_col is not None:
            y = pd.to_numeric(x[target_col], errors="coerce")
            valid = y.isin([0, 1])
            if valid.any():
                yy = y[valid].astype(int)
                pp = (~mask[valid]).astype(int)
                tp = int(((yy == 1) & (pp == 1)).sum())
                fp = int(((yy == 0) & (pp == 1)).sum())
                tn = int(((yy == 0) & (pp == 0)).sum())
                fn = int(((yy == 1) & (pp == 0)).sum())
                row.update({
                    "label_n": int(valid.sum()),
                    "target_event_rate": float(yy.mean()),
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "proxy_accuracy": float((tn + tp) / max(tn + fp + fn + tp, 1)),
                    "proxy_precision_bad": float(tp / max(tp + fp, 1)),
                    "proxy_recall_bad": float(tp / max(tp + fn, 1)),
                    "proxy_f1_bad": float(2 * tp / max(2 * tp + fp + fn, 1)),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def point_correction_summary_table(eval_df: pd.DataFrame, specs: Sequence[PolicySpec]) -> pd.DataFrame:
    x = make_eval_df_reviewer_compatible(eval_df)
    rows = []
    for spec in specs:
        mask = pd.Series(spec.mask, index=x.index).fillna(False).astype(bool)
        row = {
            "method": spec.method,
            "residual_col": spec.residual_col,
            "n_total": int(len(x)),
            "n_used": int(mask.sum()),
            "coverage": float(mask.mean()) if len(x) else np.nan,
        }
        if spec.residual_col in x.columns:
            row.update(residual_metrics_from_series(x.loc[mask, spec.residual_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_fusion_manifest_path(cfg: ReviewerRobustnessConfig, run_label: str) -> Path:
    """Best-effort path for the downstream fusion manifest.

    Older fusion-module versions may ignore run_label; newer ones write into a
    run-labeled subdirectory. This helper prefers the run-labeled path but falls
    back to the base output directory when needed.
    """
    base_dir = Path(cfg.fusion_output_dir)
    candidates = [
        base_dir / str(run_label) / f"{cfg.city}_qc_risk_fusion_manifest.json",
        base_dir / f"{cfg.city}_qc_risk_fusion_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _postprocess_fusion_outputs_for_reviewer(
    cfg: ReviewerRobustnessConfig,
    fusion_manifest: Mapping,
    manifest_path: str | Path | None = None,
) -> dict:
    """Normalize fused outputs and back-fill reviewer-compatible summary tables."""
    out = json.loads(json.dumps(fusion_manifest, default=str))
    outputs = dict(out.get("outputs", {}))
    fused_path = outputs.get("fused_path") or out.get("fused_path")
    if not fused_path:
        out["outputs"] = outputs
        return out

    fused_path = Path(fused_path)
    if not fused_path.exists():
        out["outputs"] = outputs
        return out

    fused = load_dataframe_auto(fused_path)
    fused = make_eval_df_reviewer_compatible(fused)
    actual_fused_path = save_dataframe(fused, fused_path)
    outputs["fused_path"] = str(actual_fused_path)

    evaluation_split = out.get("evaluation_split") or out.get("config", {}).get("evaluation_split", "test")
    if "split" in fused.columns and evaluation_split:
        eval_df = fused[fused["split"].astype(str).eq(str(evaluation_split))].copy()
    else:
        eval_df = fused.copy()

    compat_dir = actual_fused_path.parent
    compat_paths = write_fusion_compat_tables(eval_df, manifest=out, output_dir=compat_dir, city=cfg.city)
    outputs.update(compat_paths)
    out["outputs"] = outputs

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            _write_json(out, manifest_path)
            out["manifest_path"] = str(manifest_path)

    return out


def run_fusion_for_residual_manifest(
    cfg: ReviewerRobustnessConfig,
    residual_manifest: Mapping,
    risk_model_key: str,
    qrf=None,
) -> dict:
    """Run the fusion layer for a residual-risk manifest."""
    if qrf is None:
        _, qrf = import_pipeline_modules()
    manifest_path = residual_manifest.get("manifest_path") or residual_manifest.get("residual_risk_manifest")
    residual_dir = residual_manifest.get("output_dir")
    seed = residual_manifest.get("station_holdout_seed")
    run_label = (
        f"{residual_manifest.get('run_label', cfg.city)}__qc_risk_fusion_REVIEWER__"
        f"{risk_model_key}__seed{seed}"
    )

    fusion_kwargs = dict(
        city=cfg.city,
        scored_input=None,
        residual_risk_manifest=str(manifest_path),
        residual_risk_dir=str(residual_dir),
        output_dir=str(cfg.fusion_output_dir),
        run_label=run_label,
        overwrite=bool(cfg.fusion_overwrite),
        risk_model_key=risk_model_key,
        risk_model_name=f"catboost_{risk_model_key}",
        prob_col=f"pred_ref_risk_prob_{risk_model_key}",
        reliability_path=None,
        reliability_calibration_split="valid",
        use_reliability_calibration=bool(cfg.fusion_use_reliability_calibration),
        qc_lenient=cfg.qc_lenient,
        qc_strict=cfg.qc_strict,
        qc_ultra_strict=cfg.qc_ultra_strict,
        qc_dir=cfg.qc_dir,
        auto_discover_qc=True,
        evaluation_split="test",
        add_station_bias_correction_preview=True,
        save_policy_output_tables=True,
        sample_rows_per_category=250,
        random_state=int(seed) if seed is not None else 42,
    )
    fusion_cfg, dropped_kwargs = _build_fusion_config_compatible(qrf, **fusion_kwargs)
    if dropped_kwargs:
        warnings.warn(
            "FusionConfig does not support reviewer kwargs; dropping: " + ", ".join(dropped_kwargs)
        )

    out = qrf.run_fusion(fusion_cfg)
    fusion_manifest_path = _infer_fusion_manifest_path(cfg, run_label)
    out = _postprocess_fusion_outputs_for_reviewer(cfg, out, manifest_path=fusion_manifest_path)
    out["reviewer_test"] = "fusion_on_cws_station_holdout"
    out["station_holdout_seed"] = seed
    out["risk_model_key"] = risk_model_key
    out["fusionconfig_dropped_kwargs"] = dropped_kwargs
    if fusion_manifest_path.exists():
        out["manifest_path"] = str(fusion_manifest_path)
    return out


def collect_fusion_run_tables(fusion_manifests: Sequence[Mapping | str | Path]) -> dict[str, pd.DataFrame]:
    """Collect station-held-out fusion tables, recomputing summaries if needed.

    The recomputation path makes reviewer outputs robust to older local fusion
    scripts that may omit some saved tables or compatibility columns.
    """
    parts = {k: [] for k in ["category", "action", "method", "correction", "qc_agreement"]}
    manifest_rows = []

    for item in fusion_manifests:
        m = load_json(item) if isinstance(item, (str, Path)) else dict(item)
        outputs = m.get("outputs", {})
        seed = m.get("station_holdout_seed")
        risk_key = m.get("risk_model_key") or m.get("config", {}).get("risk_model_key")
        run_label = m.get("config", {}).get("run_label") or m.get("run_label")
        fused_path = outputs.get("fused_path") or m.get("fused_path")
        manifest_rows.append({
            "run_label": run_label,
            "station_holdout_seed": seed,
            "risk_model_key": risk_key,
            "n_rows": m.get("n_rows"),
            "n_eval_rows": m.get("n_eval_rows"),
            "calibrated_prob_col": m.get("calibrated_prob_col"),
            "fused_path": fused_path,
            "manifest_path": m.get("manifest_path"),
        })

        eval_df = None
        if fused_path and Path(fused_path).exists():
            fused = load_dataframe_auto(fused_path)
            evaluation_split = m.get("evaluation_split") or m.get("config", {}).get("evaluation_split", "test")
            if "split" in fused.columns and evaluation_split:
                eval_df = fused[fused["split"].astype(str).eq(str(evaluation_split))].copy()
            else:
                eval_df = fused.copy()
            eval_df = ensure_fusion_compat_columns(eval_df, manifest=m)

        summaries = {}
        if eval_df is not None:
            summaries = {
                "category": summarize_fusion_group(eval_df, "fusion_category"),
                "action": summarize_fusion_group(eval_df, "recommended_action"),
                "method": summarize_policy_specs(eval_df, manifest=m),
                "correction": summarize_correction_specs(eval_df),
                "qc_agreement": summarize_qc_category_agreement(eval_df),
            }

        for out_name in ["category", "action", "method", "correction", "qc_agreement"]:
            df = summaries.get(out_name)
            if df is None or len(df) == 0:

                path_key = {
                    "category": "category_summary_path",
                    "action": "action_summary_path",
                    "method": "method_comparison_path",
                    "correction": "correction_summary_path",
                    "qc_agreement": "qc_agreement_path",
                }[out_name]
                p = outputs.get(path_key)
                if p and Path(p).exists():
                    df = pd.read_csv(p)
            if df is not None and len(df):
                df = df.copy()
                df.insert(0, "risk_model_key", risk_key)
                df.insert(0, "station_holdout_seed", seed)
                df.insert(0, "run_label", run_label)
                parts[out_name].append(df)

    out = {"fusion_manifest_index": pd.DataFrame(manifest_rows)}
    out.update({k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()})
    return out


@dataclass(frozen=True)
class PolicySpec:
    method: str
    mask: pd.Series
    residual_col: str = "cws_ref_resid"
    table_family: str = "policy"


def _config_value(manifest: Mapping | None, key: str, default):
    if manifest is None:
        return default
    return manifest.get("config", {}).get(key, default)


def infer_calibrated_prob_col(eval_df: pd.DataFrame, manifest: Mapping | None = None, risk_model_key: str | None = None) -> str | None:
    if manifest is not None:
        c = manifest.get("calibrated_prob_col")
        if c in eval_df.columns:
            return c
        c = manifest.get("config", {}).get("prob_col")
        if c in eval_df.columns:
            return c
    if risk_model_key:
        candidates = [
            f"pred_ref_risk_prob_{risk_model_key}_reliability_calibrated",
            f"pred_ref_risk_prob_{risk_model_key}",
        ]
        for c in candidates:
            if c in eval_df.columns:
                return c
    candidates = [c for c in eval_df.columns if str(c).startswith("pred_ref_risk_prob_") and "calibrated" in str(c)]
    if candidates:
        return candidates[0]
    candidates = [c for c in eval_df.columns if str(c).startswith("pred_ref_risk_prob_")]
    return candidates[0] if candidates else None


def build_policy_specs(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> list[PolicySpec]:
    """Build method masks similar to qc_risk_fusion_audited.method_comparison_table."""
    eval_df = ensure_fusion_compat_columns(eval_df, manifest=manifest)
    specs: list[PolicySpec] = [PolicySpec("raw_all_observed", pd.Series(True, index=eval_df.index), "cws_ref_resid", "policy")]

    qc_cols = [c for c in eval_df.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
    for c in qc_cols:
        level = str(c).replace("qc_", "").replace("_is_outlier", "")
        available = eval_df[c].notna()
        flag = pd.to_numeric(eval_df[c], errors="coerce")
        specs.append(PolicySpec(f"crowdqc_{level}_clean_available_only", available & flag.eq(0), "cws_ref_resid", "policy"))
        specs.append(PolicySpec(f"crowdqc_{level}_clean_or_unavailable", flag.fillna(0).eq(0), "cws_ref_resid", "policy"))

    risk_model_key = _config_value(manifest, "risk_model_key", "context_history")
    prob_col = infer_calibrated_prob_col(eval_df, manifest=manifest, risk_model_key=risk_model_key)
    risk_keep_cutoffs = tuple(_config_value(manifest, "risk_keep_cutoffs", (0.05, 0.15, 0.30, 0.50, 0.80)))
    if prob_col is not None:
        p = pd.to_numeric(eval_df[prob_col], errors="coerce")
        for cutoff in risk_keep_cutoffs:
            specs.append(PolicySpec(f"risk_{risk_model_key}_p_le_{cutoff:g}", p <= float(cutoff), "cws_ref_resid", "policy"))

    final_keep_error_cutoffs = tuple(_config_value(manifest, "final_keep_error_cutoffs", (0.15, 0.30, 0.50, 0.70)))
    if "p_reject_as_unexplained_transient_error" in eval_df.columns:
        p_reject = pd.to_numeric(eval_df["p_reject_as_unexplained_transient_error"], errors="coerce")
        for cutoff in final_keep_error_cutoffs:
            specs.append(PolicySpec(f"fusion_reject_score_p_le_{cutoff:g}", p_reject <= float(cutoff), "cws_ref_resid", "policy"))

    for name, col in [
        ("fusion_keep_conservative", "recommended_keep_conservative"),
        ("fusion_keep_microclimate", "recommended_keep_microclimate"),
        ("fusion_bias_correctable_only", "recommended_bias_correctable"),
    ]:
        if col in eval_df.columns:
            specs.append(PolicySpec(name, _series_bool(eval_df, col), "cws_ref_resid", "policy"))
    if "recommended_reject_transient_error" in eval_df.columns:
        specs.append(PolicySpec("fusion_remove_reject_transient_only", ~_series_bool(eval_df, "recommended_reject_transient_error"), "cws_ref_resid", "policy"))
    if "analysis_weight" in eval_df.columns:
        specs.append(PolicySpec("fusion_weight_positive", pd.to_numeric(eval_df["analysis_weight"], errors="coerce").fillna(0) > 0, "cws_ref_resid", "policy"))

    return specs


def build_correction_specs(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> list[PolicySpec]:
    """Build correction policy masks for paper-facing fusion comparisons.

    This includes:
      - raw and core fusion correction policies,
      - CrowdQC-only baselines evaluated with the same station-bias correction
        columns used by the fusion policies, and
      - fusion reject-score operating points with and without policy correction.

    The corrected CrowdQC rows are important for a fair reviewer comparison:
    otherwise fusion is credited with correction while CrowdQC is not.
    """
    eval_df = ensure_fusion_compat_columns(eval_df, manifest=manifest)
    all_rows = pd.Series(True, index=eval_df.index)
    keep_micro = _series_bool(eval_df, "recommended_keep_microclimate")
    keep_cons = _series_bool(eval_df, "recommended_keep_conservative")
    keep_bias = _series_bool(eval_df, "recommended_bias_correctable")
    non_reject = ~_series_bool(eval_df, "recommended_reject_transient_error")

    specs: list[PolicySpec] = [
        PolicySpec("raw_all_observed_no_bias_correction", all_rows, "cws_ref_resid", "correction"),
        PolicySpec("fusion_keep_microclimate_no_bias_correction", keep_micro, "cws_ref_resid", "correction"),
        PolicySpec("fusion_keep_conservative_no_bias_correction", keep_cons, "cws_ref_resid", "correction"),
        PolicySpec("fusion_remove_reject_transient_only_no_bias_correction", non_reject, "cws_ref_resid", "correction"),
        PolicySpec("fusion_bias_correctable_subset_no_bias_correction", keep_bias, "cws_ref_resid", "correction"),
    ]

    corrected_policy_col = "cws_ref_resid_corrected_station_bias_policy"
    corrected_all_col = "cws_ref_resid_corrected_station_bias_all"

    if corrected_policy_col in eval_df.columns:
        specs.extend([
            PolicySpec("fusion_keep_microclimate_policy_bias_corrected", keep_micro, corrected_policy_col, "correction"),
            PolicySpec("fusion_keep_conservative_policy_bias_corrected", keep_cons, corrected_policy_col, "correction"),
            PolicySpec("fusion_remove_reject_transient_only_policy_bias_corrected", non_reject, corrected_policy_col, "correction"),
            PolicySpec("fusion_bias_correctable_subset_policy_bias_corrected", keep_bias, corrected_policy_col, "correction"),
            PolicySpec("all_rows_policy_bias_corrected", all_rows, corrected_policy_col, "correction"),
        ])

    if corrected_all_col in eval_df.columns:
        specs.append(PolicySpec("all_rows_station_bias_all_corrected_upper_bound", all_rows, corrected_all_col, "correction"))



    qc_cols = [c for c in eval_df.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
    for c in qc_cols:
        level = str(c).replace("qc_", "").replace("_is_outlier", "")
        flag = pd.to_numeric(eval_df[c], errors="coerce")
        available = flag.notna()
        masks = {
            f"crowdqc_{level}_clean_available_only": available & flag.eq(0),
            f"crowdqc_{level}_clean_or_unavailable": flag.fillna(0).eq(0),
        }
        for base_name, mask in masks.items():
            specs.append(PolicySpec(f"{base_name}_no_bias_correction", mask, "cws_ref_resid", "correction"))
            if corrected_policy_col in eval_df.columns:
                specs.append(PolicySpec(f"{base_name}_policy_bias_corrected", mask, corrected_policy_col, "correction"))
            if corrected_all_col in eval_df.columns:
                specs.append(PolicySpec(f"{base_name}_station_bias_all_corrected_upper_bound", mask, corrected_all_col, "correction"))


    if "p_reject_as_unexplained_transient_error" in eval_df.columns:
        final_keep_error_cutoffs = tuple(
            _config_value(manifest, "final_keep_error_cutoffs", (0.15, 0.30, 0.50, 0.70))
        )
        p_reject = pd.to_numeric(eval_df["p_reject_as_unexplained_transient_error"], errors="coerce")
        for cutoff in final_keep_error_cutoffs:
            mask = p_reject <= float(cutoff)
            specs.append(PolicySpec(f"fusion_reject_score_p_le_{cutoff:g}_no_bias_correction", mask, "cws_ref_resid", "correction"))
            if corrected_policy_col in eval_df.columns:
                specs.append(PolicySpec(f"fusion_reject_score_p_le_{cutoff:g}_policy_bias_corrected", mask, corrected_policy_col, "correction"))
            if corrected_all_col in eval_df.columns:
                specs.append(PolicySpec(f"fusion_reject_score_p_le_{cutoff:g}_station_bias_all_corrected_upper_bound", mask, corrected_all_col, "correction"))


    seen: set[tuple[str, str]] = set()
    unique_specs: list[PolicySpec] = []
    for spec in specs:
        key = (spec.method, spec.residual_col)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs

def make_block_key(df: pd.DataFrame, unit: str = "station") -> pd.Series:
    if not {"network", "station_id"}.issubset(df.columns):
        raise KeyError("Bootstrap requires network and station_id columns.")
    station_key = df["network"].astype(str) + "::" + df["station_id"].astype(str)
    if unit == "station":
        return station_key
    if unit == "station_day":
        if "date" not in df.columns:
            raise KeyError("station_day bootstrap requires a date column.")
        day = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        return station_key + "::" + day.astype(str)
    raise ValueError("bootstrap_unit must be 'station' or 'station_day'.")


def _block_summary_for_spec(eval_df: pd.DataFrame, block_key: pd.Series, spec: PolicySpec) -> pd.DataFrame:
    if spec.residual_col not in eval_df.columns:
        return pd.DataFrame()
    resid = pd.to_numeric(eval_df[spec.residual_col], errors="coerce")
    mask = pd.Series(spec.mask, index=eval_df.index).fillna(False).astype(bool) & resid.notna()
    abs_r = resid.abs()
    tmp = pd.DataFrame({
        "block_key": block_key.astype(str).values,
        "n_total": 1,
        "n_used": mask.astype("int64").values,
        "sum_resid": resid.where(mask, 0).fillna(0).astype("float64").values,
        "sum_abs_resid": abs_r.where(mask, 0).fillna(0).astype("float64").values,
        "sum_sq_resid": (resid ** 2).where(mask, 0).fillna(0).astype("float64").values,
        "count_abs_gt_1c": ((abs_r > 1) & mask).astype("int64").values,
        "count_abs_gt_2c": ((abs_r > 2) & mask).astype("int64").values,
        "count_abs_gt_3c": ((abs_r > 3) & mask).astype("int64").values,
    })
    g = tmp.groupby("block_key", sort=False).sum(numeric_only=True)
    return g


def _metrics_from_sums(arr: np.ndarray, columns: Sequence[str]) -> dict:
    s = dict(zip(columns, arr))
    n_total = float(s.get("n_total", 0))
    n_used = float(s.get("n_used", 0))
    if n_total <= 0 or n_used <= 0:
        return {
            "n_total": n_total,
            "n_used": n_used,
            "retention": np.nan,
            "residual_bias": np.nan,
            "residual_mae": np.nan,
            "residual_rmse": np.nan,
            "frac_abs_resid_gt_1c": np.nan,
            "frac_abs_resid_gt_2c": np.nan,
            "frac_abs_resid_gt_3c": np.nan,
        }
    return {
        "n_total": n_total,
        "n_used": n_used,
        "retention": n_used / n_total,
        "residual_bias": s.get("sum_resid", 0.0) / n_used,
        "residual_mae": s.get("sum_abs_resid", 0.0) / n_used,
        "residual_rmse": math.sqrt(max(s.get("sum_sq_resid", 0.0) / n_used, 0.0)),
        "frac_abs_resid_gt_1c": s.get("count_abs_gt_1c", 0.0) / n_used,
        "frac_abs_resid_gt_2c": s.get("count_abs_gt_2c", 0.0) / n_used,
        "frac_abs_resid_gt_3c": s.get("count_abs_gt_3c", 0.0) / n_used,
    }


def bootstrap_policy_ci(
    eval_df: pd.DataFrame,
    specs: Sequence[PolicySpec],
    unit: str = "station",
    n_bootstrap: int = 500,
    random_state: int = 2026,
    methods_regex: str | None = None,
) -> pd.DataFrame:
    """Station/block bootstrap CIs for retention and residual metrics.

    The bootstrap is exact for sums/means/RMSE/fractions and intentionally does
    not bootstrap p95/p99 quantiles, which would require row-level resampling.
    Use the original method-comparison/correction tables for point p95/p99.
    """
    block_key = make_block_key(eval_df, unit=unit)
    all_blocks = pd.Index(block_key.astype(str).unique(), name="block_key")
    n_blocks = len(all_blocks)
    if n_blocks == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(int(random_state))
    boot_indices = rng.integers(0, n_blocks, size=(int(n_bootstrap), n_blocks))
    metrics_cols = [
        "n_total", "n_used", "sum_resid", "sum_abs_resid", "sum_sq_resid",
        "count_abs_gt_1c", "count_abs_gt_2c", "count_abs_gt_3c",
    ]
    metric_names = [
        "retention", "residual_bias", "residual_mae", "residual_rmse",
        "frac_abs_resid_gt_1c", "frac_abs_resid_gt_2c", "frac_abs_resid_gt_3c",
    ]
    rows = []
    regex = re.compile(methods_regex) if methods_regex else None

    for spec in specs:
        if regex is not None and not regex.search(spec.method):
            continue
        g = _block_summary_for_spec(eval_df, block_key, spec)
        if len(g) == 0:
            continue
        g = g.reindex(all_blocks).fillna(0)
        arr = g[metrics_cols].to_numpy(dtype="float64")
        point = _metrics_from_sums(arr.sum(axis=0), metrics_cols)


        draw_sums = arr[boot_indices].sum(axis=1)
        draw_metrics = {name: [] for name in metric_names}
        for i in range(draw_sums.shape[0]):
            md = _metrics_from_sums(draw_sums[i], metrics_cols)
            for name in metric_names:
                draw_metrics[name].append(md[name])

        row = {
            "table_family": spec.table_family,
            "method": spec.method,
            "residual_col": spec.residual_col,
            "bootstrap_unit": unit,
            "n_bootstrap": int(n_bootstrap),
            "n_blocks": int(n_blocks),
            "point_n_total": point["n_total"],
            "point_n_used": point["n_used"],
        }
        for name in metric_names:
            vals = np.asarray(draw_metrics[name], dtype="float64")
            vals = vals[np.isfinite(vals)]
            row[f"point_{name}"] = point[name]
            row[f"{name}_ci_low"] = float(np.quantile(vals, 0.025)) if vals.size else np.nan
            row[f"{name}_boot_median"] = float(np.quantile(vals, 0.50)) if vals.size else np.nan
            row[f"{name}_ci_high"] = float(np.quantile(vals, 0.975)) if vals.size else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_category_fraction_ci(
    eval_df: pd.DataFrame,
    group_col: str,
    unit: str = "station",
    n_bootstrap: int = 500,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Block-bootstrap CIs for category/action fractions."""
    if group_col not in eval_df.columns:
        return pd.DataFrame()
    block_key = make_block_key(eval_df, unit=unit)
    tmp = pd.DataFrame({"block_key": block_key.astype(str), group_col: eval_df[group_col].astype(str), "n": 1})
    tab = tmp.pivot_table(index="block_key", columns=group_col, values="n", aggfunc="sum", fill_value=0)
    categories = list(tab.columns)
    arr = tab.to_numpy(dtype="float64")
    n_blocks = arr.shape[0]
    rng = np.random.default_rng(int(random_state))
    boot_indices = rng.integers(0, n_blocks, size=(int(n_bootstrap), n_blocks))
    draw_counts = arr[boot_indices].sum(axis=1)
    draw_totals = draw_counts.sum(axis=1, keepdims=True)
    draw_fracs = np.divide(draw_counts, draw_totals, out=np.zeros_like(draw_counts), where=draw_totals > 0)
    point_counts = arr.sum(axis=0)
    point_total = point_counts.sum()
    rows = []
    for j, cat in enumerate(categories):
        vals = draw_fracs[:, j]
        rows.append({
            "group_col": group_col,
            "category": str(cat),
            "bootstrap_unit": unit,
            "n_bootstrap": int(n_bootstrap),
            "n_blocks": int(n_blocks),
            "point_n": int(point_counts[j]),
            "point_fraction": float(point_counts[j] / point_total) if point_total else np.nan,
            "fraction_ci_low": float(np.quantile(vals, 0.025)),
            "fraction_boot_median": float(np.quantile(vals, 0.50)),
            "fraction_ci_high": float(np.quantile(vals, 0.975)),
        })
    return pd.DataFrame(rows).sort_values("point_fraction", ascending=False).reset_index(drop=True)


def station_level_fusion_diagnostics(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Station-level residual, QC, fusion-category, and dominance diagnostics."""
    eval_df = ensure_fusion_compat_columns(eval_df)
    if not {"network", "station_id"}.issubset(eval_df.columns):
        return {"station_summary": pd.DataFrame(), "dominance_summary": pd.DataFrame()}
    x = eval_df.copy()
    x["station_key"] = x["network"].astype(str) + "::" + x["station_id"].astype(str)
    x["raw_abs_resid"] = pd.to_numeric(x.get("cws_ref_resid"), errors="coerce").abs()
    if "cws_ref_resid_corrected_station_bias_policy" in x.columns:
        x["policy_abs_resid"] = pd.to_numeric(x["cws_ref_resid_corrected_station_bias_policy"], errors="coerce").abs()
    else:
        x["policy_abs_resid"] = np.nan

    bool_cols = [
        "recommended_keep_conservative", "recommended_keep_microclimate",
        "recommended_bias_correctable", "recommended_reject_transient_error",
    ]
    for c in bool_cols:
        x[c + "__bool"] = _series_bool(x, c) if c in x.columns else False

    qcs = [c for c in x.columns if str(c).startswith("qc_") and str(c).endswith("_is_outlier")]
    for c in qcs:
        x[c + "__flag"] = pd.to_numeric(x[c], errors="coerce").eq(1)

    category_indicators = {}
    if "fusion_category" in x.columns:
        for cat in sorted(map(str, x["fusion_category"].dropna().unique())):
            safe = _safe_name(cat).lower()
            col = f"category_frac__{safe}"
            x[col] = x["fusion_category"].astype(str).eq(cat).astype(float)
            category_indicators[col] = cat

    agg_dict = {
        "n_rows": ("station_key", "size"),
        "raw_residual_bias": ("cws_ref_resid", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        "raw_residual_mae": ("raw_abs_resid", "mean"),
        "raw_residual_p95_abs": ("raw_abs_resid", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
        "policy_corrected_mae": ("policy_abs_resid", "mean"),
    }
    for c in bool_cols:
        agg_dict[f"frac_{c}"] = (c + "__bool", "mean")
    for c in qcs:
        agg_dict[f"frac_{c}"] = (c + "__flag", "mean")


    score_cols = [c for c in x.columns if str(c).startswith("pred_ref_risk_prob_")]
    score_cols += [
        "p_reject_as_unexplained_transient_error",
        "p_reference_model_issue_no_qc",
        "p_residual_explained_by_persistent_station_context",
        "p_reference_uncertainty",
        "p_radiation_or_siting_bias_signal",
        "p_environmental_difference_signal",
        "p_context_microclimate_support",
        "expected_train_station_context_bias_c",
    ]
    for c in dict.fromkeys(score_cols):
        if c in x.columns:
            agg_dict[f"mean_{c}"] = (c, lambda s: pd.to_numeric(s, errors="coerce").mean())
    for c in category_indicators:
        agg_dict[c] = (c, "mean")

    station = x.groupby(["network", "station_id", "station_key"], dropna=False).agg(**agg_dict).reset_index()
    station["policy_correction_mae_gain"] = station["raw_residual_mae"] - station["policy_corrected_mae"]
    station["raw_abs_error_contribution"] = station["raw_residual_mae"] * station["n_rows"]
    total_abs_error = station["raw_abs_error_contribution"].sum()
    station["share_total_abs_error"] = station["raw_abs_error_contribution"] / total_abs_error if total_abs_error else np.nan
    station = station.sort_values("raw_abs_error_contribution", ascending=False).reset_index(drop=True)

    rows = []
    total_rows = station["n_rows"].sum()
    for k in [5, 10, 25, 50, 100]:
        top = station.head(k)
        rows.append({
            "top_k_stations": k,
            "row_share": float(top["n_rows"].sum() / total_rows) if total_rows else np.nan,
            "raw_abs_error_share": float(top["raw_abs_error_contribution"].sum() / total_abs_error) if total_abs_error else np.nan,
            "mean_raw_mae_top_k": float(top["raw_residual_mae"].mean()) if len(top) else np.nan,
            "mean_policy_correction_gain_top_k": float(top["policy_correction_mae_gain"].mean()) if len(top) else np.nan,
        })
    dominance = pd.DataFrame(rows)

    outputs = {
        "station_summary": station,
        "dominance_summary": dominance,
        "top_stations_by_abs_error_contribution": station.head(100),
        "top_stations_by_raw_mae_min_500_rows": station[station["n_rows"] >= 500].sort_values("raw_residual_mae", ascending=False).head(100),
        "top_stations_by_policy_correction_gain_min_500_rows": station[station["n_rows"] >= 500].sort_values("policy_correction_mae_gain", ascending=False).head(100),
    }
    return outputs


def shift_diagnostics(eval_df: pd.DataFrame, manifest: Mapping | None = None) -> dict[str, pd.DataFrame]:
    """Summarize residual/risk behavior by split, month, hour, and station support."""
    if "date" not in eval_df.columns:
        return {}
    x = eval_df.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
    x["month"] = x["date"].dt.month
    x["hour"] = x["date"].dt.hour
    if "abs_cws_ref_resid" not in x.columns and "cws_ref_resid" in x.columns:
        x["abs_cws_ref_resid"] = pd.to_numeric(x["cws_ref_resid"], errors="coerce").abs()
    prob_col = infer_calibrated_prob_col(x, manifest=manifest)
    target_col = "target_ref_risk_hiconf" if "target_ref_risk_hiconf" in x.columns else ("target_ref_risk" if "target_ref_risk" in x.columns else None)

    def _summ(group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for key, g in x.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
            row["n"] = int(len(g))
            row["n_stations"] = int(g["station_id"].nunique()) if "station_id" in g.columns else np.nan
            if "cws_ref_resid" in g.columns:
                row.update(residual_metrics_from_series(g["cws_ref_resid"]))
            if prob_col:
                row["mean_calibrated_risk_prob"] = float(pd.to_numeric(g[prob_col], errors="coerce").mean())
            if target_col:
                y = pd.to_numeric(g[target_col], errors="coerce")
                y = y[y.isin([0, 1])]
                row["target_event_rate"] = float(y.mean()) if len(y) else np.nan
                row["target_label_n"] = int(len(y))
            rows.append(row)
        return pd.DataFrame(rows)

    out = {
        "shift_by_month": _summ(["month"]),
        "shift_by_hour": _summ(["hour"]),
    }
    if "split" in x.columns:
        out["shift_by_split"] = _summ(["split"])
        out["shift_by_split_month"] = _summ(["split", "month"])
    if "ref_support_class" in x.columns:
        out["shift_by_reference_support_class"] = _summ(["ref_support_class"])
    return out


def residual_metrics_for_mask(df: pd.DataFrame, mask, residual_col: str = "cws_ref_resid") -> dict:
    """Point residual metrics for a boolean keep mask."""
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    if residual_col not in df.columns:
        return {
            "n_total": int(len(df)), "n_kept": int(mask.sum()),
            "retention": float(mask.mean()) if len(mask) else np.nan,
            "n_stations_kept": np.nan, "residual_bias": np.nan,
            "residual_mae": np.nan, "residual_rmse": np.nan,
            "residual_p95_abs": np.nan, "residual_p99_abs": np.nan,
            "frac_abs_resid_gt_1c": np.nan, "frac_abs_resid_gt_2c": np.nan,
            "frac_abs_resid_gt_3c": np.nan,
        }
    g = df.loc[mask]
    r = pd.to_numeric(g[residual_col], errors="coerce").dropna()
    abs_r = r.abs()
    return {
        "n_total": int(len(df)),
        "n_kept": int(mask.sum()),
        "retention": float(mask.mean()) if len(mask) else np.nan,
        "n_stations_kept": int(g[["network", "station_id"]].drop_duplicates().shape[0]) if {"network", "station_id"}.issubset(g.columns) else np.nan,
        "residual_bias": float(r.mean()) if len(r) else np.nan,
        "residual_mae": float(abs_r.mean()) if len(abs_r) else np.nan,
        "residual_rmse": float(np.sqrt((r ** 2).mean())) if len(r) else np.nan,
        "residual_p95_abs": float(abs_r.quantile(0.95)) if len(abs_r) else np.nan,
        "residual_p99_abs": float(abs_r.quantile(0.99)) if len(abs_r) else np.nan,
        "frac_abs_resid_gt_1c": float((abs_r > 1).mean()) if len(abs_r) else np.nan,
        "frac_abs_resid_gt_2c": float((abs_r > 2).mean()) if len(abs_r) else np.nan,
        "frac_abs_resid_gt_3c": float((abs_r > 3).mean()) if len(abs_r) else np.nan,
    }


def make_validation_matched_retention_frontier(
    fused: pd.DataFrame,
    manifest: Mapping | None = None,
    risk_model_key: str | None = None,
    validation_split: str = "valid",
    test_split: str = "test",
    qc_levels: Sequence[str] = ("lenient", "strict", "ultra_strict"),
) -> pd.DataFrame:
    """Create validation-targeted matched-retention frontier, evaluated on test.

    For each CrowdQC level, the target retention is measured on the validation
    split. Fusion/risk-score thresholds are selected on validation to match that
    target, then evaluated on test.  This avoids choosing thresholds on test.
    """
    x = ensure_fusion_compat_columns(fused, manifest=manifest)
    if "split" not in x.columns:
        raise KeyError("make_validation_matched_retention_frontier requires a split column.")
    valid = x[x["split"].astype(str).eq(str(validation_split))].copy()
    test = x[x["split"].astype(str).eq(str(test_split))].copy()
    if len(valid) == 0 or len(test) == 0:
        raise ValueError(f"Need non-empty {validation_split!r} and {test_split!r} splits.")

    risk_model_key = risk_model_key or _config_value(manifest, "risk_model_key", None)
    score_cols: dict[str, str] = {}
    if "p_reject_as_unexplained_transient_error" in valid.columns:
        score_cols["fusion_reject_score"] = "p_reject_as_unexplained_transient_error"
    cal_prob_col = infer_calibrated_prob_col(test, manifest=manifest, risk_model_key=risk_model_key)
    if cal_prob_col:
        score_cols[f"risk_prob_{risk_model_key or 'model'}"] = cal_prob_col

    corrected_cols = ["cws_ref_resid"]
    if "cws_ref_resid_corrected_station_bias_policy" in test.columns:
        corrected_cols.append("cws_ref_resid_corrected_station_bias_policy")
    if "cws_ref_resid_corrected_station_bias_all" in test.columns:
        corrected_cols.append("cws_ref_resid_corrected_station_bias_all")

    rows: list[dict] = []
    for level in qc_levels:
        qc_col = f"qc_{level}_is_outlier"
        if qc_col not in valid.columns or qc_col not in test.columns:
            continue
        valid_qc_clean = pd.to_numeric(valid[qc_col], errors="coerce").fillna(0).eq(0)
        test_qc_clean = pd.to_numeric(test[qc_col], errors="coerce").fillna(0).eq(0)
        target_retention = float(valid_qc_clean.mean())

        for residual_col in corrected_cols:
            suffix = "raw" if residual_col == "cws_ref_resid" else residual_col.replace("cws_ref_resid_corrected_", "")
            row = {
                "target_from": f"crowdqc_{level}_valid_clean_retention",
                "method": f"crowdqc_{level}_clean_test__{suffix}",
                "score_col": qc_col,
                "threshold_from_valid": np.nan,
                "target_retention_valid": target_retention,
                "score_nonmissing_valid": float(valid[qc_col].notna().mean()),
                "score_nonmissing_test": float(test[qc_col].notna().mean()),
                "residual_col": residual_col,
            }
            row.update(residual_metrics_for_mask(test, test_qc_clean, residual_col))
            rows.append(row)

        for method_name, score_col in score_cols.items():
            if score_col not in valid.columns or score_col not in test.columns:
                continue
            score_valid = pd.to_numeric(valid[score_col], errors="coerce")
            score_test = pd.to_numeric(test[score_col], errors="coerce")
            threshold = float(score_valid.quantile(target_retention))
            keep_test = score_test <= threshold
            for residual_col in corrected_cols:
                suffix = "raw" if residual_col == "cws_ref_resid" else residual_col.replace("cws_ref_resid_corrected_", "")
                row = {
                    "target_from": f"crowdqc_{level}_valid_clean_retention",
                    "method": f"{method_name}_matched_to_{level}__{suffix}",
                    "score_col": score_col,
                    "threshold_from_valid": threshold,
                    "target_retention_valid": target_retention,
                    "score_nonmissing_valid": float(score_valid.notna().mean()),
                    "score_nonmissing_test": float(score_test.notna().mean()),
                    "residual_col": residual_col,
                }
                row.update(residual_metrics_for_mask(test, keep_test, residual_col))
                rows.append(row)
    return pd.DataFrame(rows)


def policy_specs_from_matched_frontier(eval_df: pd.DataFrame, matched_frontier: pd.DataFrame) -> list[PolicySpec]:
    """Convert a matched-frontier table into PolicySpec objects for bootstrap CIs."""
    specs: list[PolicySpec] = []
    for _, r in matched_frontier.iterrows():
        score_col = str(r.get("score_col", ""))
        method = str(r.get("method", ""))
        residual_col = str(r.get("residual_col", "cws_ref_resid"))
        if score_col not in eval_df.columns or residual_col not in eval_df.columns:
            continue
        if score_col.startswith("qc_") and score_col.endswith("_is_outlier"):
            mask = pd.to_numeric(eval_df[score_col], errors="coerce").fillna(0).eq(0)
        else:
            threshold = pd.to_numeric(pd.Series([r.get("threshold_from_valid")]), errors="coerce").iloc[0]
            if pd.isna(threshold):
                continue
            mask = pd.to_numeric(eval_df[score_col], errors="coerce") <= float(threshold)
        specs.append(PolicySpec(method, mask, residual_col, "matched_retention"))
    return specs


def bootstrap_matched_retention_frontier_ci(
    test_df: pd.DataFrame,
    matched_frontier: pd.DataFrame,
    unit: str = "station",
    n_bootstrap: int = 500,
    random_state: int = 2026,
    methods_regex: str | None = None,
) -> pd.DataFrame:
    """Block-bootstrap CIs for a validation-matched retention frontier table."""
    specs = policy_specs_from_matched_frontier(test_df, matched_frontier)
    return bootstrap_policy_ci(
        test_df,
        specs,
        unit=unit,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        methods_regex=methods_regex,
    )


def load_eval_fusion_df(cfg: ReviewerRobustnessConfig) -> tuple[pd.DataFrame | None, dict | None, Path | None]:
    fused_path, manifest = resolve_fused_path_from_manifest(cfg.time_fusion_manifest, cfg.time_fused_path)
    if fused_path is None:
        return None, manifest, None
    if not fused_path.exists():
        raise FileNotFoundError(fused_path)
    fused = load_dataframe_auto(fused_path)
    fused, notes = harmonize_fusion_output_schema(fused)
    if notes:
        warnings.warn("Existing fusion output was harmonized in-memory for reviewer summaries: " + "; ".join(notes))
    evaluation_split = manifest.get("evaluation_split") if manifest else None
    evaluation_split = evaluation_split or (manifest or {}).get("config", {}).get("evaluation_split", "test")
    if "split" in fused.columns and evaluation_split:
        eval_df = fused[fused["split"].astype(str).eq(str(evaluation_split))].copy()
    else:
        eval_df = fused.copy()
    eval_df = make_eval_df_reviewer_compatible(eval_df)
    return eval_df, manifest, fused_path


def run_reviewer_robustness_suite(cfg: ReviewerRobustnessConfig, script_dir: str | Path | None = None) -> dict:
    """Run selected reviewer robustness tests and save all summary outputs."""
    outdir = _ensure_dir(cfg.output_dir)
    rrm, qrf = import_pipeline_modules(script_dir=script_dir)

    master_manifest: dict[str, object] = {
        "created_utc": _now_utc(),
        "config": asdict(cfg),
        "outputs": {},
        "station_holdout_runs": [],
        "station_holdout_fusion_runs": [],
        "fusion_compatibility": fusion_compatibility_report(qrf=qrf),
    }


    station_holdout_manifests: list[dict] = []
    if cfg.run_station_holdout:
        for seed in cfg.station_holdout_seeds:
            print(f"\n=== Running CWS station-held-out residual-risk seed={seed} ===")
            result = run_one_station_holdout_residual_risk(cfg, int(seed), rrm=rrm)
            station_holdout_manifests.append(result)
            master_manifest["station_holdout_runs"].append(result)

        collected = collect_residual_risk_run_tables(station_holdout_manifests, cfg)
        for name, df in collected.items():
            p = save_dataframe(df, outdir / f"{cfg.city}_reviewer_station_holdout_{name}.csv")
            master_manifest["outputs"][f"station_holdout_{name}_path"] = str(p)

        metric_summary = summarize_station_holdout_metrics(collected.get("metrics", pd.DataFrame()))
        p = save_dataframe(metric_summary, outdir / f"{cfg.city}_reviewer_station_holdout_metric_summary.csv")
        master_manifest["outputs"]["station_holdout_metric_summary_path"] = str(p)

        fi_stability = summarize_feature_importance_stability(collected.get("feature_importance", pd.DataFrame()))
        p = save_dataframe(fi_stability, outdir / f"{cfg.city}_reviewer_station_holdout_feature_importance_stability.csv")
        master_manifest["outputs"]["station_holdout_feature_importance_stability_path"] = str(p)


        if cfg.run_fusion_on_station_holdout:
            fusion_manifests = []
            for residual_manifest in station_holdout_manifests:
                for risk_key in cfg.fusion_risk_model_keys:
                    print(f"\n=== Running fusion on station-held-out seed={residual_manifest.get('station_holdout_seed')} risk={risk_key} ===")
                    fm = run_fusion_for_residual_manifest(cfg, residual_manifest, risk_key, qrf=qrf)
                    fusion_manifests.append(fm)
                    master_manifest["station_holdout_fusion_runs"].append(fm)
            fcollected = collect_fusion_run_tables(fusion_manifests)
            for name, df in fcollected.items():
                p = save_dataframe(df, outdir / f"{cfg.city}_reviewer_station_holdout_fusion_{name}.csv")
                master_manifest["outputs"][f"station_holdout_fusion_{name}_path"] = str(p)


    eval_df = None
    fusion_manifest = None
    fused_path = None
    if cfg.run_bootstrap_existing_fusion or cfg.run_station_diagnostics_existing_fusion or cfg.run_shift_diagnostics_existing_fusion:
        eval_df, fusion_manifest, fused_path = load_eval_fusion_df(cfg)
        if eval_df is None:
            warnings.warn("No existing fused path/manifest supplied; skipping existing-fusion diagnostics.")
        else:
            master_manifest["existing_fused_path"] = str(fused_path)
            master_manifest["existing_eval_rows"] = int(len(eval_df))

    if eval_df is not None and cfg.run_bootstrap_existing_fusion:
        print("\n=== Running block-bootstrap uncertainty for existing fusion output ===")
        policy_specs = build_policy_specs(eval_df, manifest=fusion_manifest)
        correction_specs = build_correction_specs(eval_df, manifest=fusion_manifest)
        ci_policy = bootstrap_policy_ci(
            eval_df,
            policy_specs,
            unit=cfg.bootstrap_unit,
            n_bootstrap=int(cfg.n_bootstrap),
            random_state=int(cfg.bootstrap_random_state),
            methods_regex=cfg.bootstrap_methods_regex,
        )
        ci_correction = bootstrap_policy_ci(
            eval_df,
            correction_specs,
            unit=cfg.bootstrap_unit,
            n_bootstrap=int(cfg.n_bootstrap),
            random_state=int(cfg.bootstrap_random_state) + 1,
            methods_regex=cfg.bootstrap_methods_regex,
        )
        ci_category = bootstrap_category_fraction_ci(
            eval_df,
            group_col="fusion_category",
            unit=cfg.bootstrap_unit,
            n_bootstrap=int(cfg.n_bootstrap),
            random_state=int(cfg.bootstrap_random_state) + 2,
        )
        ci_action = bootstrap_category_fraction_ci(
            eval_df,
            group_col="recommended_action",
            unit=cfg.bootstrap_unit,
            n_bootstrap=int(cfg.n_bootstrap),
            random_state=int(cfg.bootstrap_random_state) + 3,
        )
        for name, df in [
            ("existing_fusion_policy_bootstrap_ci", ci_policy),
            ("existing_fusion_correction_bootstrap_ci", ci_correction),
            ("existing_fusion_category_bootstrap_ci", ci_category),
            ("existing_fusion_action_bootstrap_ci", ci_action),
        ]:
            p = save_dataframe(df, outdir / f"{cfg.city}_reviewer_{name}.csv")
            master_manifest["outputs"][f"{name}_path"] = str(p)

    if eval_df is not None and cfg.run_station_diagnostics_existing_fusion:
        print("\n=== Running station-level diagnostics for existing fusion output ===")
        station_outputs = station_level_fusion_diagnostics(eval_df)
        for name, df in station_outputs.items():
            p = save_dataframe(df, outdir / f"{cfg.city}_reviewer_{name}.csv")
            master_manifest["outputs"][f"{name}_path"] = str(p)

    if eval_df is not None and cfg.run_shift_diagnostics_existing_fusion:
        print("\n=== Running temporal/reference-support shift diagnostics for existing fusion output ===")
        shift_outputs = shift_diagnostics(eval_df, manifest=fusion_manifest)
        for name, df in shift_outputs.items():
            p = save_dataframe(df, outdir / f"{cfg.city}_reviewer_{name}.csv")
            master_manifest["outputs"][f"{name}_path"] = str(p)

    manifest_path = _write_json(master_manifest, outdir / f"{cfg.city}_reviewer_robustness_manifest.json")
    master_manifest["manifest_path"] = str(manifest_path)
    print("\nReviewer robustness outputs written to:", outdir)
    print("Manifest:", manifest_path)
    return master_manifest


def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Run reviewer-facing robustness checks for the CWS QC residual-risk pipeline.")
    p.add_argument("--city", default="project_id")
    p.add_argument("--cws-reference-path", default=None)
    p.add_argument("--source-reference-manifest-path", default=None)
    p.add_argument("--source-reference-run-label", default=None)
    p.add_argument("--reference-method", default="catboost")
    p.add_argument("--calibration-mode", default="time_train")
    p.add_argument("--time-residual-risk-manifest", default=None)
    p.add_argument("--time-fusion-manifest", default=None)
    p.add_argument("--time-fused-path", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--residual-risk-output-dir", required=True)
    p.add_argument("--fusion-output-dir", default="outputs/qc_risk_fusion_reviewer_station_holdout")
    p.add_argument("--station-holdout-seeds", default="11,22,33,44,55")
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--feature-modes", default="context_history,context_static_met")
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-valid-rows", type=int, default=None)
    p.add_argument("--no-station-holdout", action="store_true")
    p.add_argument("--run-fusion-on-station-holdout", action="store_true")
    p.add_argument("--qc-dir", default=None)
    p.add_argument("--no-bootstrap-existing-fusion", action="store_true")
    p.add_argument("--bootstrap-unit", default="station", choices=["station", "station_day"])
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--script-dir", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    seeds = tuple(int(x.strip()) for x in str(args.station_holdout_seeds).split(",") if x.strip())
    cfg = ReviewerRobustnessConfig(
        city=args.city,
        cws_reference_path=args.cws_reference_path,
        source_reference_manifest_path=args.source_reference_manifest_path,
        source_reference_run_label=args.source_reference_run_label,
        reference_method=args.reference_method,
        calibration_mode=args.calibration_mode,
        time_residual_risk_manifest=args.time_residual_risk_manifest,
        time_fusion_manifest=args.time_fusion_manifest,
        time_fused_path=args.time_fused_path,
        output_dir=args.output_dir,
        residual_risk_output_dir=args.residual_risk_output_dir,
        fusion_output_dir=args.fusion_output_dir,
        run_station_holdout=not args.no_station_holdout,
        run_fusion_on_station_holdout=bool(args.run_fusion_on_station_holdout),
        qc_dir=args.qc_dir,
        run_bootstrap_existing_fusion=not args.no_bootstrap_existing_fusion,
        station_holdout_seeds=seeds,
        valid_frac=float(args.valid_frac),
        test_frac=float(args.test_frac),
        feature_modes=args.feature_modes,
        iterations=int(args.iterations),
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        bootstrap_unit=args.bootstrap_unit,
        n_bootstrap=int(args.n_bootstrap),
    )
    return run_reviewer_robustness_suite(cfg, script_dir=args.script_dir)


if __name__ == "__main__":
    main()
