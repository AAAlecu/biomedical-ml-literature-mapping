#!/usr/bin/env python3
"""
Sensitivity analysis for manually annotated literature-mining audit CSVs.

This script reads one manually annotated audit file such as H_sample_50.csv,
S_sample_50.csv, M_sample_50.csv, or T_sample_50.csv and writes a plain-text
report with:

1. Baseline validation metrics:
   - A_dom, A*_dom, R_amb
   - P_fam, R_fam, F1_fam for prevalence-oriented family extraction
   - PDF vs title/abstract coverage if pdf_downloaded is available

2. Sensitivity analyses using the hard-coded exact-model -> model-family taxonomy:
   - leave-one-family-out: removes one model family from automated predictions
   - leave-one-exact-model-out: removes one exact model from automated predictions
   - family-collapse scenarios: recomputes metrics under coarser taxonomies

Usage:
    python compute_validation_metrics.py --input H_sample_50.csv --output H_sensitivity_report.txt

"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


TRUE_VALUES = {"true", "t", "1", "yes", "y"}
MISSING_LABELS = {"", "none", "none/unclear", "none / unclear", "unclear", "na", "n/a", "nan"}

# -----------------------------------------------------------------------------
# Hard-coded taxonomy from Supplementary_Computational_Literature_Mining.pdf
# -----------------------------------------------------------------------------
EXACT_MODEL_TO_FAMILY: Dict[str, str] = {
    # Deep learning
    "Autoencoder": "Deep learning",
    "CNN": "Deep learning",
    "DNN": "Deep learning",
    "GRU": "Deep learning",
    "Graph neural network": "Deep learning",
    "LSTM": "Deep learning",
    "MLP": "Deep learning",
    "RNN": "Deep learning",
    "TCN": "Deep learning",
    "Transformer": "Deep learning",

    # Tree-based ensemble
    "AdaBoost": "Tree-based ensemble",
    "CatBoost": "Tree-based ensemble",
    "Extra Trees": "Tree-based ensemble",
    "Gradient Boosting": "Tree-based ensemble",
    "LightGBM": "Tree-based ensemble",
    "Random Forest": "Tree-based ensemble",
    "XGBoost": "Tree-based ensemble",

    # Linear / GLM
    "Elastic Net": "Linear / GLM",
    "LASSO": "Linear / GLM",
    "Linear Regression": "Linear / GLM",
    "Logistic Regression": "Linear / GLM",
    "Ridge Regression": "Linear / GLM",

    # Kernel / instance-based
    "Support Vector Machine": "Kernel / instance-based",
    "k-Nearest Neighbors": "Kernel / instance-based",

    # Survival models
    "Cox Proportional Hazards": "Survival models",
    "Joint Model": "Survival models",
    "Kaplan-Meier": "Survival models",
    "Random Survival Forest": "Survival models",

    # Probabilistic / statistical
    "ARIMA": "Probabilistic / statistical",
    "Bayesian Regression": "Probabilistic / statistical",
    "Generalized Estimating Equations": "Probabilistic / statistical",
    "Hidden Markov Model": "Probabilistic / statistical",
    "Kalman Filter": "Probabilistic / statistical",
    "Markov Model": "Probabilistic / statistical",
    "Mixed-Effects Model": "Probabilistic / statistical",
    "Naive Bayes": "Probabilistic / statistical",

    # Ensemble / stacking
    "Ensemble": "Ensemble / stacking",

    # Other classical ML
    "Decision Tree": "Other classical ML",
    "Federated Learning": "Other classical ML",
    "Reinforcement Learning": "Other classical ML",
}

FAMILIES: List[str] = sorted(set(EXACT_MODEL_TO_FAMILY.values()))

# Coarser taxonomy scenarios. These test whether validation conclusions depend
# on fine distinctions between adjacent families.
FAMILY_COLLAPSE_SCENARIOS: Dict[str, Dict[str, str]] = {
    "baseline_no_collapse": {},
    "classical_ml_collapsed": {
        "Linear / GLM": "Classical/statistical ML",
        "Kernel / instance-based": "Classical/statistical ML",
        "Probabilistic / statistical": "Classical/statistical ML",
        "Survival models": "Classical/statistical ML",
        "Other classical ML": "Classical/statistical ML",
    },
    "tree_and_ensemble_collapsed": {
        "Tree-based ensemble": "Tree/ensemble ML",
        "Ensemble / stacking": "Tree/ensemble ML",
        "Other classical ML": "Tree/ensemble ML",
    },
    "survival_with_linear_glm": {
        "Linear / GLM": "Linear/GLM/survival",
        "Survival models": "Linear/GLM/survival",
    },
    "all_non_deep_collapsed": {
        "Tree-based ensemble": "Non-deep learning",
        "Linear / GLM": "Non-deep learning",
        "Kernel / instance-based": "Non-deep learning",
        "Survival models": "Non-deep learning",
        "Probabilistic / statistical": "Non-deep learning",
        "Ensemble / stacking": "Non-deep learning",
        "Other classical ML": "Non-deep learning",
    },
}


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_label(x) -> str:
    s = norm_text(x)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# Normalized lookup dictionaries.
NORM_EXACT_TO_CANONICAL = {norm_label(k): k for k in EXACT_MODEL_TO_FAMILY}
NORM_FAMILY_TO_CANONICAL = {norm_label(f): f for f in FAMILIES}
NORM_MODEL_TO_NORM_FAMILY = {
    norm_label(m): norm_label(f) for m, f in EXACT_MODEL_TO_FAMILY.items()
}


def is_true(x) -> bool:
    return norm_text(x).lower() in TRUE_VALUES


def parse_set(x) -> Set[str]:
    s = norm_text(x)
    if not s:
        return set()
    if ";" in s:
        parts = s.split(";")
    elif "|" in s:
        parts = s.split("|")
    else:
        parts = s.split(",")
    out = set()
    for p in parts:
        q = norm_label(p)
        if q not in MISSING_LABELS:
            out.add(q)
    return out


def canonicalize_model_set(labels: Iterable[str]) -> Set[str]:
    out = set()
    for lab in labels:
        n = norm_label(lab)
        if n in NORM_EXACT_TO_CANONICAL:
            out.add(norm_label(NORM_EXACT_TO_CANONICAL[n]))
        elif n not in MISSING_LABELS:
            out.add(n)
    return out


def family_set_from_models(model_set: Iterable[str]) -> Set[str]:
    fams = set()
    for m in model_set:
        n = norm_label(m)
        fam = NORM_MODEL_TO_NORM_FAMILY.get(n)
        if fam:
            fams.add(fam)
    return fams


def apply_family_collapse_to_label(label: str, collapse: Mapping[str, str]) -> str:
    n = norm_label(label)
    if n in MISSING_LABELS:
        return ""
    # map canonical family labels through collapse table
    canonical = NORM_FAMILY_TO_CANONICAL.get(n, label)
    collapsed = collapse.get(canonical, canonical)
    return norm_label(collapsed)


def apply_family_collapse_to_set(labels: Iterable[str], collapse: Mapping[str, str]) -> Set[str]:
    return {x for x in (apply_family_collapse_to_label(lab, collapse) for lab in labels) if x}


def safe_div(num: float, den: float) -> float:
    return math.nan if den == 0 else num / den


def f1_from_pr(p: float, r: float) -> float:
    if math.isnan(p) or math.isnan(r):
        return math.nan
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def prf_counts(pred_sets: Sequence[Set[str]], gold_sets: Sequence[Set[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    exact = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(pred == gold)
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1_from_pr(p, r),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_match": safe_div(exact, len(pred_sets)),
    }


def compute_metrics(
    df: pd.DataFrame,
    pred_primary_family: Sequence[str],
    gold_primary_family: Sequence[str],
    ambiguous: Sequence[bool],
    pred_family_sets: Sequence[Set[str]],
    gold_family_sets: Sequence[Set[str]],
    pred_model_sets: Optional[Sequence[Set[str]]] = None,
    gold_model_sets: Optional[Sequence[Set[str]]] = None,
) -> Dict[str, float]:
    n = len(df)
    pred_pf = pd.Series([norm_label(x) for x in pred_primary_family])
    gold_pf = pd.Series([norm_label(x) for x in gold_primary_family])
    amb = pd.Series(list(ambiguous)).astype(bool)

    valid_gold = gold_pf.ne("") & ~gold_pf.isin(MISSING_LABELS)
    correct = (pred_pf == gold_pf) & valid_gold
    nonambig = ~amb
    n_nonambig = int(nonambig.sum())

    fam = prf_counts(pred_family_sets, gold_family_sets)
    out = {
        "n": n,
        "A_dom": safe_div(float(correct.sum()), n),
        "A_dom_star": safe_div(float(correct[nonambig].sum()), n_nonambig),
        "R_amb": safe_div(float(amb.sum()), n),
        "ambiguous_n": int(amb.sum()),
        "nonambiguous_n": n_nonambig,
        "P_fam": fam["precision"],
        "R_fam": fam["recall"],
        "F1_fam": fam["f1"],
        "fam_tp": fam["tp"],
        "fam_fp": fam["fp"],
        "fam_fn": fam["fn"],
        "fam_exact_match": fam["exact_match"],
    }
    if pred_model_sets is not None and gold_model_sets is not None:
        mod = prf_counts(pred_model_sets, gold_model_sets)
        out.update({
            "P_model": mod["precision"],
            "R_model": mod["recall"],
            "F1_model": mod["f1"],
            "model_tp": mod["tp"],
            "model_fp": mod["fp"],
            "model_fn": mod["fn"],
            "model_exact_match": mod["exact_match"],
        })
    return out


def fmt_num(x: float, digits: int = 3) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_pct(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    return f"{100*x:.1f}%"


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def build_baseline_inputs(df: pd.DataFrame):
    pred_primary_family = [norm_label(x) for x in df["primary_family"]]
    gold_primary_family = [norm_label(x) for x in df["manual_primary_family"]]
    ambiguous = [is_true(x) for x in df["manual_primary_ambiguous"]]

    pred_model_sets = [canonicalize_model_set(parse_set(x)) for x in df["models_detected"]]
    gold_model_sets = [canonicalize_model_set(parse_set(x)) for x in df["manual_models_any"]]

    pred_family_sets_from_models = [family_set_from_models(s) for s in pred_model_sets]
    gold_family_sets_from_models = [family_set_from_models(s) for s in gold_model_sets]

    pred_family_sets_raw = [parse_set(x) for x in df["families_detected"]]
    gold_family_sets_raw = [parse_set(x) for x in df["manual_families_any"]]

    pred_family_sets = [a if a else b for a, b in zip(pred_family_sets_from_models, pred_family_sets_raw)]
    gold_family_sets = [a if a else b for a, b in zip(gold_family_sets_from_models, gold_family_sets_raw)]

    return {
        "pred_primary_family": pred_primary_family,
        "gold_primary_family": gold_primary_family,
        "ambiguous": ambiguous,
        "pred_model_sets": pred_model_sets,
        "gold_model_sets": gold_model_sets,
        "pred_family_sets": pred_family_sets,
        "gold_family_sets": gold_family_sets,
    }


def make_report(df: pd.DataFrame, args: argparse.Namespace) -> str:
    inputs = build_baseline_inputs(df)
    baseline = compute_metrics(df, **inputs)

    regime = norm_text(df["regime"].iloc[0]) if "regime" in df.columns and len(df) else ""

    lines: List[str] = []
    lines.append(f"Taxonomy sensitivity analysis report{f' ({regime})' if regime else ''}")
    lines.append("=" * 43)
    lines.append(f"Input CSV: {args.input}")
    lines.append(f"Audited papers (N): {len(df)}")
    lines.append("")
    lines.append("Scope note")
    lines.append("----------")
    lines.append("This sensitivity analysis uses the already selected manually annotated audit sample.")
    lines.append("It tests sensitivity of model extraction, exact-model mapping, model-family aggregation,")
    lines.append("and dominant-family assignment.")
    lines.append("")

    lines.append("Hard-coded model-family taxonomy")
    lines.append("--------------------------------")
    fam_to_models = defaultdict(list)
    for model, family in EXACT_MODEL_TO_FAMILY.items():
        fam_to_models[family].append(model)
    for fam in sorted(fam_to_models):
        lines.append(f"{fam}: {', '.join(sorted(fam_to_models[fam]))}")
    lines.append("")

    lines.append("Baseline validation metrics")
    lines.append("---------------------------")
    lines.append(f"A_dom:       {fmt_num(baseline['A_dom'])} ({fmt_pct(baseline['A_dom'])})")
    lines.append(f"A*_dom:      {fmt_num(baseline['A_dom_star'])} ({fmt_pct(baseline['A_dom_star'])})")
    lines.append(f"R_amb:       {fmt_num(baseline['R_amb'])} ({fmt_pct(baseline['R_amb'])})")
    lines.append(f"Ambiguous:   {int(baseline['ambiguous_n'])}/{len(df)}")
    lines.append(f"Nonambiguous denominator for A*_dom: {int(baseline['nonambiguous_n'])}")
    lines.append("")
    lines.append("Prevalence-oriented model-family extraction, micro-averaged")
    lines.append(f"P_fam:       {fmt_num(baseline['P_fam'])}")
    lines.append(f"R_fam:       {fmt_num(baseline['R_fam'])}")
    lines.append(f"F1_fam:      {fmt_num(baseline['F1_fam'])}")
    lines.append(f"TP / FP / FN: {int(baseline['fam_tp'])} / {int(baseline['fam_fp'])} / {int(baseline['fam_fn'])}")
    lines.append(f"Exact family-set match rate: {fmt_pct(baseline['fam_exact_match'])}")
    lines.append("")
    lines.append("Exact-model extraction, micro-averaged")
    lines.append(f"P_model:     {fmt_num(baseline.get('P_model', math.nan))}")
    lines.append(f"R_model:     {fmt_num(baseline.get('R_model', math.nan))}")
    lines.append(f"F1_model:    {fmt_num(baseline.get('F1_model', math.nan))}")
    lines.append(f"TP / FP / FN: {int(baseline.get('model_tp', 0))} / {int(baseline.get('model_fp', 0))} / {int(baseline.get('model_fn', 0))}")
    lines.append("")

    if "pdf_downloaded" in df.columns:
        pdf_n = int(df["pdf_downloaded"].map(is_true).sum())
        n = len(df)
        lines.append("Manual-audit text source coverage")
        lines.append("---------------------------------")
        lines.append(f"PDF used:             {pdf_n}/{n} ({fmt_pct(safe_div(pdf_n, n))})")
        lines.append(f"Title/abstract used:  {n - pdf_n}/{n} ({fmt_pct(safe_div(n - pdf_n, n))})")
        lines.append("")

    # ------------------------------------------------------------------
    # Leave-one-family-out sensitivity
    # ------------------------------------------------------------------
    lines.append("Sensitivity 1: leave-one-family-out automated prediction ablation")
    lines.append("-----------------------------------------------------------------")
    lines.append("Each row removes one family from automated predictions only, then recomputes metrics.")
    lines.append("Manual gold labels are not altered. Large negative deltas indicate dependence on that family rule group.")
    lines.append("")
    header = f"{'removed_family':32s} {'A_dom':>7s} {'dA':>7s} {'P_fam':>7s} {'dP':>7s} {'R_fam':>7s} {'dR':>7s} {'F1':>7s} {'dF1':>7s} {'TP/FP/FN':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    for fam in FAMILIES:
        nfam = norm_label(fam)
        pred_pf = ["" if x == nfam else x for x in inputs["pred_primary_family"]]
        pred_sets = [set(s) - {nfam} for s in inputs["pred_family_sets"]]
        m = compute_metrics(
            df,
            pred_pf,
            inputs["gold_primary_family"],
            inputs["ambiguous"],
            pred_sets,
            inputs["gold_family_sets"],
        )
        lines.append(
            f"{fam:32s} {fmt_num(m['A_dom']):>7s} {fmt_num(m['A_dom']-baseline['A_dom']):>7s} "
            f"{fmt_num(m['P_fam']):>7s} {fmt_num(m['P_fam']-baseline['P_fam']):>7s} "
            f"{fmt_num(m['R_fam']):>7s} {fmt_num(m['R_fam']-baseline['R_fam']):>7s} "
            f"{fmt_num(m['F1_fam']):>7s} {fmt_num(m['F1_fam']-baseline['F1_fam']):>7s} "
            f"{int(m['fam_tp'])}/{int(m['fam_fp'])}/{int(m['fam_fn']):<5d}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Leave-one-exact-model-out sensitivity
    # ------------------------------------------------------------------
    lines.append("Sensitivity 2: leave-one-exact-model-out automated prediction ablation")
    lines.append("---------------------------------------------------------------------")
    lines.append("Each row removes one exact model from automated model predictions and remaps remaining models to families.")
    lines.append("The table is sorted by the absolute change in family-level F1.")
    lines.append("")
    rows = []
    for model, fam in EXACT_MODEL_TO_FAMILY.items():
        nmodel = norm_label(model)
        pred_model_sets = [set(s) - {nmodel} for s in inputs["pred_model_sets"]]
        pred_family_sets = [family_set_from_models(s) for s in pred_model_sets]
        pred_pf = []
        for pmod, pfam in zip(df["primary_model"], inputs["pred_primary_family"]):
            if norm_label(pmod) == nmodel:
                pred_pf.append("")
            else:
                pred_pf.append(pfam)
        m = compute_metrics(
            df,
            pred_pf,
            inputs["gold_primary_family"],
            inputs["ambiguous"],
            pred_family_sets,
            inputs["gold_family_sets"],
            pred_model_sets,
            inputs["gold_model_sets"],
        )
        rows.append((abs(m["F1_fam"] - baseline["F1_fam"]), model, fam, m))
    rows.sort(reverse=True, key=lambda x: x[0])

    header = f"{'removed_model':32s} {'family':28s} {'A_dom':>7s} {'dA':>7s} {'P_fam':>7s} {'R_fam':>7s} {'F1':>7s} {'dF1':>7s} {'TP/FP/FN':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, model, fam, m in rows:
        lines.append(
            f"{model:32s} {fam:28s} {fmt_num(m['A_dom']):>7s} {fmt_num(m['A_dom']-baseline['A_dom']):>7s} "
            f"{fmt_num(m['P_fam']):>7s} {fmt_num(m['R_fam']):>7s} {fmt_num(m['F1_fam']):>7s} "
            f"{fmt_num(m['F1_fam']-baseline['F1_fam']):>7s} "
            f"{int(m['fam_tp'])}/{int(m['fam_fp'])}/{int(m['fam_fn']):<5d}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Family-collapse scenarios
    # ------------------------------------------------------------------
    lines.append("Sensitivity 3: family-collapse taxonomy scenarios")
    lines.append("-------------------------------------------------")
    lines.append("These scenarios apply the same coarser taxonomy to automated and manual labels.")
    lines.append("They test whether conclusions depend on fine distinctions among adjacent model families.")
    lines.append("")
    header = f"{'scenario':30s} {'A_dom':>7s} {'A*':>7s} {'P_fam':>7s} {'R_fam':>7s} {'F1':>7s} {'TP/FP/FN':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    for scen, collapse in FAMILY_COLLAPSE_SCENARIOS.items():
        pred_pf = [apply_family_collapse_to_label(x, collapse) for x in inputs["pred_primary_family"]]
        gold_pf = [apply_family_collapse_to_label(x, collapse) for x in inputs["gold_primary_family"]]
        pred_sets = [apply_family_collapse_to_set(s, collapse) for s in inputs["pred_family_sets"]]
        gold_sets = [apply_family_collapse_to_set(s, collapse) for s in inputs["gold_family_sets"]]
        m = compute_metrics(
            df,
            pred_pf,
            gold_pf,
            inputs["ambiguous"],
            pred_sets,
            gold_sets,
        )
        lines.append(
            f"{scen:30s} {fmt_num(m['A_dom']):>7s} {fmt_num(m['A_dom_star']):>7s} "
            f"{fmt_num(m['P_fam']):>7s} {fmt_num(m['R_fam']):>7s} {fmt_num(m['F1_fam']):>7s} "
            f"{int(m['fam_tp'])}/{int(m['fam_fp'])}/{int(m['fam_fn']):<5d}"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute taxonomy sensitivity metrics for a manually annotated audit CSV.")

    # parser.add_argument("--input", default="./audit_papers/S/S_sample_50.csv", help="Manually annotated audit CSV.")
    # parser.add_argument("--output", default="./audit_papers/S/", help="Output text report path. Default: <input_stem>_taxonomy_sensitivity.txt")

    # parser.add_argument("--input", default="./audit_papers/T/T_sample_50.csv", help="Manually annotated audit CSV.")
    # parser.add_argument("--output", default="./audit_papers/T/", help="Output text report path. Default: <input_stem>_taxonomy_sensitivity.txt")

    # parser.add_argument("--input", default="./audit_papers/M/M_sample_50.csv", help="Manually annotated audit CSV.")
    # parser.add_argument("--output", default="./audit_papers/M/", help="Output text report path. Default: <input_stem>_taxonomy_sensitivity.txt")

    parser.add_argument("--input", default="./audit_papers/H/H_sample_50.csv", help="Manually annotated audit CSV.")
    parser.add_argument("--output", default="./audit_papers/H/", help="Output text report path. Default: <input_stem>_taxonomy_sensitivity.txt")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    require_columns(
        df,
        [
            "primary_family",
            "primary_model",
            "families_detected",
            "models_detected",
            "manual_primary_family",
            "manual_primary_model",
            "manual_primary_ambiguous",
            "manual_families_any",
            "manual_models_any",
        ],
    )

    report = make_report(df, args)
    output = Path(args.output)

    # If output has no suffix, interpret it as a directory.
    if output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
        output = output / f"{Path(args.input).stem}_taxonomy_sensitivity.txt"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(report, encoding="utf-8")
    print(f"\nSaved report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
