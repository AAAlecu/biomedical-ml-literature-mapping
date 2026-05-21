#!/usr/bin/env python3
"""
OpenAlex dataset query script
-----------------------------

Purpose
-------
Queries OpenAlex broadly for one selected dataset, optionally scoped to one of
the four prediction regimes:

    S = structured multiscale prediction
    H = high-dimensional biomedical signals
    M = multimodal learning
    T = temporal and longitudinal modeling

The script:
1) lets you choose the dataset directly in the code
2) knows which regimes each dataset supports
3) retrieves broad candidate papers from OpenAlex using dataset aliases
4) applies local keyword filtering using:
      - a large ML term list
      - regime-specific term lists
5) ranks results and retains the top 350 papers for the selected dataset (this number can be re-defined in the code)
6) saves:
      - a formatted text file (one paper per line)
      - a CSV file with metadata and keyword match details

"""

import csv
import sys
import time
from typing import Dict, List, Tuple
import requests

# ============================================================
# USER SETTINGS: EDIT THESE DIRECTLY
# ============================================================

DATASET_KEY = "NOAA"
# Valid options:
# TCGA, GTEX, UKB_IMAGING, ADNI, CAMELYON,
# MIMIC, EICU,
# MIDUS, HRS, ELSA, NHANES,
# STUDENTLIFE, MPOWER, RADAR_CNS,
# OPENAQ, NOAA

REGIME = "S"
# Set to one of: None, "S", "H", "M", "T"
# None = use all regimes supported by the selected dataset

MAX_RESULTS = 350 # adjust this number if you want more papers
FETCH_PER_ALIAS = 350 # adjust this number if you want more papers
SORT_BY = "relevance_score:desc"
MIN_ML_MATCHES = 1
MIN_REGIME_MATCHES = 0
POLITE_EMAIL = ""
SLEEP_SECONDS = 0.15

OUT_TEXT_FILE = None     # None = auto-name
OUT_CSV_FILE = None      # None = auto-name

# ============================================================
# DATASET CATALOG FROM MANUSCRIPT TABLE
# ============================================================

DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "TCGA": {
        "display_name": "TCGA",
        "aliases": [
            "TCGA",
            "The Cancer Genome Atlas",
        ],
        "primary_modality": "Omics",
        "regimes": ["H", "M"],
    },
    "GTEX": {
        "display_name": "GTEx",
        "aliases": [
            "GTEx",
            "Genotype-Tissue Expression",
            "Genotype Tissue Expression",
        ],
        "primary_modality": "Omics",
        "regimes": ["H", "M"],
    },
    "UKB_IMAGING": {
        "display_name": "UK Biobank (imaging)",
        "aliases": [
            "UK Biobank",
            "UK Biobank imaging",
            "UK Biobank imaging study",
        ],
        "primary_modality": "Imaging + clinical",
        "regimes": ["S", "H", "M"],
    },
    "ADNI": {
        "display_name": "ADNI",
        "aliases": [
            "ADNI",
            "Alzheimer's Disease Neuroimaging Initiative",
            "Alzheimers Disease Neuroimaging Initiative",
        ],
        "primary_modality": "Imaging + clinical",
        "regimes": ["H", "M", "T"],
    },
    "CAMELYON": {
        "display_name": "CAMELYON",
        "aliases": [
            "CAMELYON",
            "CAMELYON16",
            "CAMELYON17",
        ],
        "primary_modality": "Histopathology",
        "regimes": ["H"],
    },
    "MIMIC": {
        "display_name": "MIMIC-III / IV",
        "aliases": [
            "MIMIC-III",
            "MIMIC III",
            "MIMIC-IV",
            "MIMIC IV",
            "Medical Information Mart for Intensive Care",
        ],
        "primary_modality": "Clinical + time series",
        "regimes": ["S", "T"],
    },
    "EICU": {
        "display_name": "eICU",
        "aliases": [
            "eICU",
            "eICU Collaborative Research Database",
            "eICU-CRD",
        ],
        "primary_modality": "Clinical",
        "regimes": ["S", "T"],
    },
    "MIDUS": {
        "display_name": "MIDUS",
        "aliases": [
            "MIDUS",
            "Midlife in the United States",
            "Midlife in the United States study",
        ],
        "primary_modality": "Behavioral + clinical",
        "regimes": ["S", "M", "T"],
    },
    "HRS": {
        "display_name": "HRS",
        "aliases": [
            "Health and Retirement Study",
            "HRS",
        ],
        "primary_modality": "Survey + clinical",
        "regimes": ["S", "T"],
    },
    "ELSA": {
        "display_name": "ELSA",
        "aliases": [
            "ELSA",
            "English Longitudinal Study of Ageing",
            "English Longitudinal Study of Aging",
        ],
        "primary_modality": "Survey + clinical",
        "regimes": ["S", "T"],
    },
    "NHANES": {
        "display_name": "NHANES",
        "aliases": [
            "NHANES",
            "National Health and Nutrition Examination Survey",
        ],
        "primary_modality": "Clinical + survey",
        "regimes": ["S", "M", "T"],
    },
    "STUDENTLIFE": {
        "display_name": "StudentLife",
        "aliases": [
            "StudentLife",
            "StudentLife dataset",
        ],
        "primary_modality": "Smartphone sensing",
        "regimes": ["M", "T"],
    },
    "MPOWER": {
        "display_name": "mPower",
        "aliases": [
            "mPower",
            "mPower dataset",
            "mPower study",
            "mPower Parkinson",
        ],
        "primary_modality": "Mobile health",
        "regimes": ["M", "T"],
    },
    "RADAR_CNS": {
        "display_name": "RADAR-CNS",
        "aliases": [
            "RADAR-CNS",
            "RADAR CNS",
            "Remote Assessment of Disease and Relapse Central Nervous System",
        ],
        "primary_modality": "Wearables + clinical",
        "regimes": ["M", "T"],
    },
    "OPENAQ": {
        "display_name": "OpenAQ",
        "aliases": [
            "OpenAQ",
        ],
        "primary_modality": "Environmental",
        "regimes": ["S", "T"],
    },
    "NOAA": {
        "display_name": "NOAA",
        "aliases": [
            "NOAA",
            "National Oceanic and Atmospheric Administration",
        ],
        "primary_modality": "Environmental",
        "regimes": ["S", "T"],
    },
}

# ============================================================
# BROAD ML TERMS
# ============================================================

ML_TERMS = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "statistical learning",
    "data mining",
    "predictive modeling",
    "prediction model",
    "risk model",
    "risk prediction",
    "prediction",
    "predictive",
    "classifier",
    "classification",
    "regression model",
    "multivariable model",
    "multivariate model",
    "algorithm",
    "model development",
    "logistic regression",
    "linear regression",
    "ridge regression",
    "lasso",
    "elastic net",
    "generalized linear model",
    "glm",
    "cox",
    "cox proportional hazards",
    "survival model",
    "survival analysis",
    "time-to-event",
    "time to event",
    "hazard model",
    "decision tree",
    "classification tree",
    "regression tree",
    "random forest",
    "extra trees",
    "gradient boosting",
    "gradient boosting machine",
    "gbm",
    "xgboost",
    "lightgbm",
    "catboost",
    "adaboost",
    "support vector machine",
    "support vector machines",
    "svm",
    "neural network",
    "neural networks",
    "artificial neural network",
    "ann",
    "mlp",
    "multilayer perceptron",
    "deep neural network",
    "dnn",
    "convolutional neural network",
    "cnn",
    "recurrent neural network",
    "rnn",
    "lstm",
    "transformer",
    "attention model",
    "naive bayes",
    "bayesian model",
    "bayesian network",
    "hidden markov model",
    "k-nearest neighbor",
    "k nearest neighbor",
    "knn",
    "nearest neighbor",
    "ensemble model",
    "stacking",
    "bagging",
    "blending",
    "auc",
    "roc curve",
    "area under the curve",
    "model performance",
    "cross-validation",
    "cross validation",
]

# ============================================================
# REGIME TERMS
# ============================================================

REGIME_TERMS = {
    "S": [
        "tabular",
        "structured data",
        "structured",
        "cohort",
        "survey",
        "clinical variables",
        "epidemiology",
        "electronic health record",
        "ehr",
        "risk prediction",
        "risk model",
        "multivariable",
        "population study",
        "biobank",
    ],
    "H": [
        "high-dimensional",
        "high dimensional",
        "omics",
        "genomics",
        "transcriptomics",
        "proteomics",
        "metabolomics",
        "radiomics",
        "histopathology",
        "whole slide image",
        "mri",
        "fmri",
        "ct",
        "pet",
        "image classification",
        "segmentation",
        "gene expression",
        "microarray",
        "sequencing",
    ],
    "M": [
        "multimodal",
        "multi-modal",
        "multimodality",
        "data fusion",
        "fusion model",
        "sensor fusion",
        "wearable",
        "smartphone",
        "mobile health",
        "digital phenotyping",
        "behavioral sensing",
        "clinical and imaging",
        "clinical + imaging",
        "survey and clinical",
    ],
    "T": [
        "longitudinal",
        "temporal",
        "time series",
        "timeseries",
        "trajectory",
        "trajectories",
        "repeated measures",
        "recurrent event",
        "survival analysis",
        "time-to-event",
        "time to event",
        "progression",
        "forecasting",
        "follow-up",
        "follow up",
        "incident",
    ],
}

BASE_URL = "https://api.openalex.org/works"

# ============================================================
# HELPERS
# ============================================================

def build_headers(polite_email: str) -> Dict[str, str]:
    headers = {}
    if polite_email.strip():
        headers["User-Agent"] = f"OpenAlexDatasetRegimeQuery/1.0 ({polite_email.strip()})"
    return headers


def reconstruct_abstract(abstract_inverted_index) -> str:
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return ""
    position_to_word = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    if not position_to_word:
        return ""
    return " ".join(position_to_word[pos] for pos in sorted(position_to_word))


def fetch_alias_results(alias: str, limit: int, sort_by: str, headers: Dict[str, str]) -> List[dict]:
    params = {
        "search": alias,
        "per-page": 200,
        "cursor": "*",
        "sort": sort_by,
        "select": ",".join([
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "cited_by_count",
            "relevance_score",
            "authorships",
            "primary_location",
            "abstract_inverted_index",
        ]),
    }

    rows = []
    seen = set()

    while len(rows) < limit:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            oid = item.get("id")
            if not oid or oid in seen:
                continue
            seen.add(oid)
            rows.append(item)
            if len(rows) >= limit:
                break

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        params["cursor"] = next_cursor
        time.sleep(SLEEP_SECONDS)

    return rows


def normalize_title(item: dict) -> str:
    return (item.get("title") or item.get("display_name") or "Untitled").replace("\n", " ").strip()


def extract_authors(item: dict, max_authors: int = 3) -> str:
    authorships = item.get("authorships") or []
    names = []
    for a in authorships[:max_authors]:
        author = a.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    if not names:
        return "NA"
    if len(authorships) > max_authors:
        return ", ".join(names) + ", et al."
    return ", ".join(names)


def extract_source(item: dict) -> str:
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return source.get("display_name") or "NA"


def count_matches(text: str, terms: List[str]) -> Tuple[int, List[str]]:
    t = text.lower()
    matched = []
    for term in terms:
        if term.lower() in t:
            matched.append(term)
    return len(matched), matched


def dataset_mentioned(text: str, aliases: List[str]) -> bool:
    t = text.lower()
    return any(alias.lower() in t for alias in aliases)


def supported_regimes_for_dataset(dataset_key: str) -> List[str]:
    return DATASET_CONFIGS[dataset_key]["regimes"]


def active_regime_terms(dataset_key: str, selected_regime) -> List[str]:
    if selected_regime is None:
        regimes = supported_regimes_for_dataset(dataset_key)
        terms = []
        for reg in regimes:
            terms.extend(REGIME_TERMS.get(reg, []))
        return sorted(set(terms))
    return REGIME_TERMS.get(selected_regime, [])


def to_record(item: dict, dataset_key: str, aliases: List[str], selected_regime) -> dict:
    title = normalize_title(item)
    abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
    combined_text = f"{title} {abstract}".strip()

    ml_match_count, matched_ml_terms = count_matches(combined_text, ML_TERMS)
    regime_terms = active_regime_terms(dataset_key, selected_regime)
    regime_match_count, matched_regime_terms = count_matches(combined_text, regime_terms)
    has_dataset = dataset_mentioned(combined_text, aliases)

    return {
        "openalex_id": item.get("id", "NA"),
        "doi": item.get("doi", "NA"),
        "title": title,
        "year": item.get("publication_year", "NA"),
        "publication_date": item.get("publication_date", "NA"),
        "type": item.get("type", "NA"),
        "cited_by_count": item.get("cited_by_count", 0) or 0,
        "relevance_score": item.get("relevance_score", 0) or 0,
        "authors": extract_authors(item),
        "source": extract_source(item),
        "abstract": abstract,
        "dataset_mentioned": has_dataset,
        "ml_match_count": ml_match_count,
        "matched_ml_terms": "; ".join(matched_ml_terms),
        "regime_selected": selected_regime if selected_regime is not None else "ALL_SUPPORTED",
        "regime_match_count": regime_match_count,
        "matched_regime_terms": "; ".join(matched_regime_terms),
    }


def filter_and_rank(records: List[dict], selected_regime) -> List[dict]:
    kept = [r for r in records if r["ml_match_count"] >= MIN_ML_MATCHES]

    if selected_regime is not None and MIN_REGIME_MATCHES > 0:
        kept = [r for r in kept if r["regime_match_count"] >= MIN_REGIME_MATCHES]

    kept.sort(
        key=lambda r: (
            int(bool(r["dataset_mentioned"])),
            r["ml_match_count"],
            r["regime_match_count"],
            float(r["relevance_score"] or 0),
            int(r["cited_by_count"] or 0),
        ),
        reverse=True
    )
    return kept[:MAX_RESULTS]


def format_line(idx: int, rec: dict) -> str:
    return (
        f"{idx:04d}. "
        f"{rec['authors']} | "
        f"{rec['year']} | "
        f"{rec['title']} | "
        f"{rec['source']} | "
        f"ML matches: {rec['ml_match_count']} | "
        f"Regime matches: {rec['regime_match_count']} | "
        f"DOI: {rec['doi']} | "
        f"OpenAlex: {rec['openalex_id']}"
    )


def save_text(records: List[dict], path: str, dataset_name: str, aliases: List[str], selected_regime) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset aliases: {aliases}\n")
        f.write(f"Selected regime: {selected_regime if selected_regime is not None else 'ALL_SUPPORTED'}\n")
        f.write(f"ML terms used: {len(ML_TERMS)}\n")
        active_terms = active_regime_terms(dataset_key=DATASET_KEY.strip().upper(), selected_regime=selected_regime)
        f.write(f"Regime terms used: {len(active_terms)}\n")
        f.write(f"Returned records after filtering/ranking: {len(records)}\n")
        f.write("=" * 150 + "\n")
        for i, rec in enumerate(records, start=1):
            f.write(format_line(i, rec) + "\n")


def save_csv(records: List[dict], path: str) -> None:
    fieldnames = [
        "openalex_id",
        "doi",
        "title",
        "year",
        "publication_date",
        "type",
        "cited_by_count",
        "relevance_score",
        "authors",
        "source",
        "dataset_mentioned",
        "ml_match_count",
        "matched_ml_terms",
        "regime_selected",
        "regime_match_count",
        "matched_regime_terms",
        "abstract",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    dataset_key = DATASET_KEY.strip().upper()
    if dataset_key not in DATASET_CONFIGS:
        valid = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown DATASET_KEY='{DATASET_KEY}'. Valid options: {valid}")

    selected_regime = REGIME
    if selected_regime is not None:
        selected_regime = str(selected_regime).upper()
        if selected_regime not in ["S", "H", "M", "T"]:
            raise ValueError("REGIME must be one of None, 'S', 'H', 'M', 'T'")
        if selected_regime not in DATASET_CONFIGS[dataset_key]["regimes"]:
            supported = DATASET_CONFIGS[dataset_key]["regimes"]
            raise ValueError(
                f"Dataset {dataset_key} does not support regime {selected_regime}. "
                f"Supported regimes: {supported}"
            )

    cfg = DATASET_CONFIGS[dataset_key]
    dataset_name = cfg["display_name"]
    aliases = cfg["aliases"]
    headers = build_headers(POLITE_EMAIL)

    print(f"Dataset: {dataset_name}")
    print(f"Supported regimes: {cfg['regimes']}")
    print(f"Selected regime: {selected_regime if selected_regime is not None else 'ALL_SUPPORTED'}")
    print(f"Fetching up to {FETCH_PER_ALIAS} results per alias from OpenAlex...")
    print("Then applying local ML + regime filtering on title + abstract...")

    all_items = []
    seen_ids = set()

    for alias in aliases:
        print(f"  Querying alias: {alias}")
        items = fetch_alias_results(alias, FETCH_PER_ALIAS, SORT_BY, headers)
        for item in items:
            oid = item.get("id")
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)
            all_items.append(item)

    print(f"Raw unique records retrieved across aliases: {len(all_items)}")

    records = [to_record(item, dataset_key, aliases, selected_regime) for item in all_items]
    ranked = filter_and_rank(records, selected_regime)

    regime_tag = selected_regime if selected_regime is not None else "ALL"
    out_text = OUT_TEXT_FILE or f"openalex_{dataset_key.lower()}_{regime_tag}_ml_top350.txt"
    out_csv = OUT_CSV_FILE or f"openalex_{dataset_key.lower()}_{regime_tag}_ml_top350.csv"

    save_text(ranked, out_text, dataset_name, aliases, selected_regime)
    save_csv(ranked, out_csv)

    print(f"Done. Saved text file: {out_text}")
    print(f"Done. Saved CSV file:  {out_csv}")
    print(f"Final records saved: {len(ranked)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nStopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
