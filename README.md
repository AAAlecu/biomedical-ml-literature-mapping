# biomedical-ml-literature-mapping

Computational literature-mining framework for mapping machine-learning models, prediction regimes, and pipeline architectures across multiscale biomedical datasets using OpenAlex retrieval, rule-based parsing, regex-driven model extraction, dominant-model assignment, and prevalence analysis.

---

## Overview

This repository contains the code and reproducibility artifacts used to construct and analyze dataset-specific corpora of biomedical machine-learning studies. The workflow retrieves candidate papers from OpenAlex, filters them by dataset and prediction regime, extracts machine-learning model mentions, assigns model-family categories, identifies dominant models when possible, and parses higher-level pipeline architecture patterns.

The framework supports four prediction regimes:

- **S**: structured multiscale prediction
- **H**: high-dimensional biomedical signals
- **M**: multimodal learning
- **T**: temporal and longitudinal modeling

The workflow is intended for computational literature mapping rather than exhaustive systematic-review enumeration. Reported percentages should therefore be interpreted as approximate prevalence indicators within the retrieved and assessable corpora.

---

## Repository contents

```text
openalex_dataset_query_all_datasets_regimes_top350.py
    Retrieves OpenAlex candidate papers for selected datasets and regimes.

parse_regime_papers_full_updated.py
    Parses retrieved OpenAlex CSV files, extracts ML models, maps model families,
    assigns dominant models, and exports any-use and primary-only tables.

parse_pipeline_patterns_from_audit.py
    Parses audit logs to identify higher-level pipeline architecture patterns.

query_outputs/
    Retrieved OpenAlex corpus files of the form:
    openalex_*_<REGIME>_ml_top350.csv

regime_parse_outputs/
    Outputs from model extraction and dominant-model parsing.

pipeline_parse_output/
    Outputs from pipeline-architecture pattern parsing.

validation_outputs/
    Manual audit CSVs, validation reports, and taxonomy sensitivity outputs.

compute_manual_audit_metrics.py
    Computes dominant-family and prevalence-oriented validation metrics from
    manually annotated audit subsets.

compute_taxonomy_sensitivity.py
    Evaluates robustness to alternative coarser model-family taxonomies using
    taxonomy-collapse sensitivity analyses.

suplimentary/
    Supplementary documentation and rule descriptions.

requirements.txt
    Python package requirements.

LICENSE
    MIT license.
```

---

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/AAAlecu/biomedical-ml-literature-mapping.git
cd biomedical-ml-literature-mapping
pip install -r requirements.txt
```

---

## Workflow

The full workflow is:

```text
OpenAlex retrieval
→ dataset-alias deduplication
→ ML keyword filtering
→ regime filtering
→ top-350 corpus construction
→ model extraction
→ model-family mapping
→ dominant-model assignment
→ any-use and primary-only prevalence tables
→ pipeline-architecture pattern extraction
→ count, prevalence, coverage and audit
```

---

# Step 1: Retrieve OpenAlex candidate corpora

The retrieval script is:

```bash
python openalex_dataset_query_all_datasets_regimes_top350.py
```

Before running, edit the settings at the top of the script:

```python
DATASET_KEY = "NOAA"
REGIME = "S"
MAX_RESULTS = 350
FETCH_PER_ALIAS = 350
```

The script queries OpenAlex using dataset aliases, reconstructs abstracts from OpenAlex metadata, applies broad ML keyword filtering and regime-specific filtering, ranks records, and exports:

```text
openalex_<dataset>_<REGIME>_ml_top350.csv
openalex_<dataset>_<REGIME>_ml_top350.txt
```

The repository already includes retrieved CSV corpora in `query_outputs/`.

---

# Step 2: Parse models and model families

Run the model parser on the retrieved OpenAlex CSV files:

```bash
python parse_regime_papers_full_updated.py \
  --input-dir ./query_outputs/H/ \
  --input-glob "openalex_*_H_ml_top350.csv" \
  --regime H \
  --output-dir ./regime_parse_outputs/H/
```

Change `H` to `S`, `M`, or `T` as needed.

This script:

- filters papers using regime-specific regex libraries;
- extracts exact ML model mentions;
- maps exact models to broader model families;
- computes any-use model-family prevalence;
- assigns a dominant model when possible;
- exports audit logs and final tables.

Main outputs include:

```text
<REGIME>_paper_audit_log.csv
<REGIME>_summary_counts.csv
<REGIME>_any_use_table.csv
<REGIME>_primary_only_table.csv
<REGIME>_any_use_model_table.csv
<REGIME>_primary_only_model_table.csv
```

---

# Step 3: Parse pipeline architecture patterns

Run the architecture-pattern parser after the model audit logs have been generated:

```bash
python parse_pipeline_patterns_from_audit.py \
  --regime H \
  --audit-results-dir ./regime_parse_outputs \
  --openalex-base-dir ./query_outputs \
  --output-dir ./pipeline_parse_output
```

Change `H` to `S`, `M`, or `T` as needed.

This script identifies five major pipeline architecture categories:

- handcrafted feature pipelines;
- learned representation pipelines;
- fusion-centric architectures;
- sequential predictive pipelines;
- robust / transfer-aware pipelines.

Main outputs include:

```text
<REGIME>_pipeline_patterns_from_audit_prevalence_by_dataset.csv
<REGIME>_pipeline_patterns_from_audit_counts_by_dataset.csv
<REGIME>_pipeline_patterns_from_audit_coverage_by_dataset.csv
<REGIME>_pipeline_patterns_from_audit_audit_log.csv
<REGIME>_pipeline_patterns_from_audit_text_source_report.csv
```

---

## Interpretation of outputs

The main quantitative outputs use different denominators:

- **Any-use model-family tables**: papers that passed regime relevance filtering and contained at least one identifiable ML model. A paper may contribute to multiple families.
- **Primary-only tables**: papers for which a unique dominant model was identified. Each paper contributes to one family only.
- **Pipeline-pattern prevalence tables**: accessible papers with enough text for architecture-pattern assessment. A paper may contain multiple pipeline motifs.
- **Coverage tables**: track how many papers were accessible, assessable, and pattern-positive.

Because any-use and pipeline-pattern analyses allow multiple labels per paper, percentages do not necessarily sum to 100%.

---


---

# Step 4: Manual audit validation and taxonomy sensitivity analysis

Manual audit validation was performed on randomly sampled regime-specific audit subsets (50 papers per regime) drawn from the final analyzed corpora after computational relevance filtering. Manual annotations included dominant model families, dominant exact models, ambiguity flags, and all identifiable model families present within each paper.

Validation metrics include:

- dominant-family accuracy;
- ambiguity-aware dominant-family accuracy;
- ambiguity rates;
- micro-averaged precision, recall, and F1 for prevalence-oriented multi-label model-family extraction.

Random sample and retrieve audit papers for manual inspection using;

```bash
python sample_and_retrieve_audit_papers.py --regime S --audit-log ./regime_parse_outputs/S/S_paper_audit_log.csv --openalex-dir ./query_outputs/S --output-dir ./audit_papers/S
```

Taxonomy robustness analyses evaluate sensitivity to alternative coarser model-family definitions by simultaneously collapsing automated and manually annotated labels into broader aggregate categories.

Run taxonomy sensitivity analysis and compute validation metrics using:

```bash
python compute_validation_metrics.py --input H_sample_50.csv --output H_sensitivity_report.txt
```

where H_sample_50.csv is for instance the output file generated at the previous step, after it has been annotated manually. Already annoted files are available in the ./audit_papers/ directory

Sensitivity analyses include:

- collapsing multiple classical ML families into broader categories;
- merging ensemble-oriented families;
- merging survival and linear/statistical families;
- collapsing all non-deep-learning families into a single category.

The resulting reports summarize robustness of prevalence-oriented extraction metrics to alternative taxonomy definitions.


## Reproducibility notes

The framework is rule-based and uses curated keyword libraries, regex pattern sets, model-family mappings, heuristic dominant-model assignment rules, manually audited validation subsets, and taxonomy-sensitivity analyses. These rules and validation procedures are encoded directly in the source scripts and documented in the supplementary materials.

The outputs in this repository allow the analyses to be checked at multiple levels:

- retrieved OpenAlex records;
- parsed paper-level audit logs;
- detected model and family assignments;
- dominant-model assignment outcomes;
- architecture-pattern detections;
- prevalence/count tables;
- manually annotated audit subsets;
- validation metrics and ambiguity analyses;
- taxonomy-collapse sensitivity-analysis reports.

---

## Requirements

The core dependencies are listed in `requirements.txt`:

```text
pandas
requests
beautifulsoup4
PyPDF2
tqdm
```

---

## License

This repository is released under the MIT License.

---

## Citation

If you use this repository, please cite the associated manuscript once available.
