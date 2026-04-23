# Material Datasheet Entity Extraction

This repository contains the code and public research data for a bachelor thesis project on automated entity extraction from chemistry-related PDF datasheets.

The project processes two document types:

- `SDS`: Safety Data Sheet
- `MDS`: Material Data Sheet

The extraction workflow preprocesses PDFs, translates English text to German when needed, applies named-entity recognition models, and exports structured results to Excel.

The paper draft positions the spaCy-based pipeline as the main result to emphasize publicly. The transformer-based BERT path remains available for reproducibility, but the spaCy models are the stronger and more presentation-ready extraction workflow in the current project state.

## Why This Repository Exists

The project targets a concrete problem in battery cell production: relevant material properties are present in supplier-provided safety and material data sheets, but the documents differ strongly in terminology, layout, and formatting. This repository demonstrates a domain-specific NLP pipeline that converts those heterogeneous PDFs into structured material information suitable for downstream data management.

In line with the paper draft, this repository should be read as a technical feasibility study and prototype implementation for structured material data extraction, not as a finished production system.

## Repository Scope

The public repository intentionally keeps only the maintained project surface:

- PDF preprocessing
- training data preparation
- spaCy training and evaluation
- BERT training and evaluation
- inference scripts for batch extraction
- public training and validation data

The following thesis artifacts are intentionally excluded:

- plotting scripts
- plotted images
- training log CSV files
- local environment folders
- caches and machine-specific files

## Recommended Model Path

If you want to try the project quickly, use the spaCy workflow first.

- The public narrative of the project is centered on the spaCy pipeline.
- In the current validation setup, the best spaCy models clearly outperform the saved BERT checkpoints.
- The spaCy path is the one to foreground in demos, screenshots, and public explanation.

Recent validation checks in this public repo gave the following results for the saved best models:

- spaCy SDS model: precision `0.86`, recall `0.85`, F1 `0.85`
- spaCy MDS model: precision `0.86`, recall `0.75`, F1 `0.80`
- BERT SDS model: precision `0.14`, recall `0.17`, F1 `0.16`
- BERT MDS model: precision `0.22`, recall `0.27`, F1 `0.24`

## Repository Layout

```text
.
├── data/
│   ├── training/
│   │   ├── mds_training_data.xlsx
│   │   ├── sds_training_data.xlsx
│   │   ├── mds_pdfs/
│   │   └── sds_pdfs/
│   └── validation/
│       ├── mds_validation_data.xlsx
│       ├── sds_validation_data.xlsx
│       └── pdfs/
├── scripts/
├── src/
│   └── datasheet_extraction/
│       ├── cli/
│       ├── data/
│       ├── models/
│       └── processing/
├── models/
├── pyproject.toml
└── requirements.txt
```

`src/` contains the reusable Python package. `scripts/` contains the command-line entrypoints used in the examples below. `models/` is kept as a placeholder for trained checkpoints and is ignored by Git so locally trained models do not clutter repository history.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

## Data Notes

The annotated Excel workbooks still use the original sheet names from the thesis material. Training and validation are now stored in separate public workbooks, while the scripts expose English sheet aliases:

- `training` maps to `TrainingData` or `Trainingsdaten`
- `validation` maps to `ValidationData` or `Validierungsdaten`

That keeps the public interface English without rewriting the underlying annotation files.

## Run Extraction

### spaCy

```bash
python scripts/extract_with_spacy.py \
  --input-folder data/validation/pdfs \
  --mds-model models/spacy_mds_model \
  --sds-model models/spacy_sds_model \
  --output spacy_output.xlsx
```

### BERT

```bash
python scripts/extract_with_bert.py \
  --input-folder data/validation/pdfs \
  --mds-model models/bert_mds_model \
  --sds-model models/bert_sds_model \
  --output bert_output.xlsx
```

For public-facing use, the spaCy extraction path is the recommended default.

## Train Models

### Train a spaCy model

Example for SDS:

```bash
python scripts/train_spacy_model.py \
  --dataset-path data/training/sds_training_data.xlsx \
  --validation-dataset-path data/validation/sds_validation_data.xlsx \
  --train-sheet training \
  --validation-sheet validation \
  --output-dir models/spacy_sds_model
```

Example for MDS:

```bash
python scripts/train_spacy_model.py \
  --dataset-path data/training/mds_training_data.xlsx \
  --validation-dataset-path data/validation/mds_validation_data.xlsx \
  --train-sheet training \
  --validation-sheet validation \
  --output-dir models/spacy_mds_model
```

### Train a BERT model

Example for SDS:

```bash
python scripts/train_bert_model.py \
  --dataset-path data/training/sds_training_data.xlsx \
  --validation-dataset-path data/validation/sds_validation_data.xlsx \
  --train-sheet training \
  --validation-sheet validation \
  --base-model bert-base-german-cased \
  --output-dir models/bert_sds_model
```

Example for MDS:

```bash
python scripts/train_bert_model.py \
  --dataset-path data/training/mds_training_data.xlsx \
  --validation-dataset-path data/validation/mds_validation_data.xlsx \
  --train-sheet training \
  --validation-sheet validation \
  --base-model bert-base-german-cased \
  --output-dir models/bert_mds_model
```

## Evaluate Models

### Evaluate a spaCy model

```bash
python scripts/evaluate_spacy_model.py \
  --model-path models/spacy_sds_model \
  --dataset-path data/validation/sds_validation_data.xlsx \
  --evaluation-sheet validation
```

### Evaluate a BERT model

```bash
python scripts/evaluate_bert_model.py \
  --model-path models/bert_sds_model \
  --dataset-path data/validation/sds_validation_data.xlsx \
  --evaluation-sheet validation
```

## Important Implementation Detail

The preprocessing pipeline translates English source text to German before inference and training-related comparison steps, because the original models and annotations were created for German text.

## Main Takeaway

This repository is best presented as a domain-specific extraction pipeline for structured material data management in battery cell production. In the current state of the work, the spaCy-based models are the primary extraction path to highlight in demos, documentation, and public communication.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
