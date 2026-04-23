import argparse
from pathlib import Path

import pandas as pd
import spacy

from datasheet_extraction.processing import entity_postprocessing
from datasheet_extraction.processing import pdf_preprocessing


DEFAULT_COLUMNS = {
    "Filename": "",
    "PROD_NAME": "",
    "MANU_NAME": "",
    "CAS": "",
    "HAZ": "",
    "MELT_POINT": "",
    "DENSITY": "",
    "MOISTURE": "",
    "PARTICLE_SIZE": "",
    "PH": "",
    "MOL_WEIGHT": "",
}


def run_spacy_model(text, ner_model):
    document = ner_model(text)
    return document.ents


def process_folder(folder_path, mds_model_path, sds_model_path, output_path):
    mds_model = spacy.load(mds_model_path)
    sds_model = spacy.load(sds_model_path)
    rows = []

    for pdf_path in sorted(Path(folder_path).iterdir()):
        if pdf_path.suffix.lower() != ".pdf":
            continue

        row = DEFAULT_COLUMNS.copy()
        row["Filename"] = pdf_path.name
        text, is_material_data_sheet = pdf_preprocessing.extract_and_preprocess(pdf_path=str(pdf_path))

        if is_material_data_sheet:
            entities = run_spacy_model(text, mds_model)
        else:
            entities = run_spacy_model(text, sds_model)

        row = entity_postprocessing.fill_extraction_row(entities, row)
        rows.append(row)

    rows = entity_postprocessing.consolidate_document_pairs(rows)
    dataframe = pd.DataFrame(rows)
    dataframe.to_excel(output_path, index=False)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract structured entities from material and safety datasheet PDFs with a spaCy model."
    )
    parser.add_argument("--input-folder", required=True, help="Folder that contains the PDF datasheets.")
    parser.add_argument("--mds-model", required=True, help="Path to the trained MDS spaCy model.")
    parser.add_argument("--sds-model", required=True, help="Path to the trained SDS spaCy model.")
    parser.add_argument(
        "--output",
        default="spacy_output.xlsx",
        help="Excel file that will receive the extracted entities.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    process_folder(
        folder_path=arguments.input_folder,
        mds_model_path=arguments.mds_model,
        sds_model_path=arguments.sds_model,
        output_path=arguments.output,
    )
