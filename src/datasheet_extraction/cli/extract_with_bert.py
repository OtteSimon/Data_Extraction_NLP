import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

from datasheet_extraction.data.hf_training_dataset import ID_TO_LABEL
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


def predict_entities(text, model, tokenizer, max_length):
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        is_split_into_words=False,
    )

    if len(tokenized["input_ids"][0]) > max_length:
        tokenized = {key: value[:, :max_length] for key, value in tokenized.items()}

    outputs = model(**tokenized)
    predictions = outputs.logits.argmax(dim=2)
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    labels = [ID_TO_LABEL[prediction.item()] for prediction in predictions[0]]
    return tokens, labels


def combine_entities(tokens, labels):
    entities = []
    current_entity = {"label_": "", "text": ""}

    for token, label in zip(tokens, labels):
        if label.startswith("B-") and current_entity["text"] == "":
            current_entity["label_"] = label[2:]
            append_token(current_entity, token)
        elif label.startswith("I-") and current_entity["label_"] == label[2:]:
            append_token(current_entity, token)
        elif label.startswith("I-") and current_entity["label_"] != label[2:] and current_entity["label_"] != "":
            entities.append({"label_": current_entity["label_"], "text": current_entity["text"]})
            current_entity = {"label_": "", "text": ""}
        elif label.startswith("O") and current_entity["label_"] != "":
            entities.append({"label_": current_entity["label_"], "text": current_entity["text"]})
            current_entity = {"label_": "", "text": ""}

    if current_entity["label_"] != "":
        entities.append({"label_": current_entity["label_"], "text": current_entity["text"]})

    return entities


def append_token(current_entity, token):
    if token.startswith("##"):
        current_entity["text"] += token[2:]
    elif token in {")", ".", ":", "/", "-", ","}:
        current_entity["text"] += token
    else:
        if current_entity["text"].endswith(("(", ".", "/")):
            current_entity["text"] += token
        else:
            current_entity["text"] += " " + token


def run_bert_model(text, ner_model, tokenizer, max_length):
    tokens, labels = predict_entities(text, ner_model, tokenizer, max_length)
    return combine_entities(tokens, labels)


def process_folder(folder_path, mds_model_path, sds_model_path, output_path, max_length):
    mds_model = AutoModelForTokenClassification.from_pretrained(mds_model_path)
    mds_tokenizer = AutoTokenizer.from_pretrained(mds_model_path)
    sds_model = AutoModelForTokenClassification.from_pretrained(sds_model_path)
    sds_tokenizer = AutoTokenizer.from_pretrained(sds_model_path)
    rows = []

    for pdf_path in sorted(Path(folder_path).iterdir()):
        if pdf_path.suffix.lower() != ".pdf":
            continue

        row = DEFAULT_COLUMNS.copy()
        row["Filename"] = pdf_path.name
        text, is_material_data_sheet = pdf_preprocessing.extract_and_preprocess(pdf_path=str(pdf_path))

        if is_material_data_sheet:
            entities = run_bert_model(text, mds_model, mds_tokenizer, max_length)
        else:
            entities = run_bert_model(text, sds_model, sds_tokenizer, max_length)

        row = entity_postprocessing.fill_bert_extraction_row(entities, row)
        rows.append(row)

    rows = entity_postprocessing.consolidate_document_pairs(rows)
    dataframe = pd.DataFrame(rows)
    dataframe.to_excel(output_path, index=False)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract structured entities from material and safety datasheet PDFs with a BERT model."
    )
    parser.add_argument("--input-folder", required=True, help="Folder that contains the PDF datasheets.")
    parser.add_argument("--mds-model", required=True, help="Path to the trained MDS BERT model.")
    parser.add_argument("--sds-model", required=True, help="Path to the trained SDS BERT model.")
    parser.add_argument(
        "--output",
        default="bert_output.xlsx",
        help="Excel file that will receive the extracted entities.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length per inference batch.")
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    process_folder(
        folder_path=arguments.input_folder,
        mds_model_path=arguments.mds_model,
        sds_model_path=arguments.sds_model,
        output_path=arguments.output,
        max_length=arguments.max_length,
    )
