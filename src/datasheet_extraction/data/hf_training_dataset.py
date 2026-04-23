import re

import pandas as pd
from datasets import Dataset


LABEL_LIST = [
    "O",
    "B-PROD_NAME",
    "I-PROD_NAME",
    "B-MANU_NAME",
    "I-MANU_NAME",
    "B-CAS",
    "I-CAS",
    "B-HAZ",
    "I-HAZ",
    "B-MOL_WEIGHT",
    "I-MOL_WEIGHT",
    "B-MELT_POINT",
    "I-MELT_POINT",
    "B-PH",
    "I-PH",
    "B-DENSITY",
    "I-DENSITY",
    "B-PARTICLE_SIZE",
    "I-PARTICLE_SIZE",
    "B-MOISTURE",
    "I-MOISTURE",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {index: label for index, label in enumerate(LABEL_LIST)}

TRAINING_SHEET_ALIASES = ["TrainingData", "Trainingsdaten"]
VALIDATION_SHEET_ALIASES = ["ValidationData", "Validierungsdaten"]
TEXT_COLUMN_ALIASES = ["Text", "text"]


def resolve_sheet_name(excel_path, requested_sheet):
    workbook = pd.ExcelFile(excel_path)
    available_sheets = workbook.sheet_names

    if requested_sheet in available_sheets:
        return requested_sheet

    lowered_sheet = requested_sheet.lower()
    if lowered_sheet in {"training", "trainingdata"}:
        aliases = TRAINING_SHEET_ALIASES
    elif lowered_sheet in {"validation", "validationdata"}:
        aliases = VALIDATION_SHEET_ALIASES
    else:
        aliases = [requested_sheet]

    for alias in aliases:
        if alias in available_sheets:
            return alias

    raise ValueError(
        f"Could not resolve sheet '{requested_sheet}' in '{excel_path}'. "
        f"Available sheets: {available_sheets}"
    )


def get_text_column_name(dataframe):
    for alias in TEXT_COLUMN_ALIASES:
        if alias in dataframe.columns:
            return alias
    raise ValueError("The workbook does not contain a supported text column.")


def load_training_data(excel_path, sheet_name, tokenizer):
    resolved_sheet_name = resolve_sheet_name(excel_path, sheet_name)
    dataframe = pd.read_excel(excel_path, sheet_name=resolved_sheet_name, dtype=str).fillna("-1")
    text_column = get_text_column_name(dataframe)
    data = dataframe.apply(process_row, axis=1, text_column=text_column).tolist()
    return convert_to_hf_format(data=data, tokenizer=tokenizer)


def add_entity(label, value, text, entities):
    if value != "-1":
        value = value.strip()
        match = re.search(re.escape(value), text)
        if match:
            start = match.start()
            end = match.end()
            entities.append({"start": start, "end": end, "label": label})


def add_entity_multiple_times(label, value, text, entities):
    if value != "-1":
        value = value.strip()
        for match in re.finditer(re.escape(value), text):
            start = match.start()
            end = match.end()
            entities.append({"start": start, "end": end, "label": label})


def add_hazard_entities(hazard_text, text, entities):
    hazard_pattern = re.compile(r"(H\d{3})")
    end_patterns = {
        "+": re.compile(r"[+&]"),
        ".": re.compile(r"\."),
        "(": re.compile(r"\("),
        "H": hazard_pattern,
    }

    start = 0
    while start < len(hazard_text):
        match = hazard_pattern.search(hazard_text, start)
        if not match:
            add_entity("HAZ", hazard_text, text, entities)
            break

        hazard_start = match.start()
        possible_ends = [
            (pattern.search(hazard_text, hazard_start + 1).start(), end_type)
            for end_type, pattern in end_patterns.items()
            if pattern.search(hazard_text, hazard_start + 1)
        ]

        if possible_ends:
            hazard_end, end_type = min(possible_ends, key=lambda item: item[0])
            if end_type == "+":
                next_dot = end_patterns["."].search(hazard_text, hazard_end)
                if next_dot:
                    hazard_end = next_dot.end()
        else:
            hazard_end = len(hazard_text)

        add_entity("HAZ", hazard_text[hazard_start:hazard_end], text, entities)
        start = hazard_end


def process_row(row, text_column):
    text = row[text_column]
    entities = []

    for column in [
        "PROD_NAME",
        "MANU_NAME",
        "MOL_WEIGHT",
        "MELT_POINT",
        "PH",
        "DENSITY",
        "PARTICLE_SIZE",
        "MOISTURE",
    ]:
        if pd.notna(row[column]) and row[column] != "-1":
            add_entity(column, row[column], text, entities)

    if pd.notna(row["CAS"]) and row["CAS"] != "-1":
        add_entity_multiple_times("CAS", row["CAS"], text, entities)

    if pd.notna(row["HAZ"]) and row["HAZ"] != "-1":
        add_hazard_entities(row["HAZ"], text, entities)

    return {"text": text, "entities": entities}


def convert_to_hf_format(data, tokenizer):
    texts = []
    ner_tags = []
    for item in data:
        text = item["text"]
        tokenized_input = tokenizer(
            text,
            truncation=False,
            padding="max_length",
            max_length=512,
            is_split_into_words=False,
        )
        labels = ["O"] * len(tokenized_input["input_ids"])

        for entity in item["entities"]:
            start, end, label = entity["start"], entity["end"], entity["label"]
            token_start = tokenized_input.char_to_token(start)
            token_end = tokenized_input.char_to_token(end - 1)
            if token_start is not None and token_end is not None:
                labels[token_start] = f"B-{label}"
                for token_index in range(token_start + 1, token_end + 1):
                    labels[token_index] = f"I-{label}"

        texts.append(text)
        ner_tags.append([LABEL_TO_ID[label] for label in labels])

    return Dataset.from_dict({"text": texts, "ner_tags": ner_tags})
