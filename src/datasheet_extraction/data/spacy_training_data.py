import re

import pandas as pd


TRAINING_SHEET_ALIASES = ["TrainingData", "Trainingsdaten"]
VALIDATION_SHEET_ALIASES = ["ValidationData", "Validierungsdaten"]
TEXT_COLUMN_ALIASES = ["Text", "text"]


def add_entity(label, value, text, entities):
    if value != "-1":
        value = value.strip()
        match = re.search(re.escape(value), text)
        if match:
            start = match.start()
            end = match.end()
            entities.append((start, end, label))


def add_entity_multiple_times(label, value, text, entities):
    if value != "-1":
        value = value.strip()
        for match in re.finditer(re.escape(value), text):
            start = match.start()
            end = match.end()
            entities.append((start, end, label))


def add_hazard_entities(hazard_text, text, entities):
    hazard_pattern = re.compile(r"(H\d{3})")
    end_patterns = {
        "+": re.compile(r"[+&]"),
        ".": re.compile(r"\."),
        "(": re.compile(r"\("),
        "H": hazard_pattern,
    }

    start = 0
    match = hazard_pattern.search(hazard_text, start)
    if not match:
        add_entity("HAZ", hazard_text, text, entities)
        return

    while start < len(hazard_text):
        match = hazard_pattern.search(hazard_text, start)
        if not match:
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


def load_training_data(excel_path, sheet_name):
    resolved_sheet_name = resolve_sheet_name(excel_path, sheet_name)
    dataframe = pd.read_excel(excel_path, sheet_name=resolved_sheet_name, dtype=str).fillna("-1")
    text_column = get_text_column_name(dataframe)
    training_data = []

    for _, row in dataframe.iterrows():
        text = row[text_column]
        entities = []

        add_entity("PROD_NAME", row["PROD_NAME"], text, entities)
        add_entity("MANU_NAME", row["MANU_NAME"], text, entities)
        add_entity_multiple_times("CAS", row["CAS"], text, entities)
        add_entity("DENSITY", row["DENSITY"], text, entities)
        add_entity("MOISTURE", row["MOISTURE"], text, entities)
        add_entity("PARTICLE_SIZE", row["PARTICLE_SIZE"], text, entities)
        add_entity("PH", row["PH"], text, entities)
        add_entity("MOL_WEIGHT", row["MOL_WEIGHT"], text, entities)
        if row["HAZ"] != "-1":
            add_hazard_entities(hazard_text=row["HAZ"], text=text, entities=entities)

        training_data.append((text, {"entities": entities}))

    return training_data
