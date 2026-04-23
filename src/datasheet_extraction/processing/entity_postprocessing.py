from collections import Counter
from collections import defaultdict


def fill_extraction_row(entities, row):
    entity_counts = defaultdict(Counter)

    for entity in entities:
        if entity.label_ == "HAZ":
            if row["HAZ"] == "":
                row[entity.label_] += entity.text
            else:
                row[entity.label_] += " " + entity.text
        elif entity.label_ in row:
            entity_counts[entity.label_][entity.text] += 1

    for label, texts in entity_counts.items():
        if texts:
            most_common_text, _ = texts.most_common(1)[0]
            row[label] = most_common_text

    return row


def fill_bert_extraction_row(entities, row):
    entity_counts = defaultdict(Counter)

    for entity in entities:
        if entity["label_"] == "HAZ":
            if row["HAZ"] == "":
                row[entity["label_"]] += entity["text"]
            else:
                row[entity["label_"]] += " " + entity["text"]
        elif entity["label_"] in row:
            entity_counts[entity["label_"]][entity["text"]] += 1

    for label, texts in entity_counts.items():
        if texts:
            most_common_text, _ = texts.most_common(1)[0]
            row[label] = most_common_text

    return row


def consolidate_document_pairs(rows):
    consolidated_rows = {}

    for row in rows:
        base_filename = row["Filename"].lower()
        base_filename = base_filename.replace("_mds", "")
        base_filename = base_filename.replace("_mdb", "")

        if base_filename not in consolidated_rows:
            consolidated_rows[base_filename] = row.copy()
            continue

        for key, value in row.items():
            if key != "Filename" and value and consolidated_rows[base_filename][key] == "":
                consolidated_rows[base_filename][key] = value

    return list(consolidated_rows.values())
