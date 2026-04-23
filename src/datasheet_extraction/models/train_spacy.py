import argparse
import random

import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
from spacy.util import minibatch

from datasheet_extraction.data.spacy_training_data import load_training_data


SPACY_LABELS = [
    "PROD_NAME",
    "MANU_NAME",
    "CAS",
    "HAZ",
    "DENSITY",
    "MOISTURE",
    "PARTICLE_SIZE",
    "PH",
    "MOL_WEIGHT",
]


def train_ner_model(training_data, validation_data, output_dir, epochs, batch_size, learning_rate, dropout):
    nlp = spacy.load("de_core_news_sm")
    ner_pipe = nlp.get_pipe("ner")

    for label in SPACY_LABELS:
        ner_pipe.add_label(label)

    optimizer = nlp.resume_training()
    optimizer.learn_rate = learning_rate

    for epoch in range(epochs):
        losses = {}
        random.shuffle(training_data)

        for batch in minibatch(training_data, size=batch_size):
            for text, annotations in batch:
                document = nlp.make_doc(text)
                example = Example.from_dict(document, annotations)
                nlp.update([example], losses=losses, drop=dropout)

        scores = evaluate_ner(nlp, validation_data)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={losses.get('ner', 0.0):.3f}, "
            f"precision={scores['ents_p']:.2f}, "
            f"recall={scores['ents_r']:.2f}, "
            f"f1={scores['ents_f']:.2f}"
        )

    nlp.to_disk(output_dir)
    print(f"Saved spaCy model to {output_dir}")
    return nlp


def update_training_data(training_data, nlp):
    updated_data = []
    for text, annotations in training_data:
        document = nlp(text)
        updated_entities = adjust_entity_boundaries(document, annotations["entities"])
        updated_data.append((text, {"entities": updated_entities}))
    return updated_data


def adjust_entity_boundaries(document, entities):
    adjusted_entities = []
    misaligned_entities = check_entity_positions(text=document.text, entities=entities, document=document)
    if not misaligned_entities:
        return entities

    for start, end, label in entities:
        entity_text = document.text[start:end]
        token_start = None
        token_end = None
        accumulated_text = ""

        for token in document:
            if token.idx >= start and token_start is None:
                token_start = token.i
            if token_start is not None:
                accumulated_text += token.text_with_ws
                token_end = token.i + 1
                if accumulated_text.strip().startswith(entity_text):
                    break

        if token_start is not None and token_end is not None and accumulated_text.strip().startswith(entity_text):
            adjusted_start = document[token_start].idx
            adjusted_end = document[token_end - 1].idx + len(document[token_end - 1].text)
            adjusted_entities.append((adjusted_start, adjusted_end, label))
        else:
            print(
                f"Could not fully align entity '{entity_text}' "
                f"at position {start}-{end} with token boundaries."
            )

    return adjusted_entities


def check_entity_positions(text, entities, document):
    misaligned_entities = []
    for start, end, label in entities:
        entity_span = document.char_span(start, end, label=label)
        if entity_span is None:
            print(f"Misaligned entity: {label}, text='{text[start:end]}', start={start}, end={end}")
            misaligned_entities.append((start, end, label))
    return misaligned_entities


def validate_training_data(nlp, training_data):
    for text, annotations in training_data:
        document = nlp.make_doc(text)
        entities = annotations.get("entities", [])
        try:
            offsets_to_biluo_tags(document, entities)
        except ValueError as error:
            print(f"Validation error in text '{text[:80]}...': {error}")
            check_entity_positions(text=text, entities=entities, document=document)
        else:
            print(f"Validated entities for text '{text[:80]}...'")


def evaluate_ner(nlp, examples):
    scorer = Scorer()
    example_list = []

    for input_text, annotation in examples:
        document = nlp(input_text)
        example_list.append(Example.from_dict(document, annotation))

    return scorer.score(example_list)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train a spaCy NER model for MDS or SDS data.")
    parser.add_argument("--dataset-path", required=True, help="Path to the annotated Excel workbook.")
    parser.add_argument(
        "--validation-dataset-path",
        default=None,
        help="Optional path to a separate validation workbook. Defaults to --dataset-path.",
    )
    parser.add_argument(
        "--train-sheet",
        default="training",
        help="Training sheet name or alias. English aliases map to the original workbook sheets.",
    )
    parser.add_argument(
        "--validation-sheet",
        default="validation",
        help="Validation sheet name or alias. English aliases map to the original workbook sheets.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the trained model will be saved.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for spaCy training.")
    parser.add_argument("--learning-rate", type=float, default=0.002, help="Learning rate for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate during training.")
    parser.add_argument(
        "--skip-boundary-fix",
        action="store_true",
        help="Skip the entity boundary alignment pass before training.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate entity boundaries and exit without training.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()

    training_data = load_training_data(arguments.dataset_path, arguments.train_sheet)
    validation_dataset_path = arguments.validation_dataset_path or arguments.dataset_path
    validation_data = load_training_data(validation_dataset_path, arguments.validation_sheet)

    if not arguments.skip_boundary_fix:
        blank_nlp = spacy.blank("de")
        training_data = update_training_data(training_data=training_data, nlp=blank_nlp)
        validation_data = update_training_data(training_data=validation_data, nlp=blank_nlp)

    if arguments.validate_only:
        validate_training_data(spacy.blank("de"), training_data)
    else:
        train_ner_model(
            training_data=training_data,
            validation_data=validation_data,
            output_dir=arguments.output_dir,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            dropout=arguments.dropout,
        )
