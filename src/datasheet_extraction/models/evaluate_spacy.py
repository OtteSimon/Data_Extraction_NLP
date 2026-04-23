import argparse

import spacy

from datasheet_extraction.data.spacy_training_data import load_training_data
from datasheet_extraction.models.train_spacy import evaluate_ner
from datasheet_extraction.models.train_spacy import update_training_data


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained spaCy model on annotated workbook data.")
    parser.add_argument("--model-path", required=True, help="Path to the trained spaCy model.")
    parser.add_argument("--dataset-path", required=True, help="Path to the annotated Excel workbook.")
    parser.add_argument(
        "--evaluation-sheet",
        default="validation",
        help="Evaluation sheet name or alias. English aliases map to the original workbook sheets.",
    )
    parser.add_argument(
        "--skip-boundary-fix",
        action="store_true",
        help="Skip the entity boundary alignment pass before evaluation.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    evaluation_data = load_training_data(arguments.dataset_path, arguments.evaluation_sheet)
    if not arguments.skip_boundary_fix:
        evaluation_data = update_training_data(training_data=evaluation_data, nlp=spacy.blank("de"))

    nlp = spacy.load(arguments.model_path)
    scores = evaluate_ner(nlp, evaluation_data)
    print(f"Precision: {scores['ents_p']:.2f}")
    print(f"Recall: {scores['ents_r']:.2f}")
    print(f"F1 score: {scores['ents_f']:.2f}")
