from datasheet_extraction.models.train_spacy import build_argument_parser
from datasheet_extraction.models.train_spacy import train_ner_model
from datasheet_extraction.models.train_spacy import update_training_data
from datasheet_extraction.data.spacy_training_data import load_training_data

import spacy


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
        from datasheet_extraction.models.train_spacy import validate_training_data
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
