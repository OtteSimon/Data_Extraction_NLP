from transformers import AutoTokenizer

from datasheet_extraction.models.train_bert import build_argument_parser
from datasheet_extraction.models.train_bert import build_datasets
from datasheet_extraction.models.train_bert import train_ner_model


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)
    training_data, validation_data = build_datasets(
        dataset_path=arguments.dataset_path,
        train_sheet=arguments.train_sheet,
        validation_sheet=arguments.validation_sheet,
        tokenizer=tokenizer,
        validation_dataset_path=arguments.validation_dataset_path,
    )
    train_ner_model(
        tokenizer=tokenizer,
        training_data=training_data,
        validation_data=validation_data,
        base_model_name=arguments.base_model,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        dropout=arguments.dropout,
        momentum=arguments.momentum,
    )
