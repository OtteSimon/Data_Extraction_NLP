import argparse
import os

import evaluate
from torch.optim import SGD
from transformers import AutoConfig
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from datasheet_extraction.data.hf_training_dataset import ID_TO_LABEL
from datasheet_extraction.data.hf_training_dataset import LABEL_LIST
from datasheet_extraction.data.hf_training_dataset import load_training_data


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=False,
    )
    labels = []
    for index, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(prediction_output):
    metric = evaluate.load("seqeval")
    labels = prediction_output.label_ids
    predictions = prediction_output.predictions.argmax(-1)

    true_labels = [[ID_TO_LABEL[label] for label in example if label != -100] for example in labels]
    true_predictions = [
        [ID_TO_LABEL[prediction] for prediction, label in zip(example, labels[index]) if label != -100]
        for index, example in enumerate(predictions)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def build_model(base_model_name, dropout):
    config = AutoConfig.from_pretrained(base_model_name)
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    return AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        config=config,
    )


def build_datasets(dataset_path, train_sheet, validation_sheet, tokenizer, validation_dataset_path=None):
    training_data = load_training_data(dataset_path, train_sheet, tokenizer=tokenizer)
    training_data = training_data.map(lambda batch: tokenize_and_align_labels(batch, tokenizer), batched=True).shuffle()

    validation_source = validation_dataset_path or dataset_path
    validation_data = load_training_data(validation_source, validation_sheet, tokenizer=tokenizer)
    validation_data = validation_data.map(lambda batch: tokenize_and_align_labels(batch, tokenizer), batched=True)
    return training_data, validation_data


def train_ner_model(
    tokenizer,
    training_data,
    validation_data,
    base_model_name,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    dropout,
    momentum=None,
):
    model = build_model(base_model_name, dropout)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer_kwargs = {
        "train_dataset": training_data,
        "eval_dataset": validation_data,
        "compute_metrics": compute_metrics,
        "model": model,
        "args": training_args,
    }

    if momentum is not None:
        optimizer = SGD(model.parameters(), lr=training_args.learning_rate, momentum=momentum)
        trainer_kwargs["optimizers"] = (optimizer, None)

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    evaluation = trainer.evaluate()
    print(f"Saved BERT model to {output_dir}")
    print(evaluation)
    return evaluation


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train a BERT-style NER model for MDS or SDS data.")
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
    parser.add_argument(
        "--base-model",
        default="bert-base-german-cased",
        help="Hugging Face model name used as the starting checkpoint.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the trained model will be saved.")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=0.00242, help="Learning rate for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.0064, help="Dropout rate for the transformer model.")
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optional SGD momentum. When omitted, the default Trainer optimizer is used.",
    )
    return parser


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

    print(f"Loaded {len(LABEL_LIST)} labels.")
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
