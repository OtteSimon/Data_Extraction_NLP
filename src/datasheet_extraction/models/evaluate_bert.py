import argparse
import tempfile

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from datasheet_extraction.models.train_bert import build_datasets
from datasheet_extraction.models.train_bert import compute_metrics


def evaluate_model(model_path, dataset_path, evaluation_sheet, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _, evaluation_data = build_datasets(
        dataset_path=dataset_path,
        train_sheet=evaluation_sheet,
        validation_sheet=evaluation_sheet,
        tokenizer=tokenizer,
    )

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=evaluation_data,
            compute_metrics=compute_metrics,
        )
        return trainer.evaluate()


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained BERT model on annotated workbook data.")
    parser.add_argument("--model-path", required=True, help="Path to the trained BERT model.")
    parser.add_argument("--dataset-path", required=True, help="Path to the annotated Excel workbook.")
    parser.add_argument(
        "--evaluation-sheet",
        default="validation",
        help="Evaluation sheet name or alias. English aliases map to the original workbook sheets.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    metrics = evaluate_model(
        model_path=arguments.model_path,
        dataset_path=arguments.dataset_path,
        evaluation_sheet=arguments.evaluation_sheet,
        batch_size=arguments.batch_size,
    )
    print(metrics)
