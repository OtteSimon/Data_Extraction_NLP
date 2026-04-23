from datasheet_extraction.models.evaluate_bert import build_argument_parser
from datasheet_extraction.models.evaluate_bert import evaluate_model


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    metrics = evaluate_model(
        model_path=arguments.model_path,
        dataset_path=arguments.dataset_path,
        evaluation_sheet=arguments.evaluation_sheet,
        batch_size=arguments.batch_size,
    )
    print(metrics)
