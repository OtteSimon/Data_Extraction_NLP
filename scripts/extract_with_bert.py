from datasheet_extraction.cli.extract_with_bert import build_argument_parser
from datasheet_extraction.cli.extract_with_bert import process_folder


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    process_folder(
        folder_path=arguments.input_folder,
        mds_model_path=arguments.mds_model,
        sds_model_path=arguments.sds_model,
        output_path=arguments.output,
        max_length=arguments.max_length,
    )
