import argparse
from datetime import datetime
from prefect import flow

from feature_extraction import rfm_feature_extraction_flow
from extraction import data_preprocessing_flow
from train import train_flow
from prefect_utils import generate_run_name
import constants
from mlflow_setup import setup_mlflow, log_parameters, end_mlflow_run


@flow(name="full-kfz-rfm-workflow",
      description="Full KFZ RFM workflow.",
      flow_run_name=generate_run_name("full-kfz-rfm-workflow"))
def full_kfz_rfm_workflow(flow_args):
    """Full KFZ RFM workflow."""
    # Setup mlflow
    try:
        setup_mlflow(flow_args.mlflow_tracking_uri,
                     flow_args.mlflow_experiment)
        log_parameters()
        cleaned_parquet_path = data_preprocessing_flow(flow_args)
        rfm_feature_extraction_flow(
            flow_args, cleaned_parquet_path)
        train_flow(flow_args)
    except Exception as e:
        print(f"An error occurred: {e}")
        end_mlflow_run(status='FAILED')
    finally:
        end_mlflow_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mlflow_tracking_uri', type=str,
                        help='MLflow tracking URI.', required=True)
    parser.add_argument('--mlflow_experiment', type=str,
                        help='MLflow experiment name.', required=True)

    parser.add_argument('--data_dir', type=str,
                        default=constants.extraction_defaults['raw_data_path'],
                        help='Path to the raw data directory. (default: ./data)')
    parser.add_argument('--tmp_dir', type=str,
                        default=constants.extraction_defaults['tmp_file_path'],
                        help='Path to the temporary directory. (default: /tmp)')
    parser.add_argument('--file_extensions', nargs='+',
                        default=constants.extraction_defaults['raw_data_file_extensions'],
                        help='List of file extensions to be merged. (default: .TXT)')
    parser.add_argument('--encoding', type=str,
                        default=constants.extraction_defaults['raw_data_encoding'],
                        help='Encoding of the raw data files. (default: iso8859_15)')
    parser.add_argument('--output_file', type=str,
                        default=constants.extraction_defaults['output_file_path_csv'],
                        help='Name of the output file in the temporary directory. \
                            (default: data_cleaned.csv)')
    parser.add_argument('--export_path', type=str,
                        default=constants.extraction_defaults['output_file_path_parquet'],
                        help="Path to the parquet file.")
    parser.add_argument('--parquet_input_file', type=str,
                        default=constants.extraction_defaults['output_file_path_parquet'],
                        help="Path to the parquet file.")
    parser.add_argument('--rfm_output_file_path', type=str,
                        default=constants.extraction_defaults['rfm_feature_file_path'],
                        help="Name of the output file.")
    parser.add_argument('--output_train_data_path', type=str,
                        default=constants.extraction_defaults['output_train_data_path'],
                        help='Path to the output training data. (default: /tmp/train_data.parquet)')
    parser.add_argument('--date_from', type=lambda x: datetime.strptime(
        x, '%Y-%m-%d').date(), default=None, help="Start date for the analysis.")
    parser.add_argument('--date_to', type=lambda x: datetime.strptime(
        x, '%Y-%m-%d').date(), default=None, help="End date for the analysis.")

    args = parser.parse_args()

    full_kfz_rfm_workflow(args)
