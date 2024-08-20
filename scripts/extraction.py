"""
Workflow module for data preprocessing.

Raises:
    EmptyFileListException: If no files are left after filtering in merge_raw_data.
    OSError: If some provided file paths are invalid.
"""
# Module imports
import os
import argparse
from pathlib import Path

import pandas as pd
from prefect import task, flow
# Custom module imports
import constants
from exceptions import EmptyFileListException
from prefect_utils import generate_run_name
from data_cleaning import data_cleaning


@task(name="Merge raw data",
      description="Merges raw data into one file for further processing.",
      task_run_name=generate_run_name("merge-raw-data"),
      log_prints=True,
      tags=["preprocessing"])
def merge_raw_data(
        data_dir: str,
        out_file_path: str,
        *,
        data_dir_file_extensions: list,
        encoding='utf-8') -> str:
    """
    Merge contents of multiple files into one output file.

    Parameters:
        data_dir (str): Path to the raw input files.
        out_file_path (str): Full path to the output file (e.g. '/tmp/out.txt').
        encoding (str, optional): Encoding of the input files (default 'utf-8').

    Returns: 
        (str): Full path of the output file.
    """
    print("\nMerging raw data files...")
    # Full relative path for each raw file
    raw_files = sorted([os.path.join(data_dir, filename)
                       for filename in os.listdir(data_dir)])
    print("Raw files after path extensions:")
    print(raw_files)

    # Filter directories (in our case only files are necessary and allowed)
    raw_files_dir_filter = [
        file_path for file_path in raw_files if os.path.isfile(file_path)]
    if len(raw_files) != len(raw_files_dir_filter):
        print("Raw files after filtering directories:")
        print(raw_files_dir_filter)
    else:
        print("No directories had to be filtered from raw files")
    raw_files = raw_files_dir_filter

    # If specific conditions where provided (data_dir_file_extensions),
    # filter the raw_files list to only keep the permitted files
    if data_dir_file_extensions is not None and len(data_dir_file_extensions) > 0:
        raw_files_extension_filter = [filename for filename in raw_files if filename.endswith(
            tuple(data_dir_file_extensions))]
        if len(raw_files_extension_filter) != len(raw_files_dir_filter):
            print("Raw files after filtering by allowed extensions:")
            print(raw_files_extension_filter)
        else:
            print("No files had to be filtered by forbidden extension")
        raw_files = raw_files_extension_filter
    else:
        print("No allowed extensions where provided")

    # Perform merging
    if len(raw_files) == 0:
        raise EmptyFileListException("No raw_files left after filtering")
    with open(out_file_path, 'w', encoding=encoding) as merged:
        for idx, file in enumerate(raw_files):
            with open(file, 'r', encoding=encoding) as current_raw_file:
                # The header information is only needed ones
                if idx == 0:
                    merged.writelines(current_raw_file.readlines())
                else:
                    merged.writelines(current_raw_file.readlines()[1:])
                print(f"Finished processing file {file}")

    print(f"Files merged in {out_file_path}")
    return out_file_path


@task(name="Convert to csv",
      description="Converts a specific invoice data text file to a csv file.",
      task_run_name=generate_run_name("txt-to-csv"),
      log_prints=True,
      tags=["preprocessing"])
def txt_to_csv(input_file_path: str, out_file_path: str = None, encoding='utf-8') -> str:
    """
    Converts a single text file with tab as delimiter to a csv file.

    Parameters:
        input_file_path (str): Full path to the text file to be converted.
        out_file_path (str, optional): Full path to output file.
        Default is using the input_file_path with .csv extension (Default None).

    Exceptions:
        OSError: Raised if input_file_path does not exist or the specified 
        out_file_path is invalid (directory does not exist)

    Returns:
        (str): Full path to the created csv file
    """
    print("\nConverting text file to csv...")
    # Check input file path
    if not os.path.isfile(input_file_path):
        raise OSError(
            constants.error_messages['os_error'].format(input_file_path))

    # Check output file path or set it if no parameter was set
    if out_file_path is None or out_file_path == "":
        out_file_path = os.path.join(os.path.dirname(
            input_file_path), Path(input_file_path).stem + '.csv')
        print(out_file_path)
    elif not os.path.exists(os.path.dirname(out_file_path)):
        raise OSError(
            f"Path '{os.path.dirname(out_file_path)}' does not exist.")

    # Open and process text file to prepare csv
    with open(input_file_path, 'r', encoding=encoding) as merged_input:
        data = merged_input.readlines()
        # Replace unicode character U+00B7 (Middle Dot) with hyphen and split by tab delimiter
        # Replace semicolon with slash to avoid splitting the data
        # Results in a list of lists as base for converting to a csv file
        data = [x.strip().replace("·", "-").replace(";", "/").split("\t")
                for x in data]
        # Join the inner lists with semicolon as csv delimiter
        data = [';'.join(x) for x in data]

        # Remove rows which do not match the column count
        data_lines = len(data)
        print(f"The data has {data_lines} rows.")
        column_count = len(data[0].split(";"))
        print(f"The data has {column_count} columns.")
        print("Skipping unsufficient rows...")
        data = [line for line in data if len(line.split(";")) == column_count]
        print(f"Removed {data_lines - len(data)} rows.")

        # Write to output csv
        with open(out_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write("\n".join(data))

    return out_file_path


@task(name="Export output",
      description="Export the processed data.",
      task_run_name=generate_run_name("export-output"),
      log_prints=True,
      tags=["preprocessing"])
def export_output(df: pd.DataFrame, out_file_path: str) -> str:
    """Export the preprocessed data to a parquet file.

    Args:
        df (pd.DataFrame): Dataframe to be exported.
        out_file_path (str): Path to output.
        out_file_name (str, optional): Name of the output file. Defaults to "data_cleaned.parquet".

    Returns:
        str: Path to output
    """
    df.to_parquet(out_file_path, engine='fastparquet')

    return out_file_path


@flow(name="data-preprocessing-flow", flow_run_name=generate_run_name("preprocessing-flow"))
def data_preprocessing_flow(flow_args):
    """Prefect flow for data preprocessing.

    Args:
        flow_args (_type_): Arguments provided through argparse.
    """
    merged_file_path = merge_raw_data(
        flow_args.data_dir,
        os.path.join(
            flow_args.tmp_dir, constants.extraction_defaults['raw_data_merged_file_name']),
        data_dir_file_extensions=flow_args.file_extensions,
        encoding=flow_args.encoding)

    converted_file_path = txt_to_csv(
        merged_file_path, flow_args.output_file, encoding=flow_args.encoding)

    df_cleaned_file = data_cleaning(
        converted_file_path, encoding='utf-8')

    exported_path = export_output(
        df_cleaned_file, flow_args.export_path)

    return exported_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
                        help='Path to the parquet output file. \
                            (default: /tmp/data_cleaned.parquet)')
    args = parser.parse_args()

    data_preprocessing_flow(args)
