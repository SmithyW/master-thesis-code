import argparse
from datetime import datetime, date
import pandas as pd
from prefect import flow, task

from prefect_utils import generate_run_name
from extraction import export_output
import constants


@task(name="Parquet to DataFrame",
      description="Load parquet file to a pandas dataframe.",
      task_run_name=generate_run_name("parquet-to-df"))
def parquet_to_df(file_path: str) -> pd.DataFrame:
    """Load parquet file to a pandas dataframe.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: DataFrame loaded from the parquet file.
    """
    return pd.read_parquet(file_path, engine='fastparquet')


@task(name="Remove TUEV rows",
      description="Remove rows with TUEV in the description.",
      task_run_name=generate_run_name("remove-tuev-rows"))
def remove_tuev_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with item_ids associated with TUEV checks.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    tuev_item_ids = ["11109", "11110", "11111", "11200", "33333", "S33333"]
    df_with_tuev = df[df["item_id"].isin(tuev_item_ids) == True]
    df_tuev_invoices = df_with_tuev["invoice_num"].unique()

    df_without_tuev = df[~df["invoice_num"].isin(df_tuev_invoices)]

    return df_without_tuev


@task(name="RFM data preparation.",
      description="Prepare data for RFM feature extraction.",
      task_run_name=generate_run_name("rfm-data-preparation"))
def rfm_data_preparation(df: pd.DataFrame, date_from: type[date], date_to: type[date]) -> pd.DataFrame:
    """Prepare data for RFM feature extraction.

    Args:
        df (pd.DataFrame): DataFrame to be prepared.

    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    # Only the total sum of each invoice is relevant for the analysis, so we can exclude the other rows
    df_total = df[df["is_total"] == True]
    df_total.reset_index(drop=True, inplace=True)
    date_col = pd.to_datetime(df_total["date"])
    df_filtered = df_total[(date_col.dt.date >= date_from) & (
        date_col.dt.date <= date_to)]

    # Aggregate same day invoices
    df_total_aggregated = df_filtered.groupby(["customer_id", "date"]).agg({
        "total": "sum"}).reset_index()

    df_rfm_base = df_total_aggregated.drop(df_total_aggregated.columns.difference([
                                           "date", "customer_id", "total"]), axis=1)
    return df_rfm_base


@task(name="Recency feature extraction",
      description="Extract recency feature.",
      task_run_name=generate_run_name("recency-feature-extraction"))
def recency_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract recency feature."""
    df_last_visit = df.groupby('customer_id').date.max().reset_index()
    df_last_visit.columns = ['customer_id', 'last_visit']
    df_last_visit['recency'] = (
        df_last_visit['last_visit'].max() - df_last_visit['last_visit']).dt.days
    return df_last_visit.drop('last_visit', axis=1)


@task(name="Frequency feature extraction",
      description="Extract frequency feature.",
      task_run_name=generate_run_name("frequency-feature-extraction"))
def frequency_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract frequency feature."""
    df_frequency = df.groupby('customer_id').date.count().reset_index()
    df_frequency.columns = ['customer_id', 'frequency']
    df_frequency.drop(
        df_frequency[df_frequency['frequency'] == 1].index, inplace=True)

    return df_frequency


@task(name="Monetary feature extraction",
      description="Extract monetary feature.",
      task_run_name=generate_run_name("monetary-feature-extraction"))
def monetary_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract monetary feature."""
    df_monetary = df.groupby('customer_id').total.sum().reset_index()
    df_monetary.columns = ['customer_id', 'monetary']

    return df_monetary


@task(name="Merge RFM features",
      description="Merge RFM features into a single DataFrame.",
      task_run_name=generate_run_name("merge-rfm-features"))
def merge_rfm_features(df_rfm_base: pd.DataFrame, df_recency: pd.DataFrame,
                       df_frequency: pd.DataFrame, df_monetary: pd.DataFrame) -> pd.DataFrame:
    """Merge RFM features into a single DataFrame."""
    df_customer = pd.DataFrame(
        df_rfm_base['customer_id'].unique(), columns=['customer_id'])

    df_rfm = df_customer.merge(df_recency, on='customer_id')
    df_rfm = df_rfm.merge(df_frequency, on='customer_id')
    df_rfm = df_rfm.merge(df_monetary, on='customer_id')

    return df_rfm


@flow(name="RFM Feature extraction flow",
      description="Prefect flow for RFM feature extraction.",
      flow_run_name=generate_run_name("rfm-feature-extraction"))
def rfm_feature_extraction_flow(flow_args, parquet_input_file_path: str = ''):
    """Prefect flow for RFM feature extraction."""
    if flow_args.date_from is None:
        flow_args.date_from = date.today().replace(year=date.today().year - 5)
    if flow_args.date_to is None:
        flow_args.date_to = date.today()

    df = parquet_to_df(parquet_input_file_path if len(
        parquet_input_file_path) > 0 else flow_args.parquet_input_file_path)
    df_without_tuev = remove_tuev_rows(df)
    df_rfm_base = rfm_data_preparation(
        df_without_tuev, flow_args.date_from, flow_args.date_to)
    df_recency = recency_feature_extraction(df_rfm_base)
    df_frequency = frequency_feature_extraction(df_rfm_base)
    df_monetary = monetary_feature_extraction(df_rfm_base)

    df_rfm = merge_rfm_features(
        df_rfm_base, df_recency, df_frequency, df_monetary)

    export_output(df_rfm, flow_args.rfm_output_file_path)

    return flow_args.rfm_output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--parquet_input_file_path', type=str,
                        default=constants.extraction_defaults["output_file_path_parquet"],
                        help="Path to the parquet file.")
    parser.add_argument('--tmp_dir', type=str,
                        default="/tmp",
                        help="Path to the temporary directory.")
    parser.add_argument('--rfm_output_file_path', type=str,
                        default=constants.extraction_defaults['rfm_feature_file_path'],
                        help="Path to the output file.")
    parser.add_argument('--date_from', type=lambda x: datetime.strptime(
        x, '%Y-%m-%d').date(), default=None, help="Start date for the analysis.")
    parser.add_argument('--date_to', type=lambda x: datetime.strptime(
        x, '%Y-%m-%d').date(), default=None, help="End date for the analysis.")

    args = parser.parse_args()

    rfm_feature_extraction_flow(args)
