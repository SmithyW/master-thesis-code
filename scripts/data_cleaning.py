"""Contains tasks for data cleaning and preprocessing."""
# Standard imports
import hashlib
import time
# Third party imports
import pandas as pd
import numpy as np
from prefect import task, flow
from rapidfuzz import fuzz
# Custom imports
from prefect_utils import generate_run_name


def calc_sim_matrix_input_hash(column_name: str, df: pd.DataFrame) -> tuple:
    """Calculate the hash for the input of the calculate_similarity_matrix task."""
    return hashlib.sha1(str((column_name, len(df))).encode('utf-8'), usedforsecurity=False).hexdigest()


@task(name="Calculate Similarity Matrix",
      description="Calculate the similarity matrix for a column in a DataFrame.",
      task_run_name=generate_run_name("calculate-similarity-matrix"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning", "data-privacy"],
      cache_key_fn=calc_sim_matrix_input_hash)
def calculate_similarity_matrix(*, column_name: str, df: type[pd.DataFrame]) -> list[list[float]]:
    df_len = len(df)
    print(
        f"Calculating similarity matrix for column '{column_name}' of dataframe with {df_len} rows")
    similarity_matrix = np.zeros((df_len, df_len))
    time_per_row = []
    for i in range(df_len):
        start_time_row = time.time()
        for j in range(i+1, df_len):
            similarity_matrix[i, j] = fuzz.ratio(
                df[column_name].iloc[i], df[column_name].iloc[j], score_cutoff=90)
        end_time_row = time.time()
        time_per_row.append(end_time_row - start_time_row)
    print(f"Fastest processed row: {min(time_per_row)} seconds.")
    print(f"Slowest processed row: {max(time_per_row)} seconds.")
    print(
        f"Average processing time per row: {(sum(time_per_row) / len(time_per_row))} seconds")
    return similarity_matrix


@task(name="Read Data",
      description="Read csv data from a file and return a DataFrame.",
      task_run_name=generate_run_name("read-data"))
def read_data(input_file_path: str, csv_delimiter: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read data from a file and return a DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the file to read.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the data.
    """
    # Specify the column names and dtypes before reading the csv file
    column_names = ['date', "invoice_num", "name", "license_plate", "mileage", "vin",
                    "manufacturer", "type", "first_registration", "address_company",
                    "address_name", "address_street_house_number", "address_postal_code_city",
                    "address_other", "item_id", "item_description", "is_total", "amount",
                    "price", "total", "balance"]
    column_dtypes = ['object', 'string', 'string', 'string', 'string', 'string',
                     'string', 'string', 'object', 'string',
                     'string', 'string', 'string',
                     'string', 'string', 'string', 'string', 'float64',
                     'float64', 'float64', 'float64']

    # Create the dtype_dict for pandas read_csv
    dtype_dict = dict(zip(column_names, column_dtypes))

    df = pd.read_csv(input_file_path, delimiter=csv_delimiter, parse_dates=['date'],
                     dayfirst=True, names=column_names, dtype=dtype_dict, header=0,
                     na_values=[None, '', ' '], decimal=',', thousands='.', encoding=encoding)
    return df


@task(name="Type Conversion",
      description="Convert data types of columns in a DataFrame.",
      task_run_name=generate_run_name("type-conversion"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning"])
def type_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert data types of columns in a DataFrame.

    Args:
        df (pd.DataFrame): Imported DataFrame.

    Returns:
        pd.DataFrame: DataFrame with converted data types.
    """
    # Parse the date column
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    # Convert the is_total column to boolean by checking for the string "-SPLITTEN-"
    df["is_total"] = df["is_total"] == "-SPLITTEN-"
    return df


@task(name="Apply Vehicle Privacy",
      description="Apply privacy by hashing the license_plate and vin columns.",
      task_run_name=generate_run_name("apply-vehicle-privacy"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning"])
def apply_vehicle_privacy(df: pd.DataFrame) -> pd.DataFrame:
    """Appy privacy by hashing the license_plate and vin columns.

    Args:
        df (pd.DataFrame): Imported and type-converted DataFrame.

    Returns:
        pd.DataFrame: DataFrame with hashed license_plate and vin columns.
    """
    # Apply privacy by hashing the license_plate and vin columns
    df["license_plate"] = df["license_plate"].fillna("").map(lambda x: hashlib.sha1(
        x.encode('utf-8')).hexdigest()[:16] if len(x) > 0 else None)
    df["vin"] = df["vin"].fillna("").map(lambda x: hashlib.sha1(
        x.encode('utf-8')).hexdigest()[:20] if len(x) > 0 else None)

    return df


@task(name="Float Cleaning",
      description="Clean the float columns in the DataFrame.",
      task_run_name=generate_run_name("float-cleaning"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning"])
def float_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the float columns in the DataFrame.
    Multiplies negatives with -1 and drops the balance column.

    Args:
        df (pd.DataFrame): DataFrame with hashed license_plate and vin columns.

    Returns:
        pd.DataFrame: DataFrame with cleaned float columns.
    """
    # Convert the negative values of the columns amount and total to positive
    df["amount"] = df["amount"] * -1
    df.loc[df["total"] < 0, "total"] = df.loc[df["total"] < 0, "total"] * -1
    # Drop the balance column
    df.drop(columns=["balance"], inplace=True)

    return df


@task(name="Group Customer Entries",
      description="Group customer entries based on the address columns.",
      task_run_name=generate_run_name("group-customers"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning"])
def group_customer_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply privacy by hashing the customer information after identifying related customer
    entries based on the address columns.

    Args:
        df (pd.DataFrame): DataFrame with cleaned float columns.

    Returns:
        pd.DataFrame: DataFrame with hashed address_name column.
    """
    # Split the name column into customer_number and customer_description
    df[['customer_number', 'customer_description']
       ] = df['name'].str.split(' - ', n=1, expand=True)
    df.drop(['name'], axis=1, inplace=True)

    df_filtered = df[df["customer_number"] != "10"]

    # Keep only relevant customer data
    df_name = df_filtered.drop(df_filtered.columns.difference(
        ["address_company", "address_name", "address_street_house_number",
         "address_postal_code_city", "address_other", "customer_number"]), axis=1)

    # Filter out duplicates regarding the customer data
    df_name_distinct = df_name.drop_duplicates(["customer_number"])

    # Filter entries without information in address_company and address_name
    df_multiple = df_name_distinct[df_name_distinct["address_name"].notnull(
    ) | df_name_distinct["address_company"].notnull()]
    df_multiple[["address_company", "address_name"]] = df_multiple[[
        "address_company", "address_name"]].fillna('')

    # Processing of data with poor quality caused by accidently switched columns

    # Check if the address_street_house_number column contains a postal code
    # Then proceed to shift the columns
    condition = df_name_distinct["address_street_house_number"].str.contains(
        r"^[0-9]{5} ")
    df_name_distinct.loc[condition, [
        "address_name", "address_street_house_number", "address_postal_code_city"
    ]] = df_name_distinct.loc[condition,
                              ["address_company", "address_name",
                               "address_street_house_number"]].values
    df_name_distinct.loc[condition, "address_company"] = "Firma"

    # Check if the address_street_house_number column contains a valid address
    # if not, the data is possibly shifted and has to be corrected accordingly

    # Regex for a valid german address
    regex_street_house_number = r".*[0-9]{1,4}([\s]{0,1})[a-zA-Z]?[\s]?$"
    # Condition for a string not being a valid address (or postal code)
    condition_not_street_house_number = ~df_name_distinct[
        "address_street_house_number"
    ].str.contains(regex_street_house_number) & \
        df_name_distinct["address_street_house_number"].notnull() & \
        ~df_name_distinct["address_street_house_number"].str.contains(
            r"^[0-9]{5} ")
    # Condition for an address being in the postal_code_city column
    condition_street_in_city = df_name_distinct[
        "address_postal_code_city"
    ].str.contains(regex_street_house_number) & \
        df_name_distinct["address_postal_code_city"].notnull()

    # Assign the address_name to address_company if address_name is not null and
    # address_street_house_number does not contain a valid address
    df_name_distinct.loc[
        condition_not_street_house_number & df_name_distinct["address_name"].notnull(
        ),
        ["address_company", "address_name"]
    ] = df_name_distinct.loc[
        condition_not_street_house_number & df_name_distinct["address_name"].notnull(
        ),
        ["address_name", "address_street_house_number"]].values

    df_name_distinct.loc[
        condition_street_in_city,
        ["address_street_house_number", "address_postal_code_city"]
    ] = df_name_distinct.loc[
        condition_street_in_city, ["address_postal_code_city", "address_other"]].values
    df_name_distinct.loc[condition_street_in_city, "address_other"] = None

    # Move the remaining existing postal code and city entries in address_other to
    # address_postal_code_city
    condition_postal_code_city = df_name_distinct["address_other"].str.contains(
        r"^[0-9]{5} ")
    df_name_distinct.loc[condition_postal_code_city,
                         "address_postal_code_city"] = df_name_distinct.loc[
                             condition_postal_code_city, "address_other"]

    df_name_distinct.loc[condition_postal_code_city, "address_other"] = None

    # Combine the address_company and address_name columns for distinction of customer data
    df_combined_address = df_name_distinct.copy()
    df_combined_address['address'] = df_multiple["address_company"] + \
        " " + df_multiple["address_name"]
    # Filter out unnecessary keywords in the address column

    df_combined_address['address'] = df_combined_address['address'].str.lower().str.replace(
        'herr ', '').str.replace('frau ', '').str.replace('herrn ', '').str.strip()

    # Fuzzy matching due to different spellings of the same customers

    # Calculate the similarity matrix for the address column and get the indices out of
    # the non-zero values for further processing
    similarity_matrix_address = calculate_similarity_matrix(
        column_name='address', df=df_combined_address)
    # Get the indices of the non-zero values of the similarity matrix
    # to get a 2-dim list of related customers
    indices = np.nonzero(similarity_matrix_address)

    # Group the related customers in the second index by the first index
    groups: dict[int, list[int]] = {}
    for i in range(0, len(indices[0])):
        # If a group already exists: append, otherwise create a new group
        if indices[0][i] in groups:
            groups[indices[0][i]].append(indices[1][i])
        else:
            groups[indices[0][i]] = [indices[1][i]]

    # Iterate over all groups and merge matching groups
    for key, value in groups.copy().items():
        if key in groups:
            for v in value:
                if v in groups:
                    groups[key].extend(groups[v])
                    del groups[v]
    # Deduplicate the groups
    groups = {k: sorted(list(set(v)))
              for k, v in groups.items()}

    # Built a hash for each group as unique group identifier
    df_hashed_address = df_combined_address.copy()
    df_hashed_address['customer_id'] = None
    for key, value in groups.copy().items():
        # Hash the built address entry
        group_hash = hashlib.sha256(
            df_hashed_address['address'].iloc[key].encode(), usedforsecurity=False).hexdigest()
        # set the hash as customer for the key and all values
        indices_to_set = [df_hashed_address.index[x] for x in [key] + value]

        df_hashed_address.loc[indices_to_set, ['customer_id']] = group_hash

    df_hashed_address.loc[df_hashed_address['customer_id'].isnull(), 'customer_id'] = df_hashed_address.loc[df_hashed_address['customer_id'].isnull(
    ), 'address'].map(lambda x: hashlib.sha256(x.encode(), usedforsecurity=False).hexdigest())

    # Merge back
    df_merged = pd.merge(df_filtered, df_hashed_address[[
                         "customer_number", "customer_id"]], how='left', left_on='customer_number', right_on='customer_number')

    return df_merged


@task(name="Apply Customer Privacy",
      description="After grouping customer entries, delete the address columns.",
      task_run_name=generate_run_name("apply-customer-privacy"),
      log_prints=True,
      tags=["preprocessing", "data-cleaning", "data-privacy"])
def apply_customer_privacy(df: pd.DataFrame) -> pd.DataFrame:
    """After grouping customer entries, delete the address columns.
    Assign a customer_id to customer_number 10.

    Args:
        df (pd.DataFrame): DataFrame with grouped customer entries.

    Returns:
        pd.DataFrame: DataFrame with hashed customer_number and customer_id columns.
    """
    # Create hash for customer_number 10 and merge it back
    df_bar = df[df["customer_number"] == "10"]
    df_bar["customer_id"] = 10
    df = pd.concat([df, df_bar], ignore_index=True)

    # Remove customer data for privacy reasons
    df.drop(['address_company', 'address_name', 'address_street_house_number',
             'address_postal_code_city', 'address_other', 'customer_description'], axis=1, inplace=True)

    return df


@flow(name="data-cleaning-flow",
      flow_run_name=generate_run_name("data-cleaning-flow"),
      description="Flow for data cleaning and privacy.")
def data_cleaning(input_file_path: str, csv_delimiter: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """Flow for data cleaning and privacy.

    Args:
        input_file_path (str): Path to the input file.

    Returns:
        pd.DataFrame: DataFrame with cleaned and privacy-applied data.
    """
    df_init = read_data(
        input_file_path, csv_delimiter=csv_delimiter, encoding=encoding)
    df_converted = type_conversion(df_init)
    df_vehicle_privacy = apply_vehicle_privacy(df_converted)
    df_prepared = float_cleaning(df_vehicle_privacy)
    df_customer_hashed = group_customer_entries(df_prepared)
    df = apply_customer_privacy(df_customer_hashed)

    return df
