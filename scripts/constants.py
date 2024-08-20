
error_messages = {
    'os_error': 'File "{0}" not found.',
}

extraction_defaults = {
    'raw_data_path': 'data',
    'tmp_file_path': '/tmp',
    'raw_data_file_extensions': ['.TXT'],
    'raw_data_merged_file_name': 'data_merged.TXT',
    'raw_data_encoding': 'iso8859_15',
    'output_file_path_csv': '/tmp/data_cleaned.csv',
    'output_file_path_parquet': '/tmp/data_cleaned.parquet',
    'rfm_feature_file_path': '/tmp/rfm_features.parquet',
    'output_train_data_path': '/tmp/train_data.parquet',
}

training_defaults = {
    'rfm_years_lookback': 3,
    'rfm_clusters': {
        'recency': 4,
        'frequency': 4,
        'monetary': 4,
    },
    'rfm_score_segment_gt': {
        "low": 0,
        "medium": 3,
        "high": 6,
        "top": 8,
    }
}

if __name__ == '__main__':
    print("Hello world!")
