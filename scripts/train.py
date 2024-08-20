import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import AutoLocator
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import mlflow
from prefect import flow, task
from prefect_utils import generate_run_name

from feature_extraction import parquet_to_df
import constants
from extraction import export_output

# function for ordering cluster numbers


@task(name="Order cluster",
      description="Order clusters based on the median values of the features.",
      task_run_name=generate_run_name("order-cluster"))
def order_cluster(df: pd.DataFrame, feature_sort: dict[str, bool]):
    # Order clusters
    # Calculate the median value for each feature in each cluster
    features = list(feature_sort.keys())
    cluster_order = df.groupby(
        'cluster')[features].median().reset_index()
    cluster_order['rfm_score'] = 0

    # Sort the clusters based on the median values of the features
    # and add the index values to the rfm_score
    for col in features:
        cluster_order.sort_values(
            by=col, inplace=True, ascending=feature_sort[col])
        cluster_order['rfm_score'] += cluster_order.reset_index().index

    cluster_order['new_cluster'] = cluster_order.sort_values(
        'rfm_score').reset_index().index

    df_cluster_sorted = df.copy()
    df_cluster_sorted = df_cluster_sorted.merge(
        cluster_order[['cluster', 'new_cluster']], on='cluster')
    # Assign the new cluster number to the original DataFrame
    df_cluster_order = df_cluster_sorted.drop(
        'cluster', axis=1).rename(columns={'new_cluster': 'cluster'})

    return df_cluster_order


@task(name="Plot distribution",
      description="Plot distribution of features.",
      task_run_name=generate_run_name("plot-distribution"),
      tags=["visualization"])
def plot_distribution(df, features, *, return_figure: bool = True, path_to_save: str = None):
    plt.switch_backend('agg')
    fig, ax = plt.subplots(len(features), 2, figsize=(15, 15))
    for idx, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=ax[idx][0])
        sns.boxplot(df[feature], ax=ax[idx][1])
    plt.tight_layout()

    if path_to_save is not None and path_to_save != '':
        plt.savefig(path_to_save)

    if return_figure:
        return fig
    return path_to_save


@task(name="Plot 3D clusters",
      description="Plot 3D clusters.",
      task_run_name=generate_run_name("plot-3d-clusters"),
      tags=["visualization"])
def plot_3d_clusters(df: pd.DataFrame, color_feature_name: str, size_feature_name: str, *, return_figure: bool = True, path_to_save: str = None) -> str:
    print("Plotting 3D clusters...")
    fig = px.scatter_3d(df, x='recency', y='frequency', z='monetary',
                        opacity=0.8, color=df[color_feature_name], size=df[size_feature_name], title='RFM Clusters')

    if path_to_save is not None and path_to_save != '':
        fig.write_html(path_to_save, auto_open=False)

    if return_figure:
        return fig
    return path_to_save


@task(name="RFM Boxplot",
      description="Plot RFM boxplot.",
      task_run_name=generate_run_name("plot-rfm-boxplot"),
      tags=["visualization"])
def rfm_boxplot(df: pd.DataFrame, *, return_figure: bool = True, path_to_save: str = None) -> str:
    sns.set_theme(style="darkgrid")
    sns.set_style(style='white')

    fig, axs = plt.subplots(3, 1, figsize=(18, 8))
    axs[0].boxplot(df['recency'], widths=0.5, vert=False)
    axs[1].boxplot(df['frequency'], widths=0.5, vert=False)
    axs[2].boxplot(df['monetary'], widths=0.5, vert=False)

    axs[0].xaxis.set_major_locator(AutoLocator())
    axs[0].xlim = (0, 2000)

    axs[0].set_ylabel('Recency', rotation=90, ha='center', va='center')
    axs[1].set_ylabel('Frequency', rotation=90, ha='center', va='center')
    axs[2].set_ylabel('Monetary', rotation=90, ha='center', va='center')

    if path_to_save is not None and path_to_save != '':
        plt.savefig(path_to_save)

    if return_figure:
        return fig
    return path_to_save


@task(name="Plot elbow",
      description="Plot elbow method for optimal number of clusters.",
      task_run_name=generate_run_name("plot-elbow"),
      tags=["visualization"])
def plot_elbow(sse, *, return_figure: bool = True, path_to_save: str = None):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.title("Elbow method for optimal number of clusters")

    if path_to_save is not None and path_to_save != '':
        plt.savefig(path_to_save)

    if return_figure:
        return fig
    return path_to_save


@task(name="Plot KNN scores",
      description="Plot KNN scores for different number of neighbors.",
      task_run_name=generate_run_name("plot-knn-scores"),
      tags=["visualization"])
def plot_knn_scores(score_dict: dict, *, return_figure: bool = True, path_to_save: str = None):
    fig = plt.figure()
    sns.set_style(style='white')
    sns.lineplot(x=list(score_dict.keys()), y=list(score_dict.values()))
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")

    if path_to_save is not None and path_to_save != '':
        plt.savefig(path_to_save)

    if return_figure:
        return fig
    return path_to_save


def identify_outliers(df, threshold=1.5):
    outliers_indices = {}
    for col in df.columns:
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

        outliers_indices[col] = outliers

    return outliers


def transform_rfm_for_outlier_detection(df: pd.DataFrame, features: list, *, n_quantiles=50, random_state=42) -> pd.DataFrame:
    """Transform the RFM features for outlier detection using a quantile transformer.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with transformed RFM features.
    """
    transformer = QuantileTransformer(
        output_distribution='normal', n_quantiles=n_quantiles, random_state=random_state)

    df_customer_trans = df.copy()
    df_customer_trans[features] = transformer.fit_transform(df[features])

    return df_customer_trans


@task(name="Remove outliers",
      description="Remove outliers from the dataframe.",
      task_run_name=generate_run_name("remove-outliers"))
def remove_outliers(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Remove outliers from the dataframe."""

    df_transformed = transform_rfm_for_outlier_detection(df, features)
    outliers = identify_outliers(
        df_transformed[features], threshold=1.5)

    df_without_outliers = df.copy()
    df_without_outliers['outlier'] = 0
    df_without_outliers.loc[outliers, 'outlier'] = 1

    df_without_outliers.drop(
        df_without_outliers[df_without_outliers['outlier'] == 1].index, inplace=True)
    df_without_outliers.drop('outlier', axis=1, inplace=True)

    return df_without_outliers


@task(name="Scale data",
      description="Scale the data using the provided scaler.",
      task_run_name=generate_run_name("scale-data"))
def scale_data(df: pd.DataFrame, features: list, scaler=StandardScaler()):
    """Scale the data using the provided scaler.

    Args:
        df (pd.DataFrame): DataFrame to be scaled.
        features (list): Features of DataFrame to be scaled.
        scaler (sklearn scaler, optional): Scaler to be used for scaling the input data. Defaults to StandardScaler().

    Returns:
        pd.DataFrame: scaled DataFrame
        scaler: fitted scaler
    """
    df_scaled = df.copy()

    df_scaled[features] = scaler.fit_transform(df[features])

    fig_scaled_distribution = plot_distribution(
        df_scaled, ['recency', 'frequency', 'monetary'])
    mlflow.log_figure(fig_scaled_distribution, "scaled_distribution.png")

    return df_scaled, scaler


@task(name="Prepare data",
      description="Prepare data for training.",
      task_run_name=generate_run_name("prepare-data"))
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for training."""
    print("Preparing data...")
    df_prepare = df.copy()
    fig_initial_distribution = plot_distribution(
        df_prepare, ['recency', 'frequency', 'monetary'], return_figure=True)
    mlflow.log_figure(fig_initial_distribution, "initial_distribution.png")
    print(f"Got {len(df_prepare)} rows.\n")
    print("Performing outlier detection...")

    df_filtered = remove_outliers(
        df_prepare, ['recency', 'frequency', 'monetary'])
    print(f"Removed {df_prepare.shape[0] - df_filtered.shape[0]} outliers.\n")

    fig_non_outliers = plot_distribution(
        df_filtered, ['recency', 'frequency', 'monetary'])
    mlflow.log_figure(fig_non_outliers, "no_outliers_distribution.png")

    print("Finished data preparation.\n")

    return df_filtered


@task(name="Find clusters",
      description="Find clusters in the scaled data using K-Means.",
      task_run_name=generate_run_name("find-clusters"))
def find_clusters(df: pd.DataFrame, features: list, *, km_n_clusters: int = 4, km_init: str = 'k-means++', km_algorithm: str = "lloyd", km_random_state: int = 42) -> pd.DataFrame:
    """Find clusters in the provided data using kmeans."""

    # The following part is for the elbow method and silhouette score of different cluster nums
    # The cluster number used for the final model has to be provided as a parameter
    sse = {}
    silhouettes = {}

    for n_clusters in range(1, 15):
        clusterer = KMeans(n_clusters=n_clusters, init=km_init,
                           algorithm=km_algorithm, random_state=km_random_state)
        preds = clusterer.fit_predict(df[features])
        sse[n_clusters] = clusterer.inertia_

        if n_clusters > 1:
            score = silhouette_score(df[features], preds)
            silhouettes[n_clusters] = score
            print(f"For n_clusters = {
                n_clusters}, silhouette score is {score})")
    with mlflow.start_run(nested=True):
        # Find the clusters with the provided number of clusters
        kmeans = KMeans(n_clusters=km_n_clusters, init=km_init,
                        algorithm=km_algorithm, random_state=km_random_state)
        preds = kmeans.fit_predict(df[features])

        # Get the metrics for the clustering
        score = silhouette_score(df[features], preds)
        db_score = davies_bouldin_score(df[features], preds)

        # Save the computed clusters in a new column
        df_cluster = df.copy()
        df_cluster['cluster'] = preds

        # Log mlflow parameters, metrics and artifcats

        mlflow.log_params({"n_clusters": km_n_clusters,
                           "random_state": km_random_state, "init": km_init, "algorithm": km_algorithm})
        mlflow.log_param("max_iter", kmeans.max_iter)
        mlflow.log_param("n_init", kmeans.n_init)

        mlflow.log_metric("inertia", kmeans.inertia_)
        mlflow.log_metric("silhouette_score", score)
        mlflow.log_metric("davies_bouldin_score", db_score)

        mlflow.log_dict(sse, "test_sse.json")
        mlflow.log_dict(silhouettes, "test_silhouettes.json")

        fig_elbow = plot_elbow(sse)
        mlflow.log_figure(fig_elbow, "k_means_elbow.png")

        np.savetxt('/tmp/km_cluster_centers.txt',
                   kmeans.cluster_centers_)
        mlflow.log_artifact('/tmp/km_cluster_centers.txt')

    return df_cluster


@task(name="Train KNN",
      description="Train a KNN model.",
      task_run_name=generate_run_name("train-knn"))
def train_knn(df: pd.DataFrame, features: list, *, test_size: float = 0.3, random_state: int = 512, k_fold_split: int = 4, range_k=range(1, 49, 2), knn_weights='distance', knn_algorithm='auto') -> KNeighborsClassifier:
    """Train a KNN model.

    Args:
        df (pd.DataFrame): Scaled input data.
        features (list): List of features to be used from df.
        test_size (float, optional): Size of the test dataset. Defaults to 0.3.
        random_state (int, optional): random_state for the train_test_split and KFold implementation. Defaults to 123.
        k_fold_split (int, optional): The amount of splits from the train data to be used for cross-validation. Defaults to 4.

    Returns:
        _type_: Trained KNN model
    """
    # TODO: Temp

    X, y = df[features], df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    k_folds = KFold(n_splits=k_fold_split, shuffle=True,
                    random_state=random_state)

    score_dict = {}
    n_neighbors = range_k
    for n in n_neighbors:
        knn = KNeighborsClassifier(
            n_neighbors=n, weights=knn_weights, algorithm=knn_algorithm)
        scores = cross_val_score(knn, X_train, y_train,
                                 cv=k_folds, scoring='accuracy')
        score_dict[n] = scores.mean()
        print(f"n: {n}, score: {scores.mean()}")
    # Get the best k
    best_k = max(score_dict, key=score_dict.get)

    mlflow.autolog()
    # Train the model with the best k
    knn = KNeighborsClassifier(
        n_neighbors=best_k, weights=knn_weights, algorithm=knn_algorithm).fit(X_train, y_train)

    report = classification_report(
        y_test, knn.predict(X_test), output_dict=True)
    cm = confusion_matrix(y_test, knn.predict(X_test), normalize='true')

    fig_knn_scores = plot_knn_scores(score_dict)

    disp = ConfusionMatrixDisplay.from_estimator(
        knn, X_test, y_test, display_labels=knn.classes_)
    disp.figure_.savefig("/tmp/confusion_matrix.png")

    mlflow.log_metric("knn_best_k", best_k)
    mlflow.log_text(str(cm), "confusion_matrix.txt")
    mlflow.log_artifact("/tmp/confusion_matrix.png")
    mlflow.log_figure(fig_knn_scores, "knn_scores.png")
    mlflow.log_dict(score_dict, "knn_scores.json")
    mlflow.log_dict(str(report), "classification_report.txt")

    return knn


@flow(name="Train flow",
      description="Train a model on the provided data.",
      flow_run_name=generate_run_name("train"))
def train_flow(flow_args):

    # Read the parquet file and convert to a pandas DataFrame
    df = parquet_to_df(flow_args.rfm_output_file_path)
    # Prepare the rfm data for training
    df_prepared = prepare_data(df)
    # Scale the data
    df_scaled, scaler = scale_data(
        df_prepared, ['recency', 'frequency', 'monetary'])

    # Build K-Means clustering model
    df_clustered = find_clusters(
        df_scaled, ['recency', 'frequency', 'monetary'])

    # Sort K-Means cluster
    df_clustered_sorted = order_cluster(
        df_clustered, {'recency': False, 'frequency': True, 'monetary': True})

    # Plot 3D clusters
    df_plot_base = pd.merge(df_prepared, df_clustered_sorted[[
        'customer_id', 'cluster']], on='customer_id', how='left')

    df_plot_base.to_parquet("/tmp/filtered_data.parquet")
    mlflow.log_artifact("/tmp/filtered_data.parquet")

    fig_3d_clusters = plot_3d_clusters(
        df_plot_base, color_feature_name='cluster', size_feature_name='monetary')
    mlflow.log_figure(fig_3d_clusters, "3d_clusters.html")

    # Build KNN
    knn = train_knn(df_clustered_sorted, ['recency', 'frequency', 'monetary'])

    # Build sklearn pipeline with scaler and model
    pipeline = Pipeline(steps=[('scaler', scaler), ('knn', knn)])
    mlflow.sklearn.log_model(pipeline, "knn_pipeline")

    # TODO: Remove
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rfm_output_file_path', type=str,
                        default=constants.extraction_defaults['rfm_feature_file_path'],
                        help='Path to the parquet file. (default: data_cleaned.parquet)')
    parser.add_argument('--output_train_data_path', type=str,
                        default=constants.extraction_defaults['output_train_data_path'],
                        help='Path to the output training data. (default: /tmp/train_data.parquet)')
    args = parser.parse_args()
    train_flow(args)
