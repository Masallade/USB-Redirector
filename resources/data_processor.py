"""
data_processor.py
This module handles data processing tasks such as cleaning, transforming, and analyzing datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, data):
        """
        Initialize the DataProcessor with a dataset.
        
        :param data: A pandas DataFrame or list of dictionaries.
        """
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            self.data = data

    def clean_data(self):
        """
        Clean the dataset by removing duplicates, handling missing values, and standardizing formats.
        """
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)
        
        # Fill missing values with the mean for numeric columns
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)
        
        # Fill missing values with "Unknown" for categorical columns
        for col in self.data.select_dtypes(include=[object]).columns:
            self.data[col].fillna("Unknown", inplace=True)
        
        print("Data cleaned successfully!")

    def transform_data(self):
        """
        Transform the dataset by normalizing numeric columns and encoding categorical columns.
        """
        # Normalize numeric columns
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())
        
        # One-hot encode categorical columns
        self.data = pd.get_dummies(self.data, drop_first=True)
        
        print("Data transformed successfully!")

    def analyze_data(self):
        """
        Perform basic analysis on the dataset, such as calculating summary statistics.
        """
        summary = self.data.describe(include='all')
        return summary

    def filter_data(self, column: str, value: Any):
        """
        Filter the dataset based on a column and value.
        
        :param column: The column to filter on.
        :param value: The value to filter by.
        :return: A filtered DataFrame.
        """
        return self.data[self.data[column] == value]

    def sort_data(self, column: str, ascending: bool = True):
        """
        Sort the dataset based on a column.
        
        :param column: The column to sort by.
        :param ascending: Whether to sort in ascending order.
        :return: A sorted DataFrame.
        """
        return self.data.sort_values(by=column, ascending=ascending)

    def group_data(self, by: List[str], agg: Dict[str, Any]):
        """
        Group the dataset and apply aggregation functions.
        
        :param by: The columns to group by.
        :param agg: The aggregation functions to apply.
        :return: A grouped DataFrame.
        """
        return self.data.groupby(by).agg(agg)

    def merge_data(self, other: pd.DataFrame, on: str, how: str = "inner"):
        """
        Merge this dataset with another dataset.
        
        :param other: The other DataFrame to merge with.
        :param on: The column to merge on.
        :param how: The type of merge (inner, outer, left, right).
        :return: A merged DataFrame.
        """
        return pd.merge(self.data, other, on=on, how=how)

    def save_data(self, file_path: str, format: str = "csv"):
        """
        Save the dataset to a file.
        
        :param file_path: The path to save the file.
        :param format: The file format (csv, excel, json).
        """
        if format == "csv":
            self.data.to_csv(file_path, index=False)
        elif format == "excel":
            self.data.to_excel(file_path, index=False)
        elif format == "json":
            self.data.to_json(file_path, orient="records")
        else:
            raise ValueError("Unsupported file format.")
        
        print(f"Data saved to {file_path}")

    def load_data(self, file_path: str, format: str = "csv"):
        """
        Load a dataset from a file.
        
        :param file_path: The path to the file.
        :param format: The file format (csv, excel, json).
        """
        if format == "csv":
            self.data = pd.read_csv(file_path)
        elif format == "excel":
            self.data = pd.read_excel(file_path)
        elif format == "json":
            self.data = pd.read_json(file_path, orient="records")
        else:
            raise ValueError("Unsupported file format.")
        
        print(f"Data loaded from {file_path}")

    def add_column(self, column_name: str, data: List[Any]):
        """
        Add a new column to the dataset.
        
        :param column_name: The name of the new column.
        :param data: The data for the new column.
        """
        if len(data) != len(self.data):
            raise ValueError("Data length does not match the dataset.")
        self.data[column_name] = data

    def remove_column(self, column_name: str):
        """
        Remove a column from the dataset.
        
        :param column_name: The name of the column to remove.
        """
        self.data.drop(columns=[column_name], inplace=True)

    def rename_column(self, old_name: str, new_name: str):
        """
        Rename a column in the dataset.
        
        :param old_name: The current name of the column.
        :param new_name: The new name of the column.
        """
        self.data.rename(columns={old_name: new_name}, inplace=True)

    def calculate_correlation(self):
        """
        Calculate the correlation matrix for numeric columns.
        
        :return: A correlation matrix.
        """
        return self.data.corr()

    def calculate_covariance(self):
        """
        Calculate the covariance matrix for numeric columns.
        
        :return: A covariance matrix.
        """
        return self.data.cov()

    def calculate_percentiles(self, column: str, percentiles: List[float]):
        """
        Calculate percentiles for a numeric column.
        
        :param column: The column to calculate percentiles for.
        :param percentiles: The percentiles to calculate.
        :return: A DataFrame with the percentiles.
        """
        return self.data[column].quantile(percentiles)

    def calculate_moving_average(self, column: str, window: int):
        """
        Calculate the moving average for a numeric column.
        
        :param column: The column to calculate the moving average for.
        :param window: The window size for the moving average.
        :return: A Series with the moving average.
        """
        return self.data[column].rolling(window=window).mean()

    def calculate_exponential_moving_average(self, column: str, span: int):
        """
        Calculate the exponential moving average for a numeric column.
        
        :param column: The column to calculate the exponential moving average for.
        :param span: The span for the exponential moving average.
        :return: A Series with the exponential moving average.
        """
        return self.data[column].ewm(span=span, adjust=False).mean()

    def calculate_z_scores(self, column: str):
        """
        Calculate the z-scores for a numeric column.
        
        :param column: The column to calculate z-scores for.
        :return: A Series with the z-scores.
        """
        return (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def calculate_outliers(self, column: str, threshold: float = 3):
        """
        Identify outliers in a numeric column using z-scores.
        
        :param column: The column to identify outliers in.
        :param threshold: The z-score threshold for identifying outliers.
        :return: A DataFrame with the outliers.
        """
        z_scores = self.calculate_z_scores(column)
        return self.data[abs(z_scores) > threshold]

    def calculate_skewness(self, column: str):
        """
        Calculate the skewness of a numeric column.
        
        :param column: The column to calculate skewness for.
        :return: The skewness value.
        """
        return self.data[column].skew()

    def calculate_kurtosis(self, column: str):
        """
        Calculate the kurtosis of a numeric column.
        
        :param column: The column to calculate kurtosis for.
        :return: The kurtosis value.
        """
        return self.data[column].kurtosis()

    def calculate_entropy(self, column: str):
        """
        Calculate the entropy of a categorical column.
        
        :param column: The column to calculate entropy for.
        :return: The entropy value.
        """
        value_counts = self.data[column].value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))

    def calculate_mutual_information(self, column1: str, column2: str):
        """
        Calculate the mutual information between two categorical columns.
        
        :param column1: The first column.
        :param column2: The second column.
        :return: The mutual information value.
        """
        from sklearn.metrics import mutual_info_score
        return mutual_info_score(self.data[column1], self.data[column2])

    def calculate_chi_square(self, column1: str, column2: str):
        """
        Calculate the chi-square statistic between two categorical columns.
        
        :param column1: The first column.
        :param column2: The second column.
        :return: The chi-square statistic.
        """
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(self.data[column1], self.data[column2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2

    def calculate_anova(self, column1: str, column2: str):
        """
        Calculate the ANOVA F-statistic between a categorical and a numeric column.
        
        :param column1: The categorical column.
        :param column2: The numeric column.
        :return: The ANOVA F-statistic.
        """
        from scipy.stats import f_oneway
        groups = [self.data[self.data[column1] == group][column2] for group in self.data[column1].unique()]
        return f_oneway(*groups)

    def calculate_pearson_correlation(self, column1: str, column2: str):
        """
        Calculate the Pearson correlation coefficient between two numeric columns.
        
        :param column1: The first numeric column.
        :param column2: The second numeric column.
        :return: The Pearson correlation coefficient.
        """
        return self.data[column1].corr(self.data[column2], method='pearson')

    def calculate_spearman_correlation(self, column1: str, column2: str):
        """
        Calculate the Spearman correlation coefficient between two numeric columns.
        
        :param column1: The first numeric column.
        :param column2: The second numeric column.
        :return: The Spearman correlation coefficient.
        """
        return self.data[column1].corr(self.data[column2], method='spearman')

    def calculate_kendall_correlation(self, column1: str, column2: str):
        """
        Calculate the Kendall correlation coefficient between two numeric columns.
        
        :param column1: The first numeric column.
        :param column2: The second numeric column.
        :return: The Kendall correlation coefficient.
        """
        return self.data[column1].corr(self.data[column2], method='kendall')

    def calculate_autocorrelation(self, column: str, lag: int = 1):
        """
        Calculate the autocorrelation of a numeric column.
        
        :param column: The column to calculate autocorrelation for.
        :param lag: The lag for the autocorrelation.
        :return: The autocorrelation value.
        """
        return self.data[column].autocorr(lag=lag)

    def calculate_partial_autocorrelation(self, column: str, lag: int = 1):
        """
        Calculate the partial autocorrelation of a numeric column.
        
        :param column: The column to calculate partial autocorrelation for.
        :param lag: The lag for the partial autocorrelation.
        :return: The partial autocorrelation value.
        """
        from statsmodels.tsa.stattools import pacf
        return pacf(self.data[column], nlags=lag)[-1]

    def calculate_cross_correlation(self, column1: str, column2: str, lag: int = 1):
        """
        Calculate the cross-correlation between two numeric columns.
        
        :param column1: The first numeric column.
        :param column2: The second numeric column.
        :param lag: The lag for the cross-correlation.
        :return: The cross-correlation value.
        """
        from scipy.signal import correlate
        return correlate(self.data[column1], self.data[column2], mode='same')[lag]

    def calculate_fft(self, column: str):
        """
        Calculate the Fast Fourier Transform of a numeric column.
        
        :param column: The column to calculate the FFT for.
        :return: The FFT values.
        """
        return np.fft.fft(self.data[column])

    def calculate_histogram(self, column: str, bins: int = 10):
        """
        Calculate the histogram of a numeric column.
        
        :param column: The column to calculate the histogram for.
        :param bins: The number of bins for the histogram.
        :return: The histogram values and bin edges.
        """
        return np.histogram(self.data[column], bins=bins)

    def calculate_kde(self, column: str):
        """
        Calculate the Kernel Density Estimate of a numeric column.
        
        :param column: The column to calculate the KDE for.
        :return: The KDE values.
        """
        from scipy.stats import gaussian_kde
        return gaussian_kde(self.data[column])

    def calculate_pca(self, n_components: int = 2):
        """
        Perform Principal Component Analysis on the numeric columns.
        
        :param n_components: The number of components to reduce to.
        :return: The transformed data.
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.data.select_dtypes(include=[np.number]))

    def calculate_tsne(self, n_components: int = 2):
        """
        Perform t-SNE on the numeric columns.
        
        :param n_components: The number of components to reduce to.
        :return: The transformed data.
        """
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components)
        return tsne.fit_transform(self.data.select_dtypes(include=[np.number]))

    def calculate_umap(self, n_components: int = 2):
        """
        Perform UMAP on the numeric columns.
        
        :param n_components: The number of components to reduce to.
        :return: The transformed data.
        """
        from umap import UMAP
        umap = UMAP(n_components=n_components)
        return umap.fit_transform(self.data.select_dtypes(include=[np.number]))

    def calculate_clustering(self, n_clusters: int = 3):
        """
        Perform K-Means clustering on the numeric columns.
        
        :param n_clusters: The number of clusters to form.
        :return: The cluster labels.
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(self.data.select_dtypes(include=[np.number]))

    def calculate_dbscan(self, eps: float = 0.5, min_samples: int = 5):
        """
        Perform DBSCAN clustering on the numeric columns.
        
        :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: The cluster labels.
        """
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(self.data.select_dtypes(include=[np.number]))

    def calculate_hierarchical_clustering(self, n_clusters: int = 3):
        """
        Perform Hierarchical clustering on the numeric columns.
        
        :param n_clusters: The number of clusters to form.
        :return: The cluster labels.
        """
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        return hierarchical.fit_predict(self.data.select_dtypes(include=[np.number]))

    def calculate_silhouette_score(self, labels):
        """
        Calculate the Silhouette Score for clustering results.
        
        :param labels: The cluster labels.
        :return: The Silhouette Score.
        """
        from sklearn.metrics import silhouette_score
        return silhouette_score(self.data.select_dtypes(include=[np.number]), labels)

    def calculate_calinski_harabasz_score(self, labels):
        """
        Calculate the Calinski-Harabasz Score for clustering results.
        
        :param labels: The cluster labels.
        :return: The Calinski-Harabasz Score.
        """
        from sklearn.metrics import calinski_harabasz_score
        return calinski_harabasz_score(self.data.select_dtypes(include=[np.number]), labels)

    def calculate_davies_bouldin_score(self, labels):
        """
        Calculate the Davies-Bouldin Score for clustering results.
        
        :param labels: The cluster labels.
        :return: The Davies-Bouldin Score.
        """
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(self.data.select_dtypes(include=[np.number]), labels)

    