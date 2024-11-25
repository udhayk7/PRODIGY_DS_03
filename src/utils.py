"""
Utility functions for bank customer segmentation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial cleaning of the banking dataset.
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    df = df.drop_duplicates()
    
    return df

def preprocess_features(df: pd.DataFrame, 
                       numeric_cols: List[str], 
                       categorical_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess features for clustering.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_cols (List[str]): List of numerical column names
        categorical_cols (List[str]): List of categorical column names
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Preprocessed data and fitted scaler
    """
    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    return df_encoded, scaler

def plot_elbow_curve(data: pd.DataFrame, max_k: int = 10) -> None:
    """
    Plot elbow curve for K-means clustering.
    
    Args:
        data (pd.DataFrame): Preprocessed data for clustering
        max_k (int): Maximum number of clusters to try
    """
    inertias = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def analyze_clusters(df: pd.DataFrame, 
                    cluster_labels: np.ndarray, 
                    feature_cols: List[str]) -> Dict:
    """
    Analyze characteristics of each cluster.
    
    Args:
        df (pd.DataFrame): Original dataframe
        cluster_labels (np.ndarray): Cluster assignments
        feature_cols (List[str]): Features used for clustering
        
    Returns:
        Dict: Dictionary containing cluster profiles
    """
    df['Cluster'] = cluster_labels
    cluster_profiles = {}
    
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_data = df[df['Cluster'] == cluster]
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'mean_values': cluster_data[feature_cols].mean().to_dict()
        }
        cluster_profiles[f'Cluster_{cluster}'] = profile
    
    return cluster_profiles

def plot_cluster_distributions(df: pd.DataFrame, 
                             feature: str, 
                             title: str = None) -> None:
    """
    Plot distribution of a feature across different clusters.
    
    Args:
        df (pd.DataFrame): Dataframe with cluster assignments
        feature (str): Feature to plot
        title (str, optional): Plot title
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(title or f'Distribution of {feature} across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()
