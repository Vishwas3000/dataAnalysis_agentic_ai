from langchain.tools import BaseTool
import json
import pandas as pd
import numpy as np
from typing import Optional, Type, Dict, List, Any, Union, ClassVar
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import StringIO

class DataFrameSchema(BaseModel):
    """Schema for pandas DataFrame JSON representation."""
    data: List[Dict[str, Any]] = Field(..., description="List of data records")

class StatisticalAnalysisSchema(BaseModel):
    """Schema for statistical analysis tool."""
    data: List[Dict[str, Any]] = Field(..., description="Data records to analyze")
    numeric_columns: List[str] = Field(..., description="List of numeric columns to analyze")
    categorical_columns: Optional[List[str]] = Field(None, description="List of categorical columns to analyze")

class StatisticalAnalysisTool(BaseTool):
    """Tool for performing statistical analysis on data."""
    
    name: ClassVar[str] = "statistical_analysis"
    description: ClassVar[str] = """
    Perform statistical analysis on tabular data.
    Calculates descriptive statistics, correlations, and distributions for numeric columns.
    For categorical columns, calculates frequency counts and proportions.
    """
    args_schema: ClassVar[Type[BaseModel]] = StatisticalAnalysisSchema
    
    def _run(self, data: List[Dict[str, Any]], numeric_columns: List[str], 
             categorical_columns: Optional[List[str]] = None) -> str:
        """Perform statistical analysis on data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            result = {"analysis": {}}
            
            # Basic info
            result["column_info"] = {
                "total_columns": len(df.columns),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "row_count": len(df)
            }
            
            # Missing data analysis
            missing_data = df[numeric_columns + (categorical_columns or [])].isnull().sum()
            result["missing_data"] = {col: int(count) for col, count in missing_data.items() if count > 0}
            
            # Numeric analysis
            if numeric_columns:
                # Filter just numeric columns
                numeric_df = df[numeric_columns]
                
                # Basic stats
                stats = numeric_df.describe().to_dict()
                result["analysis"]["numeric"] = stats
                
                # Additional metrics
                result["analysis"]["skewness"] = {col: float(numeric_df[col].skew()) 
                                              for col in numeric_columns if pd.api.types.is_numeric_dtype(numeric_df[col])}
                result["analysis"]["kurtosis"] = {col: float(numeric_df[col].kurt()) 
                                               for col in numeric_columns if pd.api.types.is_numeric_dtype(numeric_df[col])}
                
                # Correlation matrix
                if len(numeric_columns) > 1:
                    corr_matrix = numeric_df.corr().to_dict()
                    # Convert NaN to None for JSON serialization
                    for col in corr_matrix:
                        for subcol in corr_matrix[col]:
                            if pd.isna(corr_matrix[col][subcol]):
                                corr_matrix[col][subcol] = None
                    result["analysis"]["correlation_matrix"] = corr_matrix
            
            # Categorical analysis
            if categorical_columns:
                result["analysis"]["categorical"] = {}
                for col in categorical_columns:
                    if col in df.columns:
                        # Get value counts and convert to dictionaries
                        value_counts = df[col].value_counts()
                        proportions = df[col].value_counts(normalize=True)
                        
                        # Convert to dictionary with counts and percentages
                        category_stats = {
                            "counts": {str(k): int(v) for k, v in value_counts.items()},
                            "percentages": {str(k): float(v) for k, v in proportions.items()},
                            "unique_values": int(len(value_counts))
                        }
                        result["analysis"]["categorical"][col] = category_stats
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class ClusteringAnalysisSchema(BaseModel):
    """Schema for clustering analysis tool."""
    data: List[Dict[str, Any]] = Field(..., description="Data records to analyze")
    columns: List[str] = Field(..., description="Columns to use for clustering")
    n_clusters: int = Field(3, description="Number of clusters to generate")

class ClusteringAnalysisTool(BaseTool):
    """Tool for performing clustering analysis on data."""
    
    name: ClassVar[str] = "clustering_analysis"
    description: ClassVar[str] = """
    Perform clustering analysis on numeric data using K-means algorithm.
    Identifies natural groupings in the data and returns cluster assignments.
    """
    args_schema: ClassVar[Type[BaseModel]] = ClusteringAnalysisSchema
    
    def _run(self, data: List[Dict[str, Any]], columns: List[str], n_clusters: int = 3) -> str:
        """Perform clustering analysis on data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Select features for clustering
            features = df[columns].copy()
            
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Get cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Convert centers back to original scale
            original_centers = scaler.inverse_transform(cluster_centers)
            
            # Prepare cluster profile
            cluster_profiles = {}
            for i in range(n_clusters):
                cluster_df = df[df['cluster'] == i]
                profile = {
                    "size": len(cluster_df),
                    "percentage": float(len(cluster_df) / len(df) * 100),
                    "center": {columns[j]: float(original_centers[i][j]) for j in range(len(columns))}
                }
                
                # Get statistics for each feature within this cluster
                stats = {}
                for col in columns:
                    stats[col] = {
                        "mean": float(cluster_df[col].mean()),
                        "min": float(cluster_df[col].min()),
                        "max": float(cluster_df[col].max()),
                        "std": float(cluster_df[col].std())
                    }
                profile["stats"] = stats
                
                cluster_profiles[f"cluster_{i}"] = profile
            
            # Prepare result
            result = {
                "n_clusters": n_clusters,
                "features_used": columns,
                "cluster_sizes": df['cluster'].value_counts().to_dict(),
                "cluster_profiles": cluster_profiles,
                "cluster_assignments": df['cluster'].to_list()
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class TimeSeriesAnalysisSchema(BaseModel):
    """Schema for time series analysis tool."""
    data: List[Dict[str, Any]] = Field(..., description="Data records to analyze")
    date_column: str = Field(..., description="Column containing date/time values")
    value_column: str = Field(..., description="Column containing the values to analyze")
    freq: str = Field("D", description="Frequency for resampling (D=daily, W=weekly, M=monthly, etc.)")
    
class TimeSeriesAnalysisTool(BaseTool):
    """Tool for performing time series analysis on data."""
    
    name: ClassVar[str] = "time_series_analysis"
    description: ClassVar[str] = """
    Perform time series analysis on temporal data.
    Identifies trends, seasonality, and provides forecasting metrics.
    """
    args_schema: ClassVar[Type[BaseModel]] = TimeSeriesAnalysisSchema
    
    def _run(self, data: List[Dict[str, Any]], date_column: str, value_column: str, freq: str = "D") -> str:
        """Perform time series analysis on data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            # Set date as index
            df = df.set_index(date_column)
            
            # Check if we have enough data
            if len(df) < 3:
                return json.dumps({"error": "Not enough data points for time series analysis"})
            
            # Resample to specified frequency
            resampled = df[value_column].resample(freq).mean()
            
            # Handle missing values after resampling
            resampled = resampled.fillna(method='ffill')
            
            # Basic statistics
            result = {
                "original_records": len(df),
                "resampled_records": len(resampled),
                "frequency": freq,
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d"),
                    "end": df.index.max().strftime("%Y-%m-%d")
                },
                "statistics": {
                    "mean": float(resampled.mean()),
                    "min": float(resampled.min()),
                    "max": float(resampled.max()),
                    "std": float(resampled.std())
                }
            }
            
            # Calculate rolling metrics
            rolling_mean = resampled.rolling(window=3).mean()
            rolling_std = resampled.rolling(window=3).std()
            
            # Trend analysis
            result["trend"] = {
                "start_value": float(resampled.iloc[0]),
                "end_value": float(resampled.iloc[-1]),
                "overall_change": float(resampled.iloc[-1] - resampled.iloc[0]),
                "percent_change": float((resampled.iloc[-1] / resampled.iloc[0] - 1) * 100)
            }
            
            # Generate key data points for visualization
            time_points = [d.strftime("%Y-%m-%d") for d in resampled.index]
            data_points = [float(v) for v in resampled.values]
            trend_points = [float(v) if not pd.isna(v) else None for v in rolling_mean.values]
            
            result["visualization_data"] = {
                "time_points": time_points,
                "values": data_points,
                "trend": trend_points
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

def get_analysis_tools():
    """Get all data analysis tools."""
    return [
        StatisticalAnalysisTool(),
        ClusteringAnalysisTool(),
        TimeSeriesAnalysisTool()
    ]