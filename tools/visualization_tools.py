from langchain.tools import BaseTool
import json
import pandas as pd
import numpy as np
from typing import Optional, Type, Dict, List, Any, Union, ClassVar
from pydantic import BaseModel, Field
from config.settings import MAX_POINTS_PER_CHART, MAX_CATEGORIES_PIE_CHART

class VisualizationDataSchema(BaseModel):
    """Schema for visualization data preparation tool."""
    data: List[Dict[str, Any]] = Field(..., description="Data records to visualize")
    chart_type: str = Field(..., description="Type of chart to prepare data for (bar, line, scatter, pie, heatmap)")
    x_column: Optional[str] = Field(None, description="Column to use for X-axis or categories")
    y_column: Optional[str] = Field(None, description="Column to use for Y-axis or values")
    group_by: Optional[str] = Field(None, description="Column to group data by (for multi-series charts)")
    aggregation: Optional[str] = Field("mean", description="Aggregation function (mean, sum, count, min, max)")
    limit: Optional[int] = Field(None, description="Limit the number of data points")

class VisualizationDataTool(BaseTool):
    """Tool for preparing data for visualization."""
    
    name: ClassVar[str] = "prepare_visualization"
    description: ClassVar[str] = """
    Prepare data for visualization by structuring it in the format needed for charts.
    Handles data aggregation, filtering, and formatting for different chart types.
    """
    args_schema: ClassVar[Type[BaseModel]] = VisualizationDataSchema
    
    def _run(self, data: List[Dict[str, Any]], chart_type: str, 
             x_column: Optional[str] = None, y_column: Optional[str] = None,
             group_by: Optional[str] = None, aggregation: str = "mean",
             limit: Optional[int] = None) -> str:
        """Prepare data for visualization."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Apply limit if needed
            if limit and limit > 0:
                df = df.head(limit)
            
            # Initialize result
            result = {
                "chart_type": chart_type,
                "data_points": len(df),
                "visualization_data": {}
            }
            
            # Process based on chart type
            if chart_type.lower() == "bar":
                result["visualization_data"] = self._prepare_bar_chart_data(
                    df, x_column, y_column, group_by, aggregation
                )
                
            elif chart_type.lower() == "line":
                result["visualization_data"] = self._prepare_line_chart_data(
                    df, x_column, y_column, group_by, aggregation
                )
                
            elif chart_type.lower() == "scatter":
                result["visualization_data"] = self._prepare_scatter_chart_data(
                    df, x_column, y_column, group_by
                )
                
            elif chart_type.lower() == "pie":
                result["visualization_data"] = self._prepare_pie_chart_data(
                    df, x_column, y_column, aggregation
                )
                
            elif chart_type.lower() == "heatmap":
                result["visualization_data"] = self._prepare_heatmap_data(
                    df, x_column, y_column, group_by, aggregation
                )
                
            else:
                return json.dumps({"error": f"Unsupported chart type: {chart_type}"})
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _prepare_bar_chart_data(self, df, x_column, y_column, group_by, aggregation):
        """Prepare data for bar chart visualization."""
        if not x_column or not y_column:
            return {"error": "Both x_column and y_column are required for bar charts"}
        
        # Handle aggregation function
        agg_func = self._get_aggregation_function(aggregation)
        
        if group_by:
            # Create multi-series bar chart data
            pivot_df = df.pivot_table(
                index=x_column, 
                columns=group_by, 
                values=y_column,
                aggfunc=agg_func
            ).reset_index()
            
            # Convert to needed format
            categories = pivot_df[x_column].tolist()
            series_names = [col for col in pivot_df.columns if col != x_column]
            series_data = []
            
            for series in series_names:
                series_data.append({
                    "name": str(series),
                    "data": pivot_df[series].tolist()
                })
            
            return {
                "categories": categories,
                "series": series_data
            }
        else:
            # Create single-series bar chart data
            grouped = df.groupby(x_column)[y_column].agg(agg_func).reset_index()
            
            # Limit categories if needed
            if len(grouped) > MAX_CATEGORIES_PIE_CHART:
                # Sort by value and keep top categories
                grouped = grouped.sort_values(y_column, ascending=False).head(MAX_CATEGORIES_PIE_CHART)
                
            return {
                "categories": grouped[x_column].tolist(),
                "series": [{
                    "name": y_column,
                    "data": grouped[y_column].tolist()
                }]
            }
    
    def _prepare_line_chart_data(self, df, x_column, y_column, group_by, aggregation):
        """Prepare data for line chart visualization."""
        if not x_column or not y_column:
            return {"error": "Both x_column and y_column are required for line charts"}
        
        # Handle aggregation function
        agg_func = self._get_aggregation_function(aggregation)
        
        # Sort by x column
        df = df.sort_values(x_column)
        
        # Sample if too many points
        if len(df) > MAX_POINTS_PER_CHART:
            sample_size = MAX_POINTS_PER_CHART
            df = df.iloc[np.linspace(0, len(df) - 1, sample_size, dtype=int)]
        
        if group_by:
            # Create multi-series line chart data
            series_data = []
            
            for name, group in df.groupby(group_by):
                series_data.append({
                    "name": str(name),
                    "data": [
                        {"x": str(x), "y": float(y)} 
                        for x, y in zip(group[x_column], group[y_column])
                    ]
                })
            
            return {"series": series_data}
        else:
            # Create single-series line chart data
            return {
                "xAxis": df[x_column].tolist(),
                "series": [{
                    "name": y_column,
                    "data": df[y_column].tolist()
                }]
            }
    
    def _prepare_scatter_chart_data(self, df, x_column, y_column, group_by):
        """Prepare data for scatter chart visualization."""
        if not x_column or not y_column:
            return {"error": "Both x_column and y_column are required for scatter charts"}
        
        # Sample if too many points
        if len(df) > MAX_POINTS_PER_CHART:
            sample_size = MAX_POINTS_PER_CHART
            df = df.sample(sample_size, random_state=42)
        
        if group_by:
            # Create multi-series scatter chart data
            series_data = []
            
            for name, group in df.groupby(group_by):
                series_data.append({
                    "name": str(name),
                    "data": [
                        {"x": float(x), "y": float(y)} 
                        for x, y in zip(group[x_column], group[y_column])
                    ]
                })
            
            return {"series": series_data}
        else:
            # Create single-series scatter chart data
            return {
                "series": [{
                    "name": f"{x_column} vs {y_column}",
                    "data": [
                        {"x": float(x), "y": float(y)} 
                        for x, y in zip(df[x_column], df[y_column])
                    ]
                }]
            }
    
    def _prepare_pie_chart_data(self, df, x_column, y_column, aggregation):
        """Prepare data for pie chart visualization."""
        if not x_column or not y_column:
            return {"error": "Both x_column and y_column are required for pie charts"}
        
        # Handle aggregation function
        agg_func = self._get_aggregation_function(aggregation)
        
        # Group data
        grouped = df.groupby(x_column)[y_column].agg(agg_func).reset_index()
        
        # Limit categories if needed
        if len(grouped) > MAX_CATEGORIES_PIE_CHART:
            # Keep top categories and group others
            top = grouped.sort_values(y_column, ascending=False).head(MAX_CATEGORIES_PIE_CHART - 1)
            other_sum = grouped.sort_values(y_column, ascending=False).iloc[MAX_CATEGORIES_PIE_CHART - 1:][y_column].sum()
            
            # Add "Other" category
            other_row = pd.DataFrame({x_column: ["Other"], y_column: [other_sum]})
            grouped = pd.concat([top, other_row])
        
        # Format for pie chart
        return {
            "series": [{
                "name": y_column,
                "data": [
                    {"name": str(name), "value": float(value)}
                    for name, value in zip(grouped[x_column], grouped[y_column])
                ]
            }]
        }
    
    def _prepare_heatmap_data(self, df, x_column, y_column, value_column, aggregation):
        """Prepare data for heatmap visualization."""
        # For heatmaps, parameters work differently:
        # x_column: Column for x-axis categories
        # y_column: Column for y-axis categories
        # value_column (reusing group_by param): Column for values (colors)
        
        if not x_column or not y_column or not value_column:
            return {"error": "x_column, y_column, and value_column (passed as group_by) are required for heatmaps"}
        
        # Handle aggregation function
        agg_func = self._get_aggregation_function(aggregation)
        
        # Create pivot table
        pivot_df = df.pivot_table(
            index=y_column, 
            columns=x_column, 
            values=value_column,
            aggfunc=agg_func
        )
        
        # Limit dimensions if needed
        if pivot_df.shape[0] > MAX_CATEGORIES_PIE_CHART:
            pivot_df = pivot_df.iloc[:MAX_CATEGORIES_PIE_CHART, :]
            
        if pivot_df.shape[1] > MAX_CATEGORIES_PIE_CHART:
            pivot_df = pivot_df.iloc[:, :MAX_CATEGORIES_PIE_CHART]
        
        # Format for heatmap
        return {
            "xAxis": pivot_df.columns.tolist(),
            "yAxis": pivot_df.index.tolist(),
            "data": [
                [i, j, float(value) if not pd.isna(value) else None]
                for i, col in enumerate(pivot_df.columns)
                for j, idx in enumerate(pivot_df.index)
                for value in [pivot_df.loc[idx, col]]
            ]
        }
    
    def _get_aggregation_function(self, aggregation):
        """Convert aggregation string to pandas function."""
        agg_map = {
            "mean": "mean",
            "sum": "sum",
            "count": "count",
            "min": "min",
            "max": "max"
        }
        return agg_map.get(aggregation.lower(), "mean")

def get_visualization_tools():
    """Get all visualization tools."""
    return [
        VisualizationDataTool()
    ]