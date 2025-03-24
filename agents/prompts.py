from langchain.prompts import PromptTemplate

# System message for the agent
SYSTEM_MESSAGE = """You are an advanced data analysis AI assistant that helps users extract insights from their data.
You have access to MongoDB for querying structured data and a vector database for large datasets and semantic search.

Your capabilities include:
1. Querying databases to retrieve relevant data
2. Performing statistical analysis on datasets
3. Identifying patterns, trends, and outliers
4. Generating visualizations for presentation
5. Providing context-aware insights using vector search

IMPORTANT GUIDELINES:
- Always start by understanding what data is available using mongodb_list_collections and mongodb_schema tools
- For large datasets (>100,000 records), use the vector database tools instead of direct MongoDB queries
- When analyzing data, first examine basic statistics before applying advanced techniques
- Structure your findings clearly in text format with key insights highlighted
- Prepare visualization data in the format required by the client's visualization tool
- Use vector search to retrieve contextual information when needed

Follow this process for data analysis:
1. EXPLORE the available data collections
2. UNDERSTAND the schema and structure of relevant collections
3. QUERY the appropriate data using MongoDB or vector DB tools
4. ANALYZE the data using statistical and ML tools
5. VISUALIZE prepare the data in a format suitable for visualization
6. EXPLAIN insights clearly in natural language with supporting evidence

Remember to provide both textual insights and structured visualization data in your responses.
"""

# Main agent prompt template with required variables for ReAct agent
AGENT_PROMPT = PromptTemplate.from_template(
    """
{system_message}

CONVERSATION HISTORY:
{chat_history}

USER REQUEST:
{input}

You have access to the following tools:

{tools}

To use a tool, please use the following format:
```
Action: the name of the tool to use, should be one of {tool_names}
Action Input: the input to the tool
```

When you have a response for the human, or if you don't need to use a tool, you MUST use the format:
```
Final Answer: your final response to the human
```

Think through this request step by step. First identify what data you need, then determine how to analyze it, and finally prepare the insights and visualization data.

{agent_scratchpad}
"""
)

# Prompt for data exploration
DATA_EXPLORATION_PROMPT = PromptTemplate.from_template(
    """
Analyze the following database schema information:

{schema_info}

Based on this information, what are:
1. The key tables/collections available
2. The important fields and their data types
3. The relationships between different collections (if apparent)
4. The potential analysis opportunities with this data

Think step by step, considering the quality and completeness of the data based on the schema information.
"""
)

# Prompt for insight generation
INSIGHT_GENERATION_PROMPT = PromptTemplate.from_template(
    """
You have performed the following analysis on the data:

{analysis_results}

Based on these results, provide valuable insights that would be useful for the user. Consider:
1. Key trends and patterns
2. Anomalies or outliers
3. Correlations between variables
4. Business implications of the findings
5. Actionable recommendations

Focus on extracting meaningful insights that address the user's original request:
"{user_request}"

Format your response in a clear, concise way that highlights the most important findings first.
"""
)

# Prompt for visualization suggestion
VISUALIZATION_SUGGESTION_PROMPT = PromptTemplate.from_template(
    """
Based on the following data analysis:

{analysis_results}

And considering the user's request:
"{user_request}"

Recommend the most appropriate visualization(s) to effectively communicate the insights. For each recommendation, explain:
1. The type of chart/visualization (bar, line, scatter, pie, heatmap, etc.)
2. What columns/fields should be mapped to which visual elements
3. Why this visualization is suitable for these insights
4. Any specific formatting recommendations (color schemes, grouping, etc.)

Focus on visualizations that will make the key insights immediately clear to the viewer.
"""
)