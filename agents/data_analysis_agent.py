from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from typing import List, Dict, Any, Optional
import json

from llm.ollama_client import get_ollama_llm
from memory.conversation_memory import MemoryManager
from tools.mongodb_tools import get_mongodb_tools
from tools.vectordb_tools import get_vectordb_tools
from tools.analysis_tools import get_analysis_tools
from tools.visualization_tools import get_visualization_tools
from agents.prompts import (
    SYSTEM_MESSAGE,
    AGENT_PROMPT,
    DATA_EXPLORATION_PROMPT,
    INSIGHT_GENERATION_PROMPT,
    VISUALIZATION_SUGGESTION_PROMPT
)
from config.settings import AGENT_MAX_ITERATIONS, INPUT_KEY, OUTPUT_KEY, MEMORY_KEY

class DataAnalysisAgent:
    """Agent for data analysis and insight generation.
    
    This agent combines MongoDB queries, vector database retrieval,
    statistical analysis, and visualization preparation to provide
    comprehensive data insights."""
    
    def __init__(self, memory_type="buffer_window", streaming=False):
        """Initialize the data analysis agent.""""""
        
        Args:
            memory_type: Type of memory to use (buffer, buffer_window, vector)
            streaming: Whether to use streaming LLM responses
        """
        self.streaming = streaming
        self.llm = get_ollama_llm(streaming=streaming)
        
        # Initialize memory
        self.memory_manager = MemoryManager(memory_type=memory_type)
        self.memory = self.memory_manager.get_memory()
        
        # Initialize tools
        self.tools = self._get_all_tools()
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # Initialize additional chains
        self.exploration_chain = LLMChain(
            llm=self.llm,
            prompt=DATA_EXPLORATION_PROMPT
        )
        
        self.insight_chain = LLMChain(
            llm=self.llm,
            prompt=INSIGHT_GENERATION_PROMPT
        )
        
        self.visualization_chain = LLMChain(
            llm=self.llm,
            prompt=VISUALIZATION_SUGGESTION_PROMPT
        )
    
    def _get_all_tools(self):
        """Get all available tools for the agent."""
        mongodb_tools = get_mongodb_tools()
        vectordb_tools = get_vectordb_tools()
        analysis_tools = get_analysis_tools()
        visualization_tools = get_visualization_tools()
        
        return mongodb_tools + vectordb_tools + analysis_tools + visualization_tools
    
    def _initialize_agent(self):
        """Initialize the agent with tools and memory."""
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=AGENT_PROMPT.partial(system_message=SYSTEM_MESSAGE)
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            max_iterations=AGENT_MAX_ITERATIONS,
            early_stopping_method="generate",
            verbose=True
        )
    
    async def analyze_data(self, query: str):
        """Perform data analysis based on user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict containing analysis results, insights, and visualization data
        """
        # Step 1: Run the agent to analyze the data
        agent_result = await self.agent.ainvoke({INPUT_KEY: query})
        
        # Extract the agent's output
        agent_output = agent_result[OUTPUT_KEY]
        
        # Step 2: Generate additional insights if needed
        insight_result = await self.insight_chain.ainvoke({
            "analysis_results": agent_output,
            "user_request": query
        })
        
        # Step 3: Generate visualization suggestions
        viz_result = await self.visualization_chain.ainvoke({
            "analysis_results": agent_output,
            "user_request": query
        })
        
        # Combine results
        result = {
            "query": query,
            "analysis": agent_output,
            "insights": insight_result["text"],
            "visualization_suggestions": viz_result["text"],
            "raw_agent_result": agent_result
        }
        
        # Save context
        self.memory_manager.save_context(
            {INPUT_KEY: query},
            {OUTPUT_KEY: f"Analysis: {agent_output}\n\nInsights: {insight_result['text']}"}
        )
        
        return result
    
    def analyze_data_sync(self, query: str):
        """Synchronous version of analyze_data.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict containing analysis results, insights, and visualization data
        """
        # Step 1: Run the agent to analyze the data
        agent_result = self.agent.invoke({INPUT_KEY: query})
        
        # Extract the agent's output
        agent_output = agent_result[OUTPUT_KEY]
        
        # Step 2: Generate additional insights if needed
        insight_result = self.insight_chain.invoke({
            "analysis_results": agent_output,
            "user_request": query
        })
        
        # Step 3: Generate visualization suggestions
        viz_result = self.visualization_chain.invoke({
            "analysis_results": agent_output,
            "user_request": query
        })
        
        # Combine results
        result = {
            "query": query,
            "analysis": agent_output,
            "insights": insight_result["text"],
            "visualization_suggestions": viz_result["text"],
            "raw_agent_result": agent_result
        }
        
        # Save context
        self.memory_manager.save_context(
            {INPUT_KEY: query},
            {OUTPUT_KEY: f"Analysis: {agent_output}\n\nInsights: {insight_result['text']}"}
        )
        
        return result
    
    def explore_data_collections(self):
        """Explore available data collections.
        
        Returns:
            Dict containing discovered collections and their schemas
        """
        # Use the MongoDB list collections tool
        list_tool = next(tool for tool in self.tools if tool.name == "mongodb_list_collections")
        collections_result = list_tool._run()
        collections_data = json.loads(collections_result)
        
        # If there are collections, get schema for each
        result = {"collections": []}
        
        if "collections" in collections_data and collections_data["collections"]:
            schema_tool = next(tool for tool in self.tools if tool.name == "mongodb_schema")
            
            for collection in collections_data["collections"]:
                schema_result = schema_tool._run(collection=collection, sample_size=100)
                schema_data = json.loads(schema_result)
                
                result["collections"].append({
                    "name": collection,
                    "schema": schema_data
                })
        
        # Generate exploration insights
        if result["collections"]:
            exploration_result = self.exploration_chain.invoke({
                "schema_info": json.dumps(result, indent=2)
            })
            result["exploration_insights"] = exploration_result["text"]
        
        return result
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory_manager.clear()