from langchain.tools import BaseTool
from pymongo import MongoClient
import json
import pandas as pd
from typing import Optional, Type, Dict, List, Any, Union, ClassVar, Literal
from pydantic import BaseModel, Field
import os

from config.settings import MONGO_URI, MONGO_DB_NAME, LARGE_DATA_THRESHOLD

class MongoDBConnectionManager:
    """Manager for MongoDB connections."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnectionManager, cls).__new__(cls)
            cls._instance.client = MongoClient(MONGO_URI)
            cls._instance.db = cls._instance.client[MONGO_DB_NAME]
        return cls._instance
    
    def get_db(self):
        """Get database instance."""
        return self.db
    
    def get_collection(self, collection_name: str):
        """Get collection by name."""
        return self.db[collection_name]
    
    def list_collections(self):
        """List all available collections."""
        return self.db.list_collection_names()
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection."""
        return self.db.command('collStats', collection_name)
    
    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()

class MongoDBQuerySchema(BaseModel):
    """Schema for MongoDB query tool."""
    collection: str = Field(..., description="Name of the MongoDB collection to query")
    query: Dict = Field({}, description="MongoDB query filter criteria")
    projection: Optional[Dict] = Field(None, description="Fields to include or exclude in the result")
    limit: int = Field(100, description="Maximum number of results to return")
    sort: Optional[List[Dict]] = Field(None, description="Sort criteria [{'field': 'asc/desc'}]")

class MongoDBQueryTool(BaseTool):
    """Tool for executing MongoDB queries for small datasets."""
    
    name: ClassVar[str] = "mongodb_query"
    description: ClassVar[str] = """
    Query a MongoDB collection based on provided filter criteria.
    Use this tool when you need to retrieve small to medium-sized datasets.
    Not suitable for collections with more than 100,000 documents - use vector database for those.
    """
    args_schema: ClassVar[Type[BaseModel]] = MongoDBQuerySchema
    
    def _run(self, collection: str, query: Dict = {}, projection: Optional[Dict] = None, 
             limit: int = 100, sort: Optional[List[Dict]] = None) -> str:
        """Execute MongoDB query and return results as a JSON string."""
        try:
            mongo_manager = MongoDBConnectionManager()
            coll = mongo_manager.get_collection(collection)
            
            # Check collection size first
            count = coll.count_documents(query)
            if count > LARGE_DATA_THRESHOLD:
                return json.dumps({
                    "error": f"Collection contains {count} documents matching query, exceeding the threshold for direct queries. Please use the vector database tools instead or refine your query further."
                })
            
            # Parse sort parameters if provided
            mongo_sort = []
            if sort:
                for sort_item in sort:
                    for field, direction in sort_item.items():
                        # Convert 'asc'/'desc' to pymongo's 1/-1
                        direction_value = 1 if direction.lower() == 'asc' else -1
                        mongo_sort.append((field, direction_value))
            
            # Execute the query
            cursor = coll.find(query, projection)
            
            # Apply sorting if specified
            if mongo_sort:
                cursor = cursor.sort(mongo_sort)
                
            # Apply limit
            cursor = cursor.limit(limit)
            
            # Convert to list of dictionaries
            results = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return json.dumps({"count": len(results), "data": results}, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class MongoDBSchemaSchema(BaseModel):
    """Schema for MongoDB collection schema tool."""
    collection: str = Field(..., description="Name of the MongoDB collection to analyze")
    sample_size: int = Field(100, description="Number of documents to sample for schema inference")

class MongoDBSchemaInferenceTool(BaseTool):
    """Tool for inferring the schema of a MongoDB collection."""
    
    name: ClassVar[str] = "mongodb_schema"
    description: ClassVar[str] = """
    Analyze the schema and structure of a MongoDB collection.
    Use this tool to understand the available fields and data types.
    """
    args_schema: ClassVar[Type[BaseModel]] = MongoDBSchemaSchema
    
    def _run(self, collection: str, sample_size: int = 100) -> str:
        """Infer the schema of a MongoDB collection."""
        try:
            mongo_manager = MongoDBConnectionManager()
            coll = mongo_manager.get_collection(collection)
            
            # Sample documents
            sample = list(coll.aggregate([{"$sample": {"size": sample_size}}]))
            
            if not sample:
                return json.dumps({"error": "Collection is empty"})
            
            # Infer schema from sample
            schema = {}
            for doc in sample:
                for key, value in doc.items():
                    if key not in schema:
                        schema[key] = {"types": set(), "sample_values": []}
                    
                    # Add type
                    value_type = type(value).__name__
                    schema[key]["types"].add(value_type)
                    
                    # Add sample value if we have fewer than 5
                    if len(schema[key]["sample_values"]) < 5:
                        if value_type not in ['dict', 'list']:
                            schema[key]["sample_values"].append(str(value))
            
            # Convert sets to lists for JSON serialization
            for key in schema:
                schema[key]["types"] = list(schema[key]["types"])
            
            # Get collection stats
            stats = mongo_manager.get_collection_stats(collection)
            
            result = {
                "collection": collection,
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "schema": schema
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class MongoDBListCollectionsSchema(BaseModel):
    """Schema for MongoDB list collections tool."""
    pass

class MongoDBListCollectionsTool(BaseTool):
    """Tool for listing all available MongoDB collections."""
    
    name: ClassVar[str] = "mongodb_list_collections"
    description: ClassVar[str] = """
    List all available collections in the MongoDB database.
    Use this tool to discover what data is available for analysis.
    """
    args_schema: ClassVar[Type[BaseModel]] = MongoDBListCollectionsSchema
    
    def _run(self) -> str:
        """List all collections in the database."""
        try:
            mongo_manager = MongoDBConnectionManager()
            collections = mongo_manager.list_collections()
            
            return json.dumps({"collections": collections})
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class LLMQuerySchema(BaseModel):
    """Schema for LLM-based query generation tool."""
    prompt: str = Field(..., description="Natural language prompt describing the data to query")
    collection: str = Field(..., description="Name of the MongoDB collection to query")
    max_results: int = Field(100, description="Maximum number of results to return")

class LLMQueryTool(BaseTool):
    """Tool for generating and executing MongoDB queries based on natural language prompts."""
    
    name: ClassVar[str] = "llm_query"
    description: ClassVar[str] = """
    Generate and execute MongoDB queries based on natural language prompts.
    Use this tool when you need to convert user requests into database queries.
    """
    args_schema: ClassVar[Type[BaseModel]] = LLMQuerySchema
    
    def _parse_python_code_to_query(self, python_code: str) -> Dict:
        """Parse Python code to extract MongoDB query parameters.
        
        Handles various MongoDB query patterns in Python code including:
        - find() with conditions
        - sort() with single or multiple fields
        - limit()
        - projection
        - skip()
        """
        query_dict = {
            "query": {},
            "projection": None,
            "sort": [],
            "limit": 40  # Default to max_results if not specified
        }
        
        # Extract find conditions if present
        if "find(" in python_code:
            find_part = python_code.split("find(")[1].split(")")[0]
            if find_part.strip() and find_part.strip() != "{}":
                try:
                    # Handle both string and dict formats
                    if find_part.startswith("{"):
                        # Convert Python dict syntax to JSON
                        find_part = find_part.replace("'", '"')
                        query_dict["query"] = json.loads(find_part)
                    else:
                        # Handle string conditions
                        conditions = find_part.strip("'").split(",")
                        for condition in conditions:
                            if "=" in condition:
                                field, value = condition.split("=")
                                query_dict["query"][field.strip()] = value.strip()
                except Exception as e:
                    print(f"Warning: Could not parse find conditions: {e}")
        
        # Extract projection if present
        if "projection=" in python_code:
            proj_part = python_code.split("projection=")[1].split(",")[0]
            try:
                if proj_part.startswith("{"):
                    proj_part = proj_part.replace("'", '"')
                    query_dict["projection"] = json.loads(proj_part)
                else:
                    # Handle simple field inclusion/exclusion
                    fields = proj_part.strip("{}").split(",")
                    query_dict["projection"] = {}
                    for field in fields:
                        field = field.strip()
                        if field.startswith("-"):
                            query_dict["projection"][field[1:]] = 0
                        else:
                            query_dict["projection"][field] = 1
            except Exception as e:
                print(f"Warning: Could not parse projection: {e}")
        
        # Extract sort fields and directions
        if "sort(" in python_code:
            sort_parts = python_code.split("sort(")[1].split(")")[0]
            if sort_parts.startswith("[("):
                # Handle multiple sort fields
                sort_fields = sort_parts.strip("[]").split("),(")
                for sort_field in sort_fields:
                    if "," in sort_field:
                        field, direction = sort_field.split(",")
                        field = field.strip("'")
                        direction = -1 if "-1" in direction else 1
                        query_dict["sort"].append({
                            field: "desc" if direction == -1 else "asc"
                        })
            else:
                # Handle single sort field
                field, direction = sort_parts.split(",")
                field = field.strip("'")
                direction = -1 if "-1" in direction else 1
                query_dict["sort"].append({
                    field: "desc" if direction == -1 else "asc"
                })
        
        # Extract limit
        if "limit(" in python_code:
            try:
                limit = int(python_code.split("limit(")[1].split(")")[0])
                query_dict["limit"] = limit
            except Exception as e:
                print(f"Warning: Could not parse limit: {e}")
        
        # Extract skip
        if "skip(" in python_code:
            try:
                skip = int(python_code.split("skip(")[1].split(")")[0])
                query_dict["skip"] = skip
            except Exception as e:
                print(f"Warning: Could not parse skip: {e}")
        
        # Clean up empty values
        if not query_dict["query"]:
            query_dict["query"] = {}
        if not query_dict["sort"]:
            query_dict["sort"] = None
        
        return query_dict
    
    def _run(self, prompt: str, collection: str, max_results: int = 100) -> str:
        """Generate and execute a MongoDB query based on the prompt."""
        try:
            # First get the schema of the collection
            schema_tool = MongoDBSchemaInferenceTool()
            schema_result = schema_tool._run(collection=collection)
            schema_data = json.loads(schema_result)
            if "error" in schema_data:
                return json.dumps({"error": f"Failed to get schema: {schema_data['error']}"})
            
            # Create a prompt for the LLM to generate the query
            query_prompt = f"""Given the following MongoDB collection schema and user request, generate a MongoDB query.
            
Collection Schema:
{json.dumps(schema_data['schema'], indent=2)}

User Request:
{prompt}

Generate a MongoDB query that will retrieve the relevant data. The query should be in JSON format with the following structure:
{{
    "query": {{}},  // MongoDB query filter
    "projection": {{}},  // Fields to include (1) or exclude (0). For example: {{"title": 1, "year": 1, "vote_count": 1}}
    "sort": []  // Sort criteria as [{{"field": "asc/desc"}}]
}}

IMPORTANT: Always include a projection to specify exactly which fields to return. Do not return all fields.
Only include the JSON query, no other text."""

            # Use the LLM to generate the query
            from llm.ollama_client import get_ollama_llm
            llm = get_ollama_llm()
            query_json = llm.invoke(query_prompt)
            print("Query JSON:")
            print(query_json)

            try:
                # Try to parse as JSON first
                try:
                    query_dict = json.loads(query_json)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to parse as Python code
                    print("Failed to parse as JSON, attempting to parse as Python code...")
                    query_dict = self._parse_python_code_to_query(query_json)
                
                # Ensure projection is set correctly
                if not query_dict.get("projection"):
                    # Extract fields from prompt
                    fields = []
                    if "title" in prompt.lower():
                        fields.append("Movie_Name")
                    if "year" in prompt.lower():
                        fields.append("Release_Date")
                    if "vote" in prompt.lower():
                        fields.append("Vote_Count")
                    if "genre" in prompt.lower():
                        fields.append("Genres")
                    
                    # Create projection
                    if fields:
                        query_dict["projection"] = {field: 1 for field in fields}
                        query_dict["projection"]["_id"] = 0  # Exclude _id by default
                
                # Execute the query using MongoDBQueryTool
                query_tool = MongoDBQueryTool()
                result = query_tool._run(
                    collection=collection,
                    query=query_dict.get("query", {}),
                    projection=query_dict.get("projection"),
                    limit=max_results,
                    sort=query_dict.get("sort")
                )
                
                return result
                
            except Exception as e:
                return json.dumps({
                    "error": "Failed to generate valid query",
                    "llm_output": query_json,
                    "parse_error": str(e)
                })
            
        except Exception as e:
            return json.dumps({"error": str(e)})

def get_mongodb_tools():
    """Get all MongoDB tools."""
    return [
        MongoDBQueryTool(),
        MongoDBSchemaInferenceTool(),
        MongoDBListCollectionsTool(),
        LLMQueryTool()  # Add the new LLM query tool
    ]