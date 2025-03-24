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

def get_mongodb_tools():
    """Get all MongoDB tools."""
    return [
        MongoDBQueryTool(),
        MongoDBSchemaInferenceTool(),
        MongoDBListCollectionsTool()
    ]